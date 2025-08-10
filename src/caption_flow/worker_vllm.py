"""Worker node with vLLM integration for distributed captioning."""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "64"

import asyncio
import io
import json
import logging
import ssl
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread

import websockets
from websockets.client import WebSocketClientProtocol
from PIL import Image
import numpy as np

from .models import Job, JobStatus
from .utils import CaptionUtils
from .worker import Worker

logger = logging.getLogger(__name__)

# Constants from original script
MAX_MODEL_LEN = 16384
LIMIT_IMAGES_PER_PROMPT = 1

@dataclass
class InferenceConfig:
    """Configuration for vLLM inference."""
    gpu_id: int = 0
    precision: str = "fp16"
    batch_size: int = 8
    coalesce_ms: int = 30
    max_retries: int = 3
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    temperature: float = 0.7

class VLLMWorker(Worker):
    """Worker node that processes captioning jobs using vLLM."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")
        
        # vLLM configuration
        self.inference_config = InferenceConfig(
            gpu_id=config.get("gpu_id", 0),
            precision=config.get("precision", "fp16"),
            batch_size=config.get("batch_size", 8),
            coalesce_ms=config.get("coalesce_ms", 30),
            max_retries=config.get("max_retries", 3),
            model_name=config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
            temperature=config.get("temperature", 0.7)
        )
        
        # SSL configuration
        self.ssl_context = self._setup_ssl()
        
        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.running = False
        
        # Inference components (initialized in setup)
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None
        
        # Batching queue for inference
        self.inference_queue = Queue(maxsize=256)
        self.result_queue = Queue()
        
        # Metrics
        self.processed_count = 0
        self.error_count = 0
        
    def _setup_vllm(self):
        """Initialize vLLM components."""
        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.inference_config.gpu_id)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoProcessor
        
        logger.info(f"Loading model {self.inference_config.model_name} on GPU {self.inference_config.gpu_id}")
        
        # Initialize tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.inference_config.model_name, 
            trust_remote_code=True, 
            use_fast=True
        )
        self.processor = AutoProcessor.from_pretrained(self.inference_config.model_name)
        
        # Configure LLM based on precision
        dtype = "float16" if self.inference_config.precision == "fp16" else "bfloat16"
        
        if self.inference_config.precision == "awq":
            self.llm = LLM(
                model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=MAX_MODEL_LEN,
                enforce_eager=True,
                quantization="awq",
                gpu_memory_utilization=0.95,
                dtype="float16",
                limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
                disable_mm_preprocessor_cache=True,
            )
        elif self.inference_config.precision == "fp8":
            self.llm = LLM(
                model=self.inference_config.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=MAX_MODEL_LEN,
                enforce_eager=True,
                enable_chunked_prefill=True,
                gpu_memory_utilization=0.92,
                dtype="bfloat16",
                quantization="fp8",
                limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
                disable_mm_preprocessor_cache=True,
            )
        else:
            self.llm = LLM(
                model=self.inference_config.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_num_seqs=512,
                max_num_batched_tokens=MAX_MODEL_LEN,
                max_model_len=MAX_MODEL_LEN,
                enforce_eager=True,
                enable_chunked_prefill=False,
                gpu_memory_utilization=0.92,
                dtype=dtype,
                limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
                disable_mm_preprocessor_cache=True,
            )
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.inference_config.temperature,
            top_p=0.95,
            max_tokens=256,
            stop=[
                "<|end|>", "<|endoftext|>", "<|im_end|>", 
                "<|end_of_text|>", "|assistant|", "<|assistant_end|>"
            ],
            stop_token_ids=[151643, 151645],
            repetition_penalty=1.05,
            frequency_penalty=0.1,
            skip_special_tokens=True,
        )
        
        logger.info("vLLM initialization complete")
    
    def _build_inputs(self, pil_img: Image.Image, question: str) -> Dict[str, Any]:
        """Build input for vLLM following Qwen format."""
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            
            return {
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": image_inputs},
            }
        except ImportError:
            # Fallback if qwen_vl_utils not available
            return {
                "prompt": f"<|user|>\n<|image_pad|>\n{question}<|end|>\n<|assistant|>",
                "multi_modal_data": {"image": [pil_img]},
            }
    
    def _clean_output(self, text: str) -> str:
        """Clean any remaining tokens that slipped through."""
        if not text:
            return ""
        
        end_tokens = [
            "<|end|>", "<|endoftext|>", "<|im_end|>", "<|user|>", 
            "<|assistant|>", "<|assistant>", "|", ".<", "</",
            "I'm sorry", "The content is inappropriate", "I apologize",
            "The image does not", "If you"
        ]
        
        cleaned = text
        for token in end_tokens:
            if token in cleaned:
                cleaned = cleaned.split(token)[0]
        
        if "I'm sorry" in cleaned or "I cannot" in cleaned:
            parts = cleaned.split("I'm sorry")[0].split("I cannot")[0]
            if len(parts.strip()) > 50:
                cleaned = parts
        
        return cleaned.strip()
    
    def _is_refusal(self, text: str) -> bool:
        """Check if the output is likely a refusal or too short."""
        if not text or len(text) < 20:
            return True
        
        refusal_patterns = [
            "i'm sorry", "i cannot", "i apologize", "inappropriate",
            "i can't", "unable to", "not able to", "refuse to",
            "digital artwork", "stylized"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in refusal_patterns)
    
    def _inference_worker(self):
        """Background thread that runs vLLM inference with batching."""
        prompt_sets = [
            ["describe in detail without speculating. don't write anything except the caption."],
            ["provide a detailed description of the visual content in this image."],
            ["what is shown in this image? provide a comprehensive description."],
        ]
        
        buffer: List[Tuple[Job, Image.Image]] = []
        last_flush = time.monotonic()
        
        def flush_batch():
            nonlocal buffer, last_flush
            
            if not buffer:
                return
            
            logger.info(f"Processing batch of {len(buffer)} images")
            
            retry_items = []
            
            for retry_attempt in range(self.inference_config.max_retries):
                items_to_process = buffer if retry_attempt == 0 else retry_items
                retry_items = []
                
                if not items_to_process:
                    break
                
                if retry_attempt > 0:
                    logger.info(f"Retry {retry_attempt} for {len(items_to_process)} images")
                
                prompts = prompt_sets[min(retry_attempt, len(prompt_sets) - 1)]
                
                # Build batch requests
                reqs = []
                index = []
                
                for job, img in items_to_process:
                    try:
                        # Resize for consistency
                        img.thumbnail((512, 512), Image.BILINEAR)
                        
                        for q in prompts:
                            reqs.append(self._build_inputs(img, q))
                        index.append((job, img))
                    except Exception as e:
                        logger.error(f"Failed to prepare image for job {job.job_id}: {e}")
                        self.result_queue.put(("error", job, str(e)))
                
                if not reqs:
                    continue
                
                # Adjust temperature for retries
                temp = self.inference_config.temperature + (0.1 * retry_attempt)
                self.sampling_params.temperature = temp
                
                try:
                    # Run vLLM batch inference
                    outputs = self.llm.generate(reqs, self.sampling_params)
                except Exception as e:
                    logger.error(f"Batch generation failed: {e}")
                    for job, _ in items_to_process:
                        self.result_queue.put(("error", job, str(e)))
                    continue
                
                # Process outputs
                texts = [
                    (self._clean_output(o.outputs[0].text) if o.outputs else "")
                    for o in outputs
                ]
                
                # Map back to jobs (2 prompts per image)
                for i, (job, img) in enumerate(index):
                    d1 = texts[2 * i] if 2 * i < len(texts) else ""
                    d2 = texts[2 * i + 1] if (2 * i + 1) < len(texts) else ""
                    
                    # Check if both are refusals
                    if self._is_refusal(d1) and self._is_refusal(d2):
                        if retry_attempt < self.inference_config.max_retries - 1:
                            retry_items.append((job, img))
                            logger.debug(f"Job {job.job_id} queued for retry")
                        else:
                            self.result_queue.put(("error", job, "All attempts refused"))
                        continue
                    
                    # Combine and clean captions
                    valids = [t for t in (d1, d2) if not self._is_refusal(t)]
                    combined = CaptionUtils.combine(valids) if valids else max([d1, d2], key=len, default="")
                    caption = CaptionUtils.clean_caption(combined)
                    
                    if self._is_refusal(caption) and retry_attempt < self.inference_config.max_retries - 1:
                        retry_items.append((job, img))
                    else:
                        self.result_queue.put(("success", job, caption))
            
            buffer = []
            last_flush = time.monotonic()
        
        # Main inference loop
        while self.running:
            now = time.monotonic()
            timeout = max(0.0, (self.inference_config.coalesce_ms / 1000.0) - (now - last_flush))
            
            try:
                item = self.inference_queue.get(timeout=timeout if buffer else 0.1)
                if item == "STOP":
                    break
                buffer.append(item)
                
                if len(buffer) >= self.inference_config.batch_size:
                    flush_batch()
            except Empty:
                if buffer and (time.monotonic() - last_flush) * 1000.0 >= self.inference_config.coalesce_ms:
                    flush_batch()
        
        # Final flush
        flush_batch()
        logger.info("Inference worker stopped")
    
    async def start(self):
        """Start the worker and connect to orchestrator."""
        # Initialize vLLM
        self._setup_vllm()
        
        # Start inference worker thread
        self.running = True
        inference_thread = Thread(target=self._inference_worker, daemon=True)
        inference_thread.start()
        
        while self.running:
            try:
                await self._connect_and_run()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def _connect_and_run(self):
        """Connect to orchestrator and process jobs."""
        logger.info(f"Connecting to {self.server_url}")
        
        async with websockets.connect(
            self.server_url,
            ssl=self.ssl_context
        ) as websocket:
            self.websocket = websocket
            
            # Authenticate
            await websocket.send(json.dumps({
                "token": self.token,
                "name": self.name
            }))
            
            # Wait for welcome
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            
            if "error" in welcome_data:
                logger.error(f"Authentication failed: {welcome_data['error']}")
                self.running = False
                return
            
            self.worker_id = welcome_data.get("worker_id")
            logger.info(f"Connected as {self.worker_id}")
            
            # Start processing
            await asyncio.gather(
                self._heartbeat_loop(),
                self._job_processing_loop(),
                self._result_processing_loop()
            )
    
    async def _job_processing_loop(self):
        """Request and queue jobs for inference."""
        while self.running and self.websocket:
            try:
                # Request a job
                await self.websocket.send(json.dumps({
                    "type": "request_job"
                }))
                
                # Wait for response
                message = await self.websocket.recv()
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "job":
                    job_data = data["job"]
                    job = Job(**job_data)
                    logger.info(f"Received job {job.job_id}")
                    
                    # Load image (this would come from actual dataset)
                    # For now, create placeholder
                    img = await self._load_image_for_job(job)
                    if img:
                        # Queue for inference
                        self.inference_queue.put((job, img))
                    else:
                        # Report error
                        await self.websocket.send(json.dumps({
                            "type": "job_failed",
                            "job_id": job.job_id,
                            "error": "Failed to load image"
                        }))
                
                elif msg_type == "no_jobs":
                    logger.debug("No jobs available")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Job processing error: {e}")
                await asyncio.sleep(1)
    
    async def _result_processing_loop(self):
        """Process results from inference and send to orchestrator."""
        while self.running and self.websocket:
            try:
                # Check for results (non-blocking)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.result_queue.get, True, 0.1
                )
                
                status, job, payload = result
                
                if status == "success":
                    # Submit caption
                    await self.websocket.send(json.dumps({
                        "type": "submit_caption",
                        "job_id": job.job_id,
                        "dataset": job.dataset,
                        "shard": job.shard,
                        "item_key": job.item_key,
                        "caption": payload
                    }))
                    self.processed_count += 1
                    logger.info(f"Submitted caption for job {job.job_id}")
                else:
                    # Report error
                    await self.websocket.send(json.dumps({
                        "type": "job_failed",
                        "job_id": job.job_id,
                        "error": payload
                    }))
                    self.error_count += 1
                    
            except Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Result processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _load_image_for_job(self, job: Job) -> Optional[Image.Image]:
        """Load image for a job from dataset."""
        # TODO: Implement actual image loading from WebDataset or other source
        # This would use job.dataset, job.shard, job.item_key to locate the image
        
        # For now, return a placeholder
        # In production, this would:
        # 1. Connect to the dataset source (HuggingFace, S3, etc.)
        # 2. Load the specific shard
        # 3. Extract the image for the given item_key
        
        try:
            # Placeholder: create a dummy image
            img = Image.new('RGB', (256, 256), color='white')
            return img
        except Exception as e:
            logger.error(f"Failed to load image for job {job.job_id}: {e}")
            return None
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        while self.running and self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": "heartbeat",
                    "processed": self.processed_count,
                    "errors": self.error_count
                }))
                await asyncio.sleep(30)
            except:
                break
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        self.running = False
        
        # Stop inference worker
        self.inference_queue.put("STOP")
        
        if self.websocket:
            await self.websocket.close()