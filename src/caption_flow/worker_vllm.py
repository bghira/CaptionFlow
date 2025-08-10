"""Improved vLLM worker that processes shard chunks directly.

The worker now:
1. Receives shard chunk assignments from orchestrator
2. Directly streams WebDataset shards with readahead
3. Extracts metadata from images
4. Submits captions + metadata back to orchestrator
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import asyncio
import io
import json
import logging
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator
from queue import Queue, Empty
from threading import Thread, Lock
from collections import deque

import websockets
from websockets.client import WebSocketClientProtocol
from PIL import Image
import numpy as np
import webdataset as wds
from huggingface_hub import get_token

from .models import JobStatus
from .utils import CaptionUtils

logger = logging.getLogger(__name__)


@dataclass
class ShardChunk:
    """Shard chunk assignment from orchestrator."""

    chunk_id: str
    shard_url: str
    shard_name: str
    start_index: int
    chunk_size: int


@dataclass
class ProcessingItem:
    """Item being processed."""

    chunk_id: str
    item_key: str
    image: Image.Image
    image_data: bytes


@dataclass
class ProcessedResult:
    """Result with caption and metadata."""

    chunk_id: str
    shard_name: str
    item_key: str
    caption: str
    image_width: int
    image_height: int
    image_format: str
    file_size: int
    processing_time_ms: float


class VLLMWorker:
    """Worker that processes shard chunks directly."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")
        self.dataset_type = config.get("dataset_type", "huggingface")

        # vLLM configuration
        self.gpu_id = config.get("gpu_id", 0)
        self.batch_size = config.get("batch_size", 8)
        self.model_name = config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.temperature = config.get("temperature", 0.7)
        self.max_retries = config.get("max_retries", 3)

        # Readahead configuration
        self.readahead_size = config.get("readahead_size", 256)
        self.prefetch_batches = config.get("prefetch_batches", 4)

        # HuggingFace token for dataset access
        self.hf_token = get_token()

        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.running = False

        # Shard processing
        self.assigned_chunks: deque = deque()
        self.current_chunk: Optional[ShardChunk] = None
        self.chunk_lock = Lock()

        # Processing queues
        self.readahead_queue = Queue(maxsize=self.readahead_size)
        self.inference_queue = Queue(maxsize=self.batch_size * 2)
        self.result_queue = Queue()

        # vLLM components
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None

        # Metrics
        self.items_processed = 0
        self.items_failed = 0
        self.current_chunk_progress = 0

    def _setup_vllm(self):
        """Initialize vLLM components."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoProcessor

        logger.info(f"Loading {self.model_name} on GPU {self.gpu_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Initialize LLM with optimized settings
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=16384,
            enforce_eager=True,
            gpu_memory_utilization=0.92,
            dtype="float16",
            limit_mm_per_prompt={"image": 1},
            disable_mm_preprocessor_cache=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=256,
            stop=["<|end|>", "<|endoftext|>", "<|im_end|>"],
            repetition_penalty=1.05,
            skip_special_tokens=True,
        )

        logger.info("vLLM initialization complete")

    async def start(self):
        """Start the worker."""
        # Initialize vLLM
        self._setup_vllm()

        # Start background threads
        self.running = True

        # Start shard reader thread
        reader_thread = Thread(target=self._shard_reader_thread, daemon=True)
        reader_thread.start()

        # Start inference thread
        inference_thread = Thread(target=self._inference_thread, daemon=True)
        inference_thread.start()

        # Connect to orchestrator
        while self.running:
            try:
                await self._connect_and_run()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if self.running:
                    await asyncio.sleep(5)

    async def _connect_and_run(self):
        """Connect to orchestrator and process chunks."""
        logger.info(f"Connecting to {self.server_url}")

        async with websockets.connect(self.server_url) as websocket:
            self.websocket = websocket

            # Authenticate
            await websocket.send(json.dumps({"token": self.token, "name": self.name}))

            # Wait for welcome
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                logger.error(f"Authentication failed: {welcome_data['error']}")
                self.running = False
                return

            self.worker_id = welcome_data.get("worker_id")
            config = welcome_data.get("config", {})

            logger.info(f"Connected as {self.worker_id}")

            # Request initial chunks
            await websocket.send(json.dumps({"type": "request_chunks"}))

            # Start processing
            await asyncio.gather(
                self._message_handler(), self._result_sender(), self._heartbeat_loop()
            )

    async def _message_handler(self):
        """Handle messages from orchestrator."""
        async for message in self.websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "shard_assignment":
                    chunks = data["chunks"]
                    for chunk_data in chunks:
                        chunk = ShardChunk(**chunk_data)
                        with self.chunk_lock:
                            self.assigned_chunks.append(chunk)
                        logger.info(f"Received chunk assignment: {chunk.chunk_id}")

                    # Request more chunks if queue is low
                    if len(self.assigned_chunks) < 2:
                        await self.websocket.send(json.dumps({"type": "request_chunks"}))

            except Exception as e:
                logger.error(f"Message handler error: {e}")

    def _shard_reader_thread(self):
        """Background thread that reads from WebDataset shards."""
        logger.info("Starting shard reader thread")

        while self.running:
            # Get next chunk to process
            with self.chunk_lock:
                if not self.current_chunk and self.assigned_chunks:
                    self.current_chunk = self.assigned_chunks.popleft()
                    self.current_chunk_progress = 0
                    logger.info(f"Starting chunk {self.current_chunk.chunk_id}")

            if not self.current_chunk:
                time.sleep(1)
                continue

            try:
                # Process the chunk
                self._process_shard_chunk(self.current_chunk)

                # Mark chunk as complete
                logger.info(f"Completed chunk {self.current_chunk.chunk_id}")

                # Notify orchestrator
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(
                        json.dumps(
                            {"type": "chunk_complete", "chunk_id": self.current_chunk.chunk_id}
                        )
                    ),
                    asyncio.get_event_loop(),
                )

                with self.chunk_lock:
                    self.current_chunk = None

            except Exception as e:
                logger.error(f"Error processing chunk {self.current_chunk.chunk_id}: {e}")

                # Notify orchestrator of failure
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(
                        json.dumps(
                            {
                                "type": "chunk_failed",
                                "chunk_id": self.current_chunk.chunk_id,
                                "error": str(e),
                            }
                        )
                    ),
                    asyncio.get_event_loop(),
                )

                with self.chunk_lock:
                    self.current_chunk = None

    def _process_shard_chunk(self, chunk: ShardChunk):
        """Process a single shard chunk."""
        logger.info(f"Processing shard {chunk.shard_name} from index {chunk.start_index}")

        # Create WebDataset pipeline
        if self.dataset_type == "huggingface":
            # Use curl with auth for HuggingFace
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.hf_token)}' {shlex.quote(chunk.shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp"),
            )
        else:
            # Local file
            ds = wds.DataPipeline(
                wds.SimpleShardList(chunk.shard_url),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp"),
            )

        # Process items with readahead
        items_processed = 0
        items_to_skip = chunk.start_index

        for key, image_data in ds:
            if not self.running:
                break

            # Skip to start index
            if items_to_skip > 0:
                items_to_skip -= 1
                continue

            # Check if we've processed enough
            if items_processed >= chunk.chunk_size:
                break

            try:
                # Load image
                img = Image.open(io.BytesIO(image_data))

                # Create processing item
                item = ProcessingItem(
                    chunk_id=chunk.chunk_id, item_key=key, image=img, image_data=image_data
                )

                # Add to readahead queue (blocks if full - provides backpressure)
                self.readahead_queue.put(item, timeout=30)

                items_processed += 1
                self.current_chunk_progress = items_processed

                # Batch items for inference
                if self.readahead_queue.qsize() >= self.batch_size:
                    self._batch_for_inference()

            except Exception as e:
                logger.error(f"Error processing item {key}: {e}")
                self.items_failed += 1

        # Process remaining items
        self._batch_for_inference()

        logger.info(f"Chunk {chunk.chunk_id} processed {items_processed} items")

    def _batch_for_inference(self):
        """Batch items from readahead queue for inference."""
        batch = []

        try:
            while len(batch) < self.batch_size:
                item = self.readahead_queue.get_nowait()
                batch.append(item)
        except Empty:
            pass

        if batch:
            self.inference_queue.put(batch)

    def _inference_thread(self):
        """Background thread for vLLM inference."""
        logger.info("Starting inference thread")

        prompts = [
            "describe this image in detail",
            "provide a comprehensive description of the visual content",
        ]

        while self.running:
            try:
                # Get batch from queue
                batch = self.inference_queue.get(timeout=1)

                if not batch:
                    continue

                logger.debug(f"Processing batch of {len(batch)} images")
                start_time = time.time()

                # Prepare vLLM inputs
                requests = []
                for item in batch:
                    # Resize for consistency
                    item.image.thumbnail((512, 512), Image.BILINEAR)

                    for prompt in prompts:
                        req = self._build_vllm_input(item.image, prompt)
                        requests.append(req)

                # Run inference
                outputs = self.llm.generate(requests, self.sampling_params)

                # Process outputs
                for i, item in enumerate(batch):
                    # Get both prompt outputs
                    idx = i * len(prompts)
                    descriptions = [
                        self._clean_output(outputs[idx + j].outputs[0].text)
                        for j in range(len(prompts))
                        if idx + j < len(outputs) and outputs[idx + j].outputs
                    ]

                    # Combine captions
                    caption = CaptionUtils.combine(descriptions)

                    # Extract metadata
                    result = ProcessedResult(
                        chunk_id=item.chunk_id,
                        shard_name=Path(item.chunk_id).stem.rsplit("_chunk_", 1)[0],
                        item_key=item.item_key,
                        caption=caption,
                        image_width=item.image.width,
                        image_height=item.image.height,
                        image_format=item.image.format or "unknown",
                        file_size=len(item.image_data),
                        processing_time_ms=(time.time() - start_time) * 1000 / len(batch),
                    )

                    self.result_queue.put(result)
                    self.items_processed += 1

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}")

    def _build_vllm_input(self, image: Image.Image, prompt: str) -> Dict:
        """Build vLLM input."""
        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
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
            return {
                "prompt": f"<|user|>\n<|image_pad|>\n{prompt}<|end|>\n<|assistant|>",
                "multi_modal_data": {"image": [image]},
            }

    def _clean_output(self, text: str) -> str:
        """Clean model output."""
        if not text:
            return ""

        # Remove common artifacts
        for token in ["<|end|>", "<|endoftext|>", "<|im_end|>", "I'm sorry", "I cannot"]:
            if token in text:
                text = text.split(token)[0]

        return text.strip()

    async def _result_sender(self):
        """Send results back to orchestrator."""
        while self.running:
            try:
                # Get result (with timeout to allow checking self.running)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.result_queue.get, True, 1
                )

                # Send to orchestrator with metadata
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": "submit_caption",
                            "chunk_id": result.chunk_id,
                            "dataset": self.config.get("dataset_path", "unknown"),
                            "shard": result.shard_name,
                            "item_key": result.item_key,
                            "caption": result.caption,
                            "image_width": result.image_width,
                            "image_height": result.image_height,
                            "image_format": result.image_format,
                            "file_size": result.file_size,
                            "processing_time_ms": result.processing_time_ms,
                        }
                    )
                )

                if self.items_processed % 100 == 0:
                    logger.info(f"Processed {self.items_processed} items")

            except Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                import traceback

                logger.error(f"Error sending result: {e}")
                logger.debug(traceback.format_exc())
                self.items_failed += 1

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running and self.websocket:
            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": "heartbeat",
                            "processed": self.items_processed,
                            "failed": self.items_failed,
                            "current_chunk": (
                                self.current_chunk.chunk_id if self.current_chunk else None
                            ),
                            "chunk_progress": self.current_chunk_progress,
                            "queue_sizes": {
                                "readahead": self.readahead_queue.qsize(),
                                "inference": self.inference_queue.qsize(),
                                "results": self.result_queue.qsize(),
                            },
                        }
                    )
                )
                await asyncio.sleep(30)
            except:
                break

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        self.running = False

        if self.websocket:
            await self.websocket.close()
