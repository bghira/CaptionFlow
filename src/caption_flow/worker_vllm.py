"""Improved vLLM worker with proper connection recovery and chunk abandonment.

Key improvements:
1. Detects disconnection and stops current chunk processing
2. Clears all queues and abandons current chunk on disconnect
3. Maintains vLLM instance across reconnections
4. Properly handles connection state in all threads
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import asyncio
import io
import json
import logging
import ssl
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from threading import Thread, Lock, Event
from collections import deque

import websockets
from websockets.client import WebSocketClientProtocol
from PIL import Image
import numpy as np
import webdataset as wds
from huggingface_hub import get_token

from .models import JobStatus, Job
from .utils import CaptionUtils
from .utils.dataset_loader import DatasetLoader

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
    """Result with multiple captions and metadata."""

    chunk_id: str
    shard_name: str
    item_key: str
    captions: List[str]
    image_width: int
    image_height: int
    image_format: str
    file_size: int
    processing_time_ms: float


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


class VLLMWorker:
    """Worker that processes shard chunks directly with proper reconnection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")

        # Dataset configuration will be received from orchestrator
        self.dataset_config = None
        self.dataset_loader = None
        self.dataset_type = None
        self.hf_token = get_token()

        # vLLM configuration
        self.inference_config = InferenceConfig(
            gpu_id=config.get("gpu_id", 0),
            precision=config.get("precision", "fp16"),
            batch_size=config.get("batch_size", 8),
            coalesce_ms=config.get("coalesce_ms", 30),
            max_retries=config.get("max_retries", 3),
            model_name=config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
            temperature=config.get("temperature", 0.7),
        )

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None  # Store main event loop

        # Connection state events
        self.connected = Event()
        self.should_stop_processing = Event()

        # Inference components (initialized in setup)
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None

        # Shard chunk processing
        self.chunk_lock = Lock()
        self.assigned_chunks = deque()
        self.current_chunk = None
        self.current_chunk_progress = 0

        # Batching queues - will be cleared on disconnect
        self.readahead_queue = Queue(maxsize=256)
        self.inference_queue = Queue(maxsize=128)
        self.result_queue = Queue()

        # Metrics
        self.items_processed = 0
        self.items_failed = 0
        self.chunks_completed = 0

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context."""
        if self.server_url.startswith("ws://"):
            logger.warning("Using insecure WebSocket connection")
            return None

        if not self.config.get("verify_ssl", True):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

        return ssl.create_default_context()

    def _setup_dataset_loader(self, dataset_config: Dict[str, Any]):
        """Initialize dataset loader with config from orchestrator."""
        dataset_path = dataset_config.get("dataset_path") or dataset_config.get("path")
        dataset_type = dataset_config.get("dataset_type") or dataset_config.get(
            "type", "huggingface"
        )

        if dataset_path:
            logger.info(f"Initializing dataset loader for {dataset_type}: {dataset_path}")
            self.dataset_loader = DatasetLoader(dataset_path, dataset_type)
            self.dataset_config = dataset_config
            self.dataset_type = dataset_type
        else:
            logger.warning("No dataset path provided by orchestrator")

    def _setup_vllm(self):
        """Initialize vLLM components."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.inference_config.gpu_id)

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoProcessor

        logger.info(
            f"Loading {self.inference_config.model_name} on GPU {self.inference_config.gpu_id}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.inference_config.model_name, trust_remote_code=True, use_fast=True
        )
        self.processor = AutoProcessor.from_pretrained(self.inference_config.model_name)

        # Initialize LLM with optimized settings
        self.llm = LLM(
            model=self.inference_config.model_name,
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
            temperature=self.inference_config.temperature,
            top_p=0.95,
            max_tokens=256,
            stop=["<|end|>", "<|endoftext|>", "<|im_end|>"],
            repetition_penalty=1.05,
            skip_special_tokens=True,
        )

        logger.info("vLLM initialization complete")

    def _clear_state_on_disconnect(self):
        """Clear all processing state when disconnected."""
        logger.info("Clearing state due to disconnection")

        # Signal threads to stop current processing
        self.should_stop_processing.set()

        with self.chunk_lock:
            # Clear assigned chunks
            self.assigned_chunks.clear()
            self.current_chunk = None
            self.current_chunk_progress = 0

        # Clear all queues
        self._clear_queue(self.readahead_queue)
        self._clear_queue(self.inference_queue)
        self._clear_queue(self.result_queue)

        logger.info("State cleared, ready for reconnection")

    def _clear_queue(self, queue: Queue):
        """Clear all items from a queue."""
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass

    async def start(self):
        """Start the worker with automatic reconnection."""
        # Initialize vLLM once
        self._setup_vllm()

        # Capture the main event loop for use in background threads
        self.main_loop = asyncio.get_running_loop()

        # Start background threads
        self.running = True

        # Start shard reader thread
        reader_thread = Thread(target=self._shard_reader_thread, daemon=True)
        reader_thread.start()

        # Start inference thread
        inference_thread = Thread(target=self._inference_thread, daemon=True)
        inference_thread.start()

        # Reconnection with exponential backoff
        reconnect_delay = 5
        max_delay = 60

        # Connect to orchestrator with retries
        while self.running:
            try:
                # Clear stop signal before connecting
                self.should_stop_processing.clear()

                await self._connect_and_run()

                # Reset delay on successful connection
                reconnect_delay = 5

            except Exception as e:
                logger.error(f"Connection error: {e}")

                # Mark as disconnected
                self.connected.clear()
                self.websocket = None

                # Clear all state on disconnect
                self._clear_state_on_disconnect()

            if self.running:
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)

                # Exponential backoff
                reconnect_delay = min(reconnect_delay * 2, max_delay)

    async def _connect_and_run(self):
        """Connect to orchestrator and process chunks."""
        logger.info(f"Connecting to {self.server_url}")

        async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
            self.websocket = websocket
            self.connected.set()

            # Authenticate
            await websocket.send(json.dumps({"token": self.token, "name": self.name}))

            # Wait for welcome message with dataset config
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                logger.error(f"Authentication failed: {welcome_data['error']}")
                self.running = False
                return

            self.worker_id = welcome_data.get("worker_id")
            logger.info(f"Connected as {self.worker_id}")

            # Extract and setup dataset configuration from orchestrator
            dataset_config = welcome_data.get("dataset_config", {})
            if dataset_config:
                self._setup_dataset_loader(dataset_config)
                logger.info(f"Received dataset config: {dataset_config}")
            else:
                logger.warning("No dataset configuration received from orchestrator")

            # Request initial chunks
            await websocket.send(json.dumps({"type": "request_chunks", "count": 2}))

            # Start processing
            try:
                # Create tasks
                tasks = [
                    asyncio.create_task(self._heartbeat_loop()),
                    asyncio.create_task(self._message_handler()),
                    asyncio.create_task(self._result_sender()),
                ]

                # Wait for any task to complete (likely due to disconnection)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            finally:
                # Ensure we mark as disconnected
                self.connected.clear()
                self.websocket = None

    async def _message_handler(self):
        """Handle messages from orchestrator."""
        try:
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

                    elif msg_type == "no_chunks":
                        logger.info("No chunks available from orchestrator")
                        await asyncio.sleep(10)
                        # Request again after waiting
                        if self.websocket:
                            await self.websocket.send(
                                json.dumps({"type": "request_chunks", "count": 2})
                            )

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid message format: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed by orchestrator: {e}")
            raise  # Re-raise to trigger cleanup
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            raise

    def _shard_reader_thread(self):
        """Background thread that reads from WebDataset shards."""
        logger.info("Starting shard reader thread")

        while self.running:
            # Check if we should stop processing
            if self.should_stop_processing.is_set():
                logger.info("Shard reader stopping due to disconnection")
                self.should_stop_processing.wait()  # Wait until cleared
                continue

            # Only process if connected
            if not self.connected.is_set():
                time.sleep(1)
                continue

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

                # Only mark complete if still connected
                if self.connected.is_set() and not self.should_stop_processing.is_set():
                    logger.info(f"Completed chunk {self.current_chunk.chunk_id}")
                    self.chunks_completed += 1

                    # Notify orchestrator if connected
                    if self.websocket and self.main_loop:
                        try:
                            # Notify completion
                            asyncio.run_coroutine_threadsafe(
                                self.websocket.send(
                                    json.dumps(
                                        {
                                            "type": "chunk_complete",
                                            "chunk_id": self.current_chunk.chunk_id,
                                        }
                                    )
                                ),
                                self.main_loop,
                            ).result(timeout=5)

                            # Request more chunks if queue is low
                            with self.chunk_lock:
                                queue_size = len(self.assigned_chunks)

                            if queue_size < 2:
                                logger.info(f"Requesting more chunks (queue size: {queue_size})")
                                asyncio.run_coroutine_threadsafe(
                                    self.websocket.send(
                                        json.dumps({"type": "request_chunks", "count": 2})
                                    ),
                                    self.main_loop,
                                ).result(timeout=5)

                        except Exception as e:
                            logger.warning(f"Could not notify orchestrator: {e}")

                with self.chunk_lock:
                    self.current_chunk = None

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

                # Only notify of failure if still connected
                if self.connected.is_set() and self.websocket and self.main_loop:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket.send(
                                json.dumps(
                                    {
                                        "type": "chunk_failed",
                                        "chunk_id": (
                                            self.current_chunk.chunk_id
                                            if self.current_chunk
                                            else "unknown"
                                        ),
                                        "error": str(e),
                                    }
                                )
                            ),
                            self.main_loop,
                        ).result(timeout=5)
                    except Exception as send_error:
                        logger.warning(
                            f"Could not notify orchestrator of chunk failure: {send_error}"
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
            # Check if we should stop
            if not self.running or self.should_stop_processing.is_set():
                logger.info(f"Stopping chunk processing early due to disconnect")
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
                # Use timeout to allow checking for disconnection
                timeout_end = time.time() + 30
                while self.running and not self.should_stop_processing.is_set():
                    try:
                        self.readahead_queue.put(item, timeout=1)
                        break
                    except:
                        if time.time() > timeout_end:
                            raise TimeoutError("Queue put timeout")
                        continue

                items_processed += 1
                self.current_chunk_progress = items_processed

                # Batch items for inference
                if self.readahead_queue.qsize() >= self.inference_config.batch_size:
                    self._batch_for_inference()

            except Exception as e:
                if self.should_stop_processing.is_set():
                    break
                logger.error(f"Error processing item {key}: {e}")
                self.items_failed += 1

        # Process remaining items only if still connected
        if not self.should_stop_processing.is_set():
            self._batch_for_inference()

        logger.info(f"Chunk {chunk.chunk_id} processed {items_processed} items")

    def _batch_for_inference(self):
        """Batch items from readahead queue for inference."""
        batch = []

        try:
            while len(batch) < self.inference_config.batch_size:
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
            "what are the key elements in this image?",
        ]

        while self.running:
            try:
                # Get batch from queue with timeout
                batch = self.inference_queue.get(timeout=1)

                if not batch:
                    continue

                # Skip if disconnected
                if self.should_stop_processing.is_set():
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

                # Process outputs only if still connected
                if not self.should_stop_processing.is_set():
                    for i, item in enumerate(batch):
                        # Get all prompt outputs as a list
                        idx = i * len(prompts)
                        captions = []

                        for j in range(len(prompts)):
                            if idx + j < len(outputs) and outputs[idx + j].outputs:
                                caption_text = self._clean_output(outputs[idx + j].outputs[0].text)
                                if caption_text:  # Only add non-empty captions
                                    captions.append(caption_text)

                        # Only create result if we have at least one caption
                        if captions:
                            result = ProcessedResult(
                                chunk_id=item.chunk_id,
                                shard_name=Path(item.chunk_id).stem.rsplit("_chunk_", 1)[0],
                                item_key=item.item_key,
                                captions=captions,
                                image_width=item.image.width,
                                image_height=item.image.height,
                                image_format=item.image.format or "unknown",
                                file_size=len(item.image_data),
                                processing_time_ms=(time.time() - start_time) * 1000 / len(batch),
                            )

                            self.result_queue.put(result)
                            self.items_processed += 1
                        else:
                            logger.warning(f"No valid captions generated for item {item.item_key}")
                            self.items_failed += 1

            except Empty:
                continue
            except Exception as e:
                if self.should_stop_processing.is_set():
                    continue
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
        pending_results = []  # Buffer for results during disconnection

        while self.running:
            try:
                # Get result (with timeout to allow checking self.running)
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.result_queue.get, True, 1
                    )
                    pending_results.append(result)
                except Empty:
                    pass

                # Only try to send if connected
                if pending_results and self.websocket and self.connected.is_set():
                    sent_results = []
                    for result in pending_results:
                        try:
                            # Send result with all captions
                            await self.websocket.send(
                                json.dumps(
                                    {
                                        "type": "submit_captions",
                                        "chunk_id": result.chunk_id,
                                        "dataset": self.dataset_config.get(
                                            "dataset_path", "unknown"
                                        ),
                                        "shard": result.shard_name,
                                        "item_key": result.item_key,
                                        "captions": result.captions,
                                        "caption_count": len(result.captions),
                                        "image_width": result.image_width,
                                        "image_height": result.image_height,
                                        "image_format": result.image_format,
                                        "file_size": result.file_size,
                                        "processing_time_ms": result.processing_time_ms,
                                    }
                                )
                            )
                            sent_results.append(result)

                            if self.items_processed % 100 == 0:
                                logger.info(
                                    f"Processed {self.items_processed} items "
                                    f"(~{self.items_processed * 3} captions)"
                                )
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(f"Connection lost while sending result: {e}")
                            self.connected.clear()
                            self.websocket = None
                            break
                        except Exception as e:
                            logger.error(f"Error sending result: {e}")
                            break

                    # Remove successfully sent results
                    for result in sent_results:
                        pending_results.remove(result)

                # Clear pending results if disconnected and buffer is too large
                if not self.connected.is_set() and len(pending_results) > 1000:
                    logger.warning(
                        f"Clearing {len(pending_results)} pending results due to prolonged disconnection"
                    )
                    pending_results.clear()

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Unexpected error in result sender: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats with connection checking."""
        while self.running and self.connected.is_set():
            try:
                if self.websocket:
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "processed": self.items_processed,
                                "failed": self.items_failed,
                                "chunks_completed": self.chunks_completed,
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
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Connection lost during heartbeat: {e}")
                self.connected.clear()
                self.websocket = None
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.connected.clear()
                self.websocket = None
                break

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        self.running = False
        self.connected.clear()
        self.should_stop_processing.set()

        # Stop processing threads by adding stop signals
        self.readahead_queue.put(None)
        self.inference_queue.put(None)

        # Close websocket if connected
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        logger.info("Worker shutdown complete")
