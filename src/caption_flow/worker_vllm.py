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
from .utils.vllm_config import VLLMConfigManager
from .utils.image_processor import ImageProcessor

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


class VLLMWorker:
    """Worker that processes shard chunks directly with proper reconnection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")
        batch_image_processing = config.get("batch_image_processing", False)

        # Dataset configuration will be received from orchestrator
        self.dataset_config = None
        self.dataset_loader = None
        self.dataset_type = None
        self.hf_token = get_token()

        # vLLM configuration will be received from orchestrator
        self.vllm_config = None
        self.inference_prompts = None
        self.vllm_config_manager = VLLMConfigManager()

        # Backward compatibility: local config for GPU selection
        self.gpu_id = config.get("gpu_id", 0)

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
        self.image_processor = None
        # when we use batch processing for image processor, we'll have to use a persistent instance later.
        if batch_image_processing:
            self.image_processor = ImageProcessor()

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

        # Job mode for shards vs jobs and job queue.
        self.job_mode = config.get("job_mode", False)
        self.job_queue = Queue(maxsize=32)

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
        dataset_split = dataset_config.get("dataset_split") or dataset_config.get("split", "train")
        dataset_image_column = dataset_config.get("dataset_image_column") or dataset_config.get(
            "image_column", "image"
        )

        if dataset_path:
            logger.info(
                f"Initializing dataset loader for {dataset_type}: {dataset_path} "
                f"(split: {dataset_split}, image_column: {dataset_image_column})"
            )
            self.dataset_loader = DatasetLoader(
                dataset_path, dataset_type, dataset_split, dataset_image_column
            )
            self.dataset_config = dataset_config
            self.dataset_type = dataset_type
            self.dataset_split = dataset_split
            self.dataset_image_column = dataset_image_column
        else:
            logger.warning("No dataset path provided by orchestrator")

    def _setup_vllm(self):
        """Initialize vLLM components."""
        if not self.vllm_config:
            raise RuntimeError("vLLM config not received from orchestrator")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoProcessor

        model_name = self.vllm_config["model"]
        logger.info(f"Loading {model_name} on GPU {self.gpu_id}")

        # Always reload tokenizer/processor (they're model-specific)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Initialize LLM with settings from orchestrator using config manager
        vllm_params = self.vllm_config_manager.get_vllm_init_params(self.vllm_config)
        self.llm = LLM(**vllm_params)

        # Create sampling params from orchestrator config
        self.sampling_params = self.vllm_config_manager.create_sampling_params(self.vllm_config)

        logger.info("vLLM initialization complete")

        # Update config manager's tracking
        self.vllm_config_manager.current_config = self.vllm_config

    async def _handle_job_assignment(self, job_data: Dict):
        """Handle job assignment from orchestrator."""
        try:
            # Convert to processing item
            image = Image.open(io.BytesIO(job_data["image_data"]))

            item = ProcessingItem(
                chunk_id=job_data["job_id"],
                item_key=job_data["sample_id"],
                image=image,
                image_data=job_data["image_data"],
            )

            # Add to inference queue
            self.readahead_queue.put(item)
            logger.debug(f"Queued job {job_data['job_id']} for processing")

        except Exception as e:
            logger.error(f"Error handling job assignment: {e}")

    async def _job_request_loop(self):
        """Request jobs from orchestrator in job mode."""
        while self.running and self.connected.is_set():
            try:
                # Check if we need more work
                if self.readahead_queue.qsize() < self.vllm_config.get("batch_size", 8):
                    await self.websocket.send(json.dumps({"type": "request_job"}))

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Job request error: {e}")
                await asyncio.sleep(5)

    def _handle_vllm_config_update(self, new_config: Dict[str, Any]) -> bool:
        """
        Handle vLLM configuration updates.

        Returns:
            True if config was updated successfully, False if reload is needed
        """
        if not new_config:
            return True

        # Check what changed
        change = self.vllm_config_manager.analyze_config_change(self.vllm_config, new_config)

        if not change.changed_fields:
            # No changes
            return True

        if change.requires_reload:
            # Need to reload vLLM
            logger.info(f"vLLM config changes require reload: {change.changed_fields}")

            # Save old config
            old_config = self.vllm_config
            self.vllm_config = new_config

            try:
                # Reload vLLM with new config
                logger.info("Reloading vLLM with new configuration...")

                # Clean up old instance
                if hasattr(self, "llm") and self.llm:
                    del self.llm

                # Also clean up tokenizer/processor if model changed
                if change.model_changed:
                    if hasattr(self, "tokenizer"):
                        del self.tokenizer
                    if hasattr(self, "processor"):
                        del self.processor

                import gc

                gc.collect()

                # Reload with new config
                self._setup_vllm()

                # Update prompts
                self.inference_prompts = new_config.get("inference_prompts", self.inference_prompts)

                logger.info("vLLM reload complete")
                return True

            except Exception as e:
                logger.error(f"Failed to reload vLLM: {e}")
                # Restore old config
                self.vllm_config = old_config
                return False

        else:
            # Can update without reload
            logger.info(f"Updating vLLM config without reload: {change.changed_fields}")

            # Update sampling params if changed
            if change.sampling_changed:
                self.sampling_params = self.vllm_config_manager.create_sampling_params(new_config)

            # Update prompts if changed
            if change.prompts_changed:
                self.inference_prompts = new_config.get("inference_prompts", self.inference_prompts)
                logger.info(f"Updated inference prompts: {len(self.inference_prompts)} prompts")

            # Update config
            self.vllm_config = new_config
            logger.info("vLLM configuration updated successfully without reload")
            return True

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
        self.running = True

        # Wait for initial connection to get vLLM config
        logger.info("Connecting to orchestrator for configuration...")

        # Try initial connection to get config
        config_received = False
        while not config_received and self.running:
            try:
                await self._initial_connect_for_config()
                config_received = True
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                await asyncio.sleep(5)

        # Initialize vLLM once we have config
        self._setup_vllm()

        # Capture the main event loop for use in background threads
        self.main_loop = asyncio.get_running_loop()

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

    async def _initial_connect_for_config(self):
        """Connect initially just to get configuration."""
        async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
            # Authenticate
            await websocket.send(json.dumps({"token": self.token, "name": self.name}))

            # Wait for welcome message with config
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                raise RuntimeError(f"Authentication failed: {welcome_data['error']}")

            # Extract vLLM configuration
            self.vllm_config = welcome_data.get("vllm_config")
            if not self.vllm_config:
                raise RuntimeError("No vLLM configuration received from orchestrator")

            self.inference_prompts = self.vllm_config.get(
                "inference_prompts",
                [
                    "describe this image in detail",
                    "provide a comprehensive description of the visual content",
                    "what are the key elements in this image?",
                ],
            )

            # Store config in manager
            self.vllm_config_manager.current_config = self.vllm_config

            # Extract dataset configuration
            dataset_config = welcome_data.get("dataset_config", {})
            if dataset_config:
                self._setup_dataset_loader(dataset_config)

            logger.info("Received configuration from orchestrator")
            # Disconnect after getting config

    async def _connect_and_run(self):
        """Connect to orchestrator and process chunks."""
        logger.info(f"Connecting to {self.server_url}")

        async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
            self.websocket = websocket
            self.connected.set()

            # Clear stop signal now that we're connected
            self.should_stop_processing.clear()

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

            # Update vLLM config if provided (in case it changed)
            new_vllm_config = welcome_data.get("vllm_config")
            if new_vllm_config and new_vllm_config != self.vllm_config:
                logger.info("Received updated vLLM configuration")

                # Handle config update (may trigger reload)
                if not self._handle_vllm_config_update(new_vllm_config):
                    logger.error("Failed to update vLLM configuration")
                    # Continue with existing config

            if self.job_mode:
                # In job mode, request individual jobs instead of chunks
                tasks.append(asyncio.create_task(self._job_request_loop()))
            else:
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
                        reason = data.get("reason", "unknown")
                        logger.info(f"No chunks available from orchestrator (reason: {reason})")

                        # Different wait times based on reason
                        wait_time = 2 if reason == "state_restoring" else 10
                        await asyncio.sleep(wait_time)

                        # Request again after waiting
                        if self.websocket and self.connected.is_set():
                            await self.websocket.send(
                                json.dumps({"type": "request_chunks", "count": 2})
                            )

                    elif msg_type == "reload_vllm":
                        # Orchestrator requested vLLM reload
                        logger.info("Orchestrator requested vLLM reload")
                        new_config = data.get("vllm_config")
                        if new_config:
                            self._handle_vllm_config_update(new_config)

                    elif msg_type == "job_assignment":
                        await self._handle_job_assignment(data["job"])

                    elif msg_type == "no_jobs":
                        logger.debug("No jobs available")
                        await asyncio.sleep(2)

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
                logger.info("Shard reader waiting for reconnection")
                time.sleep(1)
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

        # Check if this is a virtual HuggingFace dataset shard
        if chunk.shard_url.startswith("hf_dataset:"):
            # Use dataset loader's iterate_shard method directly
            # It knows how to handle virtual shards
            if not self.dataset_loader:
                logger.error("No dataset loader configured for HuggingFace dataset shard")
                return

            items_processed = 0

            # For HF dataset chunks, we need to construct the proper virtual shard URL
            # The chunk has the actual start index in its chunk.start_index field
            # We need to create a virtual shard URL that includes this offset
            parts = chunk.shard_url.split("_chunk_")
            if len(parts) == 2:
                base_path = parts[0]  # e.g., "hf_dataset:RareConcepts/pixelvision-670k"
                # Construct virtual shard URL with the correct offset
                virtual_shard_url = f"{base_path}:chunk:{chunk.start_index}"
            else:
                # Fallback to original URL
                virtual_shard_url = chunk.shard_url

            logger.debug(f"Using virtual shard URL: {virtual_shard_url}")

            # Iterate through the virtual shard
            for key, url, image_data in self.dataset_loader.iterate_shard(virtual_shard_url):
                # Check if we should stop
                if (
                    not self.running
                    or self.should_stop_processing.is_set()
                    or not self.connected.is_set()
                ):
                    logger.info(f"Stopping chunk processing early due to disconnect")
                    break

                # Check if we've processed enough for this chunk
                if items_processed >= chunk.chunk_size:
                    break

                try:
                    # Load image
                    img = Image.open(io.BytesIO(image_data))

                    # Create processing item
                    item = ProcessingItem(
                        chunk_id=chunk.chunk_id, item_key=key, image=img, image_data=image_data
                    )

                    # Add to readahead queue
                    timeout_end = time.time() + 30
                    while (
                        self.running
                        and not self.should_stop_processing.is_set()
                        and self.connected.is_set()
                    ):
                        try:
                            self.readahead_queue.put(item, timeout=1)
                            break
                        except:
                            if time.time() > timeout_end:
                                raise TimeoutError("Queue put timeout")
                            continue

                    # If we couldn't queue due to disconnection, skip this item
                    if not self.connected.is_set() or self.should_stop_processing.is_set():
                        logger.debug(f"Skipping item {key} due to disconnection")
                        break

                    items_processed += 1
                    self.current_chunk_progress = items_processed

                    # Batch items for inference
                    batch_size = self.vllm_config.get("batch_size", 8)
                    if self.readahead_queue.qsize() >= batch_size:
                        self._batch_for_inference()

                except Exception as e:
                    if self.should_stop_processing.is_set():
                        break
                    logger.error(f"Error processing item {key}: {e}")
                    self.items_failed += 1

        else:
            # Regular WebDataset shard processing
            # Create WebDataset pipeline
            if self.dataset_type == "huggingface" and not chunk.shard_url.startswith("hf_dataset:"):
                # Use curl with auth for HuggingFace WebDataset
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
                if (
                    not self.running
                    or self.should_stop_processing.is_set()
                    or not self.connected.is_set()
                ):
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
                    while (
                        self.running
                        and not self.should_stop_processing.is_set()
                        and self.connected.is_set()
                    ):
                        try:
                            self.readahead_queue.put(item, timeout=1)
                            break
                        except:
                            if time.time() > timeout_end:
                                raise TimeoutError("Queue put timeout")
                            continue

                    # If we couldn't queue due to disconnection, skip this item
                    if not self.connected.is_set() or self.should_stop_processing.is_set():
                        logger.debug(f"Skipping item {key} due to disconnection")
                        break

                    items_processed += 1
                    self.current_chunk_progress = items_processed

                    # Batch items for inference
                    batch_size = self.vllm_config.get("batch_size", 8)
                    if self.readahead_queue.qsize() >= batch_size:
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
        batch_size = self.vllm_config.get("batch_size", 8)

        try:
            while len(batch) < batch_size:
                item = self.readahead_queue.get_nowait()
                batch.append(item)
        except Empty:
            pass

        if batch:
            self.inference_queue.put(batch)

    def _process_batch_with_retries(self, batch, max_attempts=3):
        """Process a batch of items with per-item retry logic for failed captions.

        Returns:
            List of ProcessedResult objects for successful items
        """
        results = []

        # Track which items need processing and their retry counts
        items_to_process = [(i, item, 0) for i, item in enumerate(batch)]

        while items_to_process:
            # Build requests for current items
            current_batch = []
            current_indices = []
            requests = []

            for idx, (original_idx, item, attempt_count) in enumerate(items_to_process):
                current_batch.append((original_idx, item, attempt_count))
                current_indices.append(idx)

                # Prepare image with background replacement if needed
                converted_img = ImageProcessor.prepare_for_inference(item.image)

                # Build requests for all prompts
                for prompt in self.inference_prompts:
                    req = self._build_vllm_input(converted_img, prompt)
                    requests.append(req)

            # Run inference
            outputs = self.llm.generate(requests, self.sampling_params)

            # Process outputs
            successful_items = []
            failed_items = []

            for idx, (original_idx, item, attempt_count) in enumerate(current_batch):
                # Check if we should stop
                if self.should_stop_processing.is_set():
                    return results

                # Extract captions for this item
                base_idx = idx * len(self.inference_prompts)
                captions = []

                for j in range(len(self.inference_prompts)):
                    if base_idx + j < len(outputs) and outputs[base_idx + j].outputs:
                        original_caption = outputs[base_idx + j].outputs[0].text
                        caption_text = self._clean_output(original_caption)
                        if caption_text:
                            captions.append(caption_text)
                        else:
                            logger.warning(
                                f"(item {item.item_key}) caption destroyed: {original_caption}"
                            )

                if captions:
                    # Success - add to results
                    result = ProcessedResult(
                        chunk_id=item.chunk_id,
                        shard_name=Path(item.chunk_id).stem.rsplit("_chunk_", 1)[0],
                        item_key=item.item_key,
                        captions=captions,
                        image_width=item.image.width,
                        image_height=item.image.height,
                        image_format=item.image.format or "unknown",
                        file_size=len(item.image_data),
                        processing_time_ms=0,  # Will be calculated by caller
                    )
                    results.append(result)
                    self.items_processed += 1
                else:
                    # Failed - check if we should retry
                    if attempt_count + 1 < max_attempts:
                        failed_items.append((original_idx, item, attempt_count + 1))
                        logger.warning(
                            f"Item {item.item_key} failed (attempt {attempt_count + 1}/{max_attempts}), will retry"
                        )
                    else:
                        logger.error(f"Item {item.item_key} failed after {max_attempts} attempts")
                        self.items_failed += 1

            # Update items to process for next iteration
            items_to_process = failed_items

            # Log retry status if we have items to retry
            if items_to_process:
                logger.info(f"Retrying {len(items_to_process)} failed items")

        return results

    def _inference_thread(self):
        """Background thread for vLLM inference."""
        logger.info("Starting inference thread")

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

                # Process batch with retries
                results = self._process_batch_with_retries(batch)

                # Calculate processing time per item
                if results:
                    processing_time_per_item = (time.time() - start_time) * 1000 / len(batch)

                    # Update processing time and queue results
                    for result in results:
                        result.processing_time_ms = processing_time_per_item
                        self.result_queue.put(result)

                logger.debug(
                    f"Batch processing complete: {len(results)} successful, {len(batch) - len(results)} failed"
                )

            except Empty:
                continue
            except Exception as e:
                if self.should_stop_processing.is_set():
                    continue
                logger.error(f"Inference error: {e}", exc_info=True)

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

        try:
            while self.running and self.connected.is_set():
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
                                raise  # Re-raise to trigger task completion
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
                    if isinstance(e, websockets.exceptions.ConnectionClosed):
                        raise  # Re-raise connection errors
                    logger.error(f"Unexpected error in result sender: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.debug("Result sender cancelled")
            raise

    async def _heartbeat_loop(self):
        """Send periodic heartbeats with connection checking."""
        try:
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
                    raise  # Re-raise to trigger task completion
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    raise  # Re-raise to trigger task completion
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
            raise

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        self.running = False
        self.connected.clear()
        self.should_stop_processing.set()

        # Stop processing threads by adding stop signals
        self.readahead_queue.put(None)
        self.inference_queue.put(None)

        # Shutdown image processor
        if self.image_processor is not None:
            self.image_processor.shutdown()

        # Close websocket if connected
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        logger.info("Worker shutdown complete")
