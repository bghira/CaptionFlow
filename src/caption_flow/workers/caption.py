"""Caption worker for vLLM-based distributed image captioning."""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import asyncio
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from threading import Thread, Lock, Event
from collections import deque

from PIL import Image
import numpy as np
from huggingface_hub import get_token

from .base_worker import BaseWorker
from .models import JobStatus, Job
from .utils import CaptionUtils
from .utils.dataset_loader import DatasetLoader
from .utils.vllm_config import VLLMConfigManager
from .utils.image_processor import ImageProcessor
from .utils.shard_processor import HFDatasetShardProcessor, WebDatasetShardProcessor

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


class CaptionWorker(BaseWorker):
    """Worker that processes shard chunks for image captioning using vLLM."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        batch_image_processing = config.get("batch_image_processing", False)

        # Dataset configuration will be received from orchestrator
        self.dataset_config = None
        self.dataset_loader = None
        self.dataset_type = None
        self.dataset_split = None
        self.dataset_image_column = None
        self.hf_token = get_token()

        # vLLM configuration will be received from orchestrator
        self.vllm_config = None
        self.inference_prompts = None
        self.vllm_config_manager = VLLMConfigManager()

        # Backward compatibility: local config for GPU selection
        self.gpu_id = config.get("gpu_id", 0)

        # Connection state events
        self.should_stop_processing = Event()

        # Inference components (initialized in setup)
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None
        self.image_processor = None

        if batch_image_processing:
            self.image_processor = ImageProcessor()

        # Shard chunk processing
        self.hf_processor = HFDatasetShardProcessor()
        self.webdataset_processor = WebDatasetShardProcessor(
            hf_token=self.hf_token, dataset_type=self.dataset_type
        )
        self.chunk_lock = Lock()
        self.assigned_chunks = deque()
        self.current_chunk = None
        self.current_chunk_progress = 0

        # Batching queues - will be cleared on disconnect
        self.readahead_queue = Queue(maxsize=256)
        self.inference_queue = Queue(maxsize=128)
        self.result_queue = Queue()

        # Job mode for shards vs jobs and job queue
        self.job_mode = config.get("job_mode", False)
        self.job_queue = Queue(maxsize=32)

    def _init_metrics(self):
        """Initialize worker metrics."""
        self.items_processed = 0
        self.items_failed = 0
        self.chunks_completed = 0

    def _get_auth_data(self) -> Dict[str, Any]:
        """Get authentication data."""
        return {"token": self.token, "name": self.name}

    async def _pre_start(self):
        """Initialize before starting connection loop."""
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

        # Start background threads
        reader_thread = Thread(target=self._shard_reader_thread, daemon=True)
        reader_thread.start()

        inference_thread = Thread(target=self._inference_thread, daemon=True)
        inference_thread.start()

    async def _handle_welcome(self, welcome_data: Dict[str, Any]):
        """Handle welcome message from orchestrator."""
        # Extract and setup dataset configuration
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
            if not self._handle_vllm_config_update(new_vllm_config):
                logger.error("Failed to update vLLM configuration")

        # Clear stop signal now that we're connected
        self.should_stop_processing.clear()

        # Request initial chunks if not in job mode
        if not self.job_mode and self.websocket:
            await self.websocket.send(json.dumps({"type": "request_chunks", "count": 2}))

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle message from orchestrator."""
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

            wait_time = 2 if reason == "state_restoring" else 10
            await asyncio.sleep(wait_time)

            if self.websocket and self.connected.is_set():
                await self.websocket.send(json.dumps({"type": "request_chunks", "count": 2}))

        elif msg_type == "reload_vllm":
            logger.info("Orchestrator requested vLLM reload")
            new_config = data.get("vllm_config")
            if new_config:
                self._handle_vllm_config_update(new_config)

        elif msg_type == "job_assignment":
            await self._handle_job_assignment(data["job"])

        elif msg_type == "no_jobs":
            logger.debug("No jobs available")
            await asyncio.sleep(2)

    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Get heartbeat data."""
        return {
            "type": "heartbeat",
            "processed": self.items_processed,
            "failed": self.items_failed,
            "chunks_completed": self.chunks_completed,
            "current_chunk": self.current_chunk.chunk_id if self.current_chunk else None,
            "chunk_progress": self.current_chunk_progress,
            "queue_sizes": {
                "readahead": self.readahead_queue.qsize(),
                "inference": self.inference_queue.qsize(),
                "results": self.result_queue.qsize(),
            },
        }

    async def _create_tasks(self) -> list:
        """Create async tasks to run."""
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._base_message_handler()),
            asyncio.create_task(self._result_sender()),
        ]

        if self.job_mode:
            tasks.append(asyncio.create_task(self._job_request_loop()))

        return tasks

    async def _on_disconnect(self):
        """Handle disconnection."""
        self._clear_state_on_disconnect()

    async def _pre_shutdown(self):
        """Cleanup before shutdown."""
        # Stop processing threads by adding stop signals
        self.readahead_queue.put(None)
        self.inference_queue.put(None)

        # Shutdown image processor
        if self.image_processor is not None:
            self.image_processor.shutdown()

    async def _initial_connect_for_config(self):
        """Connect initially just to get configuration."""
        logger.info(f"Connecting to {self.server_url}")
        async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
            await websocket.send(json.dumps(self._get_auth_data()))

            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                raise RuntimeError(f"Authentication failed: {welcome_data['error']}")

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

            self.vllm_config_manager.current_config = self.vllm_config

            dataset_config = welcome_data.get("dataset_config", {})
            if dataset_config:
                self._setup_dataset_loader(dataset_config)

            logger.info("Received configuration from orchestrator")

    def _clear_state_on_disconnect(self):
        """Clear all processing state when disconnected."""
        logger.info("Clearing state due to disconnection")

        self.should_stop_processing.set()

        with self.chunk_lock:
            self.assigned_chunks.clear()
            self.current_chunk = None
            self.current_chunk_progress = 0

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

        # Select appropriate processor
        if chunk.shard_url.startswith("hf_dataset:"):
            processor = self.hf_processor
        else:
            processor = self.webdataset_processor

        items_processed = 0

        # Iterate through chunk items using the processor
        for key, url, image_data in processor.iterate_chunk(
            chunk, self.dataset_loader, self.should_stop_processing, self.connected
        ):
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
