"""Caption worker for vLLM-based distributed image captioning with multi-stage processing."""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import asyncio
import io
import json
import logging
import websockets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from queue import Queue, Empty
from threading import Thread, Lock, Event
from collections import deque, defaultdict

from PIL import Image
import numpy as np
from huggingface_hub import get_token

from .base import BaseWorker
from ..models import JobStatus, Job
from ..utils import CaptionUtils
from ..utils.dataset_loader import DatasetLoader
from ..utils.vllm_config import VLLMConfigManager
from ..utils.image_processor import ImageProcessor
from ..utils.shard_processor import HFDatasetShardProcessor, WebDatasetShardProcessor
from ..utils.prompt_template import PromptTemplateManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """Configuration for a single processing stage."""

    name: str
    model: str
    prompts: List[str]
    output_field: str
    requires: List[str] = field(default_factory=list)
    sampling: Optional[Dict[str, Any]] = None

    # Model-specific overrides
    tensor_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None


@dataclass
class StageResult:
    """Results from a single stage."""

    stage_name: str
    output_field: str
    outputs: List[str]  # Multiple outputs from multiple prompts


@dataclass
class ShardChunk:
    """Shard chunk assignment with unprocessed ranges."""

    chunk_id: str
    shard_url: str
    shard_name: str
    start_index: int
    chunk_size: int
    unprocessed_ranges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ProcessingItem:
    """Item being processed."""

    chunk_id: str
    item_key: str
    image: Image.Image
    image_data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)  # Accumulated results


@dataclass
class ProcessedResult:
    """Result with multi-stage outputs."""

    chunk_id: str
    shard_name: str
    item_key: str
    outputs: Dict[str, List[str]]  # field_name -> list of outputs
    image_width: int
    image_height: int
    image_format: str
    file_size: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiStageVLLMManager:
    """Manages multiple vLLM instances for different models."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.models: Dict[str, Any] = {}  # model_name -> LLM instance
        self.processors: Dict[str, Any] = {}  # model_name -> processor
        self.tokenizers: Dict[str, Any] = {}  # model_name -> tokenizer
        self.sampling_params: Dict[str, Any] = {}  # stage_name -> SamplingParams

    def load_model(self, model_name: str, stage: ProcessingStage, base_config: Dict[str, Any]):
        """Load a model if not already loaded."""
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded, reusing instance")
            return

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoProcessor

        logger.info(f"Loading model {model_name} for stage {stage.name}")

        # Build model-specific config by merging base config with stage overrides
        model_config = base_config.copy()
        model_config["model"] = model_name

        # Apply stage-specific overrides
        if stage.tensor_parallel_size is not None:
            model_config["tensor_parallel_size"] = stage.tensor_parallel_size
        if stage.max_model_len is not None:
            model_config["max_model_len"] = stage.max_model_len
        if stage.dtype is not None:
            model_config["dtype"] = stage.dtype
        if stage.gpu_memory_utilization is not None:
            model_config["gpu_memory_utilization"] = stage.gpu_memory_utilization

        # Load tokenizer and processor
        self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        self.processors[model_name] = AutoProcessor.from_pretrained(model_name)

        # Initialize LLM
        vllm_params = {
            "model": model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": model_config.get("tensor_parallel_size", 1),
            "max_model_len": model_config.get("max_model_len", 16384),
            "enforce_eager": model_config.get("enforce_eager", True),
            "gpu_memory_utilization": model_config.get("gpu_memory_utilization", 0.92),
            "dtype": model_config.get("dtype", "float16"),
            "limit_mm_per_prompt": model_config.get("limit_mm_per_prompt", {"image": 1}),
            "disable_mm_preprocessor_cache": model_config.get(
                "disable_mm_preprocessor_cache", True
            ),
        }

        self.models[model_name] = LLM(**vllm_params)
        logger.info(f"Model {model_name} loaded successfully")

    def create_sampling_params(self, stage: ProcessingStage, base_sampling: Dict[str, Any]):
        """Create sampling params for a stage."""
        from vllm import SamplingParams

        # Start with base sampling config
        sampling_config = base_sampling.copy()

        # Override with stage-specific sampling if provided
        if stage.sampling:
            sampling_config.update(stage.sampling)

        params = SamplingParams(
            temperature=sampling_config.get("temperature", 0.7),
            top_p=sampling_config.get("top_p", 0.95),
            max_tokens=sampling_config.get("max_tokens", 256),
            stop=sampling_config.get("stop", ["<|end|>", "<|endoftext|>", "<|im_end|>"]),
            repetition_penalty=sampling_config.get("repetition_penalty", 1.05),
            skip_special_tokens=sampling_config.get("skip_special_tokens", True),
        )

        self.sampling_params[stage.name] = params
        return params

    def get_model_for_stage(self, stage_name: str, model_name: str) -> Tuple[Any, Any, Any, Any]:
        """
        Get model components for a stage.

        Returns:
            tuple: A tuple containing:
                - llm: The language model instance for the given model name.
                - processor: The processor associated with the model.
                - tokenizer: The tokenizer for the model.
                - sampling_params: The sampling parameters for the given stage.
        """
        return (
            self.models[model_name],
            self.processors[model_name],
            self.tokenizers[model_name],
            self.sampling_params[stage_name],
        )

    def cleanup(self):
        """Clean up all loaded models."""
        for model_name in list(self.models.keys()):
            del self.models[model_name]
            del self.processors[model_name]
            del self.tokenizers[model_name]
        self.sampling_params.clear()

        import gc

        gc.collect()


class CaptionWorker(BaseWorker):
    """Worker that processes shard chunks for image captioning using multi-stage vLLM."""

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
        self.stages: List[ProcessingStage] = []
        self.stage_order: List[str] = []  # Topologically sorted stage names
        self.vllm_config_manager = VLLMConfigManager()
        self.model_manager = None

        # Backward compatibility: local config for GPU selection
        self.gpu_id = config.get("gpu_id", 0)

        # Connection state events
        self.should_stop_processing = Event()

        # Image processor
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

    def _parse_stages_config(self, vllm_config: Dict[str, Any]) -> List[ProcessingStage]:
        """Parse stages configuration from vLLM config."""
        stages_config = vllm_config.get("stages", [])

        if not stages_config:
            # Backward compatibility: create single stage from old config
            return [
                ProcessingStage(
                    name="default",
                    model=vllm_config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
                    prompts=vllm_config.get("inference_prompts", ["describe this image"]),
                    output_field="captions",
                    requires=[],
                )
            ]

        # Parse stages
        stages = []
        for stage_cfg in stages_config:
            stage = ProcessingStage(
                name=stage_cfg["name"],
                model=stage_cfg.get("model", vllm_config.get("model")),
                prompts=stage_cfg.get("prompts", []),
                output_field=stage_cfg.get("output_field", "captions"),
                requires=stage_cfg.get("requires", []),
                sampling=stage_cfg.get("sampling"),
                tensor_parallel_size=stage_cfg.get("tensor_parallel_size"),
                max_model_len=stage_cfg.get("max_model_len"),
                dtype=stage_cfg.get("dtype"),
                gpu_memory_utilization=stage_cfg.get("gpu_memory_utilization"),
            )
            stages.append(stage)

        return stages

    def _topological_sort_stages(self, stages: List[ProcessingStage]) -> List[str]:
        """Sort stages by dependencies."""
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        stage_map = {s.name: s for s in stages}

        for stage in stages:
            in_degree[stage.name] = len(stage.requires)
            for dep in stage.requires:
                if dep not in stage_map:
                    raise ValueError(f"Stage '{stage.name}' requires missing dependency '{dep}'")
                graph[dep].append(stage.name)

        # Topological sort using Kahn's algorithm
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(stages):
            raise ValueError("Circular dependency detected in stages")

        return result

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

        elif msg_type == "config_update":
            # Soft config update without reload
            if data.get("vllm_config"):
                self._handle_vllm_config_update(data["vllm_config"])

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
            "stages": len(self.stages),
            "models_loaded": len(self.model_manager.models) if self.model_manager else 0,
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

        # Cleanup model manager
        if self.model_manager:
            self.model_manager.cleanup()

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

            # Parse stages configuration
            self.stages = self._parse_stages_config(self.vllm_config)
            self.stage_order = self._topological_sort_stages(self.stages)

            logger.info(f"Configured {len(self.stages)} processing stages: {self.stage_order}")

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
        """Initialize multi-stage vLLM components."""
        if not self.vllm_config:
            raise RuntimeError("vLLM config not received from orchestrator")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Initialize model manager
        self.model_manager = MultiStageVLLMManager(self.gpu_id)

        # Get base config for models
        base_config = {
            "tensor_parallel_size": self.vllm_config.get("tensor_parallel_size", 1),
            "max_model_len": self.vllm_config.get("max_model_len", 16384),
            "dtype": self.vllm_config.get("dtype", "float16"),
            "gpu_memory_utilization": self.vllm_config.get("gpu_memory_utilization", 0.92),
            "enforce_eager": self.vllm_config.get("enforce_eager", True),
            "disable_mm_preprocessor_cache": self.vllm_config.get(
                "disable_mm_preprocessor_cache", True
            ),
            "limit_mm_per_prompt": self.vllm_config.get("limit_mm_per_prompt", {"image": 1}),
        }

        base_sampling = self.vllm_config.get("sampling", {})

        # Load models for all stages
        unique_models = set()
        for stage in self.stages:
            unique_models.add(stage.model)

        logger.info(f"Loading {len(unique_models)} unique models for {len(self.stages)} stages")

        for stage in self.stages:
            self.model_manager.load_model(stage.model, stage, base_config)
            self.model_manager.create_sampling_params(stage, base_sampling)

        logger.info("Multi-stage vLLM initialization complete")

        # Update config manager's tracking
        self.vllm_config_manager.current_config = self.vllm_config

    def _handle_vllm_config_update(self, new_config: Dict[str, Any]) -> bool:
        """Handle vLLM configuration updates for multi-stage."""
        if not new_config:
            return True

        # Parse new stages
        new_stages = self._parse_stages_config(new_config)

        # Check if stages changed significantly
        stages_changed = len(new_stages) != len(self.stages)
        if not stages_changed:
            for old, new in zip(self.stages, new_stages):
                if (
                    old.name != new.name
                    or old.model != new.model
                    or old.prompts != new.prompts
                    or old.output_field != new.output_field
                ):
                    stages_changed = True
                    break

        if stages_changed:
            logger.info("Stage configuration changed, reloading all models")

            # Save old config
            old_config = self.vllm_config
            self.vllm_config = new_config
            self.stages = new_stages
            self.stage_order = self._topological_sort_stages(self.stages)

            try:
                # Cleanup old models
                if self.model_manager:
                    self.model_manager.cleanup()

                # Reload with new config
                self._setup_vllm()

                logger.info("Multi-stage vLLM reload complete")
                return True

            except Exception as e:
                logger.error(f"Failed to reload vLLM: {e}")
                # Restore old config
                self.vllm_config = old_config
                return False
        else:
            # Just update sampling params for existing stages
            logger.info("Updating sampling parameters without model reload")

            base_sampling = new_config.get("sampling", {})
            for stage in self.stages:
                self.model_manager.create_sampling_params(stage, base_sampling)

            self.vllm_config = new_config
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

    def _process_shard_chunk(self, chunk: ShardChunk):
        """Process a single shard chunk with item-level tracking."""
        logger.info(
            f"Processing shard {chunk.shard_name} with unprocessed ranges: {chunk.unprocessed_ranges}"
        )

        # Select appropriate processor
        if chunk.shard_url.startswith("hf_dataset:"):
            processor = self.hf_processor
        else:
            processor = self.webdataset_processor

        items_processed = 0

        # Let the processor handle the range filtering
        for key, url, image_data, metadata in processor.iterate_chunk_with_metadata(
            chunk, self.dataset_loader, self.should_stop_processing, self.connected
        ):
            try:
                # Load image
                img = Image.open(io.BytesIO(image_data))

                # Create processing item
                item = ProcessingItem(
                    chunk_id=chunk.chunk_id,
                    item_key=key,
                    image=img,
                    image_data=image_data,
                    metadata=metadata,
                )

                # Store absolute item index for tracking
                # The processor should provide the correct index in metadata
                if "_chunk_relative_index" in metadata:
                    item.metadata["_item_index"] = (
                        chunk.start_index + metadata["_chunk_relative_index"]
                    )

                # Add to readahead queue with timeout handling
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

                # If we couldn't queue due to disconnection, stop processing
                if not self.connected.is_set() or self.should_stop_processing.is_set():
                    logger.debug(f"Skipping remaining items due to disconnection")
                    break

                items_processed += 1

                # Batch items for inference
                batch_size = self.vllm_config.get("batch_size", 8)
                if self.readahead_queue.qsize() >= batch_size:
                    self._batch_for_inference()

            except Exception as e:
                if self.should_stop_processing.is_set():
                    break
                logger.error(f"Error processing item {key}: {e}")
                self.items_failed += 1

        # Process any remaining items in queue
        if not self.should_stop_processing.is_set():
            self._batch_for_inference()

        logger.info(
            f"Chunk {chunk.chunk_id} processed {items_processed} items from unprocessed ranges"
        )

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

    async def _result_sender(self):
        """Send results back to orchestrator with item index."""
        pending_results = []

        try:
            while self.running and self.connected.is_set():
                try:
                    # Get result with timeout
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
                                # Build message with item index
                                message_data = {
                                    "type": "submit_captions",
                                    "chunk_id": result.chunk_id,
                                    "dataset": self.dataset_config.get("dataset_path", "unknown"),
                                    "shard": result.shard_name,
                                    "item_key": result.item_key,
                                    "item_index": result.item_index,  # NEW: Include index
                                    "outputs": result.outputs,
                                    "captions": result.outputs.get("captions", []),  # Compatibility
                                    "caption_count": sum(len(v) for v in result.outputs.values()),
                                    "image_width": result.image_width,
                                    "image_height": result.image_height,
                                    "image_format": result.image_format,
                                    "file_size": result.file_size,
                                    "processing_time_ms": result.processing_time_ms,
                                    "metadata": result.metadata,
                                }

                                await self.websocket.send(json.dumps(message_data))
                                sent_results.append(result)

                                if self.items_processed % 100 == 0:
                                    total_outputs = sum(
                                        len(outputs) for outputs in result.outputs.values()
                                    )
                                    logger.info(
                                        f"Processed {self.items_processed} items "
                                        f"(~{total_outputs} outputs across {len(result.outputs)} fields)"
                                    )

                            except websockets.exceptions.ConnectionClosed as e:
                                logger.warning(f"Connection lost while sending result: {e}")
                                raise
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
                        raise
                    logger.error(f"Unexpected error in result sender: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.debug("Result sender cancelled")
            raise

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

    def _process_batch_multi_stage(
        self, batch: List[ProcessingItem], max_attempts: int = 3
    ) -> List[ProcessedResult]:
        """Process a batch through all stages sequentially."""
        results = []

        # Process each stage in order
        for stage_name in self.stage_order:
            stage = next(s for s in self.stages if s.name == stage_name)
            logger.debug(f"Processing batch through stage: {stage_name}")

            # Get model components for this stage
            llm, processor, tokenizer, sampling_params = self.model_manager.get_model_for_stage(
                stage_name, stage.model
            )

            # Track items for retry
            items_to_process = [(i, item, 0) for i, item in enumerate(batch)]

            while items_to_process:
                # Build requests for current items
                current_batch = []
                current_indices = []
                requests = []

                for idx, (original_idx, item, attempt_count) in enumerate(items_to_process):
                    current_batch.append((original_idx, item, attempt_count))
                    current_indices.append(idx)

                    # Prepare image
                    converted_img = ImageProcessor.prepare_for_inference(item.image)

                    # Create template manager for this stage's prompts
                    template_manager = PromptTemplateManager(stage.prompts)

                    # Build context including metadata and previous stage results
                    context = item.metadata.copy()

                    # Add previous stage outputs to context
                    for prev_stage_name, stage_result in item.stage_results.items():
                        # Add outputs with stage name prefix
                        for i, output in enumerate(stage_result.outputs):
                            context[f"{prev_stage_name}_output_{i}"] = output
                        # Also add under output field name
                        if len(stage_result.outputs) == 1:
                            context[stage_result.output_field] = stage_result.outputs[0]
                        else:
                            context[stage_result.output_field] = stage_result.outputs

                    # Format prompts with context
                    formatted_prompts = template_manager.format_all(context)

                    # Build requests for all prompts
                    for prompt in formatted_prompts:
                        req = self._build_vllm_input(converted_img, prompt, processor, tokenizer)
                        requests.append(req)

                # Run inference
                outputs = llm.generate(requests, sampling_params)

                # Process outputs
                successful_items = []
                failed_items = []

                for idx, (original_idx, item, attempt_count) in enumerate(current_batch):
                    # Check if we should stop
                    if self.should_stop_processing.is_set():
                        return results

                    # Extract outputs for this item
                    base_idx = idx * len(stage.prompts)
                    stage_outputs = []

                    for j in range(len(stage.prompts)):
                        if base_idx + j < len(outputs) and outputs[base_idx + j].outputs:
                            original_output = outputs[base_idx + j].outputs[0].text
                            cleaned_output = self._clean_output(original_output)
                            if cleaned_output:
                                stage_outputs.append(cleaned_output)
                            else:
                                logger.warning(
                                    f"(stage {stage_name}, item {item.item_key}) output destroyed: {original_output}"
                                )

                    if stage_outputs:
                        # Success - add stage result to item
                        stage_result = StageResult(
                            stage_name=stage_name,
                            output_field=stage.output_field,
                            outputs=stage_outputs,
                        )
                        item.stage_results[stage_name] = stage_result
                        successful_items.append((original_idx, item))
                    else:
                        # Failed - check if we should retry
                        if attempt_count + 1 < max_attempts:
                            failed_items.append((original_idx, item, attempt_count + 1))
                            logger.warning(
                                f"Stage {stage_name} failed for item {item.item_key} "
                                f"(attempt {attempt_count + 1}/{max_attempts}), will retry"
                            )
                        else:
                            logger.error(
                                f"Stage {stage_name} failed for item {item.item_key} "
                                f"after {max_attempts} attempts"
                            )
                            self.items_failed += 1

                # Update items to process for next iteration
                items_to_process = failed_items

                # Update batch with successful items for next stage
                batch = [item for _, item in successful_items]

                # Log retry status if we have items to retry
                if items_to_process:
                    logger.info(
                        f"Retrying {len(items_to_process)} failed items for stage {stage_name}"
                    )

        # Convert batch items to results
        for item in batch:
            # Aggregate outputs by field name
            outputs_by_field = defaultdict(list)

            for stage_result in item.stage_results.values():
                outputs_by_field[stage_result.output_field].extend(stage_result.outputs)

            result = ProcessedResult(
                chunk_id=item.chunk_id,
                shard_name=Path(item.chunk_id).stem.rsplit("_chunk_", 1)[0],
                item_key=item.item_key,
                outputs=dict(outputs_by_field),  # Convert defaultdict to dict
                image_width=item.image.width,
                image_height=item.image.height,
                image_format=item.image.format or "unknown",
                file_size=len(item.image_data),
                processing_time_ms=0,  # Will be calculated by caller
                metadata=item.metadata,
            )
            results.append(result)
            self.items_processed += 1

        return results

    def _inference_thread(self):
        """Background thread for multi-stage vLLM inference."""
        logger.info("Starting multi-stage inference thread")

        while self.running:
            try:
                # Get batch from queue with timeout
                batch = self.inference_queue.get(timeout=1)

                if not batch:
                    continue

                # Skip if disconnected
                if self.should_stop_processing.is_set():
                    continue

                logger.debug(
                    f"Processing batch of {len(batch)} images through {len(self.stages)} stages"
                )
                start_time = time.time()

                # Process batch through all stages
                results = self._process_batch_multi_stage(batch)

                # Calculate processing time per item
                if results:
                    processing_time_per_item = (time.time() - start_time) * 1000 / len(batch)

                    # Update processing time and queue results
                    for result in results:
                        result.processing_time_ms = processing_time_per_item
                        self.result_queue.put(result)

                logger.debug(
                    f"Multi-stage batch processing complete: {len(results)} successful, "
                    f"{len(batch) - len(results)} failed"
                )

            except Empty:
                continue
            except Exception as e:
                if self.should_stop_processing.is_set():
                    continue
                logger.error(f"Inference error: {e}", exc_info=True)

    def _build_vllm_input(self, image: Image.Image, prompt: str, processor, tokenizer) -> Dict:
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

            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

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
        """Send results back to orchestrator with multi-stage outputs."""
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
                                # For backward compatibility, if there's only one output field "captions"
                                # send it in the old format
                                if len(result.outputs) == 1 and "captions" in result.outputs:
                                    # Old format for single-stage compatibility
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
                                                "item_index": result.metadata.get("_item_index"),
                                                "captions": result.outputs["captions"],
                                                "caption_count": len(result.outputs["captions"]),
                                                "image_width": result.image_width,
                                                "image_height": result.image_height,
                                                "image_format": result.image_format,
                                                "file_size": result.file_size,
                                                "processing_time_ms": result.processing_time_ms,
                                            }
                                        )
                                    )
                                else:
                                    # New format for multi-stage outputs
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
                                                "outputs": result.outputs,  # Dict of field -> list of outputs
                                                "captions": result.outputs.get(
                                                    "captions", []
                                                ),  # For compatibility
                                                "caption_count": sum(
                                                    len(v) for v in result.outputs.values()
                                                ),
                                                "image_width": result.image_width,
                                                "image_height": result.image_height,
                                                "image_format": result.image_format,
                                                "file_size": result.file_size,
                                                "processing_time_ms": result.processing_time_ms,
                                                "metadata": result.metadata,
                                            }
                                        )
                                    )

                                sent_results.append(result)

                                if self.items_processed % 100 == 0:
                                    total_outputs = sum(
                                        len(outputs) for outputs in result.outputs.values()
                                    )
                                    logger.info(
                                        f"Processed {self.items_processed} items "
                                        f"(~{total_outputs} outputs across {len(result.outputs)} fields)"
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
