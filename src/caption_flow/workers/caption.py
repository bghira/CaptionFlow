"""Caption worker with processor abstraction for distributed captioning."""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import asyncio
import json
import logging
import websockets
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from queue import Queue, Empty
from threading import Thread, Event, Lock
from collections import defaultdict, deque

from PIL import Image
from huggingface_hub import get_token

from .base import BaseWorker
from ..processors import (
    ProcessorConfig,
    WorkAssignment,
    WorkUnit,
    WorkResult,
    WebDatasetWorkerProcessor,
    HuggingFaceDatasetWorkerProcessor,
    LocalFilesystemWorkerProcessor,
)
from ..utils.vllm_config import VLLMConfigManager
from ..utils.image_processor import ImageProcessor
from ..utils.prompt_template import PromptTemplateManager
from ..models import ProcessingStage, StageResult

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@dataclass
class ProcessingItem:
    """Item being processed through stages."""

    unit_id: str
    job_id: str
    chunk_id: str
    item_key: str
    item_index: int
    image: Image.Image
    image_data: bytes
    metadata: Dict[str, Any]
    stage_results: Dict[str, StageResult] = None

    def __post_init__(self):
        if self.stage_results is None:
            self.stage_results = {}


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

        # Build model-specific config
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

        sampling_config = base_sampling.copy()
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
        """Get model components for a stage."""
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
    """Worker that processes work units for image captioning using multi-stage vLLM."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Processor configuration - will be set from orchestrator
        self.processor_type = None
        self.processor: Optional[
            Union[
                WebDatasetWorkerProcessor,
                HuggingFaceDatasetWorkerProcessor,
                LocalFilesystemWorkerProcessor,
            ],
        ] = None
        self.dataset_path: Optional[str] = None

        # vLLM configuration
        self.vllm_config = None
        self.stages: List[ProcessingStage] = []
        self.stage_order: List[str] = []
        self.vllm_config_manager = VLLMConfigManager()
        self.model_manager = None

        # GPU selection
        self.gpu_id = config.get("gpu_id", 0)
        self.hf_token = get_token()

        # Image processor
        batch_image_processing = config.get("batch_image_processing", False)
        self.image_processor = ImageProcessor() if batch_image_processing else None

        # Work processing
        self.work_lock = Lock()
        self.assigned_units = deque()
        self.current_unit: Optional[WorkUnit] = None

        # Processing queues
        self.readahead_queue = Queue(maxsize=256)
        self.inference_queue = Queue(maxsize=128)
        self.result_queue = Queue()

        # Processing control
        self.should_stop_processing = Event()

    def _init_metrics(self):
        """Initialize worker metrics."""
        self.items_processed = 0
        self.items_failed = 0
        self.units_completed = 0

    def _get_auth_data(self) -> Dict[str, Any]:
        """Get authentication data."""
        return {"token": self.token, "name": self.name}

    def _get_current_unit_id(self) -> Optional[str]:
        """Get the current unit ID."""
        return self.current_unit.unit_id if self.current_unit else None

    async def _pre_start(self):
        """Initialize before starting connection loop."""
        # Wait for initial connection to get config
        logger.info("Connecting to orchestrator for configuration...")

        config_received = False
        while not config_received and self.running:
            try:
                await self._initial_connect_for_config()
                config_received = True
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                await asyncio.sleep(5)

        # Initialize vLLM once we have config
        if self.vllm_config:
            self._setup_vllm()

        # Start background threads
        Thread(target=self._unit_processor_thread, daemon=True).start()
        Thread(target=self._inference_thread, daemon=True).start()

    async def _initial_connect_for_config(self):
        """Connect initially just to get configuration."""
        logger.info(f"Connecting to {self.server_url}")
        async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
            await websocket.send(json.dumps(self._get_auth_data()))

            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                raise RuntimeError(f"Authentication failed: {welcome_data['error']}")

            # Extract vLLM config from processor config
            processor_config = welcome_data.get("processor_config", {})
            self.vllm_config = processor_config.get("vllm", {})

            if not self.vllm_config:
                raise RuntimeError("No vLLM configuration received from orchestrator")

            # Parse stages
            self.stages = self._parse_stages_config(self.vllm_config)
            self.stage_order = self._topological_sort_stages(self.stages)

            logger.info(f"Configured {len(self.stages)} processing stages: {self.stage_order}")

    async def _handle_welcome(self, welcome_data: Dict[str, Any]):
        """Handle welcome message from orchestrator."""
        with self.work_lock:
            self.assigned_units.clear()
            self.current_unit = None

        self._clear_queue(self.readahead_queue)
        self._clear_queue(self.inference_queue)
        self._clear_queue(self.result_queue)

        # Reset counters
        self.items_processed = 0
        self.items_failed = 0
        self.units_completed = 0

        # Setup processor
        self.processor_type = welcome_data.get("processor_type", None)
        assert self.processor_type is not None, "Processor type not found in welcome data"
        logger.info(f"Creating {self.processor_type} processor")
        processor_config = ProcessorConfig(
            processor_type=self.processor_type, config=welcome_data.get("processor_config", {})
        )

        if self.processor_type == "webdataset":
            self.processor = WebDatasetWorkerProcessor()
        elif self.processor_type == "huggingface_datasets":
            self.processor = HuggingFaceDatasetWorkerProcessor()
        elif self.processor_type == "local_filesystem":
            self.processor = LocalFilesystemWorkerProcessor()
        else:
            raise ValueError(f"Unknown processor type: {self.processor_type}")

        self.processor.initialize(processor_config)
        self.dataset_path = self.processor.dataset_path

        # Update vLLM config if provided
        new_vllm_config = welcome_data.get("processor_config", {}).get("vllm")
        if new_vllm_config and new_vllm_config != self.vllm_config:
            logger.info("Received updated vLLM configuration")
            self._handle_vllm_config_update(new_vllm_config)

        # Clear stop signal
        self.should_stop_processing.clear()

        # Request initial work
        if self.websocket:
            await self.websocket.send(json.dumps({"type": "request_work", "count": 2}))

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle message from orchestrator."""
        msg_type = data.get("type")

        if msg_type == "work_assignment":
            assignment = WorkAssignment.from_dict(data["assignment"])
            with self.work_lock:
                for unit in assignment.units:
                    self.assigned_units.append(unit)
            logger.info(f"Received {len(assignment.units)} work units")

        elif msg_type == "no_work":
            logger.info("No work available")
            await asyncio.sleep(10)

            if self.websocket and self.connected.is_set():
                await self.websocket.send(json.dumps({"type": "request_work", "count": 2}))

    def _parse_stages_config(self, vllm_config: Dict[str, Any]) -> List[ProcessingStage]:
        """Parse stages configuration from vLLM config."""
        stages_config = vllm_config.get("stages", [])

        if not stages_config:
            # Backward compatibility
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
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        stage_map = {s.name: s for s in stages}

        for stage in stages:
            in_degree[stage.name] = len(stage.requires)
            for dep in stage.requires:
                if dep not in stage_map:
                    raise ValueError(f"Stage '{stage.name}' requires missing dependency '{dep}'")
                graph[dep].append(stage.name)

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

    def _setup_vllm(self):
        """Initialize multi-stage vLLM components."""
        if not self.vllm_config:
            raise RuntimeError("vLLM config not received")

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

    def _handle_vllm_config_update(self, new_config: Dict[str, Any]) -> bool:
        """Handle vLLM configuration updates."""
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

            old_config = self.vllm_config
            self.vllm_config = new_config
            self.stages = new_stages
            self.stage_order = self._topological_sort_stages(self.stages)

            try:
                if self.model_manager:
                    self.model_manager.cleanup()
                self._setup_vllm()
                return True
            except Exception as e:
                logger.error(f"Failed to reload vLLM: {e}")
                self.vllm_config = old_config
                return False
        else:
            # Just update sampling params
            logger.info("Updating sampling parameters without model reload")
            base_sampling = new_config.get("sampling", {})
            for stage in self.stages:
                self.model_manager.create_sampling_params(stage, base_sampling)
            self.vllm_config = new_config
            return True

    def _unit_processor_thread(self):
        """Background thread that processes work units."""
        logger.info("Starting unit processor thread")

        while self.running:
            if self.should_stop_processing.is_set():
                time.sleep(1)
                continue

            if not self.connected.is_set():
                time.sleep(1)
                continue

            # Get next unit
            with self.work_lock:
                if not self.current_unit and self.assigned_units:
                    self.current_unit = self.assigned_units.popleft()
                    logger.info(f"Starting unit {self._get_current_unit_id()}")

            if not self.current_unit:
                time.sleep(1)
                continue

            try:
                self._process_work_unit(self.current_unit)

                if self.connected.is_set() and not self.should_stop_processing.is_set():
                    logger.info(f"Completed unit {self._get_current_unit_id()}")
                    self.units_completed += 1

                    # Request more work if needed
                    with self.work_lock:
                        queue_size = len(self.assigned_units)

                    if queue_size < 2 and self.websocket and self.main_loop:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self.websocket.send(
                                    json.dumps({"type": "request_work", "count": 2})
                                ),
                                self.main_loop,
                            ).result(timeout=5)
                        except Exception as e:
                            logger.warning(f"Could not request more work: {e}")

                with self.work_lock:
                    self.current_unit = None

            except Exception as e:
                logger.error(f"Error processing unit: {e}", exc_info=True)
                with self.work_lock:
                    self.current_unit = None

    def _process_work_unit(self, unit: WorkUnit):
        """Process a single work unit."""
        if not self.processor:
            logger.error("Processor not initialized")
            return

        items_processed = 0
        context = {}  # Will store processed indices

        # Get items from processor
        for item_data in self.processor.process_unit(unit, context):
            try:
                # Create processing item
                item = ProcessingItem(
                    unit_id=unit.unit_id,
                    chunk_id=unit.chunk_id,
                    job_id=item_data["job_id"],
                    item_key=item_data["item_key"],
                    item_index=item_data["item_index"],
                    image=item_data["image"],
                    image_data=item_data.get("image_data", b""),
                    metadata=item_data.get("metadata", {}),
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

                if not self.connected.is_set() or self.should_stop_processing.is_set():
                    break

                items_processed += 1

                # Batch items for inference
                batch_size = self.vllm_config.get("batch_size", 8)
                if self.readahead_queue.qsize() >= batch_size:
                    self._batch_for_inference()

            except Exception as e:
                if self.should_stop_processing.is_set():
                    break
                logger.error(f"Error processing item {item_data.get('item_key')}: {e}")
                self.items_failed += 1

        # Process any remaining items
        if not self.should_stop_processing.is_set():
            self._batch_for_inference()
            if self.connected.is_set():
                # Notify orchestrator that unit is complete
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(
                        json.dumps({"type": "work_complete", "unit_id": unit.unit_id})
                    ),
                    self.main_loop,
                ).result(timeout=5)

        logger.info(f"Unit {unit.unit_id} processed {items_processed} items")

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

    def _inference_thread(self):
        """Background thread for multi-stage vLLM inference."""
        logger.info("Starting multi-stage inference thread")

        while self.running:
            try:
                batch = self.inference_queue.get(timeout=1)
                if not batch:
                    continue

                if self.should_stop_processing.is_set():
                    continue

                logger.debug(
                    f"Processing batch of {len(batch)} images through {len(self.stages)} stages"
                )
                start_time = time.time()

                # Process batch through all stages
                results = self._process_batch_multi_stage(batch)

                # Calculate processing time
                if results:
                    processing_time_per_item = (time.time() - start_time) * 1000 / len(batch)

                    for item, result_outputs in results:
                        self.result_queue.put(
                            {
                                "item": item,
                                "outputs": result_outputs,
                                "processing_time_ms": processing_time_per_item,
                            }
                        )

                logger.debug(f"Batch processing complete: {len(results)} successful")

            except Empty:
                continue
            except Exception as e:
                if self.should_stop_processing.is_set():
                    continue
                logger.error(f"Inference error: {e}", exc_info=True)

    def _process_batch_multi_stage(
        self, batch: List[ProcessingItem], max_attempts: int = 3
    ) -> List[Tuple[ProcessingItem, Dict]]:
        """Process a batch through all stages sequentially."""
        results = []

        # Process each stage in order
        for stage_name in self.stage_order:
            stage = next(s for s in self.stages if s.name == stage_name)
            logger.debug(f"Processing batch through stage: {stage_name}")

            # Get model components
            llm, processor, tokenizer, sampling_params = self.model_manager.get_model_for_stage(
                stage_name, stage.model
            )

            # Track items for retry
            items_to_process = [(i, item, 0) for i, item in enumerate(batch)]

            while items_to_process:
                current_batch = []
                requests = []

                for idx, (original_idx, item, attempt_count) in enumerate(items_to_process):
                    current_batch.append((original_idx, item, attempt_count))

                    # Prepare image
                    converted_img = ImageProcessor.prepare_for_inference(item.image)

                    # Create template manager
                    template_manager = PromptTemplateManager(stage.prompts)

                    # Build context
                    context = item.metadata.copy()

                    # Add previous stage results
                    for prev_stage_name, stage_result in item.stage_results.items():
                        for i, output in enumerate(stage_result.outputs):
                            context[f"{prev_stage_name}_output_{i}"] = output
                        if len(stage_result.outputs) == 1:
                            context[stage_result.output_field] = stage_result.outputs[0]
                        else:
                            context[stage_result.output_field] = stage_result.outputs

                    # Format prompts
                    formatted_prompts = template_manager.format_all(context)

                    # Build requests
                    for prompt in formatted_prompts:
                        req = self._build_vllm_input(converted_img, prompt, processor, tokenizer)
                        requests.append(req)

                # Run inference
                outputs = llm.generate(requests, sampling_params)

                # Process outputs
                successful_items = []
                failed_items = []

                for idx, (original_idx, item, attempt_count) in enumerate(current_batch):
                    if self.should_stop_processing.is_set():
                        return results

                    # Extract outputs
                    base_idx = idx * len(stage.prompts)
                    stage_outputs = []

                    for j in range(len(stage.prompts)):
                        if base_idx + j < len(outputs) and outputs[base_idx + j].outputs:
                            original_output = outputs[base_idx + j].outputs[0].text
                            cleaned_output = self._clean_output(original_output)
                            if cleaned_output:
                                stage_outputs.append(cleaned_output)

                    if stage_outputs:
                        # Success
                        stage_result = StageResult(
                            stage_name=stage_name,
                            output_field=stage.output_field,
                            outputs=stage_outputs,
                        )
                        item.stage_results[stage_name] = stage_result
                        successful_items.append((original_idx, item))
                    else:
                        # Failed - check retry
                        if attempt_count + 1 < max_attempts:
                            failed_items.append((original_idx, item, attempt_count + 1))
                        else:
                            logger.error(f"Stage {stage_name} failed for item {item.item_key}")
                            self.items_failed += 1
                            stage_result = StageResult(
                                stage_name=stage_name,
                                output_field=stage.output_field,
                                outputs=[],
                                error=f"Failed after {max_attempts} attempts",
                            )
                            item.stage_results[stage_name] = stage_result
                            self.result_queue.put(
                                {
                                    "item": item,
                                    "outputs": {},
                                    "processing_time_ms": 0.0,
                                    "error": f"Failed stage {stage_name} after {max_attempts} attempts",
                                }
                            )

                # Update for next iteration
                items_to_process = failed_items
                batch = [item for _, item in successful_items]

        # Convert to results
        for item in batch:
            # Aggregate outputs by field
            outputs_by_field = defaultdict(list)
            for stage_result in item.stage_results.values():
                outputs_by_field[stage_result.output_field].extend(stage_result.outputs)

            results.append((item, dict(outputs_by_field)))
            self.items_processed += 1

        return results

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

        for token in ["<|end|>", "<|endoftext|>", "<|im_end|>", "I'm sorry", "I cannot"]:
            if token in text:
                text = text.split(token)[0]

        return text.strip()

    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Get heartbeat data."""
        return {
            "type": "heartbeat",
            "processed": self.items_processed,
            "failed": self.items_failed,
            "units_completed": self.units_completed,
            "current_unit": self._get_current_unit_id() if self.current_unit else None,
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
        return [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._base_message_handler()),
            asyncio.create_task(self._result_sender()),
        ]

    async def _result_sender(self):
        """Send results back to orchestrator."""
        while self.running and self.connected.is_set():
            try:
                # Get result
                result_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.result_queue.get, True, 1
                )

                if self.websocket and self.connected.is_set():
                    item = result_data["item"]
                    logger.debug(f"Handling results for item: {item}")
                    outputs = result_data["outputs"]

                    # Create work result
                    # logger.info(f"Processed item: {item}")
                    work_result = WorkResult(
                        unit_id=item.unit_id,
                        source_id=item.metadata.get("shard_name", "unknown"),
                        chunk_id=item.chunk_id,
                        sample_id=f"{item.item_key}",
                        outputs=outputs,
                        metadata={
                            "item_key": item.item_key,
                            "item_index": item.metadata.get("_item_index"),
                            "image_width": item.image.width,
                            "image_height": item.image.height,
                            "image_format": item.image.format or "unknown",
                            "file_size": len(item.image_data) if item.image_data else 0,
                            **item.metadata,
                        },
                        processing_time_ms=result_data["processing_time_ms"],
                        error=result_data.get("error", None),
                    )

                    # Send result in format that orchestrator expects
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "submit_results",
                                "unit_id": work_result.unit_id,
                                "job_id": item.job_id,
                                "dataset": self.dataset_path,
                                "sample_id": work_result.sample_id,
                                "source_id": work_result.source_id,
                                "outputs": work_result.outputs,
                                "metadata": work_result.metadata,
                                "processing_time_ms": work_result.processing_time_ms,
                            }
                        )
                    )

                    if self.items_processed % 100 == 0:
                        total_outputs = sum(len(v) for v in outputs.values())
                        logger.info(
                            f"Processed {self.items_processed} items (~{total_outputs} outputs)"
                        )

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending result: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _on_disconnect(self):
        """Handle disconnection."""
        self.should_stop_processing.set()

        with self.work_lock:
            self.assigned_units.clear()
            self.current_unit = None

        # Clear queues
        self._clear_queue(self.readahead_queue)
        self._clear_queue(self.inference_queue)
        self._clear_queue(self.result_queue)

    def _clear_queue(self, queue: Queue):
        """Clear all items from a queue."""
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass

    async def _pre_shutdown(self):
        """Cleanup before shutdown."""
        self.readahead_queue.put(None)
        self.inference_queue.put(None)

        if self.image_processor:
            self.image_processor.shutdown()

        if self.model_manager:
            self.model_manager.cleanup()
