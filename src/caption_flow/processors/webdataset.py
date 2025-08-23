"""WebDataset processor implementation."""

import logging
import threading
from typing import Dict, Any, List, Optional, Iterator, Set, Deque, Tuple
from collections import deque, defaultdict
from pathlib import Path
import json
import io
from datetime import datetime
from PIL import Image

from .base import OrchestratorProcessor, WorkerProcessor, ProcessorConfig, WorkUnit, WorkResult
from ..utils import DatasetLoader, ChunkTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WebDatasetOrchestratorProcessor(OrchestratorProcessor):
    """Orchestrator processor for WebDataset shards."""

    def __init__(self):
        logger.debug("Initializing WebDatasetOrchestratorProcessor")
        self.dataset_loader: Optional[DatasetLoader] = None
        self.chunk_tracker: Optional[ChunkTracker] = None
        self.chunk_size: int = 1000

        # Work unit management
        self.work_units: Dict[str, WorkUnit] = {}
        self.pending_units: Deque[str] = deque()
        self.assigned_units: Dict[str, Set[str]] = defaultdict(set)  # worker_id -> unit_ids
        self.lock = threading.Lock()

        # Shard processing state
        self.all_shards: List[str] = []
        self.current_shard_index = 0
        self.current_shard_items = 0

        # Background thread for creating work units
        self.unit_creation_thread: Optional[threading.Thread] = None
        self.stop_creation = threading.Event()

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize WebDataset processor."""
        logger.debug("Initializing orchestrator with config: %s", config.config)
        cfg = config.config

        # Dataset configuration
        dataset_cfg = cfg.get("dataset", {})
        dataset_path = dataset_cfg.get("dataset_path")
        dataset_type = dataset_cfg.get("dataset_type", "huggingface")
        dataset_split = dataset_cfg.get("dataset_split", "train")
        image_column = dataset_cfg.get("dataset_image_column", "image")

        # Chunk settings
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.min_buffer = cfg.get("min_chunk_buffer", 10)
        self.buffer_multiplier = cfg.get("chunk_buffer_multiplier", 3)

        logger.debug(
            "Chunk size: %d, min_buffer: %d, buffer_multiplier: %d",
            self.chunk_size,
            self.min_buffer,
            self.buffer_multiplier,
        )

        # Initialize dataset loader
        if dataset_path:
            checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Checkpoint dir: %s", checkpoint_dir)

            self.dataset_loader = DatasetLoader(
                dataset_path=dataset_path,
                dataset_type=dataset_type,
                split=dataset_split,
                image_column=image_column,
                cache_dir=checkpoint_dir,
            )
            logger.debug("DatasetLoader initialized")

            self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")
            logger.debug("ChunkTracker initialized at %s", checkpoint_dir / "chunks.json")

            # Get all shards
            self.all_shards = self.dataset_loader.get_shard_list()
            logger.debug("All shards: %s", self.all_shards)

            # Restore existing state from chunk tracker
            self._restore_state()

            # Start background unit creation
            self.unit_creation_thread = threading.Thread(
                target=self._create_units_background, daemon=True
            )
            self.unit_creation_thread.start()
            logger.debug("Unit creation thread started")
        else:
            logger.error("No dataset_path provided in config")

    def _restore_state(self) -> None:
        """Restore state from chunk tracker."""
        logger.debug("Restoring state from chunk tracker")
        if not self.chunk_tracker:
            logger.warning("No chunk_tracker available for state restore")
            return

        shards_summary = self.chunk_tracker.get_shards_summary()
        logger.debug("Shards summary from chunk tracker: %s", shards_summary)

        with self.lock:
            for shard_name, shard_info in shards_summary.items():
                for chunk_state in shard_info["chunks"]:
                    logger.debug("Restoring chunk_state: %s", chunk_state)
                    if chunk_state.status in ["pending", "failed", "assigned"]:
                        # Create work unit
                        unit = WorkUnit(
                            unit_id=chunk_state.chunk_id,
                            source_id=shard_name,
                            data={
                                "shard_url": chunk_state.shard_url,
                                "start_index": chunk_state.start_index,
                                "chunk_size": chunk_state.chunk_size,
                                "unprocessed_ranges": getattr(
                                    chunk_state,
                                    "unprocessed_ranges",
                                    [
                                        (
                                            chunk_state.start_index,
                                            chunk_state.start_index + chunk_state.chunk_size - 1,
                                        )
                                    ],
                                ),
                            },
                            metadata={
                                "shard_name": shard_name,
                                "chunk_index": chunk_state.start_index // self.chunk_size,
                            },
                        )

                        self.work_units[unit.unit_id] = unit
                        self.pending_units.append(unit.unit_id)
                        logger.debug("Restored work unit: %s", unit.unit_id)

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

        shard_iter = iter(self.all_shards)
        current_shard_url = None
        current_shard_name = None
        current_shard_items = None
        current_index = 0

        while not self.stop_creation.is_set():
            # Check if we need more units
            with self.lock:
                pending_count = len(self.pending_units)
                assigned_count = sum(len(units) for units in self.assigned_units.values())
                worker_count = max(1, len(self.assigned_units))

                target_buffer = max(self.min_buffer, worker_count * self.buffer_multiplier)
                units_needed = max(0, target_buffer - (pending_count + assigned_count))
                logger.debug(
                    "pending_count=%d assigned_count=%d worker_count=%d target_buffer=%d units_needed=%d",
                    pending_count,
                    assigned_count,
                    worker_count,
                    target_buffer,
                    units_needed,
                )

            if units_needed == 0:
                logger.debug("No units needed, sleeping for 5 seconds")
                threading.Event().wait(5)
                continue

            # Create units as needed
            units_created = 0

            while units_created < units_needed and not self.stop_creation.is_set():
                # Load next shard if needed
                if current_shard_url is None or current_index >= current_shard_items:
                    try:
                        current_shard_url = next(shard_iter)
                        current_shard_name = Path(current_shard_url).stem

                        logger.debug("Loading shard: %s", current_shard_url)
                        # Count items in shard
                        current_shard_items = sum(
                            1 for _ in self.dataset_loader.iterate_shard(current_shard_url)
                        )
                        logger.info(
                            f"Processing shard {current_shard_name} with {current_shard_items} items"
                        )
                        current_index = 0

                    except StopIteration:
                        logger.info("All shards processed")
                        break
                    except Exception as e:
                        logger.error("Error loading shard: %s", e)
                        break

                # Create work unit
                if current_shard_url and current_index < current_shard_items:
                    chunk_size = min(self.chunk_size, current_shard_items - current_index)
                    unit_id = f"{current_shard_name}_chunk_{current_index}"

                    logger.debug(
                        "Creating work unit: unit_id=%s shard=%s start_index=%d chunk_size=%d",
                        unit_id,
                        current_shard_name,
                        current_index,
                        chunk_size,
                    )

                    unit = WorkUnit(
                        unit_id=unit_id,
                        source_id=current_shard_name,
                        data={
                            "shard_url": current_shard_url,
                            "start_index": current_index,
                            "chunk_size": chunk_size,
                            "unprocessed_ranges": [(current_index, current_index + chunk_size - 1)],
                        },
                        metadata={
                            "shard_name": current_shard_name,
                            "chunk_index": current_index // self.chunk_size,
                        },
                    )

                    with self.lock:
                        self.work_units[unit_id] = unit
                        self.pending_units.append(unit_id)
                        logger.debug("Added work unit %s to pending_units", unit_id)

                    if self.chunk_tracker:
                        self.chunk_tracker.add_chunk(
                            unit_id,
                            current_shard_name,
                            current_shard_url,
                            current_index,
                            chunk_size,
                        )
                        logger.debug("Added chunk to chunk_tracker: %s", unit_id)

                    units_created += 1
                    current_index += self.chunk_size

            if units_created > 0:
                logger.info(f"Created {units_created} work units")

    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""
        logger.debug("get_work_units called: count=%d worker_id=%s", count, worker_id)
        assigned = []

        with self.lock:
            # First check if worker has existing assignments with unprocessed items
            if worker_id in self.assigned_units:
                for unit_id in list(self.assigned_units[worker_id]):
                    if len(assigned) >= count:
                        break

                    unit = self.work_units.get(unit_id)
                    if unit and self._has_unprocessed_items(unit):
                        assigned.append(unit)
                        logger.debug(
                            "Re-assigning existing unit %s to worker %s", unit_id, worker_id
                        )

            # Get new units if needed
            while len(assigned) < count and self.pending_units:
                unit_id = self.pending_units.popleft()
                unit = self.work_units.get(unit_id)

                if unit:
                    self.assigned_units[worker_id].add(unit_id)
                    assigned.append(unit)
                    logger.debug("Assigning new unit %s to worker %s", unit_id, worker_id)

                    if self.chunk_tracker:
                        self.chunk_tracker.mark_assigned(unit_id, worker_id)

        logger.debug("Returning %d work units to worker %s", len(assigned), worker_id)
        return assigned

    def _has_unprocessed_items(self, unit: WorkUnit) -> bool:
        """Check if a work unit has unprocessed items."""
        if not self.chunk_tracker:
            logger.debug("No chunk_tracker, assuming unit %s has unprocessed items", unit.unit_id)
            return True

        chunk_info = self.chunk_tracker.get_chunk_with_unprocessed_items(unit.unit_id)
        has_unprocessed = bool(chunk_info and chunk_info.get("unprocessed_ranges"))
        logger.debug("Unit %s has unprocessed items: %s", unit.unit_id, has_unprocessed)
        return has_unprocessed

    def mark_completed(self, unit_id: str, worker_id: str) -> None:
        """Mark a work unit as completed."""
        logger.debug("Marking unit %s as completed by worker %s", unit_id, worker_id)
        with self.lock:
            if unit_id in self.work_units:
                self.assigned_units[worker_id].discard(unit_id)
                logger.debug(
                    "Removed unit %s from assigned_units for worker %s", unit_id, worker_id
                )

                if self.chunk_tracker:
                    self.chunk_tracker.mark_completed(unit_id)
                    logger.debug("Marked unit %s as completed in chunk_tracker", unit_id)

    def mark_failed(self, unit_id: str, worker_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        logger.debug("Marking unit %s as failed by worker %s, error: %s", unit_id, worker_id, error)
        with self.lock:
            if unit_id in self.work_units:
                self.assigned_units[worker_id].discard(unit_id)
                self.pending_units.append(unit_id)
                logger.debug("Returned unit %s to pending_units", unit_id)

                if self.chunk_tracker:
                    self.chunk_tracker.mark_failed(unit_id)
                    logger.debug("Marked unit %s as failed in chunk_tracker", unit_id)

    def release_assignments(self, worker_id: str) -> None:
        """Release all assignments for a disconnected worker."""
        logger.debug("Releasing assignments for worker %s", worker_id)
        with self.lock:
            unit_ids = list(self.assigned_units.get(worker_id, []))

            for unit_id in unit_ids:
                if unit_id in self.work_units:
                    self.pending_units.append(unit_id)
                    logger.debug("Returned unit %s to pending_units", unit_id)

            if worker_id in self.assigned_units:
                del self.assigned_units[worker_id]
                logger.debug("Deleted worker %s from assigned_units", worker_id)

            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)
                logger.debug("Released worker %s chunks in chunk_tracker", worker_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self.lock:
            stats = {
                "total_units": len(self.work_units),
                "pending_units": len(self.pending_units),
                "assigned_units": sum(len(units) for units in self.assigned_units.values()),
                "total_shards": len(self.all_shards),
                "workers": len(self.assigned_units),
            }
            logger.debug("Stats: %s", stats)
            return stats

    def handle_result(self, result: WorkResult) -> Dict[str, Any]:
        """Handle WebDataset-specific result processing."""
        logger.debug("Handling result for unit %s", result.unit_id)
        base_result = super().handle_result(result)

        # Track processed items if we have chunk tracker
        if self.chunk_tracker and "item_indices" in result.metadata:
            indices = result.metadata["item_indices"]
            logger.debug("Result metadata item_indices: %s", indices)

            # Group consecutive indices into ranges
            if indices:
                indices.sort()
                ranges = []
                start = indices[0]
                end = indices[0]

                for i in range(1, len(indices)):
                    if indices[i] == end + 1:
                        end = indices[i]
                    else:
                        ranges.append((start, end))
                        start = indices[i]
                        end = indices[i]

                ranges.append((start, end))

                # Mark ranges as processed
                for start_idx, end_idx in ranges:
                    self.chunk_tracker.mark_items_processed(result.unit_id, start_idx, end_idx)
                    logger.debug(
                        "Marked items processed for unit %s: %d-%d",
                        result.unit_id,
                        start_idx,
                        end_idx,
                    )

        return base_result


class WebDatasetWorkerProcessor(WorkerProcessor):
    """Worker processor for WebDataset shards."""

    def __init__(self):
        logger.debug("Initializing WebDatasetWorkerProcessor")
        self.dataset_loader: Optional[DatasetLoader] = None
        self.dataset_config: Dict[str, Any] = {}

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize WebDataset processor."""
        logger.debug("Initializing worker with config: %s", config.config)
        cfg = config.config["dataset"]

        # Store config
        self.dataset_config = cfg

        # Initialize dataset loader
        dataset_path = cfg.get("dataset_path")
        dataset_type = cfg.get("dataset_type", "huggingface")
        dataset_split = cfg.get("dataset_split", "train")
        image_column = cfg.get("dataset_image_column", "image")

        if dataset_path:
            self.dataset_loader = DatasetLoader(
                dataset_path=dataset_path,
                dataset_type=dataset_type,
                split=dataset_split,
                image_column=image_column,
            )
            logger.debug("DatasetLoader initialized for worker")
        else:
            logger.error("No dataset_path provided in worker config")

    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a WebDataset chunk, yielding items to be captioned."""
        logger.debug("Processing unit: %s", unit.unit_id)
        if not self.dataset_loader:
            logger.error("Dataset loader not initialized")
            return

        shard_url = unit.data["shard_url"]
        start_index = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get(
            "unprocessed_ranges", [(start_index, start_index + chunk_size - 1)]
        )

        logger.info(f"Processing unit {unit.unit_id} with ranges: {unprocessed_ranges}")

        # Create set of indices to process
        indices_to_process = set()
        for start, end in unprocessed_ranges:
            indices_to_process.update(range(start, end + 1))
        logger.debug("Indices to process: %s", indices_to_process)

        processed_indices = []

        # Iterate through shard
        for idx, (key, url, image_data, metadata) in enumerate(
            self._iterate_shard_with_metadata(shard_url)
        ):
            # Skip if not in our chunk range
            if idx < start_index or idx >= start_index + chunk_size:
                logger.debug(f"Skipping idx={idx} not in chunk range")
                continue

            # Skip if already processed
            if idx not in indices_to_process:
                logger.debug(f"Skipping idx={idx} already processed")
                continue

            try:
                # Load image
                image = Image.open(io.BytesIO(image_data))

                # Clean metadata - remove sensitive and redundant fields
                clean_metadata = {
                    k: v
                    for k, v in metadata.items()
                    if k not in ["url", "_shard_url", "shard_name"]  # Remove these fields
                }

                # Add only necessary index information
                clean_metadata.update(
                    {
                        "_item_index": idx,
                        "_chunk_relative_index": idx - start_index,
                    }
                )

                # Prepare item for captioning
                logger.debug("Yielding item idx=%d key=%s", idx, key)
                yield {
                    "image": image,
                    "item_key": key,
                    "item_index": idx,
                    "metadata": clean_metadata,
                }

                processed_indices.append(idx)

            except Exception as e:
                logger.error(f"Error processing item {key}: {e}")

        # Store processed indices in context for result preparation
        context["_processed_indices"] = processed_indices
        logger.debug("Processed indices for unit %s: %s", unit.unit_id, processed_indices)

    def _iterate_shard_with_metadata(
        self, shard_url: str
    ) -> Iterator[Tuple[str, str, bytes, Dict]]:
        """Iterate through a shard with metadata."""
        logger.debug("Iterating shard with metadata: %s", shard_url)

        if not self.dataset_loader:
            logger.error("Dataset loader not initialized")
            return

        # Use the DatasetLoader that returns full samples
        for sample in self.dataset_loader.iterate_shard(shard_url):
            if not isinstance(sample, dict):
                logger.warning("Unexpected sample format: %s", type(sample))
                continue

            key = sample.get("__key__", "unknown")
            url = sample.get("__url__", "")  # Don't use shard_url as default

            # Find image data
            image_data = None
            image_ext = None
            for ext in ["jpg", "jpeg", "png", "webp", "bmp", "jxl"]:
                if ext in sample:
                    image_data = sample[ext]
                    image_ext = ext
                    break

            if not image_data:
                logger.debug(
                    "No image data found for item key=%s, available keys: %s",
                    key,
                    list(sample.keys()),
                )
                continue

            # Extract metadata (all non-system and non-image keys)
            metadata = {
                k: v
                for k, v in sample.items()
                if not k.startswith("__") and k not in ["jpg", "jpeg", "png", "webp", "bmp", "jxl"]
            }

            # Add image format but not URLs
            if image_ext:
                metadata["_image_format"] = image_ext

            yield key, url, image_data, metadata

    def prepare_result(
        self, unit: WorkUnit, outputs: List[Dict[str, Any]], processing_time_ms: float
    ) -> WorkResult:
        """Prepare WebDataset-specific result."""
        logger.debug("Preparing result for unit %s", unit.unit_id)
        result = super().prepare_result(unit, outputs, processing_time_ms)

        # Add processed indices to metadata if available
        if outputs and "_processed_indices" in outputs[0].get("metadata", {}):
            result.metadata["item_indices"] = outputs[0]["metadata"]["_processed_indices"]
            logger.debug(
                "Added item_indices to result metadata: %s", result.metadata["item_indices"]
            )

        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.dataset_loader:
            info = self.dataset_loader.get_dataset_info()
            logger.debug("Dataset info: %s", info)
            return info
        info = {
            "dataset_path": self.dataset_config.get("dataset_path", "unknown"),
            "dataset_type": self.dataset_config.get("dataset_type", "unknown"),
        }
        logger.debug("Dataset info (no loader): %s", info)
        return info
