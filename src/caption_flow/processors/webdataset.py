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
from caption_flow.storage import StorageManager

from .base import OrchestratorProcessor, WorkerProcessor, ProcessorConfig, WorkUnit, WorkResult
from ..utils import DatasetLoader, ChunkTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def initialize(self, config: ProcessorConfig, storage: StorageManager) -> None:
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
            self._restore_state(storage=storage)

            # Start background unit creation
            self.unit_creation_thread = threading.Thread(
                target=self._create_units_background, daemon=True
            )
            self.unit_creation_thread.start()
            logger.debug("Unit creation thread started")
        else:
            logger.error("No dataset_path provided in config")

    def _restore_state(self, storage: StorageManager) -> None:
        """Restore state from chunk tracker."""
        logger.debug("Restoring state from chunk tracker")
        if not self.chunk_tracker:
            return

        shards_summary = self.chunk_tracker.get_shards_summary()

        # Get all processed job_ids from storage
        all_processed_jobs = storage.get_all_processed_job_ids()

        with self.lock:
            for shard_name, shard_info in shards_summary.items():
                for chunk_state in shard_info["chunks"]:
                    # Calculate actual unprocessed ranges based on what's in storage
                    chunk_range = (
                        chunk_state.start_index,
                        chunk_state.start_index + chunk_state.chunk_size - 1,
                    )

                    # Get processed indices for this chunk
                    processed_ranges = self.chunk_tracker.get_processed_indices_for_chunk(
                        chunk_state.chunk_id, all_processed_jobs
                    )

                    # Calculate unprocessed ranges
                    unprocessed_ranges = self._subtract_ranges([chunk_range], processed_ranges)

                    if unprocessed_ranges:
                        # Create work unit for unprocessed items
                        logger.debug(f"Creating WorkUnit for chunk {chunk_state}")
                        unit = WorkUnit(
                            unit_id=chunk_state.chunk_id,
                            chunk_id=chunk_state.chunk_id,
                            source_id=shard_name,
                            data={
                                "shard_url": chunk_state.shard_url,
                                "start_index": chunk_state.start_index,
                                "chunk_size": chunk_state.chunk_size,
                                "unprocessed_ranges": unprocessed_ranges,
                            },
                            metadata={
                                "shard_name": shard_name,
                                "chunk_index": chunk_state.start_index // self.chunk_size,
                            },
                        )

                        self.work_units[unit.unit_id] = unit
                        self.pending_units.append(unit.unit_id)

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

        shard_iter = iter(self.all_shards)
        current_shard_url = None
        current_shard_name = None
        current_shard_items = 0
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
                    unit_id = f"{current_shard_name}:chunk:{current_index // self.chunk_size}"

                    with self.lock:
                        # Check if this unit already exists in work_units
                        if unit_id in self.work_units:
                            logger.debug(
                                f"Unit {unit_id} already exists in work_units, skipping creation"
                            )
                            current_index += self.chunk_size
                            continue

                        # Check if chunk is already completed or has no unprocessed items
                        if self.chunk_tracker:
                            chunk_state = self.chunk_tracker.chunks.get(unit_id)

                            if chunk_state:
                                # Check if completed
                                if chunk_state.status == "completed":
                                    logger.debug(f"Unit {unit_id} already completed, skipping")
                                    current_index += self.chunk_size
                                    continue

                                # Check if has unprocessed items
                                unprocessed_ranges = chunk_state.get_unprocessed_ranges()
                                if not unprocessed_ranges:
                                    logger.debug(
                                        f"Unit {unit_id} has no unprocessed items, skipping"
                                    )
                                    current_index += self.chunk_size
                                    continue

                                # If chunk exists but has unprocessed items, use those ranges
                                logger.debug(
                                    f"Existing chunk {unit_id} has unprocessed ranges: {unprocessed_ranges}"
                                )

                                unit = WorkUnit(
                                    unit_id=unit_id,
                                    chunk_id=unit_id,
                                    source_id=current_shard_name,
                                    data={
                                        "shard_url": current_shard_url,
                                        "start_index": current_index,
                                        "chunk_size": chunk_size,
                                        "unprocessed_ranges": [
                                            (
                                                r[0] + chunk_state.start_index,
                                                r[1] + chunk_state.start_index,
                                            )
                                            for r in unprocessed_ranges
                                        ],  # Convert relative to absolute
                                    },
                                    metadata={
                                        "shard_name": current_shard_name,
                                        "chunk_index": current_index // self.chunk_size,
                                    },
                                )
                            else:
                                # New chunk
                                logger.debug(
                                    "Creating new work unit: unit_id=%s shard=%s start_index=%d chunk_size=%d",
                                    unit_id,
                                    current_shard_name,
                                    current_index,
                                    chunk_size,
                                )

                                unit = WorkUnit(
                                    unit_id=unit_id,
                                    chunk_id=unit_id,
                                    source_id=current_shard_name,
                                    data={
                                        "shard_url": current_shard_url,
                                        "start_index": current_index,
                                        "chunk_size": chunk_size,
                                        "unprocessed_ranges": [
                                            (current_index, current_index + chunk_size - 1)
                                        ],
                                    },
                                    metadata={
                                        "shard_name": current_shard_name,
                                        "chunk_index": current_index // self.chunk_size,
                                    },
                                )
                        else:
                            # No chunk tracker, create normally
                            unit = WorkUnit(
                                unit_id=unit_id,
                                chunk_id=unit_id,
                                source_id=current_shard_name,
                                data={
                                    "shard_url": current_shard_url,
                                    "start_index": current_index,
                                    "chunk_size": chunk_size,
                                    "unprocessed_ranges": [
                                        (current_index, current_index + chunk_size - 1)
                                    ],
                                },
                                metadata={
                                    "shard_name": current_shard_name,
                                    "chunk_index": current_index // self.chunk_size,
                                },
                            )

                        self.work_units[unit_id] = unit
                        self.pending_units.append(unit_id)
                        logger.debug("Added work unit %s to pending_units", unit_id)

                        if self.chunk_tracker:
                            added_chunk = self.chunk_tracker.add_chunk(
                                unit_id,
                                current_shard_name,
                                current_shard_url,
                                current_index,
                                chunk_size,
                            )
                            if added_chunk:
                                logger.debug("Added chunk to chunk_tracker: %s", unit_id)
                            else:
                                logger.debug("Chunk already exists in chunk_tracker: %s", unit_id)

                        units_created += 1

                    current_index += self.chunk_size

            if units_created > 0:
                logger.debug(f"Created {units_created} work units")

    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""
        logger.debug("get_work_units called: count=%d worker_id=%s", count, worker_id)
        assigned = []

        with self.lock:
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
                    unit = self.work_units[unit_id]

                    # Update unprocessed ranges based on what's been processed
                    if self.chunk_tracker and unit_id in self.chunk_tracker.chunks:
                        chunk_state = self.chunk_tracker.chunks[unit_id]
                        unprocessed_ranges = chunk_state.get_unprocessed_ranges()

                        # Convert relative ranges back to absolute
                        absolute_ranges = []
                        for start, end in unprocessed_ranges:
                            abs_start = chunk_state.start_index + start
                            abs_end = chunk_state.start_index + end
                            absolute_ranges.append((abs_start, abs_end))

                        # Update the work unit's data
                        unit.data["unprocessed_ranges"] = absolute_ranges

                        logger.debug(
                            f"Updated unit {unit_id} with unprocessed ranges: {absolute_ranges}"
                        )

                    self.pending_units.append(unit_id)
                    logger.debug("Returned unit %s to pending_units", unit_id)

            if worker_id in self.assigned_units:
                del self.assigned_units[worker_id]
                logger.debug("Deleted worker %s from assigned_units", worker_id)

            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)
                logger.debug("Released worker %s chunks in chunk_tracker", worker_id)

    def _subtract_ranges(
        self, total_ranges: List[Tuple[int, int]], processed_ranges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Subtract processed ranges from total ranges."""
        if not processed_ranges:
            return total_ranges

        # Create a set of all processed indices
        processed_indices = set()
        for start, end in processed_ranges:
            processed_indices.update(range(start, end + 1))

        # Find unprocessed ranges
        unprocessed_ranges = []
        for start, end in total_ranges:
            current_start = None
            for i in range(start, end + 1):
                if i not in processed_indices:
                    if current_start is None:
                        current_start = i
                else:
                    if current_start is not None:
                        unprocessed_ranges.append((current_start, i - 1))
                        current_start = None

            if current_start is not None:
                unprocessed_ranges.append((current_start, end))

        return unprocessed_ranges

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
        # logger.debug("Handling result for unit %s", result.unit_id)
        base_result = super().handle_result(result)

        # Track processed items if we have chunk tracker
        if self.chunk_tracker:
            if "item_indices" not in result.metadata:
                result.metadata["item_indices"] = [result.metadata.get("_item_index")]
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
                    logger.debug(f"Marking chunk as processed: {result.to_repr()}")
                    self.chunk_tracker.mark_items_processed(result.chunk_id, start_idx, end_idx)
                    logger.debug(
                        "Marked items processed for unit %s: %d-%d",
                        result.unit_id,
                        start_idx,
                        end_idx,
                    )
        else:
            logger.error(
                f"No chunk tracker? {self.chunk_tracker} or no item_indices in {result.metadata}"
            )

        return base_result

    def update_from_storage(self, processed_job_ids: Set[str]) -> None:
        """Update work units based on what's been processed."""
        logger.info(f"Updating work units from {len(processed_job_ids)} processed jobs")

        with self.lock:
            for unit_id, unit in self.work_units.items():
                # Extract chunk info from unit
                start_index = unit.data["start_index"]
                chunk_size = unit.data["chunk_size"]
                shard_name = unit.metadata["shard_name"]
                chunk_index = unit.metadata["chunk_index"]

                # Find processed indices for this chunk
                processed_indices = []
                for job_id in processed_job_ids:
                    # Parse job_id format: "data-0000:chunk:0:idx:42"
                    parts = job_id.split(":")
                    if (
                        len(parts) == 5
                        and parts[0] == shard_name
                        and parts[1] == "chunk"
                        and int(parts[2]) == chunk_index
                        and parts[3] == "idx"
                    ):

                        idx = int(parts[4])
                        if start_index <= idx < start_index + chunk_size:
                            processed_indices.append(idx)

                if processed_indices:
                    # Convert to ranges
                    processed_indices.sort()
                    processed_ranges = []
                    start = processed_indices[0]
                    end = processed_indices[0]

                    for idx in processed_indices[1:]:
                        if idx == end + 1:
                            end = idx
                        else:
                            processed_ranges.append((start, end))
                            start = idx
                            end = idx

                    processed_ranges.append((start, end))

                    # Calculate unprocessed ranges
                    total_range = [(start_index, start_index + chunk_size - 1)]
                    unprocessed_ranges = self._subtract_ranges(total_range, processed_ranges)

                    # Update unit
                    unit.data["unprocessed_ranges"] = unprocessed_ranges

                    logger.debug(
                        f"Updated unit {unit_id}: {len(processed_indices)} processed, "
                        f"unprocessed ranges: {unprocessed_ranges}"
                    )


class WebDatasetWorkerProcessor(WorkerProcessor):
    """Worker processor for WebDataset shards."""

    def __init__(self):
        logger.debug("Initializing WebDatasetWorkerProcessor")
        self.dataset_loader: Optional[DatasetLoader] = None
        self.dataset_config: Dict[str, Any] = {}
        self.dataset_name: Optional[str] = None

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize WebDataset processor."""
        logger.debug("Initializing worker with config: %s", config.config)
        cfg = config.config["dataset"]

        # Store config
        self.dataset_config = cfg

        # Initialize dataset loader
        dataset_path = cfg.get("dataset_path")
        self.dataset_path = dataset_path
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

        shard_name = unit.metadata["shard_name"]
        chunk_index = unit.metadata["chunk_index"]
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
                # logger.debug(f"Skipping idx={idx} not in chunk range")
                continue

            # Skip if already processed
            if idx not in indices_to_process:
                logger.debug(f"Skipping idx={idx} already processed")
                continue

            try:
                # Load image
                image = Image.open(io.BytesIO(image_data))
                job_id = f"{shard_name}:chunk:{chunk_index}:idx:{idx}"

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
                        "_job_id": job_id,
                    }
                )

                # Prepare item for captioning
                # logger.debug("Yielding item idx=%d key=%s", idx, key)
                yield {
                    "image": image,
                    "item_key": key,
                    "item_index": idx,
                    "metadata": clean_metadata,
                    "job_id": job_id,
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
            "dataset_path": self.dataset_config.get("dataset_path"),
            "dataset_type": self.dataset_config.get("type", "huggingface"),
        }
        logger.debug("Dataset info (no loader): %s", info)
        return info
