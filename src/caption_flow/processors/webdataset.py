"""WebDataset processor implementation using webshart TarDataLoader."""

import gc
import io
import logging
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Set

import cv2
import numpy as np
import webshart
from PIL import Image

from caption_flow.models import JobId
from caption_flow.storage import StorageManager

from ..utils import ChunkTracker
from .base import OrchestratorProcessor, ProcessorConfig, WorkerProcessor, WorkResult, WorkUnit

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class WebDatasetOrchestratorProcessor(OrchestratorProcessor):
    """Orchestrator processor for WebDataset shards using webshart with ChunkTracker."""

    def __init__(self):
        logger.info("Initializing WebDatasetOrchestratorProcessor with webshart + ChunkTracker")
        self.dataset: Optional[webshart.DiscoveredDataset] = None
        self.chunk_tracker: Optional[ChunkTracker] = None
        self.chunk_size: int = 1000

        # Work unit management
        self.work_units: Dict[str, WorkUnit] = {}
        self.pending_units: Deque[str] = deque()
        self.assigned_units: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()

        # Shard info cache
        self.shard_info_cache: Dict[int, Dict] = {}

        # Background thread for creating work units
        self.unit_creation_thread: Optional[threading.Thread] = None
        self.stop_creation = threading.Event()
        self.min_buffer = 10
        self.buffer_multiplier = 3

    def initialize(self, config: ProcessorConfig, storage: StorageManager) -> None:
        """Initialize with webshart dataset discovery and ChunkTracker."""
        logger.info("Initializing orchestrator with config")

        cfg = config.config
        dataset_cfg = cfg.get("dataset", {})
        self.dataset_path = dataset_cfg.get("dataset_path")
        metadata_path = dataset_cfg.get("metadata_path", None)

        # Chunk settings
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.min_buffer = cfg.get("min_chunk_buffer", 10)
        self.buffer_multiplier = cfg.get("chunk_buffer_multiplier", 3)

        # Cache configuration
        cache_dir = Path(cfg.get("cache_dir", "./webshart_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_path:
            # Initialize dataset with webshart
            self.dataset = webshart.discover_dataset(
                source=self.dataset_path,
                metadata=metadata_path,
            )

            # Enable caching for efficient access
            self.dataset.enable_metadata_cache(location=str(cache_dir / "metadata_cache"))
            self.dataset.enable_shard_cache(
                location=str(cache_dir / "shard_cache"),
                cache_limit_gb=cfg.get("shard_cache_gb", 10.0),
            )

            logger.info(f"Dataset discovered: {self.dataset.num_shards} shards")

            # Initialize chunk tracker
            checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")

            # Restore existing state from chunk tracker
            self._restore_state(storage)

            # Start background unit creation
            self.unit_creation_thread = threading.Thread(
                target=self._create_units_background, daemon=True
            )
            self.unit_creation_thread.start()
        else:
            logger.error("No dataset_path provided")

    def _get_shard_info_cached(self, shard_idx: int) -> Optional[Dict]:
        """Get shard info with caching."""
        if shard_idx not in self.shard_info_cache:
            try:
                self.shard_info_cache[shard_idx] = self.dataset.get_shard_info(shard_idx)
            except Exception as e:
                logger.error(f"Error getting shard info for idx {shard_idx}: {e}")
                return None
        return self.shard_info_cache[shard_idx]

    def _restore_state(self, storage: StorageManager) -> None:
        """Restore state from chunk tracker and synchronize with storage."""
        logger.info("Restoring state from chunk tracker and synchronizing with storage")
        if not self.chunk_tracker:
            return

        # First, update chunk tracker from storage
        processed_job_ids = storage.get_all_processed_job_ids()
        if processed_job_ids:
            logger.info(
                f"Synchronizing chunk tracker with {len(processed_job_ids)} processed items from storage"
            )
            self.update_from_storage(processed_job_ids)

        # Then restore work units from chunk tracker
        shards_summary = self.chunk_tracker.get_shards_summary()
        logger.info(f"Restoring work units from chunk tracker: {len(shards_summary)} shards")

        with self.lock:
            restored_count = 0
            for shard_name, shard_info in shards_summary.items():
                chunks = shard_info.get("chunks", [])
                for chunk_state in chunks:
                    # Only add incomplete chunks
                    if chunk_state.status == "completed":
                        logger.debug(f"Skipping completed chunk {chunk_state.chunk_id}")
                        continue

                    # Get unprocessed ranges
                    unprocessed_ranges = chunk_state.get_unprocessed_ranges()
                    if not unprocessed_ranges:
                        logger.debug(
                            f"Chunk {chunk_state.chunk_id} has no unprocessed ranges, marking as completed"
                        )
                        self.chunk_tracker.mark_completed(chunk_state.chunk_id)
                        continue

                    logger.info(
                        f"Restoring chunk {chunk_state.chunk_id} with unprocessed ranges: {unprocessed_ranges}"
                    )

                    # Convert relative ranges to absolute file indices
                    absolute_ranges = []
                    for start, end in unprocessed_ranges:
                        abs_start = chunk_state.start_index + start
                        abs_end = chunk_state.start_index + end
                        absolute_ranges.append((abs_start, abs_end))

                    # Get shard index if available
                    shard_idx = None
                    if self.dataset:
                        for idx in range(self.dataset.num_shards):
                            shard_info = self._get_shard_info_cached(idx)
                            if shard_info and shard_info["name"] == shard_name:
                                shard_idx = idx
                                break

                    unit = WorkUnit(
                        unit_id=chunk_state.chunk_id,
                        chunk_id=chunk_state.chunk_id,
                        source_id=shard_name,
                        unit_size=chunk_state.chunk_size,
                        data={
                            "shard_url": chunk_state.shard_url,
                            "shard_name": shard_name,
                            "shard_idx": shard_idx,
                            "start_index": chunk_state.start_index,
                            "chunk_size": chunk_state.chunk_size,
                            "unprocessed_ranges": absolute_ranges,
                        },
                        metadata={
                            "shard_name": shard_name,
                            "chunk_index": chunk_state.start_index // self.chunk_size,
                        },
                    )

                    self.work_units[unit.unit_id] = unit
                    self.pending_units.append(unit.unit_id)
                    restored_count += 1

            logger.info(f"Restored {restored_count} incomplete work units")

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

        current_shard_idx = 0
        current_file_idx = 0

        while not self.stop_creation.is_set():
            # Check if we need more units
            with self.lock:
                pending_count = len(self.pending_units)
                assigned_count = sum(len(units) for units in self.assigned_units.values())
                worker_count = max(1, len(self.assigned_units))

                target_buffer = max(self.min_buffer, worker_count * self.buffer_multiplier)
                units_needed = max(0, target_buffer - (pending_count + assigned_count))

            if units_needed == 0:
                threading.Event().wait(5)
                continue

            # Create units as needed
            units_created = 0
            while units_created < units_needed and not self.stop_creation.is_set():
                # Get current shard info
                if current_shard_idx >= self.dataset.num_shards:
                    threading.Event().wait(5)
                    break

                shard_info = self._get_shard_info_cached(current_shard_idx)
                if not shard_info:
                    current_shard_idx += 1
                    current_file_idx = 0
                    continue

                shard_name = shard_info["name"]
                shard_files = shard_info["num_files"]

                # Check if we need to move to next shard
                if current_file_idx >= shard_files:
                    current_shard_idx += 1
                    current_file_idx = 0
                    continue

                # Create chunk for current position
                chunk_size = min(self.chunk_size, shard_files - current_file_idx)
                self.current_chunk_index = current_file_idx // self.chunk_size
                job_id_obj = JobId(
                    shard_id=shard_name,
                    chunk_id=str(self.current_chunk_index),
                    sample_id=str(current_file_idx),
                )
                chunk_id = job_id_obj.get_chunk_str()

                with self.lock:
                    # Skip if already exists
                    if chunk_id in self.work_units:
                        current_file_idx += self.chunk_size
                        continue

                    # Check if chunk is already completed
                    if self.chunk_tracker:
                        chunk_state = self.chunk_tracker.chunks.get(chunk_id)
                        if chunk_state and chunk_state.status == "completed":
                            current_file_idx += self.chunk_size
                            continue

                    # Get shard URL (path for webshart)
                    shard_url = shard_info.get("path", f"{shard_name}.tar")

                    # Create work unit
                    unit = WorkUnit(
                        unit_id=chunk_id,
                        chunk_id=chunk_id,
                        source_id=shard_name,
                        unit_size=chunk_size,
                        data={
                            "shard_url": shard_url,
                            "shard_name": shard_name,
                            "shard_idx": current_shard_idx,
                            "start_index": current_file_idx,
                            "chunk_size": chunk_size,
                            "unprocessed_ranges": [
                                (current_file_idx, current_file_idx + chunk_size - 1)
                            ],
                        },
                        metadata={
                            "shard_name": shard_name,
                            "chunk_index": current_file_idx // self.chunk_size,
                        },
                    )

                    self.work_units[chunk_id] = unit
                    self.pending_units.append(chunk_id)

                    # Add to chunk tracker
                    if self.chunk_tracker:
                        self.chunk_tracker.add_chunk(
                            chunk_id, shard_name, shard_url, current_file_idx, chunk_size
                        )

                    units_created += 1

                current_file_idx += self.chunk_size

            if units_created > 0:
                logger.debug(f"Created {units_created} work units")

        logger.info("Work unit creation thread exiting")

    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""
        assigned = []

        with self.lock:
            units_checked = 0
            max_units_to_check = len(self.pending_units)

            while len(assigned) < count and units_checked < max_units_to_check:
                if not self.pending_units:
                    break

                unit_id = self.pending_units.popleft()
                units_checked += 1
                unit = self.work_units.get(unit_id)

                if unit:
                    # Update unprocessed ranges from chunk tracker before assigning
                    if self.chunk_tracker and unit_id in self.chunk_tracker.chunks:
                        chunk_state = self.chunk_tracker.chunks[unit_id]
                        relative_unprocessed = chunk_state.get_unprocessed_ranges()

                        # If no unprocessed ranges, mark as completed and skip
                        if not relative_unprocessed:
                            logger.info(
                                f"Chunk {unit_id} has no unprocessed ranges, marking as completed"
                            )
                            self.chunk_tracker.mark_completed(unit_id)
                            # Remove from work units
                            del self.work_units[unit_id]
                            continue

                        # Convert relative to absolute indices
                        absolute_ranges = []
                        for start, end in relative_unprocessed:
                            abs_start = chunk_state.start_index + start
                            abs_end = chunk_state.start_index + end
                            absolute_ranges.append((abs_start, abs_end))

                        # Update the work unit's unprocessed ranges
                        unit.data["unprocessed_ranges"] = absolute_ranges

                        logger.debug(
                            f"Updated unit {unit_id} with unprocessed ranges: {absolute_ranges}"
                        )

                    self.assigned_units[worker_id].add(unit_id)
                    assigned.append(unit)

                    if self.chunk_tracker:
                        self.chunk_tracker.mark_assigned(unit_id, worker_id)
                else:
                    # Put it back if we couldn't get the unit
                    self.pending_units.append(unit_id)

        logger.debug(f"Assigned {len(assigned)} units to worker {worker_id}")
        return assigned

    def mark_completed(self, unit_id: str, worker_id: str) -> None:
        """Mark a work unit as completed."""
        with self.lock:
            if unit_id in self.work_units:
                self.assigned_units[worker_id].discard(unit_id)

                if self.chunk_tracker:
                    self.chunk_tracker.mark_completed(unit_id)

                # Remove from memory
                del self.work_units[unit_id]

    def mark_failed(self, unit_id: str, worker_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        logger.error(f"Unit {unit_id} failed on {worker_id}: {error}")
        with self.lock:
            if unit_id in self.work_units:
                self.assigned_units[worker_id].discard(unit_id)
                self.pending_units.append(unit_id)

                if self.chunk_tracker:
                    self.chunk_tracker.mark_failed(unit_id)

    def release_assignments(self, worker_id: str) -> None:
        """Release all assignments for a disconnected worker."""
        with self.lock:
            unit_ids = list(self.assigned_units.get(worker_id, []))

            for unit_id in unit_ids:
                if unit_id in self.work_units and self.chunk_tracker:
                    # Get updated unprocessed ranges from chunk tracker
                    chunk_state = self.chunk_tracker.chunks.get(unit_id)
                    if chunk_state:
                        unprocessed_ranges = chunk_state.get_unprocessed_ranges()
                        # Convert relative to absolute
                        absolute_ranges = []
                        for start, end in unprocessed_ranges:
                            abs_start = chunk_state.start_index + start
                            abs_end = chunk_state.start_index + end
                            absolute_ranges.append((abs_start, abs_end))

                        # Update work unit
                        self.work_units[unit_id].data["unprocessed_ranges"] = absolute_ranges

                    self.pending_units.append(unit_id)

            if worker_id in self.assigned_units:
                del self.assigned_units[worker_id]

            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)

        logger.info(f"Released {len(unit_ids)} assignments from {worker_id}")

    def handle_result(self, result: WorkResult) -> Dict[str, Any]:
        """Handle result from worker and update chunk tracker."""
        # Extract the actual item index from the metadata
        item_index = result.metadata.get("_item_index", None)

        # If we have an item index, mark it as processed in the chunk tracker
        if self.chunk_tracker and item_index is not None and result.chunk_id:
            try:
                # Mark single item as processed
                self.chunk_tracker.mark_items_processed(result.chunk_id, item_index, item_index)
                # logger.debug(f"Marked item {item_index} as processed in chunk {result.chunk_id}")
            except Exception as e:
                logger.error(f"Error marking item {item_index} as processed: {e}")

        # Also handle batch results if present (backward compatibility)
        if self.chunk_tracker and "item_indices" in result.metadata:
            indices = result.metadata["item_indices"]

            # Convert to ranges and mark as processed
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
                    self.chunk_tracker.mark_items_processed(result.chunk_id, start_idx, end_idx)
                    logger.debug(
                        f"Marked range {start_idx}-{end_idx} as processed in chunk {result.chunk_id}"
                    )

        return {
            "source_id": result.source_id,
            "chunk_id": result.chunk_id,
            "outputs": result.outputs,
            "metadata": result.metadata,
        }

    def update_from_storage(self, processed_job_ids: Set[str]) -> None:
        """Update work units based on what's been processed."""
        logger.info(f"Updating from {len(processed_job_ids)} processed jobs")

        with self.lock:
            # Group by chunk
            processed_by_chunk = defaultdict(set)

            for job_id_str in processed_job_ids:
                try:
                    # Use JobId to parse the job ID string
                    job_id = JobId.from_str(job_id_str)
                    chunk_id = job_id.get_chunk_str()
                    sample_idx = int(job_id.sample_id)
                    processed_by_chunk[chunk_id].add(sample_idx)
                except ValueError as e:
                    logger.warning(f"Invalid job ID format: {job_id_str} - {e}")
                    continue

            # Update chunk tracker with processed items
            if self.chunk_tracker:
                for chunk_id, indices in processed_by_chunk.items():
                    if indices:
                        # Get or create chunk state
                        chunk_state = self.chunk_tracker.chunks.get(chunk_id)
                        if not chunk_state:
                            # Parse chunk_id using JobId to get shard info
                            try:
                                # chunk_id format: "shard_id:chunk:chunk_idx"
                                parts = chunk_id.split(":")
                                if len(parts) >= 3:
                                    shard_name = parts[0]
                                    chunk_idx = int(parts[2])
                                    # Infer start index from chunk index and size
                                    start_index = chunk_idx * self.chunk_size
                                    # Create chunk state
                                    self.chunk_tracker.add_chunk(
                                        chunk_id,
                                        shard_name,
                                        f"{shard_name}.tar",
                                        start_index,
                                        self.chunk_size,
                                    )
                                    logger.info(f"Created missing chunk state for {chunk_id}")
                            except (ValueError, IndexError) as e:
                                logger.error(f"Failed to create chunk state for {chunk_id}: {e}")
                                continue

                        # Sort indices and convert to ranges
                        sorted_indices = sorted(indices)
                        if not sorted_indices:
                            continue

                        # Condense into contiguous ranges
                        ranges = []
                        start_range = sorted_indices[0]
                        end_range = sorted_indices[0]

                        for i in range(1, len(sorted_indices)):
                            if sorted_indices[i] == end_range + 1:
                                end_range = sorted_indices[i]
                            else:
                                ranges.append((start_range, end_range))
                                start_range = sorted_indices[i]
                                end_range = sorted_indices[i]
                        ranges.append((start_range, end_range))

                        # Mark each contiguous range as processed
                        logger.info(f"Marking ranges {ranges} as processed in chunk {chunk_id}")
                        for start_idx, end_idx in ranges:
                            self.chunk_tracker.mark_items_processed(chunk_id, start_idx, end_idx)

                # Flush checkpoint after major update
                self.chunk_tracker.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self.lock:
            # Get chunk tracker stats if available
            if self.chunk_tracker:
                shards_summary = self.chunk_tracker.get_shards_summary()
                total_chunks = sum(len(s.get("chunks", [])) for s in shards_summary.values())
                completed_chunks = sum(
                    1
                    for s in shards_summary.values()
                    for c in s.get("chunks", [])
                    if c.status == "completed"
                )
            else:
                total_chunks = len(self.work_units)
                completed_chunks = 0

            return {
                "total_shards": self.dataset.num_shards if self.dataset else 0,
                "total_chunks": total_chunks,
                "pending_units": len(self.pending_units),
                "assigned_units": sum(len(units) for units in self.assigned_units.values()),
                "completed_chunks": completed_chunks,
                "workers": len(self.assigned_units),
            }

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up orchestrator")

        # Stop background threads
        self.stop_creation.set()
        if self.unit_creation_thread:
            self.unit_creation_thread.join(timeout=5)

        # Flush final checkpoint on cleanup
        if self.chunk_tracker:
            self.chunk_tracker.flush()


class WebDatasetWorkerProcessor(WorkerProcessor):
    """Worker processor for WebDataset shards using webshart."""

    def __init__(self):
        logger.info("Initializing WebDatasetWorkerProcessor with webshart")
        self.loader: Optional[webshart.TarDataLoader] = None
        self.dataset: Optional[webshart.DiscoveredDataset] = None
        self.mock_results = False

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize worker with webshart loader."""
        cfg = config.config
        dataset_cfg = cfg.get("dataset", {})

        self.dataset_path = dataset_cfg.get("dataset_path")
        metadata_path = dataset_cfg.get("metadata_path", None)
        self.mock_results = dataset_cfg.get("mock_results", False)
        split_worker_cache = dataset_cfg.get(
            "split_worker_cache", True
        )  # multiple workers get their own cache by default

        # Cache configuration
        cache_dir = Path(cfg.get("cache_dir", "./webshart_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_path and not self.mock_results:
            # Discover dataset
            self.dataset = webshart.discover_dataset(
                source=self.dataset_path,
                metadata=metadata_path,
            )

            # Enable caching
            self.dataset.enable_metadata_cache(location=str(cache_dir / "metadata_cache"))
            self.dataset.enable_shard_cache(
                location=(
                    str(cache_dir / "shard_cache" / str(self.gpu_id))
                    if split_worker_cache
                    else str(cache_dir / "shard_cache")
                ),
                cache_limit_gb=cfg.get("shard_cache_gb", 10.0),
            )

            # Create loader
            self.loader = webshart.TarDataLoader(
                self.dataset,
                buffer_size=cfg.get("buffer_size", 10),
                max_file_size=cfg.get("max_file_size", 100 * 1024 * 1024),
                load_file_data=True,
            )

            logger.info("webshart TarDataLoader initialized")

    def _create_mock_image(self, idx: int) -> Image.Image:
        """Create a dummy test image."""
        color = ((idx * 37) % 256, (idx * 53) % 256, (idx * 71) % 256)
        image = Image.new("RGB", (256, 256), color=color)
        return image

    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a work unit by iterating specified ranges."""
        logger.debug(f"Processing unit: {unit}")

        shard_name = unit.data["shard_name"]
        shard_idx = unit.data.get("shard_idx")
        unprocessed_ranges = unit.data.get("unprocessed_ranges", [])

        # For chunk tracking
        chunk_index = unit.metadata.get("chunk_index", 0)
        processed_indices = []

        if self.mock_results:
            # Generate mock results for unprocessed ranges
            for start_idx, end_idx in unprocessed_ranges:
                for idx in range(start_idx, end_idx + 1):
                    # Use JobId to create consistent job ID
                    job_id = JobId.from_values(
                        shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(idx)
                    )
                    job_id_str = job_id.get_sample_str()

                    yield {
                        "image": self._create_mock_image(idx),
                        "image_data": None,
                        "item_key": f"mock_{idx}",
                        "item_index": idx,
                        "metadata": {
                            "_item_index": idx,
                            "_chunk_relative_index": idx - unit.data["start_index"],
                            "_job_id": job_id_str,
                            "_mock": True,
                            "_processed_indices": processed_indices,
                        },
                        "job_id": job_id_str,
                    }

                    processed_indices.append(idx)
        else:
            # Use webshart to process unprocessed ranges
            for start_idx, end_idx in unprocessed_ranges:
                try:
                    # Jump to shard and starting position
                    if shard_idx is not None:
                        self.loader.shard(shard_idx=shard_idx, cursor_idx=start_idx)
                    else:
                        # Try to find shard by name
                        self.loader.shard(filename=shard_name, cursor_idx=start_idx)

                    # Iterate through the range
                    for idx in range(start_idx, end_idx + 1):
                        try:
                            entry = webshart.next_with_cache_wait(self.loader)

                            # Decode image
                            image = None
                            if entry.data:
                                try:
                                    # Use cv2 to decode from memory
                                    nparr = np.frombuffer(entry.data, np.uint8)
                                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                                    if img_np is not None:
                                        # Convert from BGR (OpenCV default) to RGB (PIL default)
                                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                        image = Image.fromarray(img_rgb)
                                    else:
                                        logger.warning(f"cv2.imdecode failed for {entry.path}")

                                except ImportError:
                                    logger.warning(
                                        "cv2 or numpy not installed, falling back to PIL"
                                    )
                                    image = Image.open(io.BytesIO(entry.data))
                                except Exception as img_e:
                                    logger.error(
                                        f"Error decoding image {entry.path} with cv2: {img_e}"
                                    )

                            # Generate job ID using JobId class
                            job_id = JobId.from_values(
                                shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(idx)
                            )
                            job_id_str = job_id.get_sample_str()

                            yield {
                                "image": image,
                                "image_data": entry.data,
                                "item_key": Path(entry.path).stem,
                                "item_index": idx,
                                "metadata": {
                                    "_item_index": idx,
                                    "_chunk_relative_index": idx - unit.data["start_index"],
                                    "_job_id": job_id_str,
                                    "_filename": entry.path,
                                    "_file_size": entry.size,
                                    "_processed_indices": processed_indices,
                                },
                                "job_id": job_id_str,
                            }

                            processed_indices.append(idx)

                            if len(processed_indices) % 10 == 0:
                                gc.collect()

                        except StopIteration:
                            logger.warning(f"Unexpected end of shard at index {idx}")
                            break
                        except Exception as e:
                            logger.error(f"Error processing index {idx}: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing range {start_idx}-{end_idx}: {e}")
                    continue

        # Store processed indices for result
        context["_processed_indices"] = processed_indices
        logger.info(f"Processed {len(processed_indices)} items from unit {unit.unit_id}")

    def prepare_result(
        self, unit: WorkUnit, outputs: List[Dict[str, Any]], processing_time_ms: float
    ) -> WorkResult:
        """Prepare result with processing details."""
        result = super().prepare_result(unit, outputs, processing_time_ms)

        # Add processed indices for chunk tracker
        if hasattr(self, "_last_context") and "_processed_indices" in self._last_context:
            result.metadata["item_indices"] = self._last_context["_processed_indices"]

        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.dataset:
            stats = self.dataset.get_stats()
            return {
                "dataset_name": self.dataset.name,
                "format": self.dataset.dataset_format,
                "total_shards": stats["total_shards"],
                "total_files": stats.get("total_files", "Unknown"),
                "mock_results": self.mock_results,
            }
        return {
            "dataset_name": "Mock Dataset" if self.mock_results else "Unknown",
            "mock_results": self.mock_results,
        }
