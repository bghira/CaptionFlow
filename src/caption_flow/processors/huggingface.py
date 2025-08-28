"""HuggingFace Datasets processor implementation - Memory Optimized Version."""

import logging
import threading
import re
import requests
import json
import io
import os
import gc
import psutil
from typing import Dict, Any, List, Optional, Iterator, Set, Deque, Tuple
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import hf_hub_download, get_token
from caption_flow.storage import StorageManager

from .base import OrchestratorProcessor, WorkerProcessor, ProcessorConfig, WorkUnit, WorkResult
from ..utils import ChunkTracker
from ..models import JobId

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def log_memory(location: str):
    """Log memory usage at specific location."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(
        f"Memory at {location}: RSS={mem_info.rss/1024/1024:.1f}MB, VMS={mem_info.vms/1024/1024:.1f}MB"
    )
    # Force garbage collection
    gc.collect()


class HuggingFaceDatasetOrchestratorProcessor(OrchestratorProcessor):
    """Memory-optimized orchestrator processor for HuggingFace datasets."""

    def __init__(self):
        logger.debug("Initializing HuggingFaceDatasetOrchestratorProcessor (Optimized)")
        self.dataset_name: Optional[str] = None
        self.config: Optional[str] = None
        self.split: Optional[str] = None
        self.chunk_tracker: Optional[ChunkTracker] = None
        self.chunk_size: int = 1000
        self.token = get_token()

        # Shard information
        self.shard_info: Dict[int, Dict[str, Any]] = {}
        self.total_items: int = 0

        # Work unit management - only store active units
        self.pending_units: Deque[str] = deque()
        self.assigned_units: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()

        # Track current chunk index for on-demand creation
        self.current_chunk_index = 0

        # Cache data files info instead of loading builder repeatedly
        self.data_files: List[str] = []

        # Background thread for creating work units
        self.unit_creation_thread: Optional[threading.Thread] = None
        self.stop_creation = threading.Event()

    def initialize(self, config: ProcessorConfig, storage: StorageManager) -> None:
        """Initialize HuggingFace dataset processor."""
        logger.debug("Initializing orchestrator with config: %s", config.config)
        log_memory("start of initialize")

        cfg = config.config

        # Dataset configuration
        dataset_cfg = cfg.get("dataset", {})
        self.dataset_name = dataset_cfg.get("dataset_path")
        if not self.dataset_name:
            raise ValueError("dataset_path is required in config")

        # Auto-detect config if not provided
        provided_config = dataset_cfg.get("dataset_config")
        self.config = self._detect_config(provided_config)

        # Auto-detect split if not provided
        provided_split = dataset_cfg.get("dataset_split")
        self.split = self._detect_split(provided_split)

        logger.info(
            f"Using dataset: {self.dataset_name}, config: {self.config}, split: {self.split}"
        )

        # Chunk settings
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.min_buffer = cfg.get("min_chunk_buffer", 10)
        self.buffer_multiplier = cfg.get("chunk_buffer_multiplier", 3)

        # Initialize chunk tracking
        self.checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_tracker = ChunkTracker(self.checkpoint_dir / "chunks.json")

        # Discover shards (optimized)
        self._discover_shards_optimized()

        # Restore existing state
        self._restore_state(storage=storage)

        # Start background unit creation
        self.unit_creation_thread = threading.Thread(
            target=self._create_units_background, daemon=True
        )
        self.unit_creation_thread.start()

        log_memory("end of initialize")

    def _detect_config(self, provided_config: Optional[str]) -> str:
        """Auto-detect config if not provided."""
        if provided_config:
            return provided_config

        try:
            configs = get_dataset_config_names(self.dataset_name, token=self.token)
            if not configs:
                return "default"

            preferred = ["default", "en", "train", "main"]
            for pref in preferred:
                if pref in configs:
                    logger.info(f"Auto-selected config: {pref}")
                    return pref

            logger.info(f"Auto-selected first available config: {configs[0]}")
            return configs[0]
        except Exception as e:
            logger.warning(f"Error detecting config: {e}, using 'default'")
            return "default"

    def _detect_split(self, provided_split: Optional[str]) -> str:
        """Auto-detect split if not provided."""
        if provided_split:
            return provided_split

        try:
            splits = get_dataset_split_names(
                self.dataset_name, config_name=self.config, token=self.token
            )
            if not splits:
                return "train"

            preferred = ["train", "training", "test", "validation", "dev"]
            for pref in preferred:
                if pref in splits:
                    logger.info(f"Auto-selected split: {pref}")
                    return pref

            logger.info(f"Auto-selected first available split: {splits[0]}")
            return splits[0]
        except Exception as e:
            logger.warning(f"Error detecting split: {e}, using 'train'")
            return "train"

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from HF URL format."""
        match = re.search(r"@[a-f0-9]+/(.+)$", url)
        if match:
            return match.group(1)
        return url.split("/")[-1]

    def _get_data_files_from_builder(self) -> List[str]:
        """Get data files using dataset builder with minimal memory usage."""
        # Load builder to get correct file structure
        from datasets import load_dataset_builder

        builder = load_dataset_builder(self.dataset_name, self.config)

        # Get data files for our split
        data_files = []
        if hasattr(builder.config, "data_files"):
            if isinstance(builder.config.data_files, dict):
                files = builder.config.data_files.get(self.split, [])
                if isinstance(files, str):
                    files = [files]
                data_files = files

        # Explicitly delete builder to free memory
        del builder
        gc.collect()

        return data_files

    def _discover_shards_optimized(self):
        """Discover all shards using dataset builder but release memory immediately."""
        logger.info("Discovering shards...")

        # Try to load cached shard info first
        shard_info_cache_path = (
            self.checkpoint_dir / f"{self.dataset_name}_{self.config}_{self.split}_shard_info.json"
        )

        if shard_info_cache_path.exists():
            try:
                with open(shard_info_cache_path, "r") as f:
                    cached_info = json.load(f)
                    if (
                        cached_info.get("dataset") == self.dataset_name
                        and cached_info.get("config") == self.config
                        and cached_info.get("split") == self.split
                    ):
                        self.shard_info = {int(k): v for k, v in cached_info["shards"].items()}
                        self.total_items = cached_info["total_items"]
                        self.data_files = cached_info.get("data_files", [])
                        logger.info(
                            f"Loaded cached shard info: {len(self.shard_info)} shards, {self.total_items} total items"
                        )
                        return
            except Exception as e:
                logger.warning(f"Failed to load cached shard info: {e}")

        # Get data files using dataset builder
        self.data_files = self._get_data_files_from_builder()

        if not self.data_files:
            raise ValueError(f"No data files found for split '{self.split}'")

        logger.info(f"Found {len(self.data_files)} data files")

        # Get metadata for each shard
        cumulative_offset = 0
        for i, file_url in enumerate(self.data_files):
            filename = self._extract_filename_from_url(file_url)
            logger.info(f"Discovering shard {i}: {filename}")

            try:
                # Download file to get metadata
                local_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=filename,
                    repo_type="dataset",
                    token=self.token,
                )

                # Read only metadata
                metadata = pq.read_metadata(local_path)
                size = metadata.num_rows

                self.shard_info[i] = {
                    "shard_id": i,
                    "file_url": file_url,
                    "filename": filename,
                    "start_offset": cumulative_offset,
                    "size": size,
                    "end_offset": cumulative_offset + size - 1,
                }

                cumulative_offset += size
                logger.info(f"Shard {i}: {size} rows")

            except Exception as e:
                logger.error(f"Failed to discover shard {i}: {e}")
                # Skip this shard
                continue

        self.total_items = cumulative_offset
        logger.info(f"Total items across all shards: {self.total_items}")

        # Cache shard info
        try:
            cache_data = {
                "dataset": self.dataset_name,
                "config": self.config,
                "split": self.split,
                "shards": self.shard_info,
                "total_items": self.total_items,
                "data_files": self.data_files,
            }
            with open(shard_info_cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.info(f"Cached shard info to {shard_info_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache shard info: {e}")

        # Force garbage collection
        gc.collect()
        log_memory("after discovering shards")

    def _get_shard_for_index(self, global_index: int) -> Tuple[int, int]:
        """Get shard ID and local index for a global index."""
        for shard_id, sinfo in self.shard_info.items():
            if sinfo["start_offset"] <= global_index <= sinfo["end_offset"]:
                local_index = global_index - sinfo["start_offset"]
                return shard_id, local_index
        raise ValueError(f"Global index {global_index} not found in any shard")

    def _restore_state(self, storage: StorageManager) -> None:
        """Restore state from chunk tracker."""
        logger.debug("Restoring state from chunk tracker")
        if not self.chunk_tracker:
            return

        with self.lock:
            max_chunk_index = -1

            for chunk_id, chunk_state in self.chunk_tracker.chunks.items():
                chunk_index = chunk_state.start_index // self.chunk_size
                max_chunk_index = max(max_chunk_index, chunk_index)

                # Only add incomplete chunks to pending
                if chunk_state.status != "completed":
                    self.pending_units.append(chunk_id)

            self.current_chunk_index = max_chunk_index + 1
            logger.info(f"Resuming from chunk index {self.current_chunk_index}")

    def _create_work_unit(self, chunk_index: int) -> Optional[WorkUnit]:
        """Create a single work unit for a chunk index."""
        current_index = chunk_index * self.chunk_size

        if current_index >= self.total_items:
            return None

        chunk_size = min(self.chunk_size, self.total_items - current_index)

        # Find shard for this chunk
        shard_id, _ = self._get_shard_for_index(current_index)
        shard_name = Path(self.shard_info[shard_id]["filename"]).stem

        job_id_obj = JobId(shard_id=shard_name, chunk_id=chunk_index, sample_id=current_index)
        unit_id = job_id_obj.get_chunk_str()

        unit = WorkUnit(
            unit_id=unit_id,
            chunk_id=unit_id,
            source_id=shard_name,
            data={
                "dataset_name": self.dataset_name,
                "config": self.config,
                "split": self.split,
                "start_index": current_index,
                "chunk_size": chunk_size,
                "unprocessed_ranges": [(current_index, current_index + chunk_size - 1)],
                "shard_ids": [shard_id],
                "data_files": self.data_files,  # Pass ALL data files for offset calculation
            },
            metadata={
                "dataset": self.dataset_name,
                "shard_name": shard_name,
                "chunk_index": chunk_index,
            },
        )

        return unit

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

        while not self.stop_creation.is_set():
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

            while units_created < units_needed:
                logger.debug(f"Creating work unit for chunk {self.current_chunk_index}")
                if self.current_chunk_index * self.chunk_size >= self.total_items:
                    threading.Event().wait(30)
                    break

                unit_id = f"{self.dataset_name}:chunk:{self.current_chunk_index}"

                with self.lock:
                    # Check if already tracked
                    if self.chunk_tracker and unit_id in self.chunk_tracker.chunks:
                        chunk_state = self.chunk_tracker.chunks[unit_id]
                        if chunk_state.status == "completed":
                            self.current_chunk_index += 1
                            continue

                    # Add to pending
                    self.pending_units.append(unit_id)

                    # Track in chunk tracker
                    if self.chunk_tracker:
                        start_index = self.current_chunk_index * self.chunk_size
                        chunk_size = min(self.chunk_size, self.total_items - start_index)
                        self.chunk_tracker.add_chunk(
                            unit_id,
                            self.dataset_name,
                            "",
                            start_index,
                            chunk_size,
                        )

                    units_created += 1
                    self.current_chunk_index += 1

            if units_created > 0:
                logger.debug(f"Created {units_created} work unit IDs")

        logger.info("Thread for creating units has completed. Exiting thread.")

    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""

        logger.debug(
            "get_work_units called: count=%d worker_id=%s, pending: %d",
            count,
            worker_id,
            len(self.pending_units),
        )
        assigned = []
        with self.lock:
            while len(assigned) < count and self.pending_units:
                unit_id = self.pending_units.popleft()

                # Create work unit on demand
                chunk_index = int(unit_id.split(":")[-1])
                unit = self._create_work_unit(chunk_index)

                if unit:
                    self.assigned_units[worker_id].add(unit_id)
                    assigned.append(unit)
                    logger.debug("Assigning unit %s to worker %s", unit_id, worker_id)

                    if self.chunk_tracker:
                        self.chunk_tracker.mark_assigned(unit_id, worker_id)

        logger.debug("Returning %d work units to worker %s", len(assigned), worker_id)
        return assigned

    def mark_completed(self, unit_id: str, worker_id: str) -> None:
        """Mark a work unit as completed."""
        logger.debug("Marking unit %s as completed by worker %s", unit_id, worker_id)
        with self.lock:
            self.assigned_units[worker_id].discard(unit_id)

            if self.chunk_tracker:
                self.chunk_tracker.mark_completed(unit_id)

    def mark_failed(self, unit_id: str, worker_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        logger.debug("Marking unit %s as failed by worker %s, error: %s", unit_id, worker_id, error)
        with self.lock:
            self.assigned_units[worker_id].discard(unit_id)
            self.pending_units.append(unit_id)

            if self.chunk_tracker:
                self.chunk_tracker.mark_failed(unit_id)

    def release_assignments(self, worker_id: str) -> None:
        """Release all assignments for a disconnected worker."""
        logger.debug("Releasing assignments for worker %s", worker_id)
        with self.lock:
            unit_ids = list(self.assigned_units.get(worker_id, []))

            for unit_id in unit_ids:
                self.pending_units.append(unit_id)

            if worker_id in self.assigned_units:
                del self.assigned_units[worker_id]

            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)

    def update_from_storage(self, processed_job_ids: Set[str]) -> None:
        """Update work units based on what's been processed."""
        logger.info(f"Updating from storage with {len(processed_job_ids)} processed jobs")
        # No need to update in-memory work units since we create on demand

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self.lock:
            stats = {
                "dataset": self.dataset_name,
                "config": self.config,
                "split": self.split,
                "pending_units": len(self.pending_units),
                "assigned_units": sum(len(units) for units in self.assigned_units.values()),
                "total_shards": len(self.shard_info),
                "total_items": self.total_items,
                "workers": len(self.assigned_units),
                "current_chunk_index": self.current_chunk_index,
            }
            return stats

    def handle_result(self, result: WorkResult) -> Dict[str, Any]:
        """Handle result processing."""
        base_result = super().handle_result(result)

        if self.chunk_tracker:
            if "item_indices" in result.metadata:
                indices = result.metadata["item_indices"]
                if indices:
                    # Convert to ranges for efficient tracking
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

                    for start_idx, end_idx in ranges:
                        self.chunk_tracker.mark_items_processed(result.chunk_id, start_idx, end_idx)

        return base_result

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


class HuggingFaceDatasetWorkerProcessor(WorkerProcessor):
    """Memory-optimized worker processor for HuggingFace datasets."""

    def __init__(self):
        logger.debug("Initializing HuggingFaceDatasetWorkerProcessor (Optimized)")
        self.dataset_config: Dict[str, Any] = {}
        self.token = get_token()
        self.image_column: Optional[str] = None
        self.url_column: Optional[str] = None

        # Thread-local storage for shard info to avoid repeated builder loading
        self._thread_local = threading.local()

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize processor."""
        logger.debug("Initializing worker with config: %s", config.config)
        self.dataset_config = config.config.get("dataset", {})

        self.image_column = self.dataset_config.get("dataset_image_column", "image")
        self.url_column = self.dataset_config.get("dataset_url_column", "image_url")
        self.dataset_path = self.dataset_config.get("dataset_path", None)

    def _get_shard_path(self, dataset_name: str, shard_filename: str) -> str:
        """Get local path for a shard, downloading if needed."""
        return hf_hub_download(
            repo_id=dataset_name, filename=shard_filename, repo_type="dataset", token=self.token
        )

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from HF URL format."""
        match = re.search(r"@[a-f0-9]+/(.+)$", url)
        if match:
            return match.group(1)
        return url.split("/")[-1]

    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a work unit, yielding items to be captioned."""
        logger.debug("Processing unit: %s", unit.unit_id)
        log_memory(f"start processing unit {unit.unit_id}")

        dataset_name = unit.data["dataset_name"]
        start_index = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get(
            "unprocessed_ranges", [(start_index, start_index + chunk_size - 1)]
        )
        shard_ids = unit.data.get("shard_ids", [])
        data_files = unit.data.get("data_files", [])

        logger.info(f"Processing unit {unit.unit_id} with ranges: {unprocessed_ranges}")

        # Build shard info from provided data files (no dataset builder needed)
        shard_info = {}

        if data_files:
            # Use provided data files
            for i, file_url in enumerate(data_files):
                if i in shard_ids:
                    filename = self._extract_filename_from_url(file_url)
                    shard_path = self._get_shard_path(dataset_name, filename)

                    # Get size from metadata
                    metadata = pq.read_metadata(shard_path)
                    size = metadata.num_rows

                    shard_info[i] = {
                        "path": shard_path,
                        "start_offset": 0,  # Will be set below
                        "end_offset": 0,  # Will be set below
                        "size": size,
                        "metadata": metadata,
                    }

            # Calculate offsets
            cumulative_offset = 0
            for i in range(max(shard_info.keys()) + 1):
                if i in shard_info:
                    shard_info[i]["start_offset"] = cumulative_offset
                    shard_info[i]["end_offset"] = cumulative_offset + shard_info[i]["size"] - 1
                    cumulative_offset += shard_info[i]["size"]
                else:
                    # Need to get size for offset calculation
                    filename = self._extract_filename_from_url(data_files[i])
                    shard_path = self._get_shard_path(dataset_name, filename)
                    metadata = pq.read_metadata(shard_path)
                    cumulative_offset += metadata.num_rows
        else:
            # This should never happen with the new orchestrator
            raise ValueError("No data files provided in work unit")

        # Create set of indices to process
        indices_to_process = set()
        for start, end in unprocessed_ranges:
            indices_to_process.update(range(start, end + 1))

        # Group indices by shard
        indices_by_shard = defaultdict(list)
        for global_idx in indices_to_process:
            for shard_id, sinfo in shard_info.items():
                if sinfo["start_offset"] <= global_idx <= sinfo["end_offset"]:
                    local_idx = global_idx - sinfo["start_offset"]
                    indices_by_shard[shard_id].append((global_idx, local_idx))
                    break

        processed_indices = []

        # Process items shard by shard
        for shard_id, idx_pairs in indices_by_shard.items():
            shard_path = shard_info[shard_id]["path"]

            # Process in batches to avoid loading entire table
            batch_size = 100
            for batch_start in range(0, len(idx_pairs), batch_size):
                batch_pairs = idx_pairs[batch_start : batch_start + batch_size]
                local_indices = [local_idx for _, local_idx in batch_pairs]

                # Read only specific rows using PyArrow
                try:
                    # Create row group filters based on metadata
                    metadata = shard_info[shard_id]["metadata"]
                    row_groups_to_read = set()

                    # Find which row groups contain our indices
                    current_row = 0
                    for rg_idx in range(metadata.num_row_groups):
                        rg_metadata = metadata.row_group(rg_idx)
                        rg_num_rows = rg_metadata.num_rows

                        # Check if any of our indices are in this row group
                        for local_idx in local_indices:
                            if current_row <= local_idx < current_row + rg_num_rows:
                                row_groups_to_read.add(rg_idx)

                        current_row += rg_num_rows

                    # Read only necessary row groups
                    table = pq.read_table(shard_path, row_groups=list(row_groups_to_read))

                    # Process items
                    for global_idx, local_idx in batch_pairs:
                        try:
                            # Get item as dictionary
                            item = {}
                            for col_name in table.column_names:
                                value = table[col_name][local_idx].as_py()
                                item[col_name] = value

                            # Process image
                            image = None
                            image_url = None

                            if self.url_column and self.url_column in item:
                                image_url = item[self.url_column]
                                try:
                                    response = requests.get(image_url, timeout=30)
                                    response.raise_for_status()
                                    image = Image.open(io.BytesIO(response.content))
                                except Exception as e:
                                    logger.error(f"Error downloading image from {image_url}: {e}")
                                    continue

                            elif self.image_column and self.image_column in item:
                                image_data = item[self.image_column]
                                if isinstance(image_data, dict) and "bytes" in image_data:
                                    image = Image.open(io.BytesIO(image_data["bytes"]))
                                elif isinstance(image_data, bytes):
                                    image = Image.open(io.BytesIO(image_data))

                            if image is None:
                                logger.warning(f"No image found for item at index {global_idx}")
                                continue

                            # Build job ID
                            chunk_index = unit.metadata["chunk_index"]
                            shard_name = unit.metadata["shard_name"]
                            job_id_obj = JobId(
                                shard_id=shard_name,
                                chunk_id=str(chunk_index),
                                sample_id=str(global_idx),
                            )
                            job_id = job_id_obj.get_sample_str()

                            # Clean metadata
                            clean_metadata = {
                                k: v
                                for k, v in item.items()
                                if k not in [self.image_column, self.url_column]
                                and not k.startswith("_")
                            }

                            clean_metadata.update(
                                {
                                    "_item_index": global_idx,
                                    "_chunk_relative_index": global_idx - start_index,
                                    "_job_id": job_id,
                                    "_shard_id": shard_id,
                                    "_local_index": local_idx,
                                    "_url": image_url,
                                }
                            )

                            yield {
                                "image": image,
                                "item_key": str(global_idx),
                                "item_index": global_idx,
                                "metadata": clean_metadata,
                                "job_id": job_id,
                            }

                            processed_indices.append(global_idx)

                        except Exception as e:
                            logger.error(f"Error processing item at index {global_idx}: {e}")

                    # Explicitly delete table to free memory
                    del table
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error reading batch from shard {shard_id}: {e}")

        # Store processed indices in context
        context["_processed_indices"] = processed_indices
        logger.debug("Processed indices for unit %s: %s", unit.unit_id, len(processed_indices))

        # Force garbage collection
        gc.collect()
        log_memory(f"end processing unit {unit.unit_id}")

    def prepare_result(
        self, unit: WorkUnit, outputs: List[Dict[str, Any]], processing_time_ms: float
    ) -> WorkResult:
        """Prepare result."""
        logger.debug("Preparing result for unit %s", unit.unit_id)
        result = super().prepare_result(unit, outputs, processing_time_ms)

        # Add processed indices to metadata
        if outputs and "_processed_indices" in outputs[0].get("metadata", {}):
            result.metadata["item_indices"] = outputs[0]["metadata"]["_processed_indices"]

        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "dataset_path": self.dataset_config.get("dataset_path"),
            "dataset_type": "huggingface",
            "config": self.dataset_config.get("dataset_config"),
            "split": self.dataset_config.get("dataset_split"),
        }
