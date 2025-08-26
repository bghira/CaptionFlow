"""HuggingFace Datasets processor implementation."""

import logging
import threading
import re
import requests
from typing import Dict, Any, List, Optional, Iterator, Set, Deque, Tuple
from collections import deque, defaultdict
from pathlib import Path
import json
import io
from datetime import datetime
from PIL import Image
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset_builder,
)
from huggingface_hub import hf_hub_download, get_token
from caption_flow.storage import StorageManager

from .base import OrchestratorProcessor, WorkerProcessor, ProcessorConfig, WorkUnit, WorkResult
from ..utils import ChunkTracker
from ..models import JobId

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HuggingFaceDatasetOrchestratorProcessor(OrchestratorProcessor):
    """Orchestrator processor for HuggingFace datasets."""

    def __init__(self):
        logger.debug("Initializing HuggingFaceDatasetOrchestratorProcessor")
        self.dataset_name: Optional[str] = None
        self.config: Optional[str] = None
        self.split: Optional[str] = None
        self.chunk_tracker: Optional[ChunkTracker] = None
        self.chunk_size: int = 1000
        self.token = get_token()

        # Shard information
        self.shard_info: Dict[int, Dict[str, Any]] = {}
        self.total_items: int = 0

        # Work unit management
        self.work_units: Dict[str, WorkUnit] = {}
        self.pending_units: Deque[str] = deque()
        self.assigned_units: Dict[str, Set[str]] = defaultdict(set)  # worker_id -> unit_ids
        self.lock = threading.Lock()

        # Background thread for creating work units
        self.unit_creation_thread: Optional[threading.Thread] = None
        self.stop_creation = threading.Event()

    def initialize(self, config: ProcessorConfig, storage: StorageManager) -> None:
        """Initialize HuggingFace dataset processor."""
        logger.debug("Initializing orchestrator with config: %s", config.config)
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
        checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")

        # Discover shards
        self._discover_shards()

        # Restore existing state
        self._restore_state(storage=storage)

        # Start background unit creation
        self.unit_creation_thread = threading.Thread(
            target=self._create_units_background, daemon=True
        )
        self.unit_creation_thread.start()
        logger.debug("Unit creation thread started")

    def _detect_config(self, provided_config: Optional[str]) -> str:
        """Auto-detect config if not provided."""
        if provided_config:
            return provided_config

        try:
            configs = get_dataset_config_names(self.dataset_name, token=self.token)
            if not configs:
                return "default"

            # Prefer common config names
            preferred = ["default", "en", "train", "main"]
            for pref in preferred:
                if pref in configs:
                    logger.info(f"Auto-selected config: {pref}")
                    return pref

            # Otherwise use first available
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
                logger.warning("No splits found, using 'train'")
                return "train"

            # Prefer training splits
            preferred = ["train", "training", "test", "validation", "dev"]
            for pref in preferred:
                if pref in splits:
                    logger.info(f"Auto-selected split: {pref}")
                    return pref

            # Otherwise use first available
            logger.info(f"Auto-selected first available split: {splits[0]}")
            return splits[0]
        except Exception as e:
            logger.warning(f"Error detecting split: {e}, using 'train'")
            return "train"

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from HF URL format."""
        # Format: hf://datasets/user/dataset@hash/filename
        match = re.search(r"@[a-f0-9]+/(.+)$", url)
        if match:
            return match.group(1)
        # Fallback: just get last part
        return url.split("/")[-1]

    def _discover_shards(self):
        """Discover all shards and their sizes."""
        logger.info("Discovering shards...")

        # Load dataset builder to get file info
        builder = load_dataset_builder(self.dataset_name, self.config)

        # Get data files for our split
        data_files = []
        if hasattr(builder.config, "data_files"):
            if isinstance(builder.config.data_files, dict):
                files = builder.config.data_files.get(self.split, [])
                if isinstance(files, str):
                    files = [files]
                data_files = files

        if not data_files:
            raise ValueError(f"No data files found for split '{self.split}'")

        logger.info(f"Found {len(data_files)} data files")

        # Get info about each shard
        cumulative_offset = 0
        for i, file_url in enumerate(data_files):
            filename = self._extract_filename_from_url(file_url)
            logger.info(f"Discovering shard {i}: {filename}")

            # We don't download shards here - workers will do that
            # For now, store the info we have
            self.shard_info[i] = {
                "shard_id": i,
                "file_url": file_url,
                "filename": filename,
                "start_offset": cumulative_offset,
                # Size will be determined when first worker needs it
                "size": None,
                "end_offset": None,
            }

            # Try to get size from builder info if available
            if hasattr(builder.info, "splits") and self.split in builder.info.splits:
                split_info = builder.info.splits[self.split]
                if split_info.num_examples and len(data_files) == 1:
                    # Single shard case
                    self.shard_info[i]["size"] = split_info.num_examples
                    self.shard_info[i]["end_offset"] = (
                        cumulative_offset + split_info.num_examples - 1
                    )
                    cumulative_offset += split_info.num_examples

        # If we couldn't get sizes, we'll need to load shards on demand
        if self.shard_info[0]["size"] is None:
            logger.warning("Shard sizes not available from metadata, will load on demand")
        else:
            self.total_items = cumulative_offset
            logger.info(f"Total items across all shards: {self.total_items}")

    def _get_shard_size(self, shard_id: int) -> int:
        """Get size of a shard, loading it if necessary."""
        if self.shard_info[shard_id]["size"] is not None:
            return self.shard_info[shard_id]["size"]

        # Need to load the shard to get its size
        logger.info(f"Loading shard {shard_id} to determine size...")
        filename = self.shard_info[shard_id]["filename"]

        local_path = hf_hub_download(
            repo_id=self.dataset_name, filename=filename, repo_type="dataset", token=self.token
        )

        # Load just to get size
        dataset = Dataset.from_parquet(local_path)
        size = len(dataset)

        # Update shard info
        self.shard_info[shard_id]["size"] = size

        # Update offsets for this and subsequent shards
        for sid in range(shard_id, len(self.shard_info)):
            if sid > shard_id:
                self.shard_info[sid]["start_offset"] = self.shard_info[sid - 1]["end_offset"] + 1
            self.shard_info[sid]["end_offset"] = (
                self.shard_info[sid]["start_offset"] + self.shard_info[sid]["size"] - 1
            )

        # Update total items
        if all(s["size"] is not None for s in self.shard_info.values()):
            self.total_items = sum(s["size"] for s in self.shard_info.values())
            logger.info(f"Total items: {self.total_items}")

        return size

    def _restore_state(self, storage: StorageManager) -> None:
        """Restore state from chunk tracker."""
        logger.debug("Restoring state from chunk tracker")
        if not self.chunk_tracker:
            return

        all_processed_jobs = storage.get_all_processed_job_ids()

        with self.lock:
            for chunk_id, chunk_state in self.chunk_tracker.chunks.items():
                # Calculate actual unprocessed ranges
                chunk_range = (
                    chunk_state.start_index,
                    chunk_state.start_index + chunk_state.chunk_size - 1,
                )

                # Get processed indices for this chunk
                processed_ranges = self.chunk_tracker.get_processed_indices_for_chunk(
                    chunk_id, all_processed_jobs
                )

                # Calculate unprocessed ranges
                unprocessed_ranges = self._subtract_ranges([chunk_range], processed_ranges)

                if unprocessed_ranges:
                    # Find which shard(s) this chunk belongs to
                    shard_ids = []
                    for sid, sinfo in self.shard_info.items():
                        # Need size to check
                        if sinfo["size"] is None:
                            self._get_shard_size(sid)

                        if (
                            sinfo["start_offset"]
                            <= chunk_state.start_index + chunk_state.chunk_size - 1
                            and sinfo["end_offset"] >= chunk_state.start_index
                        ):
                            shard_ids.append(sid)
                            logger.info(f"Found shard {sid} for chunk {chunk_id}: {sinfo}")

                    chunk_index = chunk_state.start_index // self.chunk_size
                    shard_name = Path(self.shard_info[shard_ids[0]]["filename"]).stem
                    unit = WorkUnit(
                        unit_id=chunk_id,
                        chunk_id=chunk_id,
                        source_id=shard_name,
                        data={
                            "dataset_name": self.dataset_name,
                            "config": self.config,
                            "split": self.split,
                            "start_index": chunk_state.start_index,
                            "chunk_size": chunk_state.chunk_size,
                            "unprocessed_ranges": unprocessed_ranges,
                            "shard_ids": shard_ids,
                        },
                        metadata={
                            "dataset": self.dataset_name,
                            "shard_name": shard_name,
                            "chunk_index": chunk_index,
                        },
                    )

                    self.work_units[unit.unit_id] = unit
                    self.pending_units.append(unit.unit_id)

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

        current_index = 0

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

            # Make sure we know total items
            if self.total_items == 0:
                # Load all shard sizes
                for sid in range(len(self.shard_info)):
                    self._get_shard_size(sid)

            # Create units as needed
            units_created = 0

            while units_created < units_needed and current_index < self.total_items:
                chunk_size = min(self.chunk_size, self.total_items - current_index)
                chunk_id = current_index // self.chunk_size

                with self.lock:
                    shard_ids = []
                    for sid, sinfo in self.shard_info.items():
                        if (
                            sinfo["start_offset"] <= current_index + chunk_size - 1
                            and sinfo["end_offset"] >= current_index
                        ):
                            shard_ids.append(sid)
                    shard_name = Path(self.shard_info[shard_ids[0]]["filename"]).stem

                    job_id_obj = JobId(
                        shard_id=shard_name, chunk_id=chunk_id, sample_id=current_index
                    )
                    unit_id = (
                        job_id_obj.get_chunk_str()
                    )  # just the chunk part, eg pixel-images:chunk:0
                    if unit_id in self.work_units:
                        current_index += self.chunk_size
                        continue

                    # Check if chunk is already completed
                    if self.chunk_tracker:
                        chunk_state = self.chunk_tracker.chunks.get(unit_id)
                        if chunk_state and chunk_state.status == "completed":
                            current_index += self.chunk_size
                            continue

                    # Find which shard(s) this chunk belongs to

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
                            "shard_ids": shard_ids,
                        },
                        metadata={
                            "dataset": self.dataset_name,
                            "shard_name": shard_name,
                            "chunk_index": chunk_id,
                        },
                    )
                    logger.debug(f"Created WorkUnit: {unit}")

                    self.work_units[unit_id] = unit
                    self.pending_units.append(unit_id)

                    if self.chunk_tracker:
                        self.chunk_tracker.add_chunk(
                            unit_id,
                            self.dataset_name,
                            "",  # No shard URL
                            current_index,
                            chunk_size,
                        )

                    units_created += 1

                current_index += self.chunk_size

            if units_created > 0:
                logger.debug(f"Created {units_created} work units")

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

    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""
        logger.debug("get_work_units called: count=%d worker_id=%s", count, worker_id)
        assigned = []

        with self.lock:
            while len(assigned) < count and self.pending_units:
                unit_id = self.pending_units.popleft()
                unit = self.work_units.get(unit_id)

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
            if unit_id in self.work_units:
                self.assigned_units[worker_id].discard(unit_id)

                if self.chunk_tracker:
                    self.chunk_tracker.mark_completed(unit_id)

    def mark_failed(self, unit_id: str, worker_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        logger.debug("Marking unit %s as failed by worker %s, error: %s", unit_id, worker_id, error)
        with self.lock:
            if unit_id in self.work_units:
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
                if unit_id in self.work_units:
                    self.pending_units.append(unit_id)

            if worker_id in self.assigned_units:
                del self.assigned_units[worker_id]

            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)

    def update_from_storage(self, processed_job_ids: Set[str]) -> None:
        """Update work units based on what's been processed."""
        logger.info(f"Updating work units from {len(processed_job_ids)} processed jobs")

        with self.lock:
            for unit_id, unit in self.work_units.items():
                # Extract chunk info from unit
                logger.debug(f"Checking unit {unit_id} for updates")
                logger.debug(f"Unit data: {unit.data}")
                logger.debug(f"Unit metadata: {unit.metadata}")
                start_index = unit.data["start_index"]
                chunk_size = unit.data["chunk_size"]
                shard_name = unit.metadata["shard_name"]
                chunk_index = unit.metadata["chunk_index"]

                # Find processed indices for this chunk
                processed_indices = []
                for job_id in processed_job_ids:
                    # Parse job_id format: "data-0000:chunk:0:idx:42"
                    job_id = JobId.from_str(job_id=job_id)
                    if job_id.shard_id == shard_name and int(job_id.chunk_id) == chunk_index:
                        idx = int(job_id.sample_id)
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

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self.lock:
            stats = {
                "dataset": self.dataset_name,
                "config": self.config,
                "split": self.split,
                "total_units": len(self.work_units),
                "pending_units": len(self.pending_units),
                "assigned_units": sum(len(units) for units in self.assigned_units.values()),
                "total_shards": len(self.shard_info),
                "total_items": self.total_items,
                "workers": len(self.assigned_units),
            }
            return stats

    def handle_result(self, result: WorkResult) -> Dict[str, Any]:
        """Handle result processing."""
        base_result = super().handle_result(result)

        # Track processed items
        if self.chunk_tracker:
            if "item_indices" not in result.metadata:
                result.metadata["item_indices"] = [result.metadata.get("_item_index")]
            indices = result.metadata["item_indices"]

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

                for start_idx, end_idx in ranges:
                    self.chunk_tracker.mark_items_processed(result.chunk_id, start_idx, end_idx)

        return base_result


class HuggingFaceDatasetWorkerProcessor(WorkerProcessor):
    """Worker processor for HuggingFace datasets."""

    def __init__(self):
        logger.debug("Initializing HuggingFaceDatasetWorkerProcessor")
        self.dataset_config: Dict[str, Any] = {}
        self.token = get_token()
        self.shard_cache: Dict[int, Dataset] = {}  # Cache loaded shards
        self.image_column: Optional[str] = None
        self.url_column: Optional[str] = None

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize processor."""
        logger.debug("Initializing worker with config: %s", config.config)
        self.dataset_config = config.config.get("dataset", {})

        # Determine if this is an image URL dataset or binary image dataset
        self.image_column = self.dataset_config.get("dataset_image_column", "image")
        self.url_column = self.dataset_config.get("dataset_url_column", "image_url")
        self.dataset_path = self.dataset_config.get("dataset_path", None)

    def _load_shard(self, dataset_name: str, shard_filename: str, shard_id: int) -> Dataset:
        """Load a shard if not already cached."""
        if shard_id in self.shard_cache:
            return self.shard_cache[shard_id]

        logger.info(f"Loading shard {shard_id}: {shard_filename}")

        local_path = hf_hub_download(
            repo_id=dataset_name, filename=shard_filename, repo_type="dataset", token=self.token
        )

        dataset = Dataset.from_parquet(local_path)
        self.shard_cache[shard_id] = dataset

        return dataset

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from HF URL format."""
        match = re.search(r"@[a-f0-9]+/(.+)$", url)
        if match:
            return match.group(1)
        return url.split("/")[-1]

    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a work unit, yielding items to be captioned."""
        logger.debug("Processing unit: %s", unit.unit_id)

        dataset_name = unit.data["dataset_name"]
        config = unit.data["config"]
        split = unit.data["split"]
        start_index = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get(
            "unprocessed_ranges", [(start_index, start_index + chunk_size - 1)]
        )
        shard_ids = unit.data.get("shard_ids", [])

        logger.info(f"Processing unit {unit.unit_id} with ranges: {unprocessed_ranges}")

        # Need to get shard info - should be passed in unit data
        # For now, we'll need to load dataset builder to get file info
        from datasets import load_dataset_builder

        builder = load_dataset_builder(dataset_name, config)

        data_files = []
        if hasattr(builder.config, "data_files"):
            if isinstance(builder.config.data_files, dict):
                files = builder.config.data_files.get(split, [])
                if isinstance(files, str):
                    files = [files]
                data_files = files

        # Build shard info
        shard_info = {}
        cumulative_offset = 0

        for i, file_url in enumerate(data_files):
            if i not in shard_ids:
                # Skip loading this shard, but we need its size for offsets
                # This is inefficient - in real implementation, orchestrator should pass this info
                filename = self._extract_filename_from_url(file_url)
                dataset = self._load_shard(dataset_name, filename, i)
                size = len(dataset)
                cumulative_offset += size
                continue

            filename = self._extract_filename_from_url(file_url)
            dataset = self._load_shard(dataset_name, filename, i)

            shard_info[i] = {
                "dataset": dataset,
                "start_offset": cumulative_offset,
                "end_offset": cumulative_offset + len(dataset) - 1,
                "columns": dataset.column_names,
            }
            cumulative_offset += len(dataset)

        # Create set of indices to process
        indices_to_process = set()
        for start, end in unprocessed_ranges:
            indices_to_process.update(range(start, end + 1))

        processed_indices = []

        # Process items
        for global_idx in sorted(indices_to_process):
            # Find which shard contains this index
            shard_id = None
            local_idx = None

            for sid, sinfo in shard_info.items():
                if sinfo["start_offset"] <= global_idx <= sinfo["end_offset"]:
                    shard_id = sid
                    local_idx = global_idx - sinfo["start_offset"]
                    break

            if shard_id is None:
                logger.warning(f"Could not find shard for global index {global_idx}")
                continue

            try:
                # Get item from shard
                item = shard_info[shard_id]["dataset"][local_idx]

                # Check if this is a URL dataset or binary image dataset
                image = None
                image_url = None

                # Try URL column first
                if self.url_column and self.url_column in item:
                    image_url = item[self.url_column]
                    # Download image from URL
                    try:
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    except Exception as e:
                        logger.error(f"Error downloading image from {image_url}: {e}")
                        continue

                # Try binary image column
                elif self.image_column and self.image_column in item:
                    image_data = item[self.image_column]
                    if isinstance(image_data, Image.Image):
                        image = image_data
                    elif isinstance(image_data, dict) and "bytes" in image_data:
                        # Handle datasets Image feature
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
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(global_idx)
                )
                job_id = job_id_obj.get_sample_str()

                # Clean metadata
                clean_metadata = {
                    k: v
                    for k, v in item.items()
                    if k not in [self.image_column, self.url_column] and not k.startswith("_")
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

        # Store processed indices in context
        context["_processed_indices"] = processed_indices
        logger.debug("Processed indices for unit %s: %s", unit.unit_id, processed_indices)

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
