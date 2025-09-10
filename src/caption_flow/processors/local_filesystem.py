"""Local filesystem datasets processor implementation."""

import asyncio
import io
import logging
import mimetypes
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple

import aiofiles
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from caption_flow.storage import StorageManager

from ..models import JobId
from ..utils import ChunkTracker
from .base import OrchestratorProcessor, ProcessorConfig, WorkerProcessor, WorkResult, WorkUnit

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".svg"}


class LocalFilesystemOrchestratorProcessor(OrchestratorProcessor):
    """Orchestrator processor for local filesystem datasets."""

    def __init__(self):
        logger.debug("Initializing LocalFilesystemOrchestratorProcessor")
        self.dataset_path: Optional[Path] = None
        self.chunk_tracker: Optional[ChunkTracker] = None
        self.chunk_size: int = 1000
        self.recursive: bool = True
        self.follow_symlinks: bool = False

        # Image file tracking
        self.all_images: List[Tuple[Path, int]] = []  # (path, size_bytes)
        self.total_images: int = 0
        self.current_index: int = 0

        # Work unit management
        self.work_units: Dict[str, WorkUnit] = {}
        self.pending_units: Deque[str] = deque()
        self.assigned_units: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()

        # Background thread for creating work units
        self.unit_creation_thread: Optional[threading.Thread] = None
        self.stop_creation = threading.Event()

        # HTTP server for serving images
        self.http_app: Optional[FastAPI] = None
        self.http_server_task: Optional[asyncio.Task] = None
        self.http_bind_address: str = "0.0.0.0"
        self.http_port: int = 8766

    def initialize(self, config: ProcessorConfig, storage: StorageManager) -> None:
        """Initialize local filesystem processor."""
        logger.debug("Initializing orchestrator with config: %s", config.config)
        cfg = config.config

        # Dataset configuration
        dataset_cfg = cfg.get("dataset", {})
        self.dataset_path = Path(dataset_cfg.get("dataset_path", "."))

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        self.recursive = dataset_cfg.get("recursive", True)
        self.follow_symlinks = dataset_cfg.get("follow_symlinks", False)

        # Chunk settings
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.min_buffer = cfg.get("min_chunk_buffer", 10)
        self.buffer_multiplier = cfg.get("chunk_buffer_multiplier", 3)

        # HTTP server settings
        self.http_bind_address = dataset_cfg.get("http_bind_address", "0.0.0.0")
        self.http_public_address = dataset_cfg.get("public_address", "127.0.0.1")
        self.http_port = dataset_cfg.get("http_port", 8766)

        logger.info(f"Root path: {self.dataset_path}, recursive: {self.recursive}")

        # Initialize chunk tracking
        checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")

        # Discover images
        self._discover_images()

        # Restore existing state
        self._restore_state(storage)

        # Start HTTP server for image serving
        self._start_http_server()

        # Start background unit creation
        self.unit_creation_thread = threading.Thread(
            target=self._create_units_background, daemon=True
        )
        self.unit_creation_thread.start()
        logger.debug("Unit creation thread started")

    def _discover_images(self):
        """Discover all image files in the filesystem."""
        logger.info("Discovering images...")

        if self.recursive:
            # Walk directory tree
            for root, dirs, files in os.walk(self.dataset_path, followlinks=self.follow_symlinks):
                root_path = Path(root)

                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for file in files:
                    if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        file_path = root_path / file
                        try:
                            size = file_path.stat().st_size
                            self.all_images.append((file_path, size))
                        except OSError as e:
                            logger.warning(f"Cannot stat {file_path}: {e}")
        else:
            # Just scan root directory
            for file_path in self.dataset_path.iterdir():
                if file_path.is_file() and any(
                    file_path.suffix.lower() == ext for ext in IMAGE_EXTENSIONS
                ):
                    try:
                        size = file_path.stat().st_size
                        self.all_images.append((file_path, size))
                    except OSError as e:
                        logger.warning(f"Cannot stat {file_path}: {e}")

        # Sort for consistent ordering
        self.all_images.sort(key=lambda x: str(x[0]))
        self.total_images = len(self.all_images)

        logger.info(f"Found {self.total_images} images")

    def _start_http_server(self):
        """Start HTTP server for serving images."""
        self.http_app = FastAPI()

        @self.http_app.get("/image/{image_index:int}")
        async def get_image(image_index: int):
            """Serve an image by index."""
            if image_index < 0 or image_index >= len(self.all_images):
                raise HTTPException(status_code=404, detail="Image not found")

            file_path, _ = self.all_images[image_index]

            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Image file not found")

            # Determine content type
            content_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"

            # Stream file
            async def stream_file():
                async with aiofiles.open(file_path, "rb") as f:
                    while chunk := await f.read(1024 * 1024):  # 1MB chunks
                        yield chunk

            return StreamingResponse(
                stream_file(),
                media_type=content_type,
                headers={"Content-Disposition": f'inline; filename="{file_path.name}"'},
            )

        @self.http_app.get("/info")
        async def get_info():
            """Get dataset info."""
            return {
                "total_images": self.total_images,
                "root_path": str(self.dataset_path),
                "http_url": f"http://{self.http_public_address}:{self.http_port}",
            }

        # Start server in background
        async def run_server():
            config = uvicorn.Config(
                app=self.http_app,
                host=self.http_bind_address,
                port=self.http_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()

        loop = asyncio.new_event_loop()
        self.http_server_task = loop.create_task(run_server())

        # Run in thread
        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=run_loop, daemon=True).start()
        logger.info(
            f"HTTP server started on {self.http_bind_address}:{self.http_port}, advertising hostname {self.http_public_address} to clients"
        )

    def _restore_state(self, storage: StorageManager) -> None:
        """Restore state from chunk tracker."""
        logger.debug("Restoring state from chunk tracker")
        if not self.chunk_tracker:
            return

        storage.get_all_processed_job_ids()

        with self.lock:
            for chunk_id, chunk_state in self.chunk_tracker.chunks.items():
                # Get unprocessed ranges (relative coordinates from ChunkTracker)
                relative_unprocessed_ranges = chunk_state.get_unprocessed_ranges()

                # Convert relative ranges to absolute ranges
                unprocessed_ranges = []
                for start, end in relative_unprocessed_ranges:
                    abs_start = chunk_state.start_index + start
                    abs_end = chunk_state.start_index + end
                    unprocessed_ranges.append((abs_start, abs_end))

                if unprocessed_ranges:
                    # Create work unit for unprocessed items
                    chunk_index = chunk_state.start_index // self.chunk_size

                    # Get filenames for this chunk
                    filenames = {}
                    for idx in range(
                        chunk_state.start_index, chunk_state.start_index + chunk_state.chunk_size
                    ):
                        if idx < len(self.all_images):
                            filenames[idx] = self.all_images[idx][0].name

                    unit = WorkUnit(
                        unit_id=chunk_id,
                        chunk_id=chunk_id,
                        source_id="local",
                        unit_size=chunk_state.chunk_size,
                        data={
                            "start_index": chunk_state.start_index,
                            "chunk_size": chunk_state.chunk_size,
                            "unprocessed_ranges": unprocessed_ranges,
                            "http_url": f"http://{self.http_public_address}:{self.http_port}",
                            "filenames": filenames,
                        },
                        metadata={
                            "dataset": str(self.dataset_path),
                            "chunk_index": chunk_index,
                        },
                    )

                    self.work_units[unit.unit_id] = unit
                    self.pending_units.append(unit.unit_id)

    def _create_units_background(self) -> None:
        """Background thread to create work units on demand."""
        logger.info("Starting work unit creation thread")

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

            while units_created < units_needed and self.current_index < self.total_images:
                chunk_size = min(self.chunk_size, self.total_images - self.current_index)
                chunk_id = self.current_index // self.chunk_size

                with self.lock:
                    job_id_obj = JobId(
                        shard_id="local", chunk_id=str(chunk_id), sample_id=str(self.current_index)
                    )
                    unit_id = job_id_obj.get_chunk_str()  # e.g. "local:chunk:0"

                    if unit_id in self.work_units:
                        self.current_index += self.chunk_size
                        continue

                    # Check if chunk is already completed
                    if self.chunk_tracker:
                        chunk_state = self.chunk_tracker.chunks.get(unit_id)
                        if chunk_state and chunk_state.status == "completed":
                            self.current_index += self.chunk_size
                            continue

                    # Get filenames for this chunk
                    filenames = {}
                    for idx in range(self.current_index, self.current_index + chunk_size):
                        if idx < len(self.all_images):
                            filenames[idx] = self.all_images[idx][0].name

                    unit = WorkUnit(
                        unit_id=unit_id,
                        chunk_id=unit_id,
                        source_id="local",
                        unit_size=chunk_size,
                        data={
                            "start_index": self.current_index,
                            "chunk_size": chunk_size,
                            "unprocessed_ranges": [
                                (self.current_index, self.current_index + chunk_size - 1)
                            ],
                            "http_url": f"http://{self.http_public_address}:{self.http_port}",
                            "filenames": filenames,
                        },
                        metadata={
                            "dataset": str(self.dataset_path),
                            "chunk_index": chunk_id,
                        },
                    )

                    self.work_units[unit_id] = unit
                    self.pending_units.append(unit_id)

                    if self.chunk_tracker:
                        self.chunk_tracker.add_chunk(
                            unit_id, "local", str(self.dataset_path), self.current_index, chunk_size
                        )

                    units_created += 1

                self.current_index += self.chunk_size

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
                start_index = unit.data["start_index"]
                chunk_size = unit.data["chunk_size"]
                chunk_index = unit.metadata["chunk_index"]

                # Find processed indices for this chunk
                processed_indices = []
                for job_id in processed_job_ids:
                    job_id_obj = JobId.from_str(job_id)
                    if job_id_obj.shard_id == "local" and int(job_id_obj.chunk_id) == chunk_index:
                        idx = int(job_id_obj.sample_id)
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
                "dataset": str(self.dataset_path),
                "total_units": len(self.work_units),
                "pending_units": len(self.pending_units),
                "assigned_units": sum(len(units) for units in self.assigned_units.values()),
                "total_images": self.total_images,
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

    def get_image_paths(self) -> List[Tuple[Path, int]]:
        """Get the list of discovered image paths and sizes."""
        return self.all_images


class LocalFilesystemWorkerProcessor(WorkerProcessor):
    """Worker processor for local filesystem datasets."""

    def __init__(self):
        logger.debug("Initializing LocalFilesystemWorkerProcessor")
        self.dataset_path: Optional[Path] = None
        self.image_paths: Optional[List[Tuple[Path, int]]] = None
        self.dataset_config: Dict[str, Any] = {}

    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize processor."""
        logger.debug("Initializing worker with config: %s", config.config)
        self.dataset_config = config.config.get("dataset", {})

        # Check if worker has local storage access
        worker_cfg = config.config.get("worker", {})
        local_path = worker_cfg.get("local_storage_path")

        self.dataset_path = None
        if local_path:
            self.dataset_path = Path(local_path)
            if self.dataset_path.exists():
                logger.info(f"Worker has local storage access at: {self.dataset_path}")
                # Could potentially cache image list here if needed
            else:
                logger.warning(f"Local storage path does not exist: {self.dataset_path}")
                self.dataset_path = None
        else:
            logger.info("Worker does not have local storage access, will use HTTP")

    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a work unit, yielding items to be captioned."""
        logger.debug("Processing unit: %s", unit.unit_id)

        start_index = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get(
            "unprocessed_ranges", [(start_index, start_index + chunk_size - 1)]
        )
        http_url = unit.data.get("http_url")
        filenames = unit.data.get("filenames", {})

        logger.info(f"Processing unit {unit.unit_id} with ranges: {unprocessed_ranges}")

        # Create set of indices to process
        indices_to_process = set()
        for start, end in unprocessed_ranges:
            indices_to_process.update(range(start, end + 1))

        processed_indices = []

        # Get orchestrator info if we need HTTP
        context.get("orchestrator")

        for idx in sorted(indices_to_process):
            try:
                image = None
                filename = filenames.get(str(idx), f"image_{idx}")

                if self.dataset_path and self.image_paths:
                    # Direct file access
                    if 0 <= idx < len(self.image_paths):
                        file_path, _ = self.image_paths[idx]
                        if file_path.exists():
                            image = Image.open(file_path)
                            filename = file_path.name
                            logger.debug(f"Loaded image from local path: {file_path}")
                        else:
                            logger.warning(f"Local file not found: {file_path}")

                if image is None and http_url:
                    # HTTP fallback
                    image_url = f"{http_url}/image/{idx}"
                    try:
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                        logger.debug(f"Loaded image via HTTP: {image_url}")
                    except Exception as e:
                        logger.error(f"Error downloading image from {image_url}: {e}")
                        continue

                if image is None:
                    logger.warning(f"Could not load image at index {idx}")
                    continue

                # Build job ID
                chunk_index = unit.metadata["chunk_index"]
                job_id_obj = JobId(shard_id="local", chunk_id=str(chunk_index), sample_id=str(idx))
                job_id = job_id_obj.get_sample_str()

                # Metadata
                clean_metadata = {
                    "_item_index": idx,
                    "_chunk_relative_index": idx - start_index,
                    "_job_id": job_id,
                    "_filename": filename,
                }

                yield {
                    "image": image,
                    "item_key": str(idx),
                    "item_index": idx,
                    "metadata": clean_metadata,
                    "job_id": job_id,
                }

                processed_indices.append(idx)

            except Exception as e:
                logger.error(f"Error processing item at index {idx}: {e}")

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
            "dataset_path": self.dataset_config.get("dataset_path", "local"),
            "dataset_type": "local_filesystem",
            "has_local_access": self.dataset_path is not None,
        }

    def set_image_paths_from_orchestrator(self, image_paths: List[Tuple[str, int]]) -> None:
        """Set the image paths list from orchestrator (for local access mode)."""
        if self.dataset_path:
            # Convert paths relative to our local storage path
            self.image_paths = []
            for path_str, size in image_paths:
                # Orchestrator sends paths relative to its root
                # We need to resolve them relative to our local_storage_path
                self.image_paths.append((self.dataset_path / path_str, size))
            logger.info(f"Set {len(self.image_paths)} image paths for local access")
