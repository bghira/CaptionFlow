"""Orchestrator with proper chunk sizing and caption list handling."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Optional, Any, List
from collections import deque

import websockets
from websockets.server import WebSocketServerProtocol

from .storage import StorageManager
from .models import Caption, Contributor
from .utils.auth import AuthManager

logger = logging.getLogger(__name__)


@dataclass
class ShardChunk:
    """A chunk of a shard assigned to a worker."""

    chunk_id: str
    dataset: str
    shard_url: str
    shard_name: str
    start_index: int
    chunk_size: int
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_items: int = 0
    status: str = "pending"


@dataclass
class ShardAssignment:
    """Track which worker is processing which shard chunks."""

    worker_id: str
    chunks: List[ShardChunk] = field(default_factory=list)
    items_processed: int = 0
    items_failed: int = 0


class Orchestrator:
    """Orchestrator with configurable chunk sizing and caption list support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)

        # Dataset configuration with sensible defaults
        self.dataset_config = config.get("dataset", {})
        self.dataset_path = self.dataset_config.get("path")
        self.dataset_type = self.dataset_config.get("type", "huggingface")

        # Chunk configuration - CRITICAL for performance
        self.chunk_size = self.dataset_config.get("chunk_size", 100)  # Default to small chunks
        self.items_per_shard = self.dataset_config.get("items_per_shard", 3900)  # Estimated
        self.readahead_chunks = self.dataset_config.get("readahead_chunks", 3)
        self.chunk_timeout_minutes = self.dataset_config.get("chunk_timeout_minutes", 15)

        logger.info(
            f"Chunk configuration: size={self.chunk_size}, timeout={self.chunk_timeout_minutes}min"
        )
        logger.info(f"Expected chunks per shard: {self.items_per_shard // self.chunk_size}")

        # Storage
        storage_config = config.get("storage", {})
        self.storage = StorageManager(
            Path(storage_config.get("data_dir", "./caption_data")),
            caption_buffer_size=storage_config.get("caption_buffer_size", 100),
        )

        # Auth
        self.auth = AuthManager(config.get("auth", {}))

        # Shard management
        self.all_shards: List[str] = []
        self.shard_queue: deque = deque()
        self.active_chunks: Dict[str, ShardChunk] = {}
        self.completed_shards: Set[str] = set()
        self.shard_progress: Dict[str, Dict] = {}  # Track per-shard progress

        # Worker management
        self.workers: Dict[str, WebSocketServerProtocol] = {}
        self.worker_assignments: Dict[str, ShardAssignment] = {}
        self.monitors: Set[WebSocketServerProtocol] = set()

        # Statistics
        self.stats = {
            "total_shards": 0,
            "completed_shards": 0,
            "active_chunks": 0,
            "pending_chunks": 0,
            "total_items_processed": 0,
            "total_captions": 0,
            "total_unique_images": 0,  # Track unique images
            "avg_captions_per_image": 0,  # Track average captions per image
            "connected_workers": 0,
            "avg_processing_time_ms": 0,
            "items_per_minute": 0,
            "captions_per_minute": 0,  # Track caption generation rate
            "estimated_completion_hours": 0,
        }

        # Performance tracking
        self.processing_start_time = None
        self.last_stats_time = datetime.utcnow()
        self.last_items_count = 0
        self.last_captions_count = 0

    async def initialize(self):
        """Initialize orchestrator and create chunks."""
        await self.storage.initialize()

        if self.dataset_path:
            self.all_shards = await self._get_shard_urls()
            self.stats["total_shards"] = len(self.all_shards)

            # Create chunks for all shards
            await self._create_all_chunks()

            # Estimate completion time
            total_items = len(self.all_shards) * self.items_per_shard
            self.stats["total_estimated_items"] = total_items

            logger.info(
                f"Initialized: {len(self.all_shards)} shards, "
                f"~{total_items:,} items, "
                f"{len(self.shard_queue)} chunks"
            )

    async def _get_shard_urls(self) -> List[str]:
        """Get list of shard URLs without opening them."""
        if self.dataset_type == "huggingface":
            from huggingface_hub import HfFileSystem, hf_hub_url

            fs = HfFileSystem()
            files = [
                fs.resolve_path(p) for p in fs.glob(f"hf://datasets/{self.dataset_path}/**/*.tar")
            ]

            urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in files]
            return sorted(urls)

        elif self.dataset_type == "local":
            path = Path(self.dataset_path)
            shards = list(path.glob("*.tar"))
            return [str(s) for s in sorted(shards)]

        return []

    async def _create_all_chunks(self):
        """Create all chunks for all shards upfront."""
        for shard_url in self.all_shards:
            shard_name = Path(shard_url).stem

            if shard_name in self.completed_shards:
                continue

            # Create multiple chunks per shard based on estimated size
            num_chunks = max(1, self.items_per_shard // self.chunk_size)

            self.shard_progress[shard_name] = {
                "total_chunks": num_chunks,
                "completed_chunks": 0,
                "total_items": 0,
                "completed_items": 0,
            }

            for chunk_idx in range(num_chunks):
                chunk = ShardChunk(
                    chunk_id=f"{shard_name}_chunk_{chunk_idx}",
                    dataset=self.dataset_path,
                    shard_url=shard_url,
                    shard_name=shard_name,
                    start_index=chunk_idx * self.chunk_size,
                    chunk_size=self.chunk_size,
                    status="pending",
                )

                self.shard_queue.append(chunk)

            logger.debug(f"Created {num_chunks} chunks for shard {shard_name}")

        self.stats["pending_chunks"] = len(self.shard_queue)
        self.stats["total_chunks"] = len(self.shard_queue)

    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting orchestrator on {self.host}:{self.port}")

        await self.initialize()

        self.processing_start_time = datetime.utcnow()

        # Start background tasks
        asyncio.create_task(self._stats_reporter())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._reassignment_checker())
        asyncio.create_task(self._performance_monitor())

        # Start WebSocket server
        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info("Orchestrator ready for connections")
            await asyncio.Future()

    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection."""
        try:
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)

            user_auth = self.auth.authenticate(auth_data.get("token"))
            role = user_auth.role
            if not role:
                await websocket.send(json.dumps({"error": "Invalid token"}))
                return

            if role == "worker":
                await self._handle_worker(websocket, auth_data)
            elif role == "monitor":
                await self._handle_monitor(websocket)
            else:
                await websocket.send(json.dumps({"error": "Unauthorized role"}))
                await websocket.close()
                return

        except Exception as e:
            logger.error(f"Connection error: {e}")
            await websocket.close()

    async def _handle_worker(self, websocket: WebSocketServerProtocol, auth_data: Dict):
        """Handle worker connection."""
        worker_id = auth_data.get("name", str(uuid.uuid4()))
        self.workers[worker_id] = websocket
        self.worker_assignments[worker_id] = ShardAssignment(worker_id=worker_id)
        self.stats["connected_workers"] = len(self.workers)

        logger.info(f"Worker {worker_id} connected")

        try:
            # Send welcome with configuration
            await websocket.send(
                json.dumps(
                    {
                        "type": "welcome",
                        "worker_id": worker_id,
                        "config": {
                            "chunk_size": self.chunk_size,
                            "dataset_type": self.dataset_type,
                            "timeout_minutes": self.chunk_timeout_minutes,
                        },
                    }
                )
            )

            # Immediately assign chunks
            await self._assign_chunks_to_worker(worker_id)

            # Process messages
            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        finally:
            del self.workers[worker_id]
            await self._handle_worker_disconnect(worker_id)
            self.stats["connected_workers"] = len(self.workers)

    async def _assign_chunks_to_worker(self, worker_id: str):
        """Assign chunks to worker."""
        assignment = self.worker_assignments.get(worker_id)
        if not assignment:
            return

        # Count active chunks
        active_chunks = len([c for c in assignment.chunks if c.status == "processing"])

        # Assign up to readahead_chunks
        chunks_to_assign = min(self.readahead_chunks - active_chunks, len(self.shard_queue))

        if chunks_to_assign <= 0:
            return

        assigned = []
        for _ in range(chunks_to_assign):
            if not self.shard_queue:
                break

            chunk = self.shard_queue.popleft()
            chunk.assigned_to = worker_id
            chunk.assigned_at = datetime.utcnow()
            chunk.status = "processing"

            assignment.chunks.append(chunk)
            self.active_chunks[chunk.chunk_id] = chunk
            assigned.append(chunk)

        if assigned:
            # Send assignment
            await self.workers[worker_id].send(
                json.dumps(
                    {
                        "type": "shard_assignment",
                        "chunks": [
                            {
                                "chunk_id": c.chunk_id,
                                "shard_url": c.shard_url,
                                "shard_name": c.shard_name,
                                "start_index": c.start_index,
                                "chunk_size": c.chunk_size,
                            }
                            for c in assigned
                        ],
                    }
                )
            )

            logger.info(f"Assigned {len(assigned)} chunks to {worker_id}")
            self.stats["active_chunks"] = len(self.active_chunks)
            self.stats["pending_chunks"] = len(self.shard_queue)

    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")

        if msg_type == "request_chunks":
            await self._assign_chunks_to_worker(worker_id)

        elif msg_type == "submit_captions":  # Changed from submit_caption to handle lists
            # Process multiple captions submission
            captions = data.get("captions", [])
            if isinstance(captions, str):
                # unpack json?
                try:
                    captions = json.loads(captions)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in captions for worker {worker_id}")
                    return

            if not isinstance(captions, list):
                logger.error(
                    f"Expected list of captions, got {type(captions)} for worker {worker_id}"
                )
                return
            caption_count = data.get("caption_count", len(captions))

            if not captions:
                logger.warning(f"Received empty captions list for item {data.get('item_key')}")
                return

            # Create a single caption entry with list of captions
            caption_data = {
                "job_id": f"{data['shard']}_{data['item_key']}",
                "dataset": data["dataset"],
                "shard": data["shard"],
                "item_key": data["item_key"],
                "captions": captions,  # Store as list
                "caption_count": caption_count,
                "contributor_id": worker_id,
                "timestamp": datetime.utcnow(),
                "quality_scores": None,  # Could be populated if we have quality metrics
                "image_width": data.get("image_width"),
                "image_height": data.get("image_height"),
                "image_format": data.get("image_format"),
                "file_size": data.get("file_size"),
                "processing_time_ms": data.get("processing_time_ms"),
            }

            # logger.info(f"Received caption data: {caption_data}")
            await self.storage.save_captions(caption_data)

            # Update statistics
            self.stats["total_items_processed"] += 1
            self.stats["total_captions"] += caption_count  # Add all captions in the list
            self.stats["total_unique_images"] = self.stats["total_items_processed"]

            # Calculate average captions per image
            if self.stats["total_unique_images"] > 0:
                self.stats["avg_captions_per_image"] = (
                    self.stats["total_captions"] / self.stats["total_unique_images"]
                )

            # Update chunk progress
            chunk_id = data.get("chunk_id")
            if chunk_id in self.active_chunks:
                chunk = self.active_chunks[chunk_id]
                chunk.completed_items += 1

                # Update shard progress
                if chunk.shard_name in self.shard_progress:
                    self.shard_progress[chunk.shard_name]["completed_items"] += 1

                # Check if chunk is complete
                if chunk.completed_items >= chunk.chunk_size:
                    await self._complete_chunk(chunk_id)

            # Update processing rate
            self._update_processing_rate()

            # Log progress periodically
            if self.stats["total_items_processed"] % 100 == 0:
                logger.info(
                    f"Progress: {self.stats['total_items_processed']} images, "
                    f"{self.stats['total_captions']} captions "
                    f"(avg {self.stats['avg_captions_per_image']:.1f} captions/image)"
                )

        elif msg_type == "chunk_complete":
            await self._complete_chunk(data["chunk_id"])

        elif msg_type == "chunk_failed":
            await self._handle_chunk_failure(data["chunk_id"], data.get("error", "Unknown"))

        elif msg_type == "heartbeat":
            assignment = self.worker_assignments.get(worker_id)
            if assignment:
                assignment.items_processed = data.get("processed", 0)
                assignment.items_failed = data.get("failed", 0)

    async def _complete_chunk(self, chunk_id: str):
        """Mark chunk as complete."""
        if chunk_id not in self.active_chunks:
            return

        chunk = self.active_chunks[chunk_id]
        chunk.status = "completed"

        # Update shard progress
        if chunk.shard_name in self.shard_progress:
            progress = self.shard_progress[chunk.shard_name]
            progress["completed_chunks"] += 1

            # Check if shard is complete
            if progress["completed_chunks"] >= progress["total_chunks"]:
                self.completed_shards.add(chunk.shard_name)
                self.stats["completed_shards"] += 1
                logger.info(f"Shard {chunk.shard_name} fully completed!")

        logger.info(f"Chunk {chunk_id} completed ({chunk.completed_items} items)")

        del self.active_chunks[chunk_id]
        self.stats["active_chunks"] = len(self.active_chunks)

        # Assign new chunk
        if chunk.assigned_to in self.workers:
            await self._assign_chunks_to_worker(chunk.assigned_to)

    async def _handle_chunk_failure(self, chunk_id: str, error: str):
        """Handle chunk failure."""
        logger.error(f"Chunk {chunk_id} failed: {error}")

        if chunk_id in self.active_chunks:
            chunk = self.active_chunks[chunk_id]

            # Reset and requeue
            chunk.status = "pending"
            chunk.assigned_to = None
            chunk.assigned_at = None
            chunk.completed_items = 0

            self.shard_queue.append(chunk)
            del self.active_chunks[chunk_id]

            self.stats["active_chunks"] = len(self.active_chunks)
            self.stats["pending_chunks"] = len(self.shard_queue)

    async def _handle_worker_disconnect(self, worker_id: str):
        """Handle worker disconnection."""
        assignment = self.worker_assignments.get(worker_id)
        if not assignment:
            return

        # Requeue active chunks
        for chunk in assignment.chunks:
            if chunk.status == "processing" and chunk.chunk_id in self.active_chunks:
                logger.info(f"Requeuing chunk {chunk.chunk_id}")
                await self._handle_chunk_failure(chunk.chunk_id, "Worker disconnected")

        del self.worker_assignments[worker_id]

    async def _reassignment_checker(self):
        """Check for stuck chunks and reassign."""
        while True:
            await asyncio.sleep(60)

            now = datetime.utcnow()
            timeout = timedelta(minutes=self.chunk_timeout_minutes)

            for chunk_id, chunk in list(self.active_chunks.items()):
                if chunk.assigned_at:
                    age = now - chunk.assigned_at
                    if age > timeout and chunk.status == "processing":
                        logger.warning(
                            f"Chunk {chunk_id} timed out after {age.total_seconds()/60:.1f} minutes"
                        )
                        await self._handle_chunk_failure(chunk_id, "Timeout")

    def _update_processing_rate(self):
        """Update items and captions per minute calculation."""
        now = datetime.utcnow()
        time_delta = (now - self.last_stats_time).total_seconds()

        if time_delta > 10:  # Update every 10 seconds
            items_delta = self.stats["total_items_processed"] - self.last_items_count
            captions_delta = self.stats["total_captions"] - self.last_captions_count

            items_per_second = items_delta / time_delta if time_delta > 0 else 0
            captions_per_second = captions_delta / time_delta if time_delta > 0 else 0

            self.stats["items_per_minute"] = round(items_per_second * 60, 1)
            self.stats["captions_per_minute"] = round(captions_per_second * 60, 1)

            # Estimate completion time
            remaining_items = (
                self.stats.get("total_estimated_items", 0) - self.stats["total_items_processed"]
            )
            if items_per_second > 0:
                remaining_hours = remaining_items / (items_per_second * 3600)
                self.stats["estimated_completion_hours"] = round(remaining_hours, 1)

            self.last_stats_time = now
            self.last_items_count = self.stats["total_items_processed"]
            self.last_captions_count = self.stats["total_captions"]

    async def _performance_monitor(self):
        """Monitor performance and alert on issues."""
        while True:
            await asyncio.sleep(60)

            # Check processing rate
            min_rate = self.config.get("monitoring", {}).get("min_items_per_minute", 20)
            if self.stats["items_per_minute"] < min_rate and self.stats["connected_workers"] > 0:
                logger.warning(
                    f"Processing rate low: {self.stats['items_per_minute']:.1f} items/min "
                    f"(expected > {min_rate})"
                )

            # Log progress
            if self.stats["total_chunks"] > 0:
                progress = (
                    (
                        self.stats["total_chunks"]
                        - self.stats["pending_chunks"]
                        - self.stats["active_chunks"]
                    )
                    / self.stats["total_chunks"]
                    * 100
                )
                logger.info(
                    f"Progress: {progress:.1f}%, "
                    f"Images: {self.stats['items_per_minute']:.1f}/min, "
                    f"Captions: {self.stats['captions_per_minute']:.1f}/min, "
                    f"Avg captions/image: {self.stats['avg_captions_per_image']:.1f}, "
                    f"ETA: {self.stats['estimated_completion_hours']:.1f} hours"
                )

    async def _checkpoint_loop(self):
        """Periodic checkpointing."""
        while True:
            await asyncio.sleep(60)

            if self.storage.caption_buffer:
                await self.storage.checkpoint()
                stats = await self.storage.get_caption_stats()
                logger.info(
                    f"Checkpoint: {stats['total_rows']} images, "
                    f"{stats['total_captions']} captions "
                    f"(avg {stats['avg_captions_per_image']:.1f} per image)"
                )

    async def _stats_reporter(self):
        """Report statistics."""
        while True:
            await asyncio.sleep(30)

            if self.monitors:
                await self._broadcast_stats()

    async def _broadcast_stats(self):
        """Broadcast stats to monitors."""
        message = json.dumps({"type": "stats", "data": self.stats})

        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except:
                disconnected.add(monitor)

        self.monitors -= disconnected

    async def _handle_monitor(self, websocket: WebSocketServerProtocol):
        """Handle monitor connection."""
        self.monitors.add(websocket)
        logger.info("Monitor connected")

        try:
            await websocket.send(json.dumps({"type": "stats", "data": self.stats}))

            async for _ in websocket:
                pass

        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        finally:
            self.monitors.discard(websocket)

    async def shutdown(self):
        """Shutdown orchestrator."""
        logger.info("Shutting down orchestrator...")

        # Close all worker connections
        for worker in self.workers.values():
            await worker.close()

        # Close monitors
        for monitor in self.monitors:
            await monitor.close()

        # Final checkpoint
        await self.storage.checkpoint()
        stats = await self.storage.get_caption_stats()
        logger.info(
            f"Final checkpoint: {stats['total_rows']} images with "
            f"{stats['total_captions']} total captions "
            f"(avg {stats['avg_captions_per_image']:.1f} captions per image)"
        )

        logger.info("Orchestrator shutdown complete.")
        await asyncio.sleep(1)  # Allow time for cleanup
