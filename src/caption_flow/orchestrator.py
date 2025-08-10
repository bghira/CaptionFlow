"""Enhanced orchestrator with shard-wise job assignment.

The orchestrator now:
1. Assigns shard chunks (not individual items) to workers
2. Never opens or processes image data
3. Receives metadata + captions from workers
4. Tracks progress at the shard chunk level
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, Any, List, Tuple
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
    start_index: int  # Starting index in the shard
    chunk_size: int  # Number of items to process
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_items: int = 0
    status: str = "pending"  # pending, processing, completed, failed


@dataclass
class ShardAssignment:
    """Track which worker is processing which shard chunks."""

    worker_id: str
    chunks: List[ShardChunk] = field(default_factory=list)
    current_chunk: Optional[ShardChunk] = None
    items_processed: int = 0
    items_failed: int = 0


@dataclass
class CaptionSubmission:
    """Caption with metadata submitted by worker."""

    dataset: str
    shard: str
    item_key: str
    caption: str
    image_width: int
    image_height: int
    image_format: str
    file_size: int
    processing_time_ms: float
    worker_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Orchestrator:
    """Orchestrator that assigns shard chunks to workers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)

        # Dataset configuration
        self.dataset_config = config.get("dataset", {})
        self.dataset_path = self.dataset_config.get("path")
        self.dataset_type = self.dataset_config.get("type", "huggingface")
        self.chunk_size = self.dataset_config.get("chunk_size", 2048)  # Items per chunk
        self.readahead_chunks = self.dataset_config.get("readahead_chunks", 2)

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
        self.shard_queue: deque = deque()  # Queue of unassigned chunks
        self.active_chunks: Dict[str, ShardChunk] = {}  # chunk_id -> ShardChunk
        self.completed_shards: Set[str] = set()

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
            "connected_workers": 0,
            "avg_processing_time_ms": 0,
            "buffer_size": 0,
            "total_written": 0,
        }

    async def initialize(self):
        """Initialize orchestrator and load shard list."""
        await self.storage.initialize()

        # Load shard list (but don't open any shards!)
        if self.dataset_path:
            self.all_shards = await self._get_shard_urls()
            self.stats["total_shards"] = len(self.all_shards)

            # Create initial chunks for queue
            await self._create_shard_chunks()

            logger.info(
                f"Initialized with {len(self.all_shards)} shards, "
                f"{len(self.shard_queue)} chunks ready"
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

        else:
            return []

    async def _create_shard_chunks(self):
        """Create chunks from shards for assignment."""
        for shard_url in self.all_shards:
            shard_name = Path(shard_url).stem

            # Skip if already completed
            if shard_name in self.completed_shards:
                continue

            # For now, create one chunk per shard
            # In production, you might estimate shard size and create multiple chunks
            chunk = ShardChunk(
                chunk_id=f"{shard_name}_chunk_0",
                dataset=self.dataset_path,
                shard_url=shard_url,
                shard_name=shard_name,
                start_index=0,
                chunk_size=self.chunk_size,
                status="pending",
            )

            self.shard_queue.append(chunk)
            self.stats["pending_chunks"] = len(self.shard_queue)

    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting improved orchestrator on {self.host}:{self.port}")

        await self.initialize()

        # Start background tasks
        asyncio.create_task(self._stats_reporter())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._reassignment_checker())

        # Start WebSocket server
        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info("Orchestrator ready")
            await asyncio.Future()

    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection."""
        try:
            # Authenticate
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)

            user_auth = self.auth.authenticate(auth_data.get("token"))
            if not user_auth.role:
                await websocket.send(json.dumps({"error": "Invalid token"}))
                return

            if user_auth.role == "worker":
                await self._handle_worker(websocket, auth_data)
            elif user_auth.role == "monitor":
                await self._handle_monitor(websocket)
            else:
                await websocket.send(json.dumps({"error": f"Unknown role. User: {user_auth}"}))

        except Exception as e:
            logger.error(f"Connection error: {e}")
            await websocket.close()

    async def _handle_worker(self, websocket: WebSocketServerProtocol, auth_data: Dict):
        """Handle worker connection with shard chunk assignment."""
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
                        },
                    }
                )
            )

            # Immediately assign initial chunks
            await self._assign_chunks_to_worker(worker_id)

            # Process messages
            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
        finally:
            del self.workers[worker_id]
            await self._handle_worker_disconnect(worker_id)
            self.stats["connected_workers"] = len(self.workers)

    async def _assign_chunks_to_worker(self, worker_id: str):
        """Assign chunk(s) to a worker for processing."""
        assignment = self.worker_assignments.get(worker_id)
        if not assignment:
            return

        # Check how many chunks the worker currently has
        active_chunks = len([c for c in assignment.chunks if c.status == "processing"])

        # Assign up to readahead_chunks
        chunks_to_assign = self.readahead_chunks - active_chunks

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
            # Send assignment to worker
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

            logger.info(f"Assigned {len(assigned)} chunks to worker {worker_id}")
            self.stats["active_chunks"] = len(self.active_chunks)
            self.stats["pending_chunks"] = len(self.shard_queue)

    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")

        if msg_type == "request_chunks":
            # Worker wants more chunks
            await self._assign_chunks_to_worker(worker_id)

        elif msg_type == "submit_caption":
            # Worker submitting a caption with metadata
            submission = CaptionSubmission(
                dataset=data["dataset"],
                shard=data["shard"],
                item_key=data["item_key"],
                caption=data["caption"],
                image_width=data["image_width"],
                image_height=data["image_height"],
                image_format=data["image_format"],
                file_size=data["file_size"],
                processing_time_ms=data["processing_time_ms"],
                worker_id=worker_id,
            )

            await self._handle_caption_submission(submission)

            # Update chunk progress
            chunk_id = data.get("chunk_id")
            if chunk_id in self.active_chunks:
                chunk = self.active_chunks[chunk_id]
                chunk.completed_items += 1

                # Check if chunk is complete
                if chunk.completed_items >= chunk.chunk_size:
                    await self._complete_chunk(chunk_id)

        elif msg_type == "chunk_complete":
            # Worker reports chunk completion
            chunk_id = data["chunk_id"]
            await self._complete_chunk(chunk_id)

        elif msg_type == "chunk_failed":
            # Worker reports chunk failure
            chunk_id = data["chunk_id"]
            error = data.get("error", "Unknown error")
            await self._handle_chunk_failure(chunk_id, error)

        elif msg_type == "heartbeat":
            # Update worker stats
            assignment = self.worker_assignments.get(worker_id)
            if assignment:
                assignment.items_processed = data.get("processed", 0)
                assignment.items_failed = data.get("failed", 0)

    async def _handle_caption_submission(self, submission: CaptionSubmission):
        """Process caption submission with metadata."""
        # Create caption record
        caption = Caption(
            job_id=f"{submission.shard}_{submission.item_key}",
            dataset=submission.dataset,
            shard=submission.shard,
            item_key=submission.item_key,
            caption=submission.caption,
            contributor_id=submission.worker_id,
            timestamp=submission.timestamp,
            quality_score=None,
        )

        # Store caption
        await self.storage.save_caption(caption)

        # Store metadata separately if needed
        # You could extend the Caption model or create a separate metadata store

        # Update statistics
        self.stats["total_items_processed"] += 1
        self.stats["total_captions"] += 1
        self.stats["buffer_size"] = len(self.storage.caption_buffer)

        # Update average processing time
        current_avg = self.stats["avg_processing_time_ms"]
        n = self.stats["total_items_processed"]
        self.stats["avg_processing_time_ms"] = (
            current_avg * (n - 1) + submission.processing_time_ms
        ) / n

        # Log progress
        if self.stats["total_captions"] % 100 == 0:
            logger.info(
                f"Collected {self.stats['total_captions']} captions, "
                f"avg time: {self.stats['avg_processing_time_ms']:.1f}ms"
            )

    async def _complete_chunk(self, chunk_id: str):
        """Mark a chunk as complete."""
        if chunk_id not in self.active_chunks:
            return

        chunk = self.active_chunks[chunk_id]
        chunk.status = "completed"

        logger.info(f"Chunk {chunk_id} completed ({chunk.completed_items} items)")

        # Check if entire shard is done
        shard_chunks = [c for c in self.active_chunks.values() if c.shard_name == chunk.shard_name]

        if all(c.status == "completed" for c in shard_chunks):
            self.completed_shards.add(chunk.shard_name)
            self.stats["completed_shards"] += 1
            logger.info(f"Shard {chunk.shard_name} fully completed!")

        # Remove from active chunks
        del self.active_chunks[chunk_id]
        self.stats["active_chunks"] = len(self.active_chunks)

        # Assign new chunk to worker
        if chunk.assigned_to in self.workers:
            await self._assign_chunks_to_worker(chunk.assigned_to)

    async def _handle_chunk_failure(self, chunk_id: str, error: str):
        """Handle chunk processing failure."""
        logger.error(f"Chunk {chunk_id} failed: {error}")

        if chunk_id in self.active_chunks:
            chunk = self.active_chunks[chunk_id]
            chunk.status = "failed"

            # Requeue the chunk
            chunk.assigned_to = None
            chunk.assigned_at = None
            chunk.status = "pending"
            chunk.completed_items = 0

            self.shard_queue.append(chunk)
            del self.active_chunks[chunk_id]

            self.stats["active_chunks"] = len(self.active_chunks)
            self.stats["pending_chunks"] = len(self.shard_queue)

    async def _handle_worker_disconnect(self, worker_id: str):
        """Handle worker disconnection - requeue chunks."""
        assignment = self.worker_assignments.get(worker_id)
        if not assignment:
            return

        # Requeue any active chunks
        for chunk in assignment.chunks:
            if chunk.status == "processing" and chunk.chunk_id in self.active_chunks:
                logger.info(f"Requeuing chunk {chunk.chunk_id} from disconnected worker")
                await self._handle_chunk_failure(chunk.chunk_id, "Worker disconnected")

        del self.worker_assignments[worker_id]

    async def _reassignment_checker(self):
        """Periodically check for stuck chunks and reassign."""
        while True:
            await asyncio.sleep(60)

            now = datetime.utcnow()
            timeout_minutes = 10

            for chunk_id, chunk in list(self.active_chunks.items()):
                if chunk.assigned_at:
                    age = (now - chunk.assigned_at).total_seconds() / 60
                    if age > timeout_minutes and chunk.status == "processing":
                        logger.warning(f"Chunk {chunk_id} timed out, requeuing")
                        await self._handle_chunk_failure(chunk_id, "Timeout")

    async def _checkpoint_loop(self):
        """Periodically checkpoint storage."""
        while True:
            await asyncio.sleep(60)

            if self.storage.caption_buffer:
                await self.storage.checkpoint()
                self.stats["total_written"] = self.storage.total_captions_written
                self.stats["buffer_size"] = 0
                logger.info(f"Checkpoint: {self.stats['total_written']} captions written")

    async def _stats_reporter(self):
        """Report statistics periodically."""
        while True:
            await asyncio.sleep(30)

            # Broadcast to monitors
            if self.monitors:
                await self._broadcast_stats()

            # Log summary
            logger.info(
                f"Stats: {self.stats['completed_shards']}/{self.stats['total_shards']} shards, "
                f"{self.stats['active_chunks']} active chunks, "
                f"{self.stats['total_captions']} captions collected"
            )

    async def _handle_monitor(self, websocket: WebSocketServerProtocol):
        """Handle monitor connection."""
        self.monitors.add(websocket)
        logger.info("Monitor connected")

        try:
            # Send initial stats
            await websocket.send(json.dumps({"type": "stats", "data": self.stats}))

            async for _ in websocket:
                pass

        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        finally:
            self.monitors.discard(websocket)

    async def _broadcast_stats(self):
        """Broadcast statistics to monitors."""
        message = json.dumps({"type": "stats", "data": self.stats})

        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except:
                disconnected.add(monitor)

        self.monitors -= disconnected

    async def shutdown(self):
        """Shutdown orchestrator and close storage."""
        logger.info("Shutting down orchestrator...")

        # Close all worker connections
        for worker in self.workers.values():
            await worker.close()

        # Close monitors
        for monitor in self.monitors:
            await monitor.close()

        # Shutdown storage
        await self.storage.close()

        logger.info("Orchestrator shutdown complete.")
