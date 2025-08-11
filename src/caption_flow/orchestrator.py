"""Enhanced orchestrator with shard chunk assignment for vLLM workers.

This orchestrator:
1. Divides dataset shards into chunks for parallel processing
2. Assigns chunks to workers on request
3. Collects captions from workers centrally
4. Manages checkpoints and fault tolerance
"""

import asyncio
import json
import logging
import ssl
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, Any, List, Deque
from collections import deque, defaultdict
import threading
from queue import Queue, Empty

import websockets
from websockets.server import WebSocketServerProtocol

from .storage import StorageManager
from .models import Caption, Contributor
from .utils.auth import AuthManager
from .utils.dataset_loader import DatasetLoader, ShardTracker
from .utils.json_utils import safe_dict, safe_json_dumps, to_json_dict
from .utils.chunk_tracker import ChunkTracker

logger = logging.getLogger(__name__)


@dataclass
class ShardChunk:
    """Represents a chunk of a shard for processing."""

    chunk_id: str
    shard_url: str
    shard_name: str
    start_index: int
    chunk_size: int
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, assigned, completed, failed
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ChunkManager:
    """Manages shard chunk creation and assignment."""

    def __init__(self, chunk_size: int = 1000, tracker: Optional[ChunkTracker] = None):
        self.chunk_size = chunk_size
        self.chunks: Dict[str, ShardChunk] = {}
        self.pending_chunks: Deque[str] = deque()
        self.assigned_chunks: Dict[str, Set[str]] = defaultdict(set)  # worker_id -> chunk_ids
        self.lock = threading.Lock()
        self.tracker = tracker  # Reference to chunk tracker

    def create_chunks_from_shard(
        self, shard_url: str, shard_name: str, total_items: int
    ) -> List[ShardChunk]:
        """Create chunks from a shard."""
        chunks = []

        for start_idx in range(0, total_items, self.chunk_size):
            chunk_id = f"{shard_name}_chunk_{start_idx}"
            chunk = ShardChunk(
                chunk_id=chunk_id,
                shard_url=shard_url,
                shard_name=shard_name,
                start_index=start_idx,
                chunk_size=min(self.chunk_size, total_items - start_idx),
            )

            with self.lock:
                self.chunks[chunk_id] = chunk
                self.pending_chunks.append(chunk_id)

            chunks.append(chunk)

        return chunks

    def get_chunks_for_worker(
        self, worker_id: str, count: int = 1, tracker: Optional["ChunkTracker"] = None
    ) -> List[ShardChunk]:
        """Get available chunks for a worker."""
        assigned = []

        with self.lock:
            while len(assigned) < count and self.pending_chunks:
                chunk_id = self.pending_chunks.popleft()
                chunk = self.chunks[chunk_id]

                chunk.assigned_to = worker_id
                chunk.status = "assigned"
                chunk.assigned_at = datetime.utcnow()

                self.assigned_chunks[worker_id].add(chunk_id)
                assigned.append(chunk)
                if tracker:
                    tracker.mark_assigned(chunk_id, worker_id)

        return assigned

    def complete_chunk(self, chunk_id: str, worker_id: str) -> bool:
        """Mark a chunk as completed."""
        with self.lock:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                if chunk.assigned_to == worker_id and chunk.status == "assigned":
                    chunk.status = "completed"
                    chunk.completed_at = datetime.utcnow()
                    self.assigned_chunks[worker_id].discard(chunk_id)
                    return True
        return False

    def fail_chunk(self, chunk_id: str, worker_id: str) -> bool:
        """Mark a chunk as failed and requeue it."""
        with self.lock:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                if chunk.assigned_to == worker_id:
                    chunk.status = "pending"
                    chunk.assigned_to = None
                    chunk.assigned_at = None
                    self.assigned_chunks[worker_id].discard(chunk_id)
                    self.pending_chunks.append(chunk_id)
                    return True
        return False

    def release_worker_chunks(self, worker_id: str):
        """Release all chunks assigned to a worker."""
        with self.lock:
            chunk_ids = list(self.assigned_chunks[worker_id])
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    if chunk.status == "assigned":
                        chunk.status = "pending"
                        chunk.assigned_to = None
                        chunk.assigned_at = None
                        self.pending_chunks.append(chunk_id)

            del self.assigned_chunks[worker_id]

    def get_stats(self) -> Dict[str, int]:
        """Get chunk statistics."""
        with self.lock:
            stats = {
                "total": len(self.chunks),
                "pending": len(self.pending_chunks),
                "assigned": sum(len(chunks) for chunks in self.assigned_chunks.values()),
                "completed": sum(1 for c in self.chunks.values() if c.status == "completed"),
                "failed": sum(1 for c in self.chunks.values() if c.status == "failed"),
            }
        return stats


class Orchestrator:
    """Enhanced orchestrator for vLLM-based distributed captioning with chunk assignment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)

        # Dataset configuration
        self.dataset_config = config.get("dataset", {})
        self.dataset_path = self.dataset_config.get("path")
        self.dataset_type = self.dataset_config.get("type", "huggingface")

        # Chunk configuration
        self.chunk_size = config.get("chunk_size", 50)
        self.chunks_per_request = config.get("chunks_per_request", 4)

        # Initialize components
        storage_config = config.get("storage", {})
        self.storage = StorageManager(
            Path(storage_config.get("data_dir", "./caption_data")),
            caption_buffer_size=storage_config.get("caption_buffer_size", 1000),
            job_buffer_size=storage_config.get("job_buffer_size", 100),
            contributor_buffer_size=storage_config.get("contributor_buffer_size", 10),
        )
        self.auth = AuthManager(config.get("auth", {}))

        # Dataset components
        self.dataset_loader = None
        self.shard_tracker = None
        self.chunk_tracker = None
        if self.dataset_path:
            self.dataset_loader = DatasetLoader(self.dataset_path, self.dataset_type)
            checkpoint_dir = Path(config.get("storage", {}).get("checkpoint_dir", "./checkpoints"))
            self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")
            self.shard_tracker = ShardTracker(checkpoint_dir / "shards.json")
        self.chunk_manager = ChunkManager(self.chunk_size, self.chunk_tracker)

        # Track connections
        self.workers: Dict[str, WebSocketServerProtocol] = {}
        self.monitors: Set[WebSocketServerProtocol] = set()

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Statistics
        self.stats = {
            "total_chunks": 0,
            "completed_chunks": 0,
            "failed_chunks": 0,
            "total_captions": 0,
            "connected_workers": 0,
            "total_shards": 0,
            "completed_shards": 0,
            "current_shard": None,
            "buffer_size": 0,
            "total_written": 0,
            "last_checkpoint": None,
        }

        # Shard processing state
        self.all_shards = []
        self.current_shard_index = 0
        self.shard_lock = threading.Lock()

        # Background chunk creation
        self.chunk_creation_thread = None
        self.stop_chunk_creation = threading.Event()

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL if certificates are provided."""
        ssl_config = self.config.get("ssl", {})
        if not ssl_config.get("cert") or not ssl_config.get("key"):
            return None

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(ssl_config["cert"], ssl_config["key"])
        return context

    def _create_chunks_from_dataset(self):
        """Background thread to create chunks from dataset shards."""
        if not self.dataset_loader:
            logger.warning("No dataset configured, skipping chunk creation")
            return

        logger.info("Starting chunk creation thread")

        # Get all shards
        self.all_shards = self.dataset_loader.get_shard_list()
        remaining_shards = self.shard_tracker.get_remaining_shards(self.all_shards)

        self.stats["total_shards"] = len(self.all_shards)
        self.stats["completed_shards"] = len(self.all_shards) - len(remaining_shards)

        logger.info(f"Total shards: {len(self.all_shards)}, " f"Remaining: {len(remaining_shards)}")

        for shard_url in remaining_shards:
            if self.stop_chunk_creation.is_set():
                break

            shard_name = Path(shard_url).stem
            self.stats["current_shard"] = shard_name

            # Check if shard is already complete in chunk tracker
            if self.chunk_tracker and self.chunk_tracker.is_shard_complete(shard_name):
                logger.info(f"Shard {shard_name} already complete, skipping")
                self.shard_tracker.mark_complete(shard_name)
                self.stats["completed_shards"] += 1
                continue

            try:
                # Count items in shard
                item_count = 0
                for _ in self.dataset_loader.iterate_shard(shard_url):
                    item_count += 1

                logger.info(f"Shard {shard_name} has {item_count} items")

                # Create chunks, checking with tracker
                created_chunks = 0
                for start_idx in range(0, item_count, self.chunk_size):
                    chunk_id = f"{shard_name}_chunk_{start_idx}"
                    chunk_size = min(self.chunk_size, item_count - start_idx)

                    # Check if chunk is already completed
                    if self.chunk_tracker:
                        if not self.chunk_tracker.add_chunk(
                            chunk_id, shard_name, start_idx, chunk_size
                        ):
                            logger.debug(f"Chunk {chunk_id} already completed, skipping")
                            continue

                    # Create chunk in memory
                    chunk = ShardChunk(
                        chunk_id=chunk_id,
                        shard_url=shard_url,
                        shard_name=shard_name,
                        start_index=start_idx,
                        chunk_size=chunk_size,
                    )

                    with self.chunk_manager.lock:
                        self.chunk_manager.chunks[chunk_id] = chunk
                        self.chunk_manager.pending_chunks.append(chunk_id)

                    created_chunks += 1

                self.stats["total_chunks"] += created_chunks
                logger.info(f"Created {created_chunks} new chunks from shard {shard_name}")

            except Exception as e:
                logger.error(f"Error processing shard {shard_name}: {e}")

        logger.info("Chunk creation thread finished")

    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting vLLM orchestrator on {self.host}:{self.port}")

        # Load existing state
        await self.storage.initialize()
        if self.chunk_tracker:
            await self.chunk_tracker.sync_with_storage(self.storage)
        await self._restore_state()

        # Start chunk creation thread if dataset is configured
        if self.dataset_loader:
            self.chunk_creation_thread = threading.Thread(
                target=self._create_chunks_from_dataset, daemon=True
            )
            self.chunk_creation_thread.start()

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._stats_update_loop())

        # Start WebSocket server
        async with websockets.serve(
            self.handle_connection, self.host, self.port, ssl=self.ssl_context
        ):
            logger.info("vLLM Orchestrator ready for connections")
            await asyncio.Future()  # Run forever

    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection."""
        try:
            # Authenticate
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)

            auth_ticket = self.auth.authenticate(auth_data.get("token"))
            if not auth_ticket:
                await websocket.send(safe_json_dumps({"error": "Invalid token"}))
                return

            # Route by role
            if auth_ticket.role == "worker":
                await self._handle_worker(websocket, auth_data)
            elif auth_ticket.role == "monitor":
                await self._handle_monitor(websocket)
            else:
                await websocket.send(safe_json_dumps({"error": "Unknown role"}))

        except Exception as e:
            logger.error(f"Connection error: {e}")
            await websocket.close()

    async def _handle_worker(self, websocket: WebSocketServerProtocol, auth_data: Dict):
        """Handle worker connection lifecycle."""
        worker_id = auth_data.get("name", str(uuid.uuid4()))
        self.workers[worker_id] = websocket
        self.stats["connected_workers"] = len(self.workers)

        # Register contributor
        contributor = Contributor(
            contributor_id=worker_id, name=worker_id, total_captions=0, trust_level=1
        )
        await self.storage.save_contributor(contributor)

        logger.info(f"Worker {worker_id} connected")
        await self._broadcast_stats()

        try:
            # Send welcome message with dataset configuration
            welcome_message = {
                "type": "welcome",
                "worker_id": worker_id,
                "dataset_config": {
                    "dataset_path": self.dataset_path,
                    "dataset_type": self.dataset_type,
                    "path": self.dataset_path,  # For compatibility
                    "type": self.dataset_type,  # For compatibility
                },
            }
            await websocket.send(safe_json_dumps(welcome_message))

            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        finally:
            del self.workers[worker_id]
            self.stats["connected_workers"] = len(self.workers)
            # Release chunks in both managers
            self.chunk_manager.release_worker_chunks(worker_id)
            if self.chunk_tracker:
                self.chunk_tracker.release_worker_chunks(worker_id)

            await self._broadcast_stats()

    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")

        if msg_type == "request_chunks":
            count = data.get("count", self.chunks_per_request)
            chunks = self.chunk_manager.get_chunks_for_worker(worker_id, count, self.chunk_tracker)

            if chunks:
                # Only send the fields that worker expects
                chunk_data = []
                for chunk in chunks:
                    chunk_data.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "shard_url": chunk.shard_url,
                            "shard_name": chunk.shard_name,
                            "start_index": chunk.start_index,
                            "chunk_size": chunk.chunk_size,
                        }
                    )

                await self.workers[worker_id].send(
                    safe_json_dumps({"type": "shard_assignment", "chunks": chunk_data})
                )
                logger.info(f"Assigned {len(chunks)} chunks to worker {worker_id}")
            else:
                await self.workers[worker_id].send(safe_json_dumps({"type": "no_chunks"}))

        elif msg_type == "chunk_complete":
            chunk_id = data["chunk_id"]
            if self.chunk_manager.complete_chunk(chunk_id, worker_id):
                self.stats["completed_chunks"] += 1

                if self.chunk_tracker:
                    self.chunk_tracker.mark_completed(chunk_id)

                logger.info(f"Chunk {chunk_id} completed by worker {worker_id}")
                await self._check_shard_completion(chunk_id)
        elif msg_type == "chunk_failed":
            chunk_id = data["chunk_id"]
            error = data.get("error", "Unknown error")
            if self.chunk_manager.fail_chunk(chunk_id, worker_id):
                self.stats["failed_chunks"] += 1

                if self.chunk_tracker:
                    self.chunk_tracker.mark_failed(chunk_id)

                logger.warning(f"Chunk {chunk_id} failed on worker {worker_id}: {error}")

        elif msg_type == "submit_captions":
            await self._handle_captions_submission(worker_id, data)

        elif msg_type == "heartbeat":
            # Update worker stats
            logger.debug(f"Heartbeat from {worker_id}: {data}")

    async def _handle_captions_submission(self, worker_id: str, data: Dict):
        """Process multiple captions submission from worker."""
        chunk_id = data.get("chunk_id")
        item_key = data["item_key"]
        captions_list = data["captions"]

        logger.debug(
            f"Received {len(captions_list)} captions for item {item_key} from worker {worker_id}"
        )

        # Create a SINGLE caption record with ALL captions as a list
        caption = Caption(
            job_id=f"{chunk_id}_{item_key}",  # Single ID for the item
            dataset=data.get("dataset"),
            shard=data.get("shard"),
            item_key=item_key,
            captions=captions_list,  # Store ALL captions as a list
            contributor_id=worker_id,
            timestamp=datetime.utcnow(),
            quality_scores=None,  # Could be a list of scores matching captions
            # Image metadata
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            image_format=data.get("image_format"),
            file_size=data.get("file_size"),
            # Processing metadata
            caption_count=len(captions_list),
            processing_time_ms=data.get("processing_time_ms"),
            chunk_id=chunk_id,
        )

        # Add to central storage buffer as a single entry
        await self.storage.save_caption(caption)

        # Update statistics
        self.stats["total_captions"] += len(captions_list)
        self.stats["buffer_size"] = len(self.storage.caption_buffer)

        # Update contributor stats
        contributor = await self.storage.get_contributor(worker_id)
        if contributor:
            contributor.total_captions += len(captions_list)
            await self.storage.save_contributor(contributor)

        # Broadcast updated stats
        await self._broadcast_stats()

        # Log progress periodically
        if self.stats["total_captions"] % 100 == 0:
            logger.info(f"Collected {self.stats['total_captions']} captions centrally")

    async def _check_shard_completion(self, chunk_id: str):
        """Check if a shard is complete after chunk completion."""
        # Extract shard name from chunk_id
        shard_name = chunk_id.rsplit("_chunk_", 1)[0]

        # Check if all chunks for this shard are complete
        chunk_stats = self.chunk_manager.get_stats()
        shard_chunks = [
            cid
            for cid, chunk in self.chunk_manager.chunks.items()
            if chunk.shard_name == shard_name
        ]

        completed_chunks = [
            cid for cid in shard_chunks if self.chunk_manager.chunks[cid].status == "completed"
        ]

        if len(completed_chunks) == len(shard_chunks):
            logger.info(f"Shard {shard_name} complete!")
            self.shard_tracker.mark_complete(shard_name)
            self.stats["completed_shards"] += 1

    async def _handle_monitor(self, websocket: WebSocketServerProtocol):
        """Handle monitor connection."""
        self.monitors.add(websocket)
        logger.info("Monitor connected")

        try:
            # Send initial stats
            await websocket.send(safe_json_dumps({"type": "stats", "data": self.stats}))

            # Send chunk stats
            chunk_stats = self.chunk_manager.get_stats()
            await websocket.send(safe_json_dumps({"type": "chunk_stats", "data": chunk_stats}))

            # Send contributor leaderboard
            contributors = await self.storage.get_top_contributors(10)
            await websocket.send(
                safe_json_dumps(
                    {"type": "leaderboard", "data": [safe_dict(c) for c in contributors]}
                )
            )

            # Keep connection alive
            async for _ in websocket:
                pass

        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        finally:
            self.monitors.discard(websocket)

    async def _broadcast_stats(self):
        """Broadcast statistics to all monitors."""
        if not self.monitors:
            return

        # Include chunk stats
        chunk_stats = self.chunk_manager.get_stats()
        self.stats.update({f"chunks_{k}": v for k, v in chunk_stats.items()})

        message = safe_json_dumps({"type": "stats", "data": self.stats})

        # Send to all monitors
        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(monitor)

        # Clean up disconnected monitors
        self.monitors -= disconnected

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections."""
        while True:
            await asyncio.sleep(30)

            # Ping workers
            disconnected = []
            for worker_id, ws in self.workers.items():
                try:
                    await ws.ping()
                except:
                    disconnected.append(worker_id)

            # Clean up disconnected workers
            for worker_id in disconnected:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.chunk_manager.release_worker_chunks(worker_id)

    async def _checkpoint_loop(self):
        """Periodically checkpoint storage."""
        interval = self.config.get("storage", {}).get("checkpoint_interval", 1000)

        while True:
            await asyncio.sleep(60)

            # Force checkpoint at regular intervals
            if self.stats["total_captions"] > 0 and self.stats["total_captions"] % interval == 0:
                logger.info(f"Triggering checkpoint at {self.stats['total_captions']} captions")
                await self.storage.checkpoint()

                # Update stats
                self.stats["last_checkpoint"] = datetime.utcnow().isoformat()
                self.stats["total_written"] = self.storage.total_captions_written
                self.stats["buffer_size"] = len(self.storage.caption_buffer)

                await self._broadcast_stats()
                logger.info(
                    f"Checkpoint complete. Total written to disk: {self.stats['total_written']}"
                )

    async def _stats_update_loop(self):
        """Periodically update and broadcast stats."""
        while True:
            await asyncio.sleep(10)

            # Update chunk stats
            chunk_stats = self.chunk_manager.get_stats()
            self.stats["total_chunks"] = chunk_stats["total"]
            self.stats["completed_chunks"] = chunk_stats["completed"]
            self.stats["failed_chunks"] = chunk_stats["failed"]

            await self._broadcast_stats()

    async def _restore_state(self):
        """Restore state from storage on startup."""
        # Update statistics
        self.stats["total_captions"] = await self.storage.count_captions()

        logger.info(f"Restored state: {self.stats['total_captions']} captions")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down orchestrator...")

        # Stop chunk creation
        self.stop_chunk_creation.set()
        if self.chunk_creation_thread:
            self.chunk_creation_thread.join(timeout=5)

        # Close all connections
        for ws in list(self.workers.values()):
            await ws.close()
        for ws in list(self.monitors):
            await ws.close()

        # Final checkpoint
        logger.info(f"Final flush: {len(self.storage.caption_buffer)} captions in buffer")
        await self.storage.checkpoint()

        # Log final statistics
        logger.info(
            f"Shutdown complete. Total captions collected: {self.storage.total_captions_written}"
        )

        await self.storage.close()
