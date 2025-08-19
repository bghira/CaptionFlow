"""Enhanced orchestrator with shard chunk assignment for vLLM workers.

This orchestrator:
1. Divides dataset shards into chunks for parallel processing
2. Assigns chunks to workers on request
3. Collects captions from workers centrally
4. Manages checkpoints and fault tolerance
"""

import time
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

from .workers import data
import websockets
from websockets.server import WebSocketServerProtocol

from .storage import StorageManager
from .models import Caption, Contributor
from .utils.auth import AuthManager
from .utils import DatasetLoader, ShardTracker, ChunkTracker
from .utils.json_utils import safe_dict, safe_json_dumps, to_json_dict

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

    @classmethod
    def create(
        cls, shard_url: str, shard_name: str, start_index: int, chunk_size: int
    ) -> "ShardChunk":
        """Factory method to create a chunk with consistent ID."""
        # Always use consistent format: dataset_chunk_startindex
        if shard_url.startswith("hf_dataset:"):
            # Extract dataset path
            parts = shard_url.split(":")
            dataset_path = parts[1] if len(parts) > 1 else "unknown"
            chunk_id = f"{dataset_path.replace('/', '_')}_chunk_{start_index}"
        else:
            # WebDataset format
            chunk_id = f"{shard_name}_chunk_{start_index}"

        return cls(
            chunk_id=chunk_id,
            shard_url=shard_url,
            shard_name=shard_name,
            start_index=start_index,
            chunk_size=chunk_size,
        )

    def belongs_to_shard(self, shard_identifier: str) -> bool:
        """Check if this chunk belongs to a given shard."""
        return self.shard_name == shard_identifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization (for workers)."""
        return {
            "chunk_id": self.chunk_id,
            "shard_url": self.shard_url,
            "shard_name": self.shard_name,
            "start_index": self.start_index,
            "chunk_size": self.chunk_size,
        }


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
            chunk = ShardChunk.create(
                shard_url=shard_url,
                shard_name=shard_name,
                start_index=start_idx,
                chunk_size=min(self.chunk_size, total_items - start_idx),
            )

            with self.lock:
                self.chunks[chunk.chunk_id] = chunk
                self.pending_chunks.append(chunk.chunk_id)

            chunks.append(chunk)

        return chunks

    def get_chunks_for_worker(
        self, worker_id: str, count: int = 1, tracker: Optional["ChunkTracker"] = None
    ) -> List[Dict[str, Any]]:
        """Get available chunks with unprocessed items for a worker."""
        assigned = []

        with self.lock:
            # FIRST PRIORITY: Check if this worker already has assigned chunks
            # Workers should complete their current chunks before getting new ones
            if worker_id in self.assigned_chunks:
                existing_chunk_ids = list(self.assigned_chunks[worker_id])
                for chunk_id in existing_chunk_ids:
                    if len(assigned) >= count:
                        break

                    chunk = self.chunks.get(chunk_id)
                    if not chunk:
                        continue

                    # Check if chunk still has unprocessed items
                    if tracker:
                        chunk_info = tracker.get_chunk_with_unprocessed_items(chunk_id)
                        if chunk_info and chunk_info["unprocessed_ranges"]:
                            assigned.append(
                                {
                                    "chunk": chunk,
                                    "unprocessed_ranges": chunk_info["unprocessed_ranges"],
                                }
                            )
                    else:
                        # No tracker, assume chunk needs processing
                        assigned.append(
                            {
                                "chunk": chunk,
                                "unprocessed_ranges": [(0, chunk.chunk_size - 1)],
                            }
                        )

            # SECOND PRIORITY: Get new pending chunks
            # Only if worker doesn't have enough chunks already
            while len(assigned) < count and self.pending_chunks:
                chunk_id = self.pending_chunks.popleft()
                chunk = self.chunks.get(chunk_id)

                if not chunk:
                    continue

                # Verify chunk is truly pending (defensive check)
                if chunk.status != "pending" or chunk.assigned_to is not None:
                    logger.warning(
                        f"Chunk {chunk_id} in pending queue but status={chunk.status}, assigned_to={chunk.assigned_to}"
                    )
                    continue

                # Assign to this worker
                chunk.assigned_to = worker_id
                chunk.status = "assigned"
                chunk.assigned_at = datetime.utcnow()
                self.assigned_chunks[worker_id].add(chunk_id)

                # Get unprocessed ranges
                unprocessed_ranges = [(0, chunk.chunk_size - 1)]  # Default
                if tracker:
                    chunk_info = tracker.get_chunk_with_unprocessed_items(chunk_id)
                    if chunk_info:
                        unprocessed_ranges = chunk_info["unprocessed_ranges"]
                    tracker.mark_assigned(chunk_id, worker_id)

                assigned.append({"chunk": chunk, "unprocessed_ranges": unprocessed_ranges})

        # Log what we're assigning
        if assigned:
            chunk_summary = ", ".join(
                [
                    f"{info['chunk'].chunk_id}[{len(info['unprocessed_ranges'])} ranges]"
                    for info in assigned
                ]
            )
            logger.info(f"Assigning to worker {worker_id}: {chunk_summary}")

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
            chunk_ids = list(self.assigned_chunks.get(worker_id, []))
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    if chunk.status == "assigned":
                        chunk.status = "pending"
                        chunk.assigned_to = None
                        chunk.assigned_at = None
                        self.pending_chunks.append(chunk_id)

            if worker_id in self.assigned_chunks:
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
        self.dataset_split = self.dataset_config.get("split", "train")  # Add split configuration
        self.dataset_image_column = self.dataset_config.get(
            "image_column", "image"
        )  # Add image column config

        # Dataset components
        self.dataset_loader = None
        self.shard_tracker = None
        self.chunk_tracker = None

        if self.dataset_path:
            self.dataset_loader = DatasetLoader(
                self.dataset_path,
                self.dataset_type,
                self.dataset_split,
                self.dataset_image_column,
            )
            checkpoint_dir = Path(config.get("storage", {}).get("checkpoint_dir", "./checkpoints"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.shard_tracker = ShardTracker(checkpoint_dir / "shards.json")
            self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")

        # vLLM configuration to distribute to workers
        self.vllm_config = config.get(
            "vllm",
            {
                "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "gpu_memory_utilization": 0.92,
                "max_model_len": 16384,
                "tensor_parallel_size": 1,
                "dtype": "float16",
                "enforce_eager": True,
                "limit_mm_per_prompt": {"image": 1},
                "disable_mm_preprocessor_cache": True,
                "sampling": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 256,
                    "repetition_penalty": 1.05,
                    "stop": ["<|end|>", "<|endoftext|>", "<|im_end|>"],
                },
                "inference_prompts": [
                    "describe this image in detail",
                    "provide a comprehensive description of the visual content",
                    "what are the key elements in this image?",
                ],
            },
        )

        # Chunk configuration
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunks_per_request = config.get("chunks_per_request", 2)

        # Demand-driven chunk creation settings
        self.chunk_buffer_multiplier = config.get("chunk_buffer_multiplier", 3)
        self.min_chunk_buffer = config.get("min_chunk_buffer", 10)

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
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.shard_tracker = ShardTracker(checkpoint_dir / "shards.json")
            self.chunk_tracker = ChunkTracker(checkpoint_dir / "chunks.json")

        # Initialize chunk manager with reference to chunk tracker
        self.chunk_manager = ChunkManager(self.chunk_size, self.chunk_tracker)
        self.pending_processed_items = defaultdict(list)  # chunk_id -> list of indices
        self.item_batch_lock = threading.Lock()
        self.last_item_batch_flush = time.time()
        self.item_batch_interval = 5  # Flush every 5 seconds
        self.item_batch_size = 100  # Or every 100 items

        # Track connections
        self.workers: Dict[str, WebSocketServerProtocol] = {}
        self.monitors: Set[WebSocketServerProtocol] = set()

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Statistics
        self.is_generating_stats = False
        self.stats = {
            "total_chunks": 0,
            "completed_chunks": 0,
            "failed_chunks": 0,
            "connected_workers": 0,
            "total_shards": 0,
            "completed_shards": 0,
            "current_shard": None,
            "last_checkpoint": None,
        }

        # Rate tracking
        self.rate_tracker = {
            "start_time": time.time(),
            "last_update_time": time.time(),
            "last_caption_count": 0,
            "current_rate": 0.0,
            "average_rate": 0.0,
            "expected_rate": 0.0,
        }

        # Data sample queue for CaptionWorker
        self.data_sample_queue = asyncio.Queue(maxsize=1000)
        self.data_workers: Dict[str, WebSocketServerProtocol] = {}

        # Backpressure threshold
        self.backpressure_threshold = config.get("backpressure_threshold", 800)

        # Shard processing state
        self.all_shards = []
        self.current_shard_index = 0
        self.shard_lock = threading.Lock()

        # Background chunk creation
        self.chunk_creation_thread = None
        self.stop_chunk_creation = threading.Event()

        # State restoration flag
        self.state_restored = threading.Event()
        # If no dataset, state is already "restored"
        if not self.dataset_loader:
            self.state_restored.set()

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL if certificates are provided."""
        ssl_config = self.config.get("ssl", {})
        if not ssl_config.get("cert") or not ssl_config.get("key"):
            return None

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(ssl_config["cert"], ssl_config["key"])
        return context

    def _create_chunks_from_dataset(self):
        """Background thread to create chunks from dataset shards on demand."""
        if not self.dataset_loader:
            logger.warning("No dataset configured, skipping chunk creation")
            self.state_restored.set()  # No state to restore
            return

        logger.info("Starting chunk creation thread")

        # Mark state as not restored until we process checkpoints
        self.state_restored.clear()

        # Get dataset info to check format
        dataset_info = self.dataset_loader.get_dataset_info()
        dataset_format = dataset_info.get("dataset_format", "unknown")
        logger.info(f"Dataset format: {dataset_format}")

        # Get all shards
        self.all_shards = self.dataset_loader.get_shard_list()
        self.stats["total_shards"] = len(self.all_shards)

        # For HuggingFace datasets, we might need to dynamically create more shards
        if dataset_format == "huggingface_datasets":
            self._is_hf_dataset = True
            self._hf_chunk_size = 10000  # Items per virtual shard
            self._next_hf_shard_index = len(self.all_shards)  # For creating new virtual shards
        else:
            self._is_hf_dataset = False

        # Get shard status from ChunkTracker
        shards_summary = self.chunk_tracker.get_shards_summary() if self.chunk_tracker else {}
        completed_shards = {
            shard_name for shard_name, info in shards_summary.items() if info["is_complete"]
        }

        # Update ShardTracker for completed shards
        for shard_name in completed_shards:
            if not self.shard_tracker.is_complete(shard_name):
                logger.info(f"Marking shard {shard_name} as complete in ShardTracker")
                self.shard_tracker.mark_complete(shard_name)

        # Get shards that need processing
        remaining_shards = self.shard_tracker.get_remaining_shards(self.all_shards)

        # Also check which shards already have chunks (partial or complete)
        shards_with_chunks = set()
        for shard_name in shards_summary.keys():
            shards_with_chunks.add(shard_name)

        # Filter out shards that already have chunks created
        remaining_shards = [
            shard
            for shard in remaining_shards
            if (shard if shard.startswith("hf_dataset:") else Path(shard).stem)
            not in shards_with_chunks
        ]

        self.stats["completed_shards"] = len(completed_shards)

        logger.info(
            f"Total shards: {len(self.all_shards)}, "
            f"Completed: {self.stats['completed_shards']}, "
            f"Shards with chunks: {len(shards_with_chunks)}, "
            f"Remaining to process: {len(remaining_shards)}"
        )

        # First, re-queue any existing pending chunks
        initial_pending = 0
        requeued_chunks_by_shard = defaultdict(list)

        for shard_name, shard_info in shards_summary.items():
            with self.chunk_manager.lock:
                for chunk_state in shard_info["chunks"]:
                    if chunk_state.status in ["pending", "failed", "assigned"]:
                        # ChunkState already has shard_url stored
                        chunk = ShardChunk(
                            chunk_id=chunk_state.chunk_id,
                            shard_url=chunk_state.shard_url,
                            shard_name=chunk_state.shard_name,
                            start_index=chunk_state.start_index,
                            chunk_size=chunk_state.chunk_size,
                        )
                        self.chunk_manager.chunks[chunk_state.chunk_id] = chunk
                        self.chunk_manager.pending_chunks.append(chunk_state.chunk_id)
                        requeued_chunks_by_shard[shard_name].append(chunk_state.chunk_id)
                        initial_pending += 1

        logger.info(f"Re-queued {initial_pending} existing pending chunks")
        for shard_name, chunk_ids in requeued_chunks_by_shard.items():
            logger.info(f"  Shard {shard_name}: {len(chunk_ids)} chunks - {chunk_ids}")

        # Mark state as restored
        self.state_restored.set()
        logger.info("State restoration complete, accepting chunk requests")

        # Process shards on-demand
        shard_iter = iter(remaining_shards)
        current_shard_url = None
        current_shard_name = None
        current_shard_items = None
        current_shard_index = 0

        while not self.stop_chunk_creation.is_set():
            # Check how many chunks we need
            with self.chunk_manager.lock:
                pending_count = len(self.chunk_manager.pending_chunks)
                assigned_count = sum(
                    len(chunks) for chunks in self.chunk_manager.assigned_chunks.values()
                )
                total_active = pending_count + assigned_count

                # Target buffer: configurable multiplier × number of workers
                worker_count = max(1, self.stats.get("connected_workers", 0))
                target_buffer = max(
                    self.min_chunk_buffer, worker_count * self.chunk_buffer_multiplier
                )

                chunks_needed = max(0, target_buffer - total_active)

            if chunks_needed == 0:
                # We have enough chunks, wait a bit
                time.sleep(5)
                continue

            logger.debug(
                f"Need {chunks_needed} more chunks (pending: {pending_count}, "
                f"assigned: {assigned_count}, workers: {worker_count})"
            )

            # Create chunks as needed
            chunks_created = 0

            while chunks_created < chunks_needed and not self.stop_chunk_creation.is_set():
                # Need to load next shard?
                if current_shard_url is None or current_shard_index >= current_shard_items:
                    try:
                        current_shard_url = next(shard_iter)

                        # Extract shard name based on type
                        if current_shard_url.startswith("hf_dataset:"):
                            current_shard_name = current_shard_url  # Use full ID for virtual shards
                        else:
                            current_shard_name = Path(current_shard_url).stem

                        self.stats["current_shard"] = current_shard_name

                        # Skip if we already have chunks from this shard
                        if current_shard_name in shards_summary:
                            logger.debug(
                                f"Skipping shard {current_shard_name} - already has chunks"
                            )
                            current_shard_url = None
                            continue

                        # Count items in new shard
                        logger.info(f"Loading new shard {current_shard_name}")

                        # For virtual HF dataset shards, use the chunk size directly
                        if current_shard_url.startswith("hf_dataset:"):
                            current_shard_items = self.dataset_loader.count_shard_items(
                                current_shard_url
                            )
                            logger.info(
                                f"Virtual shard {current_shard_name} has {current_shard_items} items"
                            )
                        else:
                            # For WebDataset, actually count items
                            current_shard_items = sum(
                                1 for _ in self.dataset_loader.iterate_shard(current_shard_url)
                            )
                            logger.info(
                                f"Shard {current_shard_name} has {current_shard_items} items"
                            )

                        current_shard_index = 0

                    except StopIteration:
                        # No more shards in the iterator
                        if self._is_hf_dataset:
                            # Before creating new virtual shards, check if we have pending chunks
                            with self.chunk_manager.lock:
                                pending_count = len(self.chunk_manager.pending_chunks)

                            if pending_count > 0:
                                # Don't create new shards if we have pending chunks
                                logger.debug(
                                    f"Have {pending_count} pending chunks, not creating new virtual shards yet"
                                )
                                current_shard_url = None
                                time.sleep(2)
                                continue

                            # For HF datasets, we can create more virtual shards on demand
                            logger.info(
                                "Creating additional virtual shards for HuggingFace dataset"
                            )

                            # Create 10 more virtual shards
                            new_shards = []
                            for i in range(10):
                                shard_id = f"hf_dataset:{self.dataset_path}:chunk:{self._next_hf_shard_index * self._hf_chunk_size}"
                                new_shards.append(shard_id)
                                self._next_hf_shard_index += 1

                            # Add to all_shards and create new iterator
                            self.all_shards.extend(new_shards)
                            self.stats["total_shards"] = len(self.all_shards)

                            # Filter for unprocessed shards
                            remaining_new_shards = [
                                s
                                for s in new_shards
                                if s not in shards_summary and s not in completed_shards
                            ]

                            if remaining_new_shards:
                                shard_iter = iter(remaining_new_shards)
                                logger.info(f"Added {len(remaining_new_shards)} new virtual shards")
                                continue

                        # No more shards to process
                        logger.info("No more shards to process")
                        break

                    except Exception as e:
                        logger.error(f"Error loading shard {current_shard_name}: {e}")
                        current_shard_url = None
                        continue

                # Create a chunk from current shard
                if current_shard_url and current_shard_index < current_shard_items:
                    # Calculate the absolute dataset index for this chunk
                    if current_shard_url.startswith("hf_dataset:"):
                        # Parse the virtual shard URL to get the base start index
                        parts = current_shard_url.split(":")
                        if len(parts) >= 4 and parts[2] == "chunk":
                            shard_base_index = int(parts[3])
                        else:
                            shard_base_index = 0

                        # The absolute start index for this chunk in the dataset
                        absolute_start_index = shard_base_index + current_shard_index
                    else:
                        # For WebDataset, current_shard_index is already absolute
                        absolute_start_index = current_shard_index

                    # Create chunk with absolute index
                    chunk = ShardChunk.create(
                        shard_url=current_shard_url,
                        shard_name=current_shard_name,
                        start_index=absolute_start_index,
                        chunk_size=min(self.chunk_size, current_shard_items - current_shard_index),
                    )

                    # Add to ChunkTracker with all required fields
                    if self.chunk_tracker and self.chunk_tracker.add_chunk(
                        chunk.chunk_id,
                        chunk.shard_name,
                        chunk.shard_url,
                        chunk.start_index,
                        chunk.chunk_size,
                    ):
                        with self.chunk_manager.lock:
                            self.chunk_manager.chunks[chunk.chunk_id] = chunk
                            self.chunk_manager.pending_chunks.append(chunk.chunk_id)

                        chunks_created += 1
                        self.stats["total_chunks"] += 1

                    current_shard_index += self.chunk_size

            if chunks_created > 0:
                logger.info(f"Created {chunks_created} chunks on demand")

            # If we couldn't create any chunks and there are no more shards, check if it's HF dataset
            if chunks_created == 0 and current_shard_url is None:
                if self._is_hf_dataset:
                    # We can always create more virtual shards for HF datasets
                    logger.debug("Will create more virtual shards on next iteration")
                else:
                    logger.info("All shards processed, chunk creation complete")
                    break

            # Brief pause to avoid spinning
            time.sleep(1)

        # Final stats
        if self.chunk_tracker:
            final_stats = self.chunk_tracker.get_stats()
            logger.info(
                f"Chunk creation thread ending. Total: {final_stats['total']}, "
                f"Pending: {final_stats['pending']}, Completed: {final_stats['completed']}"
            )

        logger.info("Chunk creation thread finished")

    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting vLLM orchestrator on {self.host}:{self.port}")
        logger.info(
            f"vLLM config: model={self.vllm_config.get('model')}, batch_size={self.vllm_config.get('batch_size')}"
        )

        # Load existing state BEFORE accepting connections
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

            # Give chunk creation thread time to restore existing chunks
            await asyncio.sleep(2)

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
            if not auth_ticket.role:
                await websocket.send(safe_json_dumps({"error": "Invalid token"}))
                return

            if auth_ticket.role == "worker":
                await self._handle_worker(websocket, auth_ticket)
            elif auth_ticket.role == "data_worker":
                await self._handle_data_worker(websocket, auth_ticket)
            elif auth_ticket.role == "monitor":
                await self._handle_monitor(websocket)
            elif auth_ticket.role == "admin":
                await self._handle_admin(websocket, auth_ticket)
            else:
                await websocket.send(
                    safe_json_dumps({"error": f"Unknown role: {auth_ticket.role}"})
                )

        except Exception as e:
            logger.error(f"Connection error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            await websocket.close()

    async def _handle_admin(self, websocket: WebSocketServerProtocol, auth_ticket):
        """Handle admin connection for configuration updates."""
        admin_id = getattr(auth_ticket, "name", "admin")
        logger.info(f"Admin {admin_id} connected")

        try:
            # Send welcome
            await websocket.send(safe_json_dumps({"type": "welcome", "role": "admin"}))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "reload_config":
                        await self._handle_config_reload(websocket, data.get("config", {}))

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid admin message: {e}")
                    await websocket.send(
                        safe_json_dumps({"type": "error", "error": "Invalid message format"})
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Admin {admin_id} disconnected")

    async def _handle_config_reload(
        self, websocket: WebSocketServerProtocol, new_config: Dict[str, Any]
    ):
        """Handle configuration reload request."""
        logger.info("Processing configuration reload request")

        updated_sections = []
        warnings = []
        requires_worker_restart = False

        try:
            # Extract orchestrator section if present
            if "orchestrator" in new_config:
                # Config has orchestrator wrapper, extract it
                orchestrator_config = new_config["orchestrator"]
            else:
                # Config is already at orchestrator level
                orchestrator_config = new_config

            # Helper function for deep comparison
            def deep_equal(a, b):
                """Deep comparison of two values including nested dicts and lists."""
                if type(a) != type(b):
                    return False
                if isinstance(a, dict):
                    if set(a.keys()) != set(b.keys()):
                        return False
                    return all(deep_equal(a[k], b[k]) for k in a.keys())
                elif isinstance(a, (list, tuple)):
                    if len(a) != len(b):
                        return False
                    return all(deep_equal(x, y) for x, y in zip(a, b))
                else:
                    return a == b

            # Update vLLM configuration
            if "vllm" in orchestrator_config:
                old_vllm = self.vllm_config.copy()
                new_vllm = orchestrator_config["vllm"]

                # Check if vLLM config actually changed using deep comparison
                vllm_changed = not deep_equal(old_vllm, new_vllm)

                if vllm_changed:
                    # Update the vLLM config
                    self.vllm_config = new_vllm.copy()
                    updated_sections.append("vllm")

                    # Check if critical changes require worker restart
                    if (
                        old_vllm.get("model") != new_vllm.get("model")
                        or old_vllm.get("gpu_memory_utilization")
                        != new_vllm.get("gpu_memory_utilization")
                        or old_vllm.get("tensor_parallel_size")
                        != new_vllm.get("tensor_parallel_size")
                        or old_vllm.get("dtype") != new_vllm.get("dtype")
                        or old_vllm.get("max_model_len") != new_vllm.get("max_model_len")
                    ):
                        requires_worker_restart = True
                        warnings.append(
                            "Critical vLLM changes detected - workers will be disconnected to reload"
                        )
                        logger.info(
                            f"Model change: {old_vllm.get('model')} -> {new_vllm.get('model')}"
                        )

            # Update dataset configuration
            if "dataset" in orchestrator_config:
                old_dataset = self.dataset_config.copy()
                new_dataset = orchestrator_config["dataset"]

                dataset_changed = not deep_equal(old_dataset, new_dataset)

                if dataset_changed:
                    self.dataset_config = new_dataset.copy()
                    self.dataset_path = self.dataset_config.get("path")
                    self.dataset_type = self.dataset_config.get("type", "huggingface")
                    updated_sections.append("dataset")
                    warnings.append("Dataset changes will apply to new chunks only")

            # Update chunk settings
            if (
                "chunk_size" in orchestrator_config
                and self.chunk_size != orchestrator_config["chunk_size"]
            ):
                self.chunk_size = orchestrator_config["chunk_size"]
                self.chunk_manager.chunk_size = self.chunk_size
                updated_sections.append("chunk_size")

            if (
                "chunks_per_request" in orchestrator_config
                and self.chunks_per_request != orchestrator_config["chunks_per_request"]
            ):
                self.chunks_per_request = orchestrator_config["chunks_per_request"]
                updated_sections.append("chunks_per_request")

            # Update auth configuration
            if "auth" in orchestrator_config:
                try:
                    self.auth = AuthManager({"auth": orchestrator_config["auth"]})
                    updated_sections.append("auth")
                except Exception as e:
                    logger.error(f"Failed to update AuthManager: {e}")
                    warnings.append(f"Auth update failed: {e}")

            # Update buffer settings
            if (
                "chunk_buffer_multiplier" in orchestrator_config
                and self.chunk_buffer_multiplier != orchestrator_config["chunk_buffer_multiplier"]
            ):
                self.chunk_buffer_multiplier = orchestrator_config["chunk_buffer_multiplier"]
                updated_sections.append("chunk_buffer_multiplier")

            if (
                "min_chunk_buffer" in orchestrator_config
                and self.min_chunk_buffer != orchestrator_config["min_chunk_buffer"]
            ):
                self.min_chunk_buffer = orchestrator_config["min_chunk_buffer"]
                updated_sections.append("min_chunk_buffer")

            # Update storage settings
            if "storage" in orchestrator_config:
                storage_config = orchestrator_config["storage"]
                storage_changed = False

                if (
                    "caption_buffer_size" in storage_config
                    and self.storage.caption_buffer_size != storage_config["caption_buffer_size"]
                ):
                    self.storage.caption_buffer_size = storage_config["caption_buffer_size"]
                    storage_changed = True

                if "checkpoint_interval" in storage_config:
                    current_interval = self.config.get("storage", {}).get(
                        "checkpoint_interval", 1000
                    )
                    if current_interval != storage_config["checkpoint_interval"]:
                        self.config.setdefault("storage", {})["checkpoint_interval"] = (
                            storage_config["checkpoint_interval"]
                        )
                        storage_changed = True

                if storage_changed:
                    updated_sections.append("storage")

            # Check if any changes were made
            if not updated_sections:
                await websocket.send(
                    safe_json_dumps(
                        {
                            "type": "reload_complete",
                            "message": "No changes applied - configuration is identical",
                        }
                    )
                )
                logger.info("Configuration reload requested but no changes detected")
                return

            # Update the main config
            if "orchestrator" in new_config:
                self.config["orchestrator"] = orchestrator_config
            else:
                self.config.update(orchestrator_config)

            # Handle worker restart if needed
            if requires_worker_restart:
                logger.info("Disconnecting all workers for configuration reload...")

                # Send reload message to workers first
                reload_msg = safe_json_dumps(
                    {
                        "type": "reload_vllm",
                        "vllm_config": self.vllm_config,
                    }
                )

                # Create a list of worker items to avoid modifying dict during iteration
                worker_items = list(self.workers.items())
                disconnected = []

                for worker_id, ws in worker_items:
                    try:
                        await ws.send(reload_msg)
                        # Give worker time to process before disconnect
                        await asyncio.sleep(0.5)
                        await ws.close(code=1012, reason="Configuration reload")
                        disconnected.append(worker_id)
                    except:
                        disconnected.append(worker_id)  # Still mark as disconnected if error

                # Now safely clear workers dict
                for worker_id in disconnected:
                    if worker_id in self.workers:
                        del self.workers[worker_id]

                warnings.append(
                    f"Sent reload message to {len(disconnected)} workers - they will reconnect with new config"
                )
            else:
                # Just notify workers about config changes without disconnecting
                config_update_msg = safe_json_dumps(
                    {
                        "type": "config_update",
                        "vllm_config": self.vllm_config if "vllm" in updated_sections else None,
                        "dataset_config": (
                            self.dataset_config if "dataset" in updated_sections else None
                        ),
                    }
                )

                # Create a list of worker items to avoid modifying dict during iteration
                worker_items = list(self.workers.items())
                disconnected = []

                for worker_id, ws in worker_items:
                    try:
                        await ws.send(config_update_msg)
                        logger.info(f"Sent config update to worker {worker_id}")
                    except:
                        disconnected.append(worker_id)

                # Now safely remove disconnected workers
                for worker_id in disconnected:
                    if worker_id in self.workers:
                        del self.workers[worker_id]

            # Send success response
            await websocket.send(
                safe_json_dumps(
                    {"type": "reload_complete", "updated": updated_sections, "warnings": warnings}
                )
            )

            logger.info(f"Configuration reloaded. Updated sections: {', '.join(updated_sections)}")

            # Broadcast stats update to monitors
            await self._broadcast_stats()
            await self._send_activity(
                f"Configuration reloaded by admin: {', '.join(updated_sections)}"
            )

        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            await websocket.send(safe_json_dumps({"type": "reload_failed", "error": str(e)}))

    async def _handle_worker(self, websocket: WebSocketServerProtocol, auth_ticket):
        """Handle worker connection lifecycle."""
        # Generate unique worker ID even if using same token
        base_name = getattr(auth_ticket, "name", "worker")
        worker_id = f"{base_name}_{str(uuid.uuid4())[:8]}"  # Add unique suffix

        # Track the original token/user for accounting
        worker_user = base_name  # Keep track of which user/token this worker belongs to

        self.workers[worker_id] = websocket
        self.stats["connected_workers"] = len(self.workers)

        # Optionally track workers by user/token
        if not hasattr(self, "workers_by_user"):
            self.workers_by_user = defaultdict(set)
        self.workers_by_user[worker_user].add(worker_id)

        # Register contributor with the base name (for aggregating stats per user)
        contributor = await self.storage.get_contributor(worker_user)
        if not contributor:
            contributor = Contributor(
                contributor_id=worker_user,
                name=worker_user,
                total_captions=0,
                trust_level=1,
            )
            await self.storage.save_contributor(contributor)

        logger.info(f"Worker {worker_id} (user: {worker_user}) connected")
        await self._broadcast_stats()
        await self._send_activity(f"Worker {worker_id} (user: {worker_user}) connected")

        try:
            # Send welcome message with dataset configuration
            welcome_message = {
                "type": "welcome",
                "worker_id": worker_id,
                "user_id": worker_user,
                "dataset_config": {
                    "dataset_path": self.dataset_path,
                    "dataset_type": self.dataset_type,
                    "dataset_split": self.dataset_split,
                    "dataset_image_column": self.dataset_image_column,
                    "path": self.dataset_path,
                    "type": self.dataset_type,
                    "split": self.dataset_split,
                    "image_column": self.dataset_image_column,
                },
                "vllm_config": self.vllm_config,
            }
            await websocket.send(safe_json_dumps(welcome_message))

            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} (user: {worker_user}) disconnected")
        finally:
            if worker_id in self.workers:
                del self.workers[worker_id]

            # Clean up user tracking
            if hasattr(self, "workers_by_user") and worker_user in self.workers_by_user:
                self.workers_by_user[worker_user].discard(worker_id)
                if not self.workers_by_user[worker_user]:
                    del self.workers_by_user[worker_user]

            self.stats["connected_workers"] = len(self.workers)

            # Release chunks
            self.chunk_manager.release_worker_chunks(worker_id)
            if self.chunk_tracker:
                released_chunks = self.chunk_tracker.release_worker_chunks(worker_id)
                logger.info(
                    f"Released {len(released_chunks) if released_chunks is not None else 0} chunks from worker {worker_id}"
                )

            await self._broadcast_stats()
            await self._send_activity(f"Worker {worker_id} (user: {worker_user}) disconnected")

    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")

        if msg_type == "request_chunks":
            # Wait for state restoration to complete
            if not self.state_restored.is_set():
                logger.info(f"Worker {worker_id} requesting chunks, but state not yet restored")
                await self.workers[worker_id].send(
                    safe_json_dumps({"type": "no_chunks", "reason": "state_restoring"})
                )
                return

            count = data.get("count", self.chunks_per_request)
            chunk_infos = self.chunk_manager.get_chunks_for_worker(
                worker_id, count, self.chunk_tracker
            )

            if chunk_infos:
                # Send chunks with unprocessed ranges
                chunks_data = []
                for info in chunk_infos:
                    chunk_dict = info["chunk"].to_dict()
                    chunk_dict["unprocessed_ranges"] = info["unprocessed_ranges"]
                    chunks_data.append(chunk_dict)

                await self.workers[worker_id].send(
                    safe_json_dumps({"type": "shard_assignment", "chunks": chunks_data})
                )

                chunk_ids = [c["chunk_id"] for c in chunks_data]
                logger.info(
                    f"Assigned {len(chunks_data)} chunks to worker {worker_id}: {chunk_ids}"
                )
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
                await self._send_activity(f"Chunk {chunk_id} completed by {worker_id}")
        elif msg_type == "chunk_failed":
            chunk_id = data["chunk_id"]
            error = data.get("error", "Unknown error")
            if self.chunk_manager.fail_chunk(chunk_id, worker_id):
                self.stats["failed_chunks"] += 1

                if self.chunk_tracker:
                    self.chunk_tracker.mark_failed(chunk_id)

                logger.warning(f"Chunk {chunk_id} failed on worker {worker_id}: {error}")
                await self._send_activity(f"Chunk {chunk_id} failed on {worker_id}: {error}")

        elif msg_type == "submit_captions":
            await self._handle_captions_submission(worker_id, data)
        elif msg_type == "request_job":
            # CaptionWorker requesting a job from data samples
            try:
                job = await asyncio.wait_for(self.data_sample_queue.get(), timeout=5)
                await self.workers[worker_id].send(
                    json.dumps({"type": "job_assignment", "job": job})
                )
                logger.debug(f"Assigned job {job['job_id']} to worker {worker_id}")
            except asyncio.TimeoutError:
                await self.workers[worker_id].send(json.dumps({"type": "no_jobs"}))
        elif msg_type == "heartbeat":
            # Update worker stats
            logger.debug(f"Heartbeat from {worker_id}: {data}")

    async def _handle_captions_submission(self, worker_id: str, data: Dict):
        """Process caption submission from worker - now handles multi-stage outputs."""
        chunk_id = data.get("chunk_id")
        item_key = data["item_key"]

        item_index = data.get("item_index")  # Worker should send this
        if item_index is None:
            # Try to extract from item_key (format: dataset_XXXXXXXX)
            try:
                item_index = int(item_key.split("_")[-1])
            except:
                logger.warning(f"Could not extract item index from key: {item_key}")

        # Extract user from worker_id (format: "username_uuid")
        worker_user = worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id

        # Handle both old format (captions list) and new format (outputs dict)
        if "outputs" in data:
            # New multi-stage format
            outputs = data["outputs"]
            captions_list = outputs.get("captions", [])
            total_outputs = sum(len(v) for v in outputs.values())

            logger.debug(
                f"Received multi-stage outputs for item {item_key} from worker {worker_id}: "
                f"{total_outputs} outputs across {len(outputs)} fields"
            )
        else:
            # Old format - single captions list
            captions_list = data["captions"]
            outputs = {"captions": captions_list}
            total_outputs = len(captions_list)

            logger.debug(
                f"Received {len(captions_list)} captions for item {item_key} from worker {worker_id}"
            )

        # Create caption record with multi-stage outputs
        caption = Caption(
            job_id=f"{chunk_id}_{item_key}",
            dataset=data.get("dataset"),
            shard=data.get("shard"),
            item_key=item_key,
            captions=captions_list,
            outputs=outputs,
            contributor_id=worker_user,
            timestamp=datetime.utcnow(),
            quality_scores=None,
            # Image metadata
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            image_format=data.get("image_format"),
            file_size=data.get("file_size"),
            # Processing metadata
            caption_count=total_outputs,
            processing_time_ms=data.get("processing_time_ms"),
            chunk_id=chunk_id,
            metadata=data.get("metadata", {}),
        )

        # Add to central storage buffer
        await self.storage.save_caption(caption)

        # Handle item tracking with fixed deadlock
        should_flush = False
        if chunk_id and item_index is not None and self.chunk_tracker:
            with self.item_batch_lock:
                self.pending_processed_items[chunk_id].append(item_index)

                # Check if we should flush
                total_pending = sum(
                    len(indices) for indices in self.pending_processed_items.values()
                )
                time_since_flush = time.time() - self.last_item_batch_flush

                if (
                    total_pending >= self.item_batch_size
                    or time_since_flush >= self.item_batch_interval
                ):
                    should_flush = True

            if should_flush:
                await self._flush_processed_items()

        # Update contributor stats (use user, not worker)
        contributor = await self.storage.get_contributor(worker_user)
        if contributor:
            contributor.total_captions += total_outputs
            await self.storage.save_contributor(contributor)

        # Broadcast updated stats
        await self._broadcast_stats()

        # Log progress periodically
        total_outputs = self.stats.get("total_outputs", 0)
        if total_outputs > 0 and total_outputs % 100 == 0:
            if (
                not hasattr(self, "_last_logged_outputs")
                or self._last_logged_outputs != total_outputs
            ):
                logger.info(f"Collected {total_outputs} outputs centrally")
                self._last_logged_outputs = total_outputs

    async def _check_shard_completion(self, chunk_id: str):
        """Check if a shard is complete after chunk completion."""
        # Get the chunk
        chunk = self.chunk_manager.chunks.get(chunk_id)
        if not chunk:
            return

        shard_name = chunk.shard_name

        # Find all chunks for this shard
        shard_chunks = [
            cid for cid, c in self.chunk_manager.chunks.items() if c.belongs_to_shard(shard_name)
        ]

        # Check if all are completed
        completed_chunks = [
            cid for cid in shard_chunks if self.chunk_manager.chunks[cid].status == "completed"
        ]

        if len(completed_chunks) == len(shard_chunks) and len(shard_chunks) > 0:
            logger.info(f"Shard {shard_name} complete!")
            # Don't mark virtual shards as complete in ShardTracker
            if not shard_name.startswith("hf_dataset:"):
                self.shard_tracker.mark_complete(shard_name)
            self.stats["completed_shards"] += 1
            await self._send_activity(f"Shard {shard_name} completed!")

    async def _handle_data_worker(self, websocket: WebSocketServerProtocol, auth_ticket):
        """Handle data worker connection."""
        worker_id = getattr(auth_ticket, "name", str(uuid.uuid4()))
        self.data_workers[worker_id] = websocket

        logger.info(f"Data worker {worker_id} connected")

        try:
            # Send welcome with storage config
            storage_config = self.config.get(
                "data_worker_storage",
                {
                    "forward_to_orchestrator": True,
                    "local": {"enabled": False},
                    "s3": {"enabled": False},
                },
            )

            await websocket.send(
                json.dumps(
                    {"type": "welcome", "worker_id": worker_id, "storage_config": storage_config}
                )
            )

            # Track if we've sent backpressure
            backpressure_sent = False

            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "submit_samples":
                    # Check queue size for backpressure
                    if self.data_sample_queue.qsize() > self.backpressure_threshold:
                        if not backpressure_sent:
                            await websocket.send(json.dumps({"type": "backpressure"}))
                            backpressure_sent = True
                            logger.warning(f"Backpressure applied to data worker {worker_id}")
                    else:
                        if backpressure_sent:
                            await websocket.send(json.dumps({"type": "resume"}))
                            backpressure_sent = False

                    # Receive image data for each sample
                    samples = data["samples"]
                    for sample in samples:
                        # Receive binary image data
                        image_data = await websocket.recv()

                        # Create job and add to queue
                        job = {
                            "job_id": f"data_{worker_id}_{sample['sample_id']}",
                            "sample_id": sample["sample_id"],
                            "image_data": image_data,
                            "metadata": sample.get("metadata", {}),
                            "source": "data_worker",
                            "worker_id": worker_id,
                        }

                        await self.data_sample_queue.put(job)

                elif msg_type == "heartbeat":
                    logger.debug(f"Data worker {worker_id} heartbeat: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Data worker {worker_id} disconnected")
        finally:
            del self.data_workers[worker_id]

    async def _send_leaderboard_to_monitor(self, websocket: WebSocketServerProtocol):
        """Send leaderboard data to a specific monitor."""
        total_start = time.time()
        try:
            if websocket not in self.monitors:
                return

            # Get contributors asynchronously
            contributors_start = time.time()
            contributors = await self.storage.get_top_contributors(10)
            logger.debug(
                f"Contributors retrieved in {(time.time() - contributors_start)*1000:.1f}ms"
            )

            # Get worker counts in thread pool
            worker_counts_start = time.time()
            loop = asyncio.get_event_loop()
            worker_counts = await loop.run_in_executor(
                None,
                lambda: (
                    self.get_workers_by_user_stats() if hasattr(self, "workers_by_user") else {}
                ),
            )
            logger.debug(
                f"Worker counts retrieved in {(time.time() - worker_counts_start)*1000:.1f}ms"
            )

            # Build enhanced contributors list
            build_start = time.time()
            enhanced_contributors = []
            for contributor in contributors:
                contrib_dict = {
                    "contributor_id": contributor.contributor_id,
                    "name": contributor.name,
                    "total_captions": contributor.total_captions,
                    "trust_level": contributor.trust_level,
                    "active_workers": len(
                        worker_counts.get(contributor.contributor_id, {}).get("worker_ids", [])
                    ),
                }
                enhanced_contributors.append(contrib_dict)
            logger.debug(f"Enhanced contributors built in {(time.time() - build_start)*1000:.1f}ms")

            # Cache for future monitors
            self._cached_leaderboard = enhanced_contributors

            # Send if still connected
            if websocket in self.monitors:
                send_start = time.time()
                await websocket.send(
                    safe_json_dumps({"type": "leaderboard", "data": enhanced_contributors})
                )
                logger.debug(
                    f"Leaderboard sent to monitor in {(time.time() - send_start)*1000:.1f}ms"
                )

            logger.debug(
                f"Leaderboard send to monitor completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Monitor disconnected during leaderboard send")
        except Exception as e:
            logger.error(f"Error sending leaderboard to monitor: {e}")

    async def _send_initial_monitor_data(self, websocket: WebSocketServerProtocol):
        """Send initial data to monitor in a separate task to avoid blocking."""
        total_start = time.time()
        try:
            # Check if websocket is still in monitors set
            if websocket not in self.monitors:
                logger.debug("Monitor disconnected before initial data send")
                return

            # Send current stats (already in memory)
            stats_start = time.time()
            await websocket.send(safe_json_dumps({"type": "stats", "data": self.stats}))
            logger.debug(f"Monitor stats sent in {(time.time() - stats_start)*1000:.1f}ms")

            # Get chunk stats asynchronously
            chunk_stats_start = time.time()
            loop = asyncio.get_event_loop()
            chunk_stats = await loop.run_in_executor(None, self.chunk_manager.get_stats)
            logger.debug(f"Chunk stats retrieved in {(time.time() - chunk_stats_start)*1000:.1f}ms")

            if websocket not in self.monitors:
                return

            chunk_send_start = time.time()
            await websocket.send(safe_json_dumps({"type": "chunk_stats", "data": chunk_stats}))
            logger.debug(f"Chunk stats sent in {(time.time() - chunk_send_start)*1000:.1f}ms")

            # For leaderboard, check if we have a cached version first
            if hasattr(self, "_cached_leaderboard") and self._cached_leaderboard:
                # Use cached leaderboard if available
                cache_send_start = time.time()
                await websocket.send(
                    safe_json_dumps({"type": "leaderboard", "data": self._cached_leaderboard})
                )
                logger.debug(
                    f"Cached leaderboard sent in {(time.time() - cache_send_start)*1000:.1f}ms"
                )
            else:
                # Schedule leaderboard update separately
                leaderboard_task_start = time.time()
                asyncio.create_task(self._send_leaderboard_to_monitor(websocket))
                logger.debug(
                    f"Leaderboard task created in {(time.time() - leaderboard_task_start)*1000:.1f}ms"
                )

            logger.debug(
                f"Monitor initial data send completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Monitor disconnected during initial data send")
        except Exception as e:
            logger.error(f"Error sending initial monitor data: {e}")

    async def _handle_monitor(self, websocket: WebSocketServerProtocol):
        """Handle monitor connection - truly non-blocking version."""
        monitor_start = time.time()
        self.monitors.add(websocket)
        logger.info(f"Monitor connected (total monitors: {len(self.monitors)})")

        try:
            # Send welcome message immediately
            welcome_start = time.time()
            await websocket.send(safe_json_dumps({"type": "welcome", "role": "monitor"}))
            logger.debug(f"Monitor welcome sent in {(time.time() - welcome_start)*1000:.1f}ms")

            # Schedule initial data send as a separate task to avoid blocking
            task_create_start = time.time()
            asyncio.create_task(self._send_initial_monitor_data(websocket))
            logger.debug(
                f"Monitor initial data task created in {(time.time() - task_create_start)*1000:.1f}ms"
            )

            # Just keep the connection alive - no blocking work here
            try:
                async for message in websocket:
                    # Handle any incoming messages from monitor if needed
                    # For now, just ignore them
                    pass
            except websockets.exceptions.ConnectionClosed:
                pass  # Normal disconnection

        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        except Exception as e:
            logger.error(f"Error in monitor handler: {e}")
        finally:
            self.monitors.discard(websocket)
            logger.debug(f"Monitor handler completed in {(time.time() - monitor_start)*1000:.1f}ms")

    async def _broadcast_stats(self):
        """Broadcast statistics to all monitors - truly non-blocking version."""
        if not self.monitors:
            return
        if self.is_generating_stats:
            return  # Already generating stats, skip this call
        self.is_generating_stats = True
        total_start = time.time()

        # Prepare all the data first
        data_prep_start = time.time()
        loop = asyncio.get_event_loop()

        # Get storage stats (already async)
        storage_stats_start = time.time()
        storage_stats = await self.storage.get_storage_stats()
        logger.debug(f"Storage stats retrieved in {(time.time() - storage_stats_start)*1000:.1f}ms")

        caption_stats_start = time.time()
        caption_stats = await self.storage.get_caption_stats()
        logger.debug(f"Caption stats retrieved in {(time.time() - caption_stats_start)*1000:.1f}ms")

        # Get chunk stats in thread pool
        chunk_stats_start = time.time()
        chunk_stats = await loop.run_in_executor(None, self.chunk_manager.get_stats)
        logger.debug(f"Chunk stats retrieved in {(time.time() - chunk_stats_start)*1000:.1f}ms")

        # Build stats dict
        build_stats_start = time.time()
        stats_update = self.stats.copy()
        stats_update.update({f"chunks_{k}": v for k, v in chunk_stats.items()})
        stats_update.update(storage_stats)
        stats_update["field_breakdown"] = caption_stats.get("field_stats", {})
        stats_update["output_fields_list"] = caption_stats.get("output_fields", [])

        # Add rate information
        stats_update.update(
            {
                "current_rate": self.rate_tracker["current_rate"],
                "average_rate": self.rate_tracker["average_rate"],
                "expected_rate": self.rate_tracker["expected_rate"],
            }
        )

        # Add vLLM info
        stats_update["vllm_model"] = self.vllm_config.get("model", "unknown")
        stats_update["vllm_batch_size"] = self.vllm_config.get("batch_size", 0)

        # Add stage information
        stages = self.vllm_config.get("stages", [])
        if stages:
            stats_update["stage_count"] = len(stages)
            stats_update["stage_names"] = [s.get("name", "unnamed") for s in stages]
        else:
            stats_update["stage_count"] = 1
            stats_update["stage_names"] = ["default"]

        # Get field stats
        field_stats_start = time.time()
        field_stats = await self.storage.get_output_field_stats()
        stats_update["output_fields"] = field_stats
        logger.debug(f"Field stats retrieved in {(time.time() - field_stats_start)*1000:.1f}ms")

        # Update our internal stats
        self.stats = stats_update
        logger.debug(f"Stats prepared in {(time.time() - build_stats_start)*1000:.1f}ms")

        logger.debug(f"Total data preparation took {(time.time() - data_prep_start)*1000:.1f}ms")

        # Create message once
        message_create_start = time.time()
        stats_message = safe_json_dumps({"type": "stats", "data": self.stats})
        logger.debug(f"Stats message created in {(time.time() - message_create_start)*1000:.1f}ms")

        # Send to all monitors asynchronously in parallel
        send_start = time.time()

        async def send_to_monitor(monitor):
            try:
                await monitor.send(stats_message)
            except websockets.exceptions.ConnectionClosed:
                return monitor  # Return for removal
            except Exception as e:
                logger.debug(f"Error sending stats to monitor: {e}")
                return monitor  # Return for removal
            return None

        # Send to all monitors in parallel
        monitors_copy = self.monitors.copy()
        results = await asyncio.gather(
            *[send_to_monitor(m) for m in monitors_copy], return_exceptions=True
        )

        # Remove disconnected monitors
        disconnected = {
            m
            for m, r in zip(monitors_copy, results)
            if r is not None and not isinstance(r, Exception)
        }
        self.monitors -= disconnected

        logger.debug(
            f"Stats sent to {len(monitors_copy)} monitors in {(time.time() - send_start)*1000:.1f}ms"
        )

        # Send leaderboard update in a separate task to avoid blocking
        leaderboard_task_start = time.time()
        asyncio.create_task(self._broadcast_leaderboard())
        self.is_generating_stats = False
        logger.debug(
            f"Leaderboard broadcast task created in {(time.time() - leaderboard_task_start)*1000:.1f}ms"
        )
        logger.debug(f"Stats broadcast completed in {(time.time() - total_start)*1000:.1f}ms")

    async def _broadcast_leaderboard(self):
        """Send leaderboard updates to monitors - separate from stats to avoid blocking."""
        if not self.monitors:
            return

        total_start = time.time()
        try:
            # Get contributors
            contributors_start = time.time()
            contributors = await self.storage.get_top_contributors(10)
            logger.debug(
                f"Contributors retrieved for broadcast in {(time.time() - contributors_start)*1000:.1f}ms"
            )

            # Get worker counts
            worker_counts_start = time.time()
            loop = asyncio.get_event_loop()
            worker_counts = await loop.run_in_executor(
                None,
                lambda: (
                    self.get_workers_by_user_stats() if hasattr(self, "workers_by_user") else {}
                ),
            )
            logger.debug(
                f"Worker counts retrieved for broadcast in {(time.time() - worker_counts_start)*1000:.1f}ms"
            )

            # Build enhanced contributors list
            build_start = time.time()
            enhanced_contributors = []
            for contributor in contributors:
                contrib_dict = {
                    "contributor_id": contributor.contributor_id,
                    "name": contributor.name,
                    "total_captions": contributor.total_captions,
                    "trust_level": contributor.trust_level,
                    "active_workers": len(
                        worker_counts.get(contributor.contributor_id, {}).get("worker_ids", [])
                    ),
                }
                enhanced_contributors.append(contrib_dict)
            logger.debug(
                f"Enhanced contributors built for broadcast in {(time.time() - build_start)*1000:.1f}ms"
            )

            # Cache it
            self._cached_leaderboard = enhanced_contributors

            # Create message once
            message_create_start = time.time()
            leaderboard_message = safe_json_dumps(
                {"type": "leaderboard", "data": enhanced_contributors}
            )
            logger.debug(
                f"Leaderboard message created in {(time.time() - message_create_start)*1000:.1f}ms"
            )

            # Send to all monitors in parallel
            send_start = time.time()

            async def send_leaderboard(monitor):
                try:
                    await monitor.send(leaderboard_message)
                except:
                    return monitor  # Mark for removal
                return None

            monitors_copy = self.monitors.copy()
            results = await asyncio.gather(
                *[send_leaderboard(m) for m in monitors_copy], return_exceptions=True
            )

            # Remove disconnected
            disconnected = {
                m
                for m, r in zip(monitors_copy, results)
                if r is not None and not isinstance(r, Exception)
            }
            self.monitors -= disconnected

            logger.debug(
                f"Leaderboard sent to {len(monitors_copy)} monitors in {(time.time() - send_start)*1000:.1f}ms"
            )
            logger.debug(
                f"Leaderboard broadcast completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error broadcasting leaderboard: {e}")

    def _get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics - synchronous helper for thread pool."""
        with self.chunk_manager.lock:
            return {
                "pending_chunks": len(self.chunk_manager.pending_chunks),
                "assigned_chunks": sum(
                    len(chunks) for chunks in self.chunk_manager.assigned_chunks.values()
                ),
            }

    async def _flush_processed_items(self):
        """Flush batched processed items to chunk tracker."""
        with self.item_batch_lock:
            if not self.pending_processed_items:
                return

            for chunk_id, indices in self.pending_processed_items.items():
                if not indices:
                    continue

                # Indices here are ABSOLUTE dataset indices
                # Sort indices
                indices.sort()

                # Group consecutive indices into ranges
                ranges = []
                start = indices[0]
                end = indices[0]

                for i in range(1, len(indices)):
                    if indices[i] == end + 1:
                        # Consecutive, extend range
                        end = indices[i]
                    else:
                        # Gap found, save current range and start new one
                        ranges.append((start, end))
                        start = indices[i]
                        end = indices[i]

                # Don't forget the last range
                ranges.append((start, end))

                # Mark ranges as processed (mark_items_processed expects absolute indices)
                for start_idx, end_idx in ranges:
                    self.chunk_tracker.mark_items_processed(chunk_id, start_idx, end_idx)

            # Clear pending items
            self.pending_processed_items.clear()
            self.last_item_batch_flush = time.time()

    def get_workers_by_user_stats(self) -> Dict[str, Any]:
        """Get statistics about workers grouped by user/token - thread-safe version."""
        if not hasattr(self, "workers_by_user"):
            return {}

        # Create a copy to avoid issues with concurrent modification
        stats = {}
        workers_snapshot = dict(self.workers_by_user)
        for user, worker_ids in workers_snapshot.items():
            stats[user] = {"worker_count": len(worker_ids), "worker_ids": list(worker_ids)}
        return stats

    async def _send_activity(self, activity: str):
        """Send activity update to monitors."""
        if not self.monitors:
            return

        message = safe_json_dumps(
            {"type": "activity", "data": f"[{datetime.now().strftime('%H:%M:%S')}] {activity}"}
        )

        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(monitor)

        self.monitors -= disconnected

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections."""
        while True:
            try:
                await asyncio.sleep(30)

                # Create a copy of worker items to avoid modification during iteration
                worker_items = list(self.workers.items())
                disconnected = []

                for worker_id, ws in worker_items:
                    try:
                        # Check if worker still exists before pinging
                        if worker_id not in self.workers:
                            continue

                        # Send ping with timeout
                        pong_waiter = await ws.ping()
                        try:
                            await asyncio.wait_for(pong_waiter, timeout=10)
                        except asyncio.TimeoutError:
                            logger.warning(f"Worker {worker_id} failed to respond to ping")
                            disconnected.append(worker_id)
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"Worker {worker_id} connection already closed")
                        disconnected.append(worker_id)
                    except Exception as e:
                        logger.error(f"Error pinging worker {worker_id}: {e}")
                        disconnected.append(worker_id)

                # Clean up disconnected workers
                for worker_id in disconnected:
                    if worker_id in self.workers:
                        logger.info(f"Removing unresponsive worker {worker_id}")
                        del self.workers[worker_id]
                        self.chunk_manager.release_worker_chunks(worker_id)

                        # Update stats
                        self.stats["connected_workers"] = len(self.workers)

                        # Also clean up from workers_by_user if it exists
                        if hasattr(self, "workers_by_user"):
                            worker_user = (
                                worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id
                            )
                            if worker_user in self.workers_by_user:
                                self.workers_by_user[worker_user].discard(worker_id)
                                if not self.workers_by_user[worker_user]:
                                    del self.workers_by_user[worker_user]

                        # Notify monitors
                        await self._broadcast_stats()
                        await self._send_activity(
                            f"Worker {worker_id} removed due to heartbeat timeout"
                        )

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}", exc_info=True)
                # Continue the loop even if there's an error
                await asyncio.sleep(5)

    async def _checkpoint_loop(self):
        """Periodically checkpoint storage."""
        interval = self.config.get("storage", {}).get("checkpoint_interval", 1000)

        while True:
            await asyncio.sleep(60)

            # Get current caption count from storage
            storage_stats = await self.storage.get_storage_stats()
            total_captions = storage_stats["total_captions"]

            # Force checkpoint at regular intervals
            if total_captions > 0 and total_captions % interval == 0:
                logger.info(f"Triggering checkpoint at {total_captions} captions")
                await self.storage.checkpoint()

                # Update stats
                self.stats["last_checkpoint"] = datetime.utcnow().isoformat()
                # No need to update total_written or buffer_size - they come from storage

                await self._broadcast_stats()
                logger.info(
                    f"Checkpoint complete. Total written to disk: {storage_stats['total_written']}"
                )

    async def _stats_update_loop(self):
        """Periodically update and broadcast stats - non-blocking version."""
        # Get the event loop for running blocking operations
        loop = asyncio.get_event_loop()

        # Track session start values
        storage_stats = await self.storage.get_storage_stats()
        session_start_outputs = storage_stats["total_captions"]  # This now counts ALL outputs
        session_start_time = time.time()

        # Track the last known total to detect flushes
        last_known_total = session_start_outputs

        while True:
            await asyncio.sleep(10)

            # Update chunk stats in thread pool to avoid blocking
            chunk_stats = await loop.run_in_executor(None, self.chunk_manager.get_stats)
            storage_stats = await self.storage.get_storage_stats()
            current_total_outputs = storage_stats["total_captions"]  # ALL outputs
            if self.chunk_tracker:
                await self._flush_processed_items()

            self.stats["total_chunks"] = chunk_stats["total"]
            self.stats["completed_chunks"] = chunk_stats["completed"]
            self.stats["failed_chunks"] = chunk_stats["failed"]

            # Update total outputs stat (rename from total_captions for clarity)
            self.stats["total_outputs"] = current_total_outputs
            self.stats["total_captions"] = current_total_outputs  # Keep for backward compatibility

            # Get queue stats in thread pool to avoid blocking
            queue_stats = await loop.run_in_executor(None, self._get_queue_stats)
            self.stats.update(queue_stats)

            # Calculate if we need more chunks
            worker_count = self.stats.get("connected_workers", 0)
            target_buffer = max(self.min_chunk_buffer, worker_count * self.chunk_buffer_multiplier)
            active_chunks = self.stats["pending_chunks"] + self.stats["assigned_chunks"]
            self.stats["chunk_buffer_status"] = f"{active_chunks}/{target_buffer}"

            # Update rate information
            current_time = time.time()
            elapsed_since_update = current_time - self.rate_tracker["last_update_time"]

            if elapsed_since_update > 0:
                # FIX: Handle the case where duplicates were skipped during save
                # If current total is less than last known, it means duplicates were skipped
                # We should not count this as negative progress
                if current_total_outputs < last_known_total:
                    logger.debug(
                        f"Detected duplicate skip during save: {last_known_total} -> {current_total_outputs}"
                    )
                    # Don't calculate negative rate, just update the baseline
                    self.rate_tracker["last_caption_count"] = current_total_outputs
                    self.rate_tracker["current_rate"] = 0.0  # Set to 0 during flush
                else:
                    # Normal rate calculation
                    output_diff = current_total_outputs - self.rate_tracker["last_caption_count"]
                    self.rate_tracker["current_rate"] = (output_diff / elapsed_since_update) * 60
                    self.rate_tracker["last_caption_count"] = current_total_outputs

                # Calculate average rate since THIS SESSION started
                session_elapsed = current_time - session_start_time
                if session_elapsed > 0:
                    # Always use the difference from session start for average
                    session_outputs = current_total_outputs - session_start_outputs
                    self.rate_tracker["average_rate"] = (session_outputs / session_elapsed) * 60

                # Calculate expected rate based on workers and stages
                batch_size = self.vllm_config.get("batch_size", 8)

                # Count total prompts across all stages
                total_prompts = 0
                stages = self.vllm_config.get("stages", [])
                if stages:
                    for stage in stages:
                        total_prompts += len(stage.get("prompts", []))
                else:
                    # Backward compatibility
                    total_prompts = len(self.vllm_config.get("inference_prompts", ["", "", ""]))

                images_per_minute = 30  # Rough estimate: 30 images/min per worker
                self.rate_tracker["expected_rate"] = (
                    worker_count * images_per_minute * total_prompts
                )

                # Update trackers
                self.rate_tracker["last_update_time"] = current_time
                last_known_total = current_total_outputs

            # Log rate information when workers are connected
            if (
                worker_count > 0 and self.rate_tracker["current_rate"] >= 0
            ):  # Only log non-negative rates
                logger.info(
                    f"Rate: {self.rate_tracker['current_rate']:.1f} outputs/min "
                    f"(avg: {self.rate_tracker['average_rate']:.1f}, "
                    f"expected: {self.rate_tracker['expected_rate']:.1f}) | "
                    f"Workers: {worker_count}, Chunks: {active_chunks}/{target_buffer}"
                )

            await self._broadcast_stats()

    async def _restore_state(self):
        """Restore state from storage on startup."""
        total_captions = await self.storage.count_captions()
        logger.info(f"Restored state: {total_captions} captions")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down orchestrator...")

        # Stop chunk creation
        if self.chunk_tracker:
            await self._flush_processed_items()
        self.stop_chunk_creation.set()
        if self.chunk_creation_thread:
            self.chunk_creation_thread.join(timeout=5)

        # Release all assigned chunks before closing connections
        for worker_id in list(self.workers.keys()):
            self.chunk_manager.release_worker_chunks(worker_id)
            if self.chunk_tracker:
                # Update chunk tracker to mark assigned chunks as pending
                with self.chunk_manager.lock:
                    for chunk_id in list(self.chunk_manager.assigned_chunks.get(worker_id, [])):
                        self.chunk_tracker.mark_pending(chunk_id)

        # Close all connections
        for ws in list(self.workers.values()):
            await ws.close()
        for ws in list(self.monitors):
            await ws.close()

        # Save chunk state
        if self.chunk_tracker:
            self.chunk_tracker.save()

        # Final checkpoint
        logger.info(f"Final flush: {len(self.storage.caption_buffer)} captions in buffer")
        await self.storage.checkpoint()

        # Log final statistics
        logger.info(
            f"Shutdown complete. Total captions collected: {self.storage.total_captions_written}"
        )

        await self.storage.close()
