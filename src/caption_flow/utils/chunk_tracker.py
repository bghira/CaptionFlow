"""Chunk tracking using CheckpointTracker base class."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Set, Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field

from .checkpoint_tracker import CheckpointTracker

logger = logging.getLogger(__name__)


@dataclass
class ChunkState:
    """State of a chunk with item-level tracking."""

    chunk_id: str
    shard_name: str
    shard_url: str
    start_index: int
    chunk_size: int
    status: str  # pending, assigned, completed, failed

    processed_ranges: List[Tuple[int, int]] = field(default_factory=list)  # [(start, end), ...]
    processed_count: int = 0

    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None

    def add_processed_range(self, start: int, end: int):
        """Add a processed range and merge if needed."""
        # Add new range
        self.processed_ranges.append((start, end))

        # Sort and merge overlapping ranges
        self.processed_ranges.sort()
        merged = []
        for start, end in self.processed_ranges:
            if merged and start <= merged[-1][1] + 1:
                # Merge with previous range
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        self.processed_ranges = merged

        # Update count
        self.processed_count = sum(end - start + 1 for start, end in self.processed_ranges)

        # Auto-complete if all items processed
        if self.processed_count >= self.chunk_size:
            self.status = "completed"
            self.completed_at = datetime.utcnow()

    def get_unprocessed_ranges(self) -> List[Tuple[int, int]]:
        """Get ranges that haven't been processed yet."""
        if not self.processed_ranges:
            return [(0, self.chunk_size - 1)]

        unprocessed = []
        current = 0

        for start, end in self.processed_ranges:
            if current < start:
                unprocessed.append((current, start - 1))
            current = max(current, end + 1)

        if current < self.chunk_size:
            unprocessed.append((current, self.chunk_size - 1))

        return unprocessed

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        if d["completed_at"]:
            d["completed_at"] = d["completed_at"].isoformat()
        if d["assigned_at"]:
            d["assigned_at"] = d["assigned_at"].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict):
        """Create from dictionary."""
        if d.get("completed_at"):
            d["completed_at"] = datetime.fromisoformat(d["completed_at"])
        if d.get("assigned_at"):
            d["assigned_at"] = datetime.fromisoformat(d["assigned_at"])
        # Ensure processed_ranges exists
        d.setdefault("processed_ranges", [])
        d.setdefault("processed_count", 0)
        return cls(**d)


class ChunkTracker(CheckpointTracker):
    """Tracks chunk processing state persistently."""

    def __init__(self, checkpoint_file: Path):
        self.chunks: Dict[str, ChunkState] = {}
        self.completed_chunks: Set[str] = set()
        super().__init__(checkpoint_file)

    def _get_default_state(self) -> Dict[str, Any]:
        """Return default state structure for new checkpoints."""
        return {"chunks": {}}

    def _deserialize_state(self, data: Dict[str, Any]) -> None:
        """Deserialize loaded data into instance state."""
        self.chunks = {}
        self.completed_chunks = set()

        # Load chunk states
        for chunk_id, chunk_data in data.get("chunks", {}).items():
            chunk_state = ChunkState.from_dict(chunk_data)
            self.chunks[chunk_id] = chunk_state
            if chunk_state.status == "completed":
                self.completed_chunks.add(chunk_id)

        logger.info(
            f"Loaded {len(self.chunks)} chunks from checkpoint, "
            f"{len(self.completed_chunks)} completed"
        )

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize instance state for saving."""
        return {"chunks": {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()}}

    def add_chunk(
        self, chunk_id: str, shard_name: str, shard_url: str, start_index: int, chunk_size: int
    ) -> bool:
        """Add a new chunk. Returns False if chunk already exists and is completed."""
        if chunk_id in self.completed_chunks:
            logger.debug(f"Chunk {chunk_id} already completed, skipping")
            return False

        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = ChunkState(
                chunk_id=chunk_id,
                shard_name=shard_name,
                shard_url=shard_url,  # Now included
                start_index=start_index,
                chunk_size=chunk_size,
                status="pending",
            )
            self.save()

        return True

    def mark_assigned(self, chunk_id: str, worker_id: str):
        """Mark chunk as assigned."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.status = "assigned"
            chunk.assigned_to = worker_id
            chunk.assigned_at = datetime.utcnow()
            self.save()

    def mark_completed(self, chunk_id: str):
        """Mark chunk as completed."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.status = "completed"
            chunk.completed_at = datetime.utcnow()
            self.completed_chunks.add(chunk_id)
            self.save()
            logger.info(f"Chunk {chunk_id} marked as completed")

    def mark_failed(self, chunk_id: str):
        """Mark chunk as failed."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.status = "pending"  # Reset to pending for retry
            chunk.assigned_to = None
            chunk.assigned_at = None
            self.save()

    def mark_pending(self, chunk_id: str):
        """Mark chunk as pending (for manual reset)."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.status = "pending"
            chunk.assigned_to = None
            chunk.assigned_at = None
            self.save()

    def release_worker_chunks(self, worker_id: str):
        """Release all chunks assigned to a worker."""
        released_chunks = []
        for chunk_id, chunk in self.chunks.items():
            if chunk.assigned_to == worker_id and chunk.status == "assigned":
                chunk.status = "pending"
                chunk.assigned_to = None
                chunk.assigned_at = None
                released_chunks.append(chunk_id)
        self.save()
        return released_chunks

    def get_pending_chunks(self, shard_name: Optional[str] = None) -> List[str]:
        """Get list of pending chunk IDs, optionally filtered by shard."""
        pending = []
        for chunk_id, chunk in self.chunks.items():
            if chunk.status == "pending":
                if shard_name is None or chunk.shard_name == shard_name:
                    pending.append(chunk_id)
        return pending

    def is_shard_complete(self, shard_name: str) -> bool:
        """Check if all chunks for a shard are complete."""
        shard_chunks = [chunk for chunk in self.chunks.values() if chunk.shard_name == shard_name]

        if not shard_chunks:
            return False

        return all(chunk.status == "completed" for chunk in shard_chunks)

    def get_stats(self) -> Dict[str, int]:
        """Get chunk statistics."""
        base_stats = super().get_stats()
        base_stats.update(
            {
                "total": len(self.chunks),
                "pending": sum(1 for c in self.chunks.values() if c.status == "pending"),
                "assigned": sum(1 for c in self.chunks.values() if c.status == "assigned"),
                "completed": len(self.completed_chunks),
                "failed": sum(1 for c in self.chunks.values() if c.status == "failed"),
            }
        )
        return base_stats

    def get_shards_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all shards and their chunk status."""
        shards = {}

        for chunk_id, chunk_state in self.chunks.items():
            shard_name = chunk_state.shard_name

            # For virtual HF dataset shards, normalize the shard name
            if shard_name.startswith("hf_dataset:"):
                parts = shard_name.split(":")
                if len(parts) >= 4 and parts[2] == "chunk":
                    # Use just the dataset identifier as the shard name
                    normalized_shard_name = ":".join(parts[:2])
                else:
                    normalized_shard_name = shard_name
            else:
                normalized_shard_name = shard_name

            if normalized_shard_name not in shards:
                shards[normalized_shard_name] = {
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "pending_chunks": 0,
                    "assigned_chunks": 0,
                    "failed_chunks": 0,
                    "is_complete": True,
                    "chunks": [],
                }

            shards[normalized_shard_name]["chunks"].append(chunk_state)
            shards[normalized_shard_name]["total_chunks"] += 1

            if chunk_state.status == "completed":
                shards[normalized_shard_name]["completed_chunks"] += 1
            elif chunk_state.status == "pending":
                shards[normalized_shard_name]["pending_chunks"] += 1
                shards[normalized_shard_name]["is_complete"] = False
            elif chunk_state.status == "assigned":
                shards[normalized_shard_name]["assigned_chunks"] += 1
                shards[normalized_shard_name]["is_complete"] = False
            elif chunk_state.status == "failed":
                shards[normalized_shard_name]["failed_chunks"] += 1
                shards[normalized_shard_name]["is_complete"] = False

        return shards

    def get_incomplete_shards(self) -> Set[str]:
        """Get set of shard names that have incomplete chunks."""
        incomplete = set()
        for chunk_id, chunk_state in self.chunks.items():
            if chunk_state.status != "completed":
                incomplete.add(chunk_state.shard_name)
        return incomplete

    async def sync_with_storage(self, storage_manager):
        """Sync chunk state with storage to detect processed items (HF and WebDataset compatible)."""
        logger.info("Syncing chunk state with storage...")

        if storage_manager.captions_path.exists():
            import pyarrow.parquet as pq

            # Read all relevant columns
            columns = ["job_id", "chunk_id", "item_key"]
            # Check if item_index column exists (new format)
            table_metadata = pq.read_metadata(storage_manager.captions_path)
            if "item_index" in table_metadata.schema.names:
                columns.append("item_index")

            table = pq.read_table(storage_manager.captions_path, columns=columns)

            # Build lookup of chunk_id -> processed indices
            chunk_indices = defaultdict(set)

            for i in range(len(table)):
                chunk_id = table["chunk_id"][i].as_py()
                if not chunk_id:
                    continue

                # Try to get item index from explicit column first
                if "item_index" in table.column_names:
                    item_index = table["item_index"][i].as_py()
                    if item_index is not None:
                        chunk_indices[chunk_id].add(item_index)
                        continue

                # Fallback to parsing from item_key or job_id
                item_key = table["item_key"][i].as_py()
                job_id = table["job_id"][i].as_py()

                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]

                    # For HF datasets, parse from item_key
                    if chunk.shard_url.startswith("hf_dataset:"):
                        try:
                            # Format: dataset_XXXXXXXX
                            abs_idx = int(item_key.split("_")[-1])
                            chunk_indices[chunk_id].add(abs_idx)
                        except:
                            pass

                    # For WebDataset, we need to maintain order
                    # Count occurrences to infer index
                    else:
                        # This is approximate - better to use explicit item_index
                        chunk_indices[chunk_id].add(len(chunk_indices[chunk_id]))

            # Update chunks with processed indices
            for chunk_id, indices in chunk_indices.items():
                if chunk_id not in self.chunks:
                    # Create chunk from ID
                    parts = chunk_id.rsplit("_chunk_", 1)
                    if len(parts) != 2:
                        continue

                    shard_name = parts[0]
                    try:
                        start_idx = int(parts[1])
                    except ValueError:
                        continue

                    # Infer shard URL
                    if shard_name.replace("_", "/") in chunk_id or "_" in shard_name:
                        # HF dataset
                        dataset_path = shard_name.replace("_", "/")
                        shard_url = f"hf_dataset:{dataset_path}:chunk:{start_idx}"
                    else:
                        # WebDataset
                        shard_url = f"unknown://{shard_name}.tar"

                    self.chunks[chunk_id] = ChunkState(
                        chunk_id=chunk_id,
                        shard_name=shard_name,
                        shard_url=shard_url,
                        start_index=start_idx,
                        chunk_size=1000,  # Default
                        status="pending",
                    )

                # Convert absolute indices to chunk-relative and mark as processed
                chunk = self.chunks[chunk_id]
                for abs_idx in sorted(indices):
                    if chunk.shard_url.startswith("hf_dataset:"):
                        # HF: absolute index to relative
                        rel_idx = abs_idx - chunk.start_index
                    else:
                        # WebDataset: indices are already relative (0-based within chunk)
                        rel_idx = abs_idx

                    if 0 <= rel_idx < chunk.chunk_size:
                        chunk.add_processed_range(rel_idx, rel_idx)

            logger.info(f"Synced {len(chunk_indices)} chunks with processed items")
            self.save()

    def mark_items_processed(self, chunk_id: str, start_idx: int, end_idx: int):
        """Mark a range of items as processed within a chunk."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.add_processed_range(start_idx, end_idx)

            # If chunk is now complete, update completed set
            if chunk.status == "completed":
                self.completed_chunks.add(chunk_id)

            self.save()
            logger.info(f"Marked items {start_idx}-{end_idx} as processed in chunk {chunk_id}")

    def get_chunk_with_unprocessed_items(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk info including unprocessed ranges."""
        if chunk_id not in self.chunks:
            return None

        chunk = self.chunks[chunk_id]
        return {"chunk": chunk.to_dict(), "unprocessed_ranges": chunk.get_unprocessed_ranges()}
