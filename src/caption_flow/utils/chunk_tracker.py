"""Chunk tracking using CheckpointTracker base class with memory optimization."""

import datetime as _datetime
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

from .checkpoint_tracker import CheckpointTracker

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


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
        processed_ranges = sorted([list(r) for r in self.processed_ranges])
        merged = []
        for start, end in processed_ranges:
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
            self.mark_completed()

    def mark_completed(self):
        """Mark chunk as completed and clear unnecessary data to save memory."""
        self.status = "completed"
        self.completed_at = datetime.now(_datetime.UTC)
        # Clear processed_ranges since we don't need them after completion
        # self.processed_ranges = []
        # self.assigned_to = None
        # self.assigned_at = None

    def get_unprocessed_ranges(self) -> List[Tuple[int, int]]:
        """Get ranges of unprocessed items within the chunk (relative indices)."""
        if self.status == "completed":
            return []

        if not self.processed_ranges:
            logger.info(f"Chunk {self.chunk_id} has no processed ranges, returning full range")
            return [(0, self.chunk_size - 1)]

        # Merge ranges first to ensure no overlaps
        merged_ranges = self._merge_ranges(self.processed_ranges)

        unprocessed = []
        current_pos = 0

        for start, end in merged_ranges:
            if current_pos < start:
                unprocessed.append((current_pos, start - 1))
            current_pos = max(current_pos, end + 1)

        # Add any remaining range
        if current_pos < self.chunk_size:
            unprocessed.append((current_pos, self.chunk_size - 1))

        # Log for debugging
        if not unprocessed:
            logger.info(
                f"Chunk {self.chunk_id} has processed ranges {merged_ranges} covering entire chunk size {self.chunk_size}"
            )
        else:
            logger.debug(f"Merged ranges for chunk {self.chunk_id}: {merged_ranges}")
            total_processed = sum(end - start + 1 for start, end in merged_ranges)
            total_unprocessed = sum(end - start + 1 for start, end in unprocessed)
            logger.debug(
                f"Chunk {self.chunk_id}: {total_processed} processed, {total_unprocessed} unprocessed"
            )

        return unprocessed

    def _merge_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent ranges."""
        if not ranges:
            return []

        # Sort ranges by start index, ensuring all are tuples
        sorted_ranges = sorted([tuple(r) for r in ranges])
        merged = [sorted_ranges[0]]

        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]

            # Check if ranges overlap or are adjacent
            if current_start <= last_end + 1:
                # Merge the ranges
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Add as new range
                merged.append((current_start, current_end))

        return merged

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
    """Tracks chunk processing state persistently with memory optimization."""

    def __init__(
        self,
        checkpoint_file: Path,
        max_completed_chunks_in_memory: int = 1000,
        archive_after_hours: int = 24,
    ):
        self.chunks: Dict[str, ChunkState] = {}
        self.max_completed_chunks_in_memory = max_completed_chunks_in_memory
        self.archive_after_hours = archive_after_hours
        self._completed_count = 0  # Track count without storing all IDs
        self.lock = Lock()
        super().__init__(checkpoint_file)

    def _get_default_state(self) -> Dict[str, Any]:
        """Return default state structure for new checkpoints."""
        return {"chunks": {}, "completed_count": 0}

    def _deserialize_state(self, data: Dict[str, Any]) -> None:
        """Deserialize loaded data into instance state."""
        self.chunks = {}
        self._completed_count = data.get("completed_count", 0)

        # Load chunk states
        completed_chunks = 0
        for chunk_id, chunk_data in data.get("chunks", {}).items():
            chunk_state = ChunkState.from_dict(chunk_data)
            with self.lock:
                self.chunks[chunk_id] = chunk_state
            if chunk_state.status == "completed":
                completed_chunks += 1

        logger.info(
            f"Loaded {len(self.chunks)} chunks from checkpoint, "
            f"{completed_chunks} completed in memory, "
            f"{self._completed_count} total completed"
        )

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize instance state for saving."""
        return {
            "chunks": {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
            "completed_count": self._completed_count,
        }

    def _archive_old_completed_chunks(self):
        """Remove old completed chunks from memory to prevent unbounded growth."""
        if not self.archive_after_hours:
            return

        cutoff_time = datetime.now(_datetime.UTC) - timedelta(hours=self.archive_after_hours)
        chunks_to_remove = []

        for chunk_id, chunk in self.chunks.items():
            if (
                chunk.status == "completed"
                and chunk.completed_at
                and chunk.completed_at < cutoff_time
            ):
                chunks_to_remove.append(chunk_id)

        if chunks_to_remove:
            for chunk_id in chunks_to_remove:
                del self.chunks[chunk_id]
            logger.info(f"Archived {len(chunks_to_remove)} old completed chunks from memory")
            self.save()

    def _limit_completed_chunks_in_memory(self):
        """Keep only the most recent completed chunks in memory."""
        completed_chunks = [
            (cid, c) for cid, c in self.chunks.items() if c.status == "completed" and c.completed_at
        ]

        if len(completed_chunks) > self.max_completed_chunks_in_memory:
            # Sort by completion time, oldest first
            completed_chunks.sort(key=lambda x: x[1].completed_at)

            # Remove oldest chunks
            to_remove = len(completed_chunks) - self.max_completed_chunks_in_memory
            for chunk_id, _ in completed_chunks[:to_remove]:
                del self.chunks[chunk_id]

            logger.info(f"Removed {to_remove} oldest completed chunks from memory")
            self.save()

    def add_chunk(
        self, chunk_id: str, shard_name: str, shard_url: str, start_index: int, chunk_size: int
    ) -> bool:
        """Add a new chunk. Returns False if chunk already exists and is completed."""
        if chunk_id in self.chunks:
            logger.debug(
                f"Chunk {chunk_id} already exists with status: {self.chunks[chunk_id].status}"
            )
            return False

        self.chunks[chunk_id] = ChunkState(
            chunk_id=chunk_id,
            shard_name=shard_name,
            shard_url=shard_url,
            start_index=start_index,
            chunk_size=chunk_size,
            status="pending",
        )
        self.save()

        # Periodically clean up old chunks
        if len(self.chunks) % 100 == 0:
            self._archive_old_completed_chunks()
            self._limit_completed_chunks_in_memory()

        return True

    def mark_assigned(self, chunk_id: str, worker_id: str):
        """Mark chunk as assigned."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            chunk.status = "assigned"
            chunk.assigned_to = worker_id
            chunk.assigned_at = datetime.now(_datetime.UTC)
            self.save()

    def mark_completed(self, chunk_id: str):
        """Mark chunk as completed."""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            was_completed = chunk.status == "completed"
            chunk.mark_completed()  # This clears processed_ranges
            if not was_completed:
                self._completed_count += 1
            self.save()
            logger.debug(f"Chunk {chunk_id} marked as completed")

            # Check if we need to clean up
            if self._completed_count % 50 == 0:
                self._limit_completed_chunks_in_memory()

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
            if chunk.status == "completed":
                self._completed_count -= 1
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

    def is_chunk_completed(self, chunk_id: str) -> bool:
        """Check if a chunk is completed (works even if chunk is archived)."""
        if chunk_id in self.chunks:
            return self.chunks[chunk_id].status == "completed"
        # If not in memory, we can't know for sure without loading from disk
        # Could implement a separate completed chunks index if needed
        return False

    def is_shard_complete(self, shard_name: str) -> bool:
        """Check if all chunks for a shard are complete."""
        shard_chunks = [chunk for chunk in self.chunks.values() if chunk.shard_name == shard_name]

        if not shard_chunks:
            return False

        return all(chunk.status == "completed" for chunk in shard_chunks)

    def get_stats(self) -> Dict[str, int]:
        """Get chunk statistics."""
        base_stats = super().get_stats()

        # Count chunks by status in memory
        status_counts = defaultdict(int)
        for chunk in self.chunks.values():
            status_counts[chunk.status] += 1

        base_stats.update(
            {
                "total_in_memory": len(self.chunks),
                "pending": status_counts["pending"],
                "assigned": status_counts["assigned"],
                "completed_in_memory": status_counts["completed"],
                "failed": status_counts["failed"],
                "total_completed": self._completed_count,
            }
        )
        return base_stats

    def get_shards_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all shards and their chunk status."""
        shards = {}

        for _chunk_id, chunk_state in self.chunks.items():
            shard_name = chunk_state.shard_name
            if shard_name not in shards:
                shards[shard_name] = {
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "pending_chunks": 0,
                    "assigned_chunks": 0,
                    "failed_chunks": 0,
                    "is_complete": True,
                    "chunks": [],
                }

            shards[shard_name]["total_chunks"] += 1
            shards[shard_name]["chunks"].append(chunk_state)

            if chunk_state.status == "completed":
                shards[shard_name]["completed_chunks"] += 1
            elif chunk_state.status == "pending":
                shards[shard_name]["pending_chunks"] += 1
                shards[shard_name]["is_complete"] = False
            elif chunk_state.status == "assigned":
                shards[shard_name]["assigned_chunks"] += 1
                shards[shard_name]["is_complete"] = False
            elif chunk_state.status == "failed":
                shards[shard_name]["failed_chunks"] += 1
                shards[shard_name]["is_complete"] = False

        return shards

    def get_incomplete_shards(self) -> Set[str]:
        """Get set of shard names that have incomplete chunks."""
        incomplete = set()
        for _chunk_id, chunk_state in self.chunks.items():
            if chunk_state.status != "completed":
                incomplete.add(chunk_state.shard_name)
        return incomplete

    async def sync_with_storage(self, storage_manager):
        """Sync chunk state with storage to detect processed items - memory efficient version."""
        logger.info("Syncing chunk state with storage...")

        if not storage_manager.captions_path.exists():
            return

        import lance

        # Check if item_index column exists
        table_metadata = lance.dataset(storage_manager.captions_path).schema
        columns = ["job_id", "chunk_id", "item_key"]
        if "item_index" in table_metadata.names:
            columns.append("item_index")

        # Process in batches to avoid loading entire table
        batch_size = 10000
        lance_dataset = lance.dataset(storage_manager.captions_path)

        chunk_indices = defaultdict(set)

        for batch in lance_dataset.to_batches(batch_size=batch_size, columns=columns):
            batch_dict = batch.to_pydict()

            for i in range(len(batch_dict["chunk_id"])):
                chunk_id = batch_dict["chunk_id"][i]
                if not chunk_id:
                    continue

                # Get or create chunk
                if chunk_id not in self.chunks:
                    parts = chunk_id.rsplit("_chunk_", 1)
                    if len(parts) != 2:
                        continue

                    shard_name = parts[0]
                    try:
                        start_idx = int(parts[1])
                    except ValueError:
                        continue

                    shard_url = f"unknown://{shard_name}.tar"

                    self.chunks[chunk_id] = ChunkState(
                        chunk_id=chunk_id,
                        shard_name=shard_name,
                        shard_url=shard_url,
                        start_index=start_idx,
                        chunk_size=10000,  # Default
                        status="pending",
                    )

                chunk = self.chunks[chunk_id]

                # Get item index
                if "item_index" in batch_dict:
                    item_index = batch_dict["item_index"][i]
                else:
                    item_key = batch_dict["item_key"][i]
                    try:
                        item_index = int(item_key.split("_")[-1])
                    except:
                        continue

                if item_index is None:
                    continue

                # Validate index belongs to chunk
                if (
                    item_index < chunk.start_index
                    or item_index >= chunk.start_index + chunk.chunk_size
                ):
                    continue

                chunk_indices[chunk_id].add(item_index)

            # Process accumulated indices periodically to avoid memory buildup
            if len(chunk_indices) > 100:
                self._process_chunk_indices(chunk_indices)
                chunk_indices.clear()

        # Process remaining indices
        if chunk_indices:
            self._process_chunk_indices(chunk_indices)

        logger.info("Sync with storage completed")
        self.save()

    def _process_chunk_indices(self, chunk_indices: Dict[str, Set[int]]):
        """Process a batch of chunk indices."""
        for chunk_id, abs_indices in chunk_indices.items():
            logger.debug(f"Processing indices: {abs_indices} for chunk {chunk_id}")
            if chunk_id not in self.chunks:
                continue

            chunk = self.chunks[chunk_id]

            # Skip if already completed
            if chunk.status == "completed":
                continue

            # Convert to relative indices and group into ranges
            rel_indices = []
            for abs_idx in sorted(abs_indices):
                rel_idx = abs_idx - chunk.start_index
                if 0 <= rel_idx < chunk.chunk_size:
                    rel_indices.append(rel_idx)

            # Group consecutive indices into ranges
            if rel_indices:
                ranges = []
                start = rel_indices[0]
                end = rel_indices[0]

                for idx in rel_indices[1:]:
                    if idx == end + 1:
                        end = idx
                    else:
                        ranges.append((start, end))
                        start = idx
                        end = idx

                ranges.append((start, end))

                # Mark ranges as processed
                for start_idx, end_idx in ranges:
                    chunk.add_processed_range(start_idx, end_idx)

    def mark_items_processed(self, chunk_id: str, start_idx: int, end_idx: int) -> None:
        """Mark a range of items as processed within a chunk."""
        if chunk_id not in self.chunks:
            logger.warning(f"Chunk {chunk_id} not found in tracker")
            return

        chunk_state = self.chunks[chunk_id]

        # Convert absolute indices to chunk-relative indices
        relative_start = start_idx - chunk_state.start_index
        relative_end = end_idx - chunk_state.start_index

        # Ensure indices are within chunk bounds and maintain valid range
        relative_start = max(0, relative_start)
        relative_end = min(chunk_state.chunk_size - 1, relative_end)

        # Skip invalid ranges where start > end
        if relative_start > relative_end:
            logger.warning(
                f"Invalid range for chunk {chunk_id}: start={relative_start}, end={relative_end}, skipping"
            )
            return

        # Add to processed ranges
        chunk_state.processed_ranges.append((relative_start, relative_end))

        # Merge overlapping ranges
        chunk_state.processed_ranges = chunk_state._merge_ranges(chunk_state.processed_ranges)

        # logger.debug(
        #     f"Marked items {start_idx}-{end_idx} as processed in chunk {chunk_id} (relative indices: {relative_start}-{relative_end})"
        # )

        # Check if chunk is now complete
        if chunk_state.get_unprocessed_ranges() == []:
            logger.info(f"Chunk {chunk_id} is now complete")
            chunk_state.status = "completed"

        # Save checkpoint after updating
        self.save()

    def get_chunk_with_unprocessed_items(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk info with unprocessed item ranges."""
        chunk_state = self.chunks.get(chunk_id)
        if not chunk_state:
            return None

        return {
            "chunk_id": chunk_id,
            "unprocessed_ranges": chunk_state.get_unprocessed_ranges(),
            "status": chunk_state.status,
        }
