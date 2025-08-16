"""Chunk tracking using CheckpointTracker base class."""

import logging
from pathlib import Path
from typing import Set, Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .checkpoint_tracker import CheckpointTracker

logger = logging.getLogger(__name__)


@dataclass
class ChunkState:
    """State of a chunk."""

    chunk_id: str
    shard_name: str
    start_index: int
    chunk_size: int
    status: str  # pending, assigned, completed, failed
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime objects to ISO format strings
        if d["completed_at"]:
            d["completed_at"] = d["completed_at"].isoformat()
        if d["assigned_at"]:
            d["assigned_at"] = d["assigned_at"].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict):
        """Create from dictionary."""
        # Convert ISO format strings back to datetime objects
        if d.get("completed_at"):
            d["completed_at"] = datetime.fromisoformat(d["completed_at"])
        if d.get("assigned_at"):
            d["assigned_at"] = datetime.fromisoformat(d["assigned_at"])
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

    def add_chunk(self, chunk_id: str, shard_name: str, start_index: int, chunk_size: int) -> bool:
        """Add a new chunk. Returns False if chunk already exists and is completed."""
        if chunk_id in self.completed_chunks:
            logger.debug(f"Chunk {chunk_id} already completed, skipping")
            return False

        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = ChunkState(
                chunk_id=chunk_id,
                shard_name=shard_name,
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
        """Sync chunk state with storage to detect already-processed chunks."""
        logger.info("Syncing chunk state with storage...")

        # Get all existing captions from storage
        if storage_manager.captions_path.exists():
            import pyarrow.parquet as pq

            # Read just the job_id column
            table = pq.read_table(storage_manager.captions_path, columns=["job_id", "chunk_id"])
            existing_job_ids = set(table["job_id"].to_pylist())

            # Also get chunk_ids if available
            if "chunk_id" in table.column_names:
                existing_chunk_ids = set(
                    cid for cid in table["chunk_id"].to_pylist() if cid is not None
                )

                # Mark existing chunks as completed
                for chunk_id in existing_chunk_ids:
                    if chunk_id in self.chunks:
                        self.mark_completed(chunk_id)
                    else:
                        # Create chunk entry for already-processed chunks
                        # Extract shard name from chunk_id (format: shard_chunk_index)
                        parts = chunk_id.rsplit("_chunk_", 1)
                        if len(parts) == 2:
                            shard_name = parts[0]
                            try:
                                start_idx = int(parts[1])
                                # We don't know exact chunk size, but mark it as completed
                                self.chunks[chunk_id] = ChunkState(
                                    chunk_id=chunk_id,
                                    shard_name=shard_name,
                                    start_index=start_idx,
                                    chunk_size=1000,  # Default chunk size
                                    status="completed",
                                    completed_at=datetime.utcnow(),
                                )
                                self.completed_chunks.add(chunk_id)
                            except ValueError:
                                logger.warning(f"Could not parse chunk_id: {chunk_id}")

                logger.info(f"Found {len(existing_chunk_ids)} completed chunks in storage")

            # Also check by job_id pattern if chunk_id column doesn't exist
            else:
                for job_id in existing_job_ids:
                    # Extract chunk_id from job_id (format: chunk_id_item_key)
                    if "_chunk_" in job_id:
                        parts = job_id.split("_")
                        # Find the chunk part
                        for i, part in enumerate(parts):
                            if part == "chunk" and i + 1 < len(parts):
                                try:
                                    # Reconstruct chunk_id
                                    chunk_idx = int(parts[i + 1])
                                    shard_parts = parts[:i]
                                    chunk_id = f"{'_'.join(shard_parts)}_chunk_{chunk_idx}"

                                    if chunk_id not in self.completed_chunks:
                                        self.completed_chunks.add(chunk_id)
                                        logger.debug(
                                            f"Marked chunk {chunk_id} as completed from job_id"
                                        )
                                    break
                                except ValueError:
                                    continue

                logger.info(f"Inferred {len(self.completed_chunks)} completed chunks from job_ids")

            self.save()
