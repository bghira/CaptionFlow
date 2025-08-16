"""Shard tracking using CheckpointTracker base class."""

from pathlib import Path
from typing import Dict, Any, List, Set

from .checkpoint_tracker import CheckpointTracker


class ShardTracker(CheckpointTracker):
    """Tracks shard processing progress."""

    def __init__(self, checkpoint_path: Path):
        """Initialize shard tracker with checkpoint file."""
        self.completed_shards: Set[str] = set()
        self.partial_shards: Dict[str, Dict[str, Any]] = {}
        super().__init__(checkpoint_path)

    def _get_default_state(self) -> Dict[str, Any]:
        """Return default state structure for new checkpoints."""
        return {"completed_shards": [], "partial_shards": {}}

    def _deserialize_state(self, data: Dict[str, Any]) -> None:
        """Deserialize loaded data into instance state."""
        self.completed_shards = set(data.get("completed_shards", []))
        self.partial_shards = data.get("partial_shards", {})

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize instance state for saving."""
        return {
            "completed_shards": list(self.completed_shards),
            "partial_shards": self.partial_shards,
        }

    def mark_complete(self, shard_name: str) -> None:
        """Mark a shard as complete."""
        self.completed_shards.add(shard_name)
        if shard_name in self.partial_shards:
            del self.partial_shards[shard_name]
        self.save()

    def update_partial(self, shard_name: str, processed_keys: List[str]) -> None:
        """Update partial progress for a shard."""
        self.partial_shards[shard_name] = {"keys": processed_keys, "count": len(processed_keys)}
        self.save()

    def get_processed_keys(self, shard_name: str) -> Set[str]:
        """Get set of processed keys for a shard."""
        if shard_name in self.completed_shards:
            return set()  # All done

        if shard_name in self.partial_shards:
            return set(self.partial_shards[shard_name].get("keys", []))

        return set()

    def is_complete(self, shard_name: str) -> bool:
        """Check if a shard is complete."""
        return shard_name in self.completed_shards

    def get_remaining_shards(self, all_shards: List[str]) -> List[str]:
        """Get list of shards that still need processing."""
        remaining = []
        for s in all_shards:
            # Extract shard name properly for both regular and virtual shards
            if s.startswith("hf_dataset:"):
                shard_name = s  # Use full virtual shard ID
            else:
                shard_name = Path(s).stem

            if shard_name not in self.completed_shards:
                remaining.append(s)

        return remaining

    def get_stats(self) -> Dict[str, Any]:
        """Get shard tracking statistics."""
        base_stats = super().get_stats()
        base_stats.update(
            {
                "completed_shards": len(self.completed_shards),
                "partial_shards": len(self.partial_shards),
                "total_partial_keys": sum(
                    len(data.get("keys", [])) for data in self.partial_shards.values()
                ),
            }
        )
        return base_stats
