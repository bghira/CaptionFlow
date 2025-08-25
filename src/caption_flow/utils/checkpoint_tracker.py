"""Base class for checkpoint tracking with persistent state."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointTracker(ABC):
    """Abstract base class for trackers that persist state to JSON checkpoints."""

    def __init__(self, checkpoint_path: Path):
        """Initialize tracker with checkpoint file path."""
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    @abstractmethod
    def _get_default_state(self) -> Dict[str, Any]:
        """Return default state structure for new checkpoints."""
        pass

    @abstractmethod
    def _deserialize_state(self, data: Dict[str, Any]) -> None:
        """Deserialize loaded data into instance state."""
        pass

    @abstractmethod
    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize instance state for saving."""
        pass

    def load(self) -> None:
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                self._deserialize_state(data)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                # Initialize with defaults on load failure
                self._deserialize_state(self._get_default_state())
        else:
            # Initialize with defaults
            self._deserialize_state(self._get_default_state())

    def save(self) -> None:
        """Save checkpoint to disk atomically."""
        try:
            # Prepare data with metadata
            data = self._serialize_state()
            data["updated_at"] = datetime.utcnow().isoformat()

            # Write atomically using temp file
            tmp_file = self.checkpoint_path.with_suffix(".tmp")

            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Ensure temp file was created
            if not tmp_file.exists():
                raise IOError(f"Failed to create temporary file: {tmp_file}")

            # Move atomically
            tmp_file.replace(self.checkpoint_path)

            logger.debug(f"Saved checkpoint to {self.checkpoint_path}")

        except Exception as e:
            # logger.error(f"Error saving checkpoint: {e}", exc_info=True)
            # Try direct write as fallback
            try:
                with open(self.checkpoint_path, "w") as f:
                    json.dump(data, f, indent=2)
                # logger.info("Saved checkpoint using fallback direct write")
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked items. Override for custom stats."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "last_modified": (
                self.checkpoint_path.stat().st_mtime if self.checkpoint_path.exists() else None
            ),
        }
