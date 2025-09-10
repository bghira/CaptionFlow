"""Base class for checkpoint tracking with persistent state."""

import datetime as _datetime
import json
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


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
            # If a save is already in progress, let it finish.
            # This prevents race conditions if save() is called rapidly.
            if hasattr(self, "_save_future") and self._save_future and not self._save_future.done():
                logger.warning("Previous save still in progress, skipping this save")
                return  # don't save this time,
            logger.info("Saving chunk tracker state...")
            # Prepare data with metadata
            with self.lock:
                data = self._serialize_state()
            data["updated_at"] = datetime.now(_datetime.UTC).isoformat()

            # Write atomically using temp file
            tmp_file = self.checkpoint_path.with_suffix(".tmp")

            # Use an executor to run the save operation in a background thread.
            # This makes the save call non-blocking.
            with ThreadPoolExecutor(max_workers=1) as executor:
                data_to_save = data.copy()
                self._save_future = executor.submit(self._write_to_disk, data_to_save, tmp_file)
        except Exception as e:
            logger.error(f"Failed to submit save task: {e}", exc_info=True)

    def _write_to_disk(self, data: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
        """Write checkpoint data to disk atomically."""
        # Create a temporary file in the same directory as the checkpoint
        tmp_file = (checkpoint_path or self.checkpoint_path).with_suffix(".tmp")
        logger.debug(f"Checkpoint {tmp_file=}")

        try:
            # Ensure the parent directory exists
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Atomically replace the checkpoint file
            tmp_file.replace(self.checkpoint_path)
            logger.debug(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint atomically: {e}", exc_info=True)
            # Try to clean up the temp file if it exists
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked items. Override for custom stats."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "last_modified": (
                self.checkpoint_path.stat().st_mtime if self.checkpoint_path.exists() else None
            ),
        }
