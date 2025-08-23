"""Dataset metadata caching for efficient HuggingFace dataset handling."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetMetadataCache:
    """Caches dataset metadata to avoid repeated full iterations."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "dataset_metadata.json"
        self.metadata: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached metadata from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded dataset metadata cache with {len(self.metadata)} datasets")
            except Exception as e:
                logger.error(f"Failed to load metadata cache: {e}")
                self.metadata = {}

    def _save_cache(self):
        """Save metadata cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug("Saved dataset metadata cache")
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")

    def get_dataset_key(self, dataset_path: str, split: str) -> str:
        """Generate a unique key for a dataset+split combination."""
        return f"{dataset_path}:{split}"

    def get_metadata(self, dataset_path: str, split: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a dataset."""
        key = self.get_dataset_key(dataset_path, split)
        return self.metadata.get(key)

    def set_metadata(self, dataset_path: str, split: str, metadata: Dict[str, Any]):
        """Cache metadata for a dataset."""
        key = self.get_dataset_key(dataset_path, split)
        metadata["cached_at"] = datetime.utcnow().isoformat()
        metadata["dataset_path"] = dataset_path
        metadata["split"] = split
        self.metadata[key] = metadata
        self._save_cache()
        logger.info(f"Cached metadata for {key}: {metadata.get('total_items', 0)} items")

    def invalidate(self, dataset_path: str, split: str):
        """Remove cached metadata for a dataset."""
        key = self.get_dataset_key(dataset_path, split)
        if key in self.metadata:
            del self.metadata[key]
            self._save_cache()
            logger.info(f"Invalidated metadata cache for {key}")
