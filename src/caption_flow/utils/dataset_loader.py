"""Dataset loading utilities for WebDataset and HuggingFace."""

import asyncio
import shlex
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple
import json

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading datasets from various sources."""

    def __init__(self, dataset_path: str, dataset_type: str = "huggingface"):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset (HF repo, local dir, etc.)
            dataset_type: Type of dataset ("huggingface", "webdataset", "local")
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.token = get_token()

        if not self.token and dataset_type == "huggingface":
            logger.warning("No HuggingFace token found; run `huggingface-cli login`")

    def get_shard_list(self) -> List[str]:
        """Get list of all shards in the dataset."""
        if self.dataset_type == "huggingface":
            return self._get_hf_shards()
        elif self.dataset_type == "local":
            return self._get_local_shards()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _get_hf_shards(self) -> List[str]:
        """Get shard URLs from HuggingFace dataset."""
        logger.info(f"Getting shard list from HuggingFace: {self.dataset_path}")

        fs = HfFileSystem()
        files = [fs.resolve_path(p) for p in fs.glob(f"hf://datasets/{self.dataset_path}/**/*.tar")]

        urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in files]

        logger.info(f"Found {len(urls)} shards")
        return sorted(urls)

    def _get_local_shards(self) -> List[str]:
        """Get shard files from local directory."""
        path = Path(self.dataset_path)
        if not path.exists():
            raise ValueError(f"Local dataset path does not exist: {path}")

        shards = list(path.glob("*.tar"))
        logger.info(f"Found {len(shards)} local shards")
        return [str(s) for s in sorted(shards)]

    def load_shard(self, shard_url: str, processed_keys: Optional[set] = None) -> wds.DataPipeline:
        """
        Load a single shard as a WebDataset pipeline.

        Args:
            shard_url: URL or path to the shard
            processed_keys: Set of already processed keys to skip
        """
        if processed_keys is None:
            processed_keys = set()

        if self.dataset_type == "huggingface":
            # Use curl with auth token for HuggingFace
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.token)}' {shlex.quote(shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "__url__", "jpg;png;jpeg;webp;jxl"),
                wds.select(lambda x: x[0] not in processed_keys),
            )
        else:
            # Local file access
            ds = wds.DataPipeline(
                wds.SimpleShardList(shard_url),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "__url__", "jpg;png;jpeg;webp;jxl"),
                wds.select(lambda x: x[0] not in processed_keys),
            )

        return ds

    def iterate_shard(
        self, shard_url: str, processed_keys: Optional[set] = None
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """
        Iterate over items in a shard.

        Yields:
            Tuple of (key, url, image_bytes)
        """
        ds = self.load_shard(shard_url, processed_keys)

        for key, url, image_data in ds:
            yield key, url, image_data

    def count_shard_items(self, shard_url: str, processed_keys: Optional[set] = None) -> int:
        """Count items in a shard (can be slow for large shards)."""
        count = 0
        try:
            for _ in self.iterate_shard(shard_url, processed_keys):
                count += 1
        except Exception as e:
            logger.error(f"Error counting shard {shard_url}: {e}")
        return count


class ShardTracker:
    """Tracks shard processing progress."""

    def __init__(self, checkpoint_path: Path):
        """Initialize shard tracker with checkpoint file."""
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.completed_shards: set = set()
        self.partial_shards: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            try:
                data = json.loads(self.checkpoint_path.read_text())
                self.completed_shards = set(data.get("completed_shards", []))
                self.partial_shards = data.get("partial_shards", {})
                logger.info(
                    f"Loaded checkpoint: {len(self.completed_shards)} completed, "
                    f"{len(self.partial_shards)} partial shards"
                )
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def save(self):
        """Save checkpoint to disk."""
        data = {
            "completed_shards": list(self.completed_shards),
            "partial_shards": self.partial_shards,
        }

        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.checkpoint_path)

    def mark_complete(self, shard_name: str):
        """Mark a shard as complete."""
        self.completed_shards.add(shard_name)
        if shard_name in self.partial_shards:
            del self.partial_shards[shard_name]
        self.save()

    def update_partial(self, shard_name: str, processed_keys: List[str]):
        """Update partial progress for a shard."""
        self.partial_shards[shard_name] = {"keys": processed_keys, "count": len(processed_keys)}
        self.save()

    def get_processed_keys(self, shard_name: str) -> set:
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
        return [s for s in all_shards if Path(s).stem not in self.completed_shards]
