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

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str = "huggingface",
        split: str = "train",
        image_column: str = "image",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset (HF repo, local dir, etc.)
            dataset_type: Type of dataset ("huggingface", "webdataset", "local")
            split: Split to use for HuggingFace datasets (default: "train")
            image_column: Column name containing image data or URLs (default: "image")
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.split = split
        self.image_column = image_column
        self.token = get_token()
        self.dataset_format = None  # Will be detected: "webdataset" or "huggingface_datasets"

        if not self.token and dataset_type == "huggingface":
            logger.warning("No HuggingFace token found; run `huggingface-cli login`")

        # Detect the actual format if it's a HuggingFace dataset
        if dataset_type == "huggingface":
            self.dataset_format = self._detect_dataset_format()
            logger.info(f"Detected dataset format: {self.dataset_format}")

    def _detect_dataset_format(self) -> str:
        """Detect whether it's WebDataset or HuggingFace datasets format."""
        fs = HfFileSystem(token=self.token)

        # Check for .tar files (WebDataset)
        tar_files = list(fs.glob(f"hf://datasets/{self.dataset_path}/**/*.tar"))
        if tar_files:
            return "webdataset"

        # Check for .parquet files (Huggingface Arrow DB)
        parquet_files = list(fs.glob(f"hf://datasets/{self.dataset_path}/**/*.parquet"))
        if parquet_files:
            return "huggingface_datasets"

        raise AssertionError(f"Could not detect dataset format for {self.dataset_path}")

    def get_shard_list(self) -> List[str]:
        """Get list of all shards in the dataset."""
        if self.dataset_type == "huggingface":
            if self.dataset_format == "webdataset":
                return self._get_hf_webdataset_shards()
            else:
                logger.error(f"Unknown dataset format: {self.dataset_format}")
                return []
        elif self.dataset_type == "local":
            return self._get_local_shards()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _get_hf_webdataset_shards(self) -> List[str]:
        """Get shard URLs from HuggingFace WebDataset."""
        logger.info(f"Getting WebDataset shard list from HuggingFace: {self.dataset_path}")

        fs = HfFileSystem(token=self.token)
        files = [fs.resolve_path(p) for p in fs.glob(f"hf://datasets/{self.dataset_path}/**/*.tar")]

        urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in files]

        logger.info(f"Found {len(urls)} WebDataset shards")
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

        if self.dataset_type == "huggingface" and self.dataset_format == "webdataset":
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
        self,
        shard_url: str,
        processed_keys: Optional[set] = None,
        unprocessed_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over items in a shard, returning full sample dictionaries.

        Args:
            shard_url: URL or identifier of the shard
            processed_keys: Set of already processed keys to skip
            unprocessed_ranges: Specific ranges to process (for range-based processing)

        Yields:
            Dictionary containing the full WebDataset sample
        """
        if processed_keys is None:
            processed_keys = set()

        if self.dataset_type == "huggingface" and self.dataset_format == "webdataset":
            # Use curl with auth token for HuggingFace
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.token)}' {shlex.quote(shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.select(lambda x: x.get("__key__", "") not in processed_keys),
            )
        else:
            # Local file access
            ds = wds.DataPipeline(
                wds.SimpleShardList(shard_url),
                wds.tarfile_to_samples(),
                wds.select(lambda x: x.get("__key__", "") not in processed_keys),
            )

        # Return full samples as dictionaries
        for sample in ds:
            # Ensure it's a dict and has required fields
            if isinstance(sample, dict) and "__key__" in sample:
                yield sample

    def count_shard_items(self, shard_url: str, processed_keys: Optional[set] = None) -> int:
        """Count items in a shard (can be slow for large shards)."""
        count = 0
        try:
            for _ in self.iterate_shard(shard_url, processed_keys):
                count += 1
        except Exception as e:
            logger.error(f"Error counting shard {shard_url}: {e}")
        return count

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            "dataset_path": self.dataset_path,
            "dataset_type": self.dataset_type,
            "dataset_format": self.dataset_format,
        }

        if self.dataset_format == "huggingface_datasets":
            # Include cached metadata if available
            if hasattr(self, "_hf_metadata"):
                info.update(self._hf_metadata)
            else:

                try:
                    # Try to get more info about the dataset
                    dataset_info = load_dataset(
                        self.dataset_path, split=self.split, streaming=True, token=self.token
                    )
                    # Get features info
                    if hasattr(dataset_info, "features"):
                        info["features"] = str(dataset_info.features)

                    # Try to get total size (might not work for all datasets)
                    try:
                        # This might be expensive for large datasets
                        total_examples = len(
                            load_dataset(self.dataset_path, split=self.split, token=self.token)
                        )
                        info["total_examples"] = total_examples
                        self._hf_total_items = total_examples
                    except:
                        info["total_examples"] = "unknown"

                except Exception as e:
                    logger.error(f"Error getting dataset info: {e}")

        return info
