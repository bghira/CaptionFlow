"""Dataset loading utilities for WebDataset and HuggingFace."""

import asyncio
import shlex
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple
import json

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from datasets import load_dataset, Dataset
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading datasets from various sources."""

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str = "huggingface",
        split: str = "train",
        image_column: str = "image",
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
        self._hf_dataset = None  # Cache for HuggingFace dataset
        self._hf_total_items = None  # Cache for total items count

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

        # Check for parquet files (HuggingFace datasets)
        parquet_files = list(fs.glob(f"hf://datasets/{self.dataset_path}/**/*.parquet"))
        if parquet_files:
            return "huggingface_datasets"

        # Check for dataset_info.json or dataset_dict.json
        if fs.exists(f"datasets/{self.dataset_path}/dataset_info.json") or fs.exists(
            f"datasets/{self.dataset_path}/dataset_dict.json"
        ):
            return "huggingface_datasets"

        logger.warning(f"Could not detect dataset format for {self.dataset_path}")
        return "unknown"

    def get_shard_list(self) -> List[str]:
        """Get list of all shards in the dataset."""
        if self.dataset_type == "huggingface":
            if self.dataset_format == "webdataset":
                return self._get_hf_webdataset_shards()
            elif self.dataset_format == "huggingface_datasets":
                return self._get_hf_dataset_shards()
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

    def _get_hf_dataset_shards(self) -> List[str]:
        """Get virtual 'shards' for HuggingFace datasets format."""
        logger.info(f"Getting HuggingFace dataset info: {self.dataset_path}")

        # For HuggingFace datasets, we'll create virtual shards based on chunks
        # Each "shard" will be a range of indices
        try:
            # First, try to get available splits
            try:
                from datasets import get_dataset_split_names

                available_splits = get_dataset_split_names(self.dataset_path, token=self.token)
                logger.info(f"Available splits: {available_splits}")

                if self.split not in available_splits:
                    logger.warning(
                        f"Requested split '{self.split}' not found. "
                        f"Available splits: {available_splits}. "
                        f"Using first available split: '{available_splits[0]}'"
                    )
                    self.split = available_splits[0]
            except Exception as e:
                logger.warning(f"Could not get split names: {e}")

            # Load dataset info without downloading data
            dataset_info = load_dataset(
                self.dataset_path, split=self.split, streaming=True, token=self.token
            )

            # Try to get the total size
            # For streaming datasets, we might need to iterate to count
            # This is expensive, so we'll use a default chunk size instead
            chunk_size = 10000  # Default chunk size for virtual shards

            # Create virtual shard identifiers
            # Format: "hf_dataset:<dataset_path>:chunk:<start_idx>"
            virtual_shards = []

            # We'll create a reasonable number of virtual shards
            # Without knowing the total size, we'll create them on-demand
            # For now, create initial batch of virtual shards
            for i in range(10):  # Start with 10 virtual shards
                shard_id = f"hf_dataset:{self.dataset_path}:chunk:{i * chunk_size}"
                virtual_shards.append(shard_id)

            logger.info(
                f"Created {len(virtual_shards)} initial virtual shards for HuggingFace dataset"
            )
            return virtual_shards

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset info: {e}")
            return []

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

        # Check if this is a virtual HuggingFace dataset shard
        if shard_url.startswith("hf_dataset:"):
            raise ValueError(
                "Virtual HuggingFace dataset shards should use iterate_shard() directly, "
                "not load_shard()"
            )

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

    def _parse_virtual_shard(self, shard_url: str) -> Tuple[str, int, int]:
        """Parse virtual shard identifier."""
        # Format: "hf_dataset:<dataset_path>:chunk:<start_idx>"
        parts = shard_url.split(":")
        if len(parts) != 4 or parts[0] != "hf_dataset" or parts[2] != "chunk":
            raise ValueError(f"Invalid virtual shard format: {shard_url}")

        dataset_path = parts[1]
        start_idx = int(parts[3])
        chunk_size = 10000  # Default chunk size

        return dataset_path, start_idx, chunk_size

    def iterate_shard(
        self, shard_url: str, processed_keys: Optional[set] = None
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """
        Iterate over items in a shard.

        Yields:
            Tuple of (key, url, image_bytes)
        """
        # Check if this is a virtual HuggingFace dataset shard
        if shard_url.startswith("hf_dataset:"):
            yield from self._iterate_hf_dataset_shard(shard_url, processed_keys)
        else:
            # Regular WebDataset shard
            ds = self.load_shard(shard_url, processed_keys)
            for key, url, image_data in ds:
                yield key, url, image_data

    def _iterate_hf_dataset_shard(
        self, shard_url: str, processed_keys: Optional[set] = None
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """Iterate over a virtual HuggingFace dataset shard."""
        if processed_keys is None:
            processed_keys = set()

        dataset_path, start_idx, chunk_size = self._parse_virtual_shard(shard_url)

        # IMPORTANT: Check if start_idx is beyond dataset bounds
        if self._hf_total_items is not None and start_idx >= self._hf_total_items:
            logger.warning(
                f"Virtual shard starts at index {start_idx} but dataset only has "
                f"{self._hf_total_items} items. Skipping this shard."
            )
            return

        # Use streaming mode for HuggingFace datasets
        logger.info(
            f"Loading HuggingFace dataset in streaming mode: {dataset_path} "
            f"(split: {self.split}, start: {start_idx}, chunk_size: {chunk_size})"
        )

        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                dataset_path,
                split=self.split,
                streaming=True,
                token=self.token,
            )

            items_processed = 0
            items_skipped = 0

            # For streaming datasets, we need to manually skip items
            for idx, item in enumerate(dataset):
                # Skip items before our start index
                if idx < start_idx:
                    items_skipped += 1
                    # Check if we're skipping too many items (indicates bounds issue)
                    if self._hf_total_items and items_skipped > self._hf_total_items:
                        logger.error(
                            f"Skipped {items_skipped} items but dataset only has "
                            f"{self._hf_total_items} total. Breaking to prevent infinite loop."
                        )
                        break
                    continue

                # Stop after processing chunk_size items
                if items_processed >= chunk_size:
                    logger.info(f"Completed chunk: processed {items_processed} items")
                    break

                # Also stop if we've reached the dataset end
                if self._hf_total_items and (start_idx + items_processed) >= self._hf_total_items:
                    logger.info(
                        f"Reached dataset end at item {start_idx + items_processed} "
                        f"(total: {self._hf_total_items})"
                    )
                    break

                # Generate a unique key for this item
                key = f"{dataset_path}_{start_idx + items_processed:08d}"

                if key in processed_keys:
                    items_processed += 1
                    continue

                try:
                    # Extract image data - check configured column name
                    if self.image_column in item:
                        img_data = item[self.image_column]

                        # Delegate image processing to ImageProcessor
                        image_bytes = ImageProcessor.process_image_data(img_data)

                        if image_bytes:
                            # URL is virtual for HF datasets
                            url = f"hf://{dataset_path}#{start_idx + items_processed}"
                            items_processed += 1
                            yield key, url, image_bytes
                        else:
                            logger.warning(f"Failed to process image for item {idx}")
                            items_processed += 1
                            continue
                    else:
                        logger.warning(
                            f"No image column '{self.image_column}' found in item {idx}. "
                            f"Available columns: {list(item.keys())}"
                        )
                        items_processed += 1

                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    items_processed += 1
                    continue

            logger.info(
                f"Virtual shard complete: processed {items_processed} items "
                f"(skipped {items_skipped}, start_idx: {start_idx})"
            )

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {e}")
            return

    def iterate_shard_with_metadata(
        self, shard_url: str, processed_keys: Optional[set] = None
    ) -> Generator[Tuple[str, str, bytes, Dict[str, Any]], None, None]:
        """
        Iterate over items in a shard, including metadata.

        Yields:
            Tuple of (key, url, image_bytes, metadata_dict)
        """
        # Check if this is a virtual HuggingFace dataset shard
        if shard_url.startswith("hf_dataset:"):
            yield from self._iterate_hf_dataset_shard_with_metadata(shard_url, processed_keys)
        else:
            # Regular WebDataset shard - no metadata by default
            for key, url, image_data in self.iterate_shard(shard_url, processed_keys):
                yield key, url, image_data, {}

    def _iterate_hf_dataset_shard_with_metadata(
        self, shard_url: str, processed_keys: Optional[set] = None
    ) -> Generator[Tuple[str, str, bytes, Dict[str, Any]], None, None]:
        """Iterate over a virtual HuggingFace dataset shard with metadata."""
        if processed_keys is None:
            processed_keys = set()

        dataset_path, start_idx, chunk_size = self._parse_virtual_shard(shard_url)

        logger.info(
            f"Loading HuggingFace dataset with metadata: {dataset_path} (split: {self.split})"
        )

        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                dataset_path,
                split=self.split,
                streaming=True,
                token=self.token,
            )

            # Skip to start index if needed
            if start_idx > 0:
                dataset = dataset.skip(start_idx)

            items_processed = 0

            for idx, item in enumerate(dataset):
                # Stop after processing chunk_size items
                if items_processed >= chunk_size:
                    break

                # Generate a unique key for this item
                key = f"{dataset_path}_{start_idx + items_processed:08d}"

                if key in processed_keys:
                    items_processed += 1
                    continue

                try:
                    # Extract image data
                    if self.image_column in item:
                        img_data = item[self.image_column]

                        # Process image to bytes
                        image_bytes = ImageProcessor.process_image_data(img_data)

                        if image_bytes:
                            # Extract all metadata (excluding the image column)
                            metadata = {k: v for k, v in item.items() if k != self.image_column}

                            # URL is virtual for HF datasets
                            url = f"hf://{dataset_path}#{start_idx + items_processed}"
                            items_processed += 1
                            yield key, url, image_bytes, metadata
                        else:
                            logger.warning(f"Failed to process image for item {idx}")
                            items_processed += 1
                            continue
                    else:
                        logger.warning(
                            f"No image column '{self.image_column}' found in item {idx}. "
                            f"Available columns: {list(item.keys())}"
                        )
                        items_processed += 1

                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    items_processed += 1
                    continue

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {e}")
            return

    def count_shard_items(self, shard_url: str, processed_keys: Optional[set] = None) -> int:
        """Count items in a shard (can be slow for large shards)."""
        if shard_url.startswith("hf_dataset:"):
            # For virtual shards, return the chunk size
            _, start_idx, chunk_size = self._parse_virtual_shard(shard_url)

            # CRITICAL: Cap chunk size by dataset bounds
            if self._hf_total_items is not None:
                # If start index is beyond dataset, return 0
                if start_idx >= self._hf_total_items:
                    logger.warning(
                        f"Virtual shard starts at {start_idx} but dataset has "
                        f"only {self._hf_total_items} items"
                    )
                    return 0

                # Otherwise, return the minimum of chunk_size and remaining items
                remaining_items = self._hf_total_items - start_idx
                actual_size = min(chunk_size, remaining_items)
                logger.debug(
                    f"Virtual shard at {start_idx}: chunk_size={chunk_size}, "
                    f"remaining={remaining_items}, actual={actual_size}"
                )
                return actual_size
            else:
                # If we don't know total size, return chunk_size
                return chunk_size
        else:
            # Regular WebDataset counting
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
