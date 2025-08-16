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

        # Use streaming mode for HuggingFace datasets to avoid loading everything into memory
        logger.info(
            f"Loading HuggingFace dataset in streaming mode: {dataset_path} (split: {self.split})"
        )
        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                dataset_path,
                split=self.split,
                streaming=True,  # This is the key change
                token=self.token,
            )
            logger.info("Loaded Huggingface dataset.")
            # Skip to start index and process chunk_size items
            items_processed = 0

            # attempt to set the starting index for iteration
            if start_idx > 0:
                dataset.skip(start_idx)

            logger.info(f"Skipping {start_idx} irrelevant chunk samples..")
            for idx, item in enumerate(dataset):
                # Skip items before our start index
                if idx < start_idx:
                    continue

                # Stop after processing chunk_size items
                if items_processed >= chunk_size:
                    logger.info(f"Skipping, {items_processed=} >= {chunk_size=}")
                    break

                # Generate a unique key for this item
                key = f"{dataset_path}_{start_idx + items_processed:08d}"

                if key in processed_keys:
                    items_processed += 1
                    logger.info(f"Skipping, {key=} in processed keys")
                    continue

                try:
                    # Extract image data - check configured column name
                    if self.image_column in item:
                        img_data = item[self.image_column]

                        # Handle different types of image data
                        if isinstance(img_data, str):
                            # It's a URL - download the image
                            try:
                                import requests
                                from io import BytesIO

                                # Download with timeout
                                response = requests.get(
                                    img_data,
                                    timeout=30,
                                    headers={
                                        "User-Agent": "Mozilla/5.0 (captionflow-dataset-loader)"
                                    },
                                )
                                response.raise_for_status()
                                image_data = response.content

                                # Verify it's an image by trying to open it
                                from PIL import Image

                                img = Image.open(BytesIO(image_data))
                                img.verify()  # Verify it's a valid image

                            except Exception as e:
                                logger.error(f"Failed to download image from {img_data}: {e}")
                                import traceback

                                logger.error(traceback.format_exc())
                                # Skip this item
                                items_processed += 1
                                continue

                        elif hasattr(img_data, "__class__") and "Image" in str(img_data.__class__):
                            # It's a PIL Image object
                            import io
                            from PIL import Image

                            # Save as PNG bytes
                            img_bytes = io.BytesIO()
                            # Convert to RGB
                            img_data = img_data.convert("RGB")
                            img_data.save(img_bytes, format="PNG")
                            image_data = img_bytes.getvalue()

                        elif isinstance(img_data, bytes):
                            # Already bytes
                            image_data = img_data

                        else:
                            logger.warning(
                                f"Unknown image data type for item {idx}: {type(img_data)}"
                            )
                            items_processed += 1
                            continue

                        # URL is virtual for HF datasets
                        url = f"hf://{dataset_path}#{start_idx + items_processed}"

                        items_processed += 1
                        yield key, url, image_data

                    else:
                        # Try common column names if configured one doesn't exist
                        found = False
                        for col in ["image", "url", "image_url", "img", "image_path"]:
                            if col in item:
                                logger.warning(
                                    f"Column '{self.image_column}' not found, but '{col}' exists. "
                                    f"Consider setting image_column='{col}' in config."
                                )
                                found = True
                                break

                        if not found:
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

            # If we have the total dataset size, cap it appropriately
            if self._hf_total_items is not None:
                return min(chunk_size, max(0, self._hf_total_items - start_idx))
            else:
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
