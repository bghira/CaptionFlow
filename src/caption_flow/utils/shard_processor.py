"""Shard processing abstraction for different dataset types."""

import io
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple, Optional
from threading import Event

import webdataset as wds
from PIL import Image

from .dataset_loader import DatasetLoader
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ShardProcessor(ABC):
    """Abstract base for processing dataset shards."""

    @abstractmethod
    def iterate_chunk(
        self,
        chunk,
        dataset_loader: Optional[DatasetLoader],
        should_stop: Event,
        connected: Event,
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """
        Iterate through items in a chunk.

        Yields:
            Tuple of (key, url, image_data)
        """
        pass


class HFDatasetShardProcessor(ShardProcessor):
    """Processor for HuggingFace virtual dataset shards."""

    def iterate_chunk(
        self,
        chunk,
        dataset_loader: Optional[DatasetLoader],
        should_stop: Event,
        connected: Event,
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """Process HuggingFace virtual shard chunk."""
        if not dataset_loader:
            logger.error("No dataset loader configured for HuggingFace dataset shard")
            return

        items_processed = 0

        # Construct proper virtual shard URL
        parts = chunk.shard_url.split("_chunk_")
        if len(parts) == 2:
            base_path = parts[0]
            virtual_shard_url = f"{base_path}:chunk:{chunk.start_index}"
        else:
            virtual_shard_url = chunk.shard_url

        logger.debug(f"Using virtual shard URL: {virtual_shard_url}")

        # Iterate through the virtual shard
        for key, url, image_data in dataset_loader.iterate_shard(virtual_shard_url):
            # Check if we should stop
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping chunk processing early due to disconnect")
                break

            # Check if we've processed enough for this chunk
            if items_processed >= chunk.chunk_size:
                break

            items_processed += 1
            yield key, url, image_data


class WebDatasetShardProcessor(ShardProcessor):
    """Processor for WebDataset tar shards."""

    def __init__(self, hf_token: Optional[str] = None, dataset_type: str = "local"):
        self.hf_token = hf_token
        self.dataset_type = dataset_type

    def iterate_chunk(
        self,
        chunk,
        dataset_loader: Optional[DatasetLoader],
        should_stop: Event,
        connected: Event,
    ) -> Generator[Tuple[str, str, bytes], None, None]:
        """Process WebDataset shard chunk."""
        import shlex

        # Create WebDataset pipeline
        if self.dataset_type == "huggingface" and not chunk.shard_url.startswith("hf_dataset:"):
            # Use curl with auth for HuggingFace WebDataset
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.hf_token)}' {shlex.quote(chunk.shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp"),
            )
        else:
            # Local file
            ds = wds.DataPipeline(
                wds.SimpleShardList(chunk.shard_url),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp"),
            )

        # Process items
        items_processed = 0
        items_to_skip = chunk.start_index

        for key, image_data in ds:
            # Check if we should stop
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping chunk processing early due to disconnect")
                break

            # Skip to start index
            if items_to_skip > 0:
                items_to_skip -= 1
                continue

            # Check if we've processed enough
            if items_processed >= chunk.chunk_size:
                break

            items_processed += 1
            # URL is the shard URL for WebDataset
            yield key, chunk.shard_url, image_data
