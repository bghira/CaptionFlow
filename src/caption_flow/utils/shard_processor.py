"""Shard processing abstraction for different dataset types."""

import io
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from threading import Event
import shlex

import webdataset as wds
from PIL import Image

from .dataset_loader import DatasetLoader

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

        # Get unprocessed ranges
        unprocessed_ranges = getattr(chunk, "unprocessed_ranges", [(0, chunk.chunk_size - 1)])

        logger.info(
            f"Processing HF dataset chunk {chunk.chunk_id} with ranges: {unprocessed_ranges}"
        )

        items_processed = 0
        current_idx = 0

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

            # Check if current index is in any unprocessed range
            in_range = any(start <= current_idx <= end for start, end in unprocessed_ranges)

            if not in_range:
                current_idx += 1
                continue  # Skip already processed items

            # Check if we've processed enough for this chunk
            if current_idx >= chunk.chunk_size:
                break

            items_processed += 1
            current_idx += 1
            yield key, url, image_data

        logger.info(
            f"HF dataset chunk {chunk.chunk_id}: yielded {items_processed} items "
            f"from ranges {unprocessed_ranges}"
        )

    def iterate_chunk_with_metadata(
        self,
        chunk,
        dataset_loader: Optional[DatasetLoader],
        should_stop: Event,
        connected: Event,
    ) -> Generator[Tuple[str, str, bytes, Dict[str, Any]], None, None]:
        """
        Process HuggingFace virtual shard chunk with metadata.

        Yields:
            Tuple of (key, url, image_data, metadata)
        """
        if not dataset_loader:
            logger.error("No dataset loader configured for HuggingFace dataset shard")
            return

        # Get unprocessed ranges
        unprocessed_ranges = getattr(chunk, "unprocessed_ranges", [(0, chunk.chunk_size - 1)])

        logger.info(
            f"Processing HF dataset chunk {chunk.chunk_id} with ranges: {unprocessed_ranges}"
        )

        items_processed = 0
        current_idx = 0

        # Construct proper virtual shard URL
        parts = chunk.shard_url.split("_chunk_")
        if len(parts) == 2:
            base_path = parts[0]
            virtual_shard_url = f"{base_path}:chunk:{chunk.start_index}"
        else:
            virtual_shard_url = chunk.shard_url

        logger.debug(f"Using virtual shard URL: {virtual_shard_url}")

        # Use the new iterate method that includes metadata
        for key, url, image_data, metadata in dataset_loader.iterate_shard_with_metadata(
            virtual_shard_url
        ):
            # Check if we should stop
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping chunk processing early due to disconnect")
                break

            # Check if current index is in any unprocessed range
            in_range = any(start <= current_idx <= end for start, end in unprocessed_ranges)

            if not in_range:
                current_idx += 1
                continue  # Skip already processed items

            # Check if we've processed enough for this chunk
            if current_idx >= chunk.chunk_size:
                break

            items_processed += 1
            current_idx += 1
            yield key, url, image_data, metadata

        logger.info(
            f"HF dataset chunk {chunk.chunk_id}: yielded {items_processed} items "
            f"from ranges {unprocessed_ranges}"
        )


class WebDatasetShardProcessor(ShardProcessor):
    """Processor for WebDataset tar shards with range support."""

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
        """Process WebDataset shard chunk with unprocessed ranges."""
        # Get unprocessed ranges
        unprocessed_ranges = getattr(chunk, "unprocessed_ranges", [(0, chunk.chunk_size - 1)])

        logger.info(
            f"Processing WebDataset chunk {chunk.chunk_id} with ranges: {unprocessed_ranges}"
        )

        # Create WebDataset pipeline
        if self.dataset_type == "huggingface" and not chunk.shard_url.startswith("hf_dataset:"):
            # Use curl with auth for HuggingFace WebDataset
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.hf_token)}' {shlex.quote(chunk.shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp;jxl"),
            )
        else:
            # Local file
            ds = wds.DataPipeline(
                wds.SimpleShardList(chunk.shard_url),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp;jxl"),
            )

        # Process items
        current_idx = 0
        items_yielded = 0

        for key, image_data in ds:
            # Check if we should stop
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping WebDataset chunk processing early due to disconnect")
                break

            # Calculate relative index within chunk
            relative_idx = current_idx - chunk.start_index

            # Skip items before chunk start
            if current_idx < chunk.start_index:
                current_idx += 1
                continue

            # Stop if beyond chunk
            if relative_idx >= chunk.chunk_size:
                break

            # Check if current index is in any unprocessed range
            in_range = any(start <= relative_idx <= end for start, end in unprocessed_ranges)

            if in_range:
                items_yielded += 1
                yield key, chunk.shard_url, image_data

            current_idx += 1

        logger.info(
            f"WebDataset chunk {chunk.chunk_id}: yielded {items_yielded} items "
            f"from ranges {unprocessed_ranges}"
        )

    def iterate_chunk_with_metadata(
        self,
        chunk,
        dataset_loader: Optional[DatasetLoader],
        should_stop: Event,
        connected: Event,
    ) -> Generator[Tuple[str, str, bytes, Dict[str, Any]], None, None]:
        """Process WebDataset shard chunk with metadata and range support."""
        # Get unprocessed ranges
        unprocessed_ranges = getattr(chunk, "unprocessed_ranges", [(0, chunk.chunk_size - 1)])

        logger.info(
            f"Processing WebDataset chunk {chunk.chunk_id} with ranges: {unprocessed_ranges}"
        )

        # Create WebDataset pipeline
        if self.dataset_type == "huggingface" and not chunk.shard_url.startswith("hf_dataset:"):
            # Use curl with auth for HuggingFace WebDataset
            url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(self.hf_token)}' {shlex.quote(chunk.shard_url)} || true"
            ds = wds.DataPipeline(
                wds.SimpleShardList(url_cmd),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp;jxl"),
            )
        else:
            # Local file
            ds = wds.DataPipeline(
                wds.SimpleShardList(chunk.shard_url),
                wds.tarfile_to_samples(),
                wds.to_tuple("__key__", "jpg;png;jpeg;webp;jxl"),
            )

        # Process items
        absolute_idx = 0  # Absolute index in the shard
        items_yielded = 0

        for key, image_data in ds:
            # Check if we should stop
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping WebDataset chunk processing early due to disconnect")
                break

            # Skip items before chunk start
            if absolute_idx < chunk.start_index:
                absolute_idx += 1
                continue

            # Calculate relative index within chunk
            relative_idx = absolute_idx - chunk.start_index

            # Stop if beyond chunk
            if relative_idx >= chunk.chunk_size:
                break

            # Check if current index is in any unprocessed range
            in_range = any(start <= relative_idx <= end for start, end in unprocessed_ranges)

            if in_range:
                # Create metadata with the relative index
                metadata = {
                    "_chunk_relative_index": relative_idx,
                }
                items_yielded += 1
                yield key, chunk.shard_url, image_data, metadata

            absolute_idx += 1

        logger.info(
            f"WebDataset chunk {chunk.chunk_id}: yielded {items_yielded} items "
            f"from ranges {unprocessed_ranges}"
        )
