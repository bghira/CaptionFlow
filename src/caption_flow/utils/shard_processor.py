"""Shard processing abstraction for different dataset types."""

import io
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datasets import load_dataset
from .image_processor import ImageProcessor
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
        Process HuggingFace virtual shard chunk with metadata, range by range.
        """
        if not dataset_loader:
            logger.error("No dataset loader configured for HuggingFace dataset shard")
            return

        # Get unprocessed ranges
        unprocessed_ranges = getattr(chunk, "unprocessed_ranges", [(0, chunk.chunk_size - 1)])

        logger.info(
            f"Processing HF dataset chunk {chunk.chunk_id} with {len(unprocessed_ranges)} ranges"
        )

        items_yielded = 0

        # Process each range independently with its own iterator
        for range_start, range_end in unprocessed_ranges:
            if should_stop.is_set() or not connected.is_set():
                logger.info(f"Stopping chunk processing early due to disconnect")
                break

            # Calculate absolute indices for this range
            abs_start = chunk.start_index + range_start
            abs_end = chunk.start_index + range_end
            range_size = range_end - range_start + 1

            logger.debug(
                f"Processing range [{range_start}, {range_end}] "
                f"(absolute: [{abs_start}, {abs_end}])"
            )

            try:
                # Create a fresh dataset iterator for this range
                dataset = load_dataset(
                    dataset_loader.dataset_path,
                    split=dataset_loader.split,
                    streaming=True,
                    token=dataset_loader.token,
                )

                # Use state_dict if available for efficient positioning
                if hasattr(dataset, "load_state_dict") and hasattr(dataset, "state_dict"):
                    try:
                        state = dataset.state_dict()
                        # Modify state to jump to abs_start
                        if "num_examples_since_previous_state" in state:
                            state["num_examples_since_previous_state"] = abs_start
                        if "examples_iterable" in state and isinstance(
                            state["examples_iterable"], dict
                        ):
                            if "shard_example_idx" in state["examples_iterable"]:
                                state["examples_iterable"]["shard_example_idx"] = abs_start
                        dataset.load_state_dict(state)
                        logger.debug(f"Positioned dataset at index {abs_start} using state_dict")
                    except Exception as e:
                        logger.debug(f"Could not use state_dict, falling back to skip: {e}")
                        dataset = dataset.skip(abs_start)
                else:
                    # Fall back to skip
                    dataset = dataset.skip(abs_start)

                # Process items in this range
                range_items = 0
                for item in dataset:
                    if range_items >= range_size:
                        break

                    if should_stop.is_set() or not connected.is_set():
                        break

                    # Generate key for this item
                    current_abs_idx = abs_start + range_items
                    key = f"{dataset_loader.dataset_path.replace('/', '_')}_{current_abs_idx:08d}"

                    try:
                        if dataset_loader.image_column in item:
                            img_data = item[dataset_loader.image_column]
                            image_bytes = ImageProcessor.process_image_data(img_data)

                            if image_bytes:
                                # Extract metadata
                                metadata = {
                                    k: v
                                    for k, v in item.items()
                                    if k != dataset_loader.image_column
                                }
                                # Add chunk-relative index to metadata
                                metadata["_chunk_relative_index"] = range_start + range_items

                                url = f"hf://{dataset_loader.dataset_path}#{current_abs_idx}"

                                items_yielded += 1
                                range_items += 1

                                yield key, url, image_bytes, metadata
                            else:
                                logger.warning(
                                    f"Failed to process image at index {current_abs_idx}"
                                )
                                range_items += 1
                        else:
                            logger.warning(
                                f"No image column '{dataset_loader.image_column}' at index {current_abs_idx}"
                            )
                            range_items += 1

                    except Exception as e:
                        logger.error(f"Error processing item at index {current_abs_idx}: {e}")
                        range_items += 1
                        continue

            except Exception as e:
                logger.error(f"Error processing range [{range_start}, {range_end}]: {e}")
                continue

        logger.info(
            f"HF dataset chunk {chunk.chunk_id}: yielded {items_yielded} items "
            f"from {len(unprocessed_ranges)} ranges"
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
