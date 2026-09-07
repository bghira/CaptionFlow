"""Image preprocessing utilities."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Iterable, Sequence

import numpy as np
import trainingsample as tsr
from PIL import Image

from ..models import ProcessingItem

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, num_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    @staticmethod
    def prepare_for_inference(item: ProcessingItem) -> Image.Image:
        """Prepare image for inference.

        Args:
        ----
            image: PIL Image to prepare

        Returns:
        -------
            Prepared PIL Image

        """
        # We used to do a lot more hand-holding here with transparency, but oh well.
        logger.debug(f"Preparing item for inference: {item}")

        if item.image is not None:
            image = item.image
            item.metadata["image_width"], item.metadata["image_height"] = image.size
            item.metadata["image_format"] = image.format or "unknown"
            # item.image = None
            return image

        item.image = None
        image = ImageProcessor.decode_image_data(item.image_data)
        item.image_data = b""
        item.metadata["image_format"] = image.format or "unknown"
        item.metadata["image_width"], item.metadata["image_height"] = image.size

        return image

    @staticmethod
    def decode_image_data(image_data: bytes) -> Image.Image:
        """Decode encoded image bytes through the Rust-backed fast path."""
        try:
            decoded = tsr.imdecode_py(image_data, 1)
            return Image.fromarray(np.asarray(decoded))
        except Exception:
            logger.debug("trainingsample decode failed; falling back to Pillow", exc_info=True)
            with Image.open(BytesIO(image_data)) as source:
                return source.convert("RGB")

    @staticmethod
    def constrained_size(size: tuple[int, int], max_dimension: int) -> tuple[int, int]:
        """Preserve aspect ratio while bounding the longest image edge."""
        width, height = size
        longest = max(width, height)
        if longest <= max_dimension:
            return width, height
        scale = max_dimension / longest
        return max(1, round(width * scale)), max(1, round(height * scale))

    @staticmethod
    def preprocess_encoded_batch(
        image_buffers: Sequence[bytes], target_sizes: Sequence[tuple[int, int]]
    ) -> list[Image.Image]:
        """Decode and resize an encoded-image batch in the Rust extension."""
        if not image_buffers:
            return []
        if len(image_buffers) != len(target_sizes):
            raise ValueError("image_buffers and target_sizes must have the same length")

        processor = tsr.PyBatchProcessor.with_config(True, min(32, len(image_buffers)))
        arrays = processor.batch_preprocess_pipeline(
            list(image_buffers),
            list(target_sizes),
            None,
            1,
            tsr.INTER_LANCZOS4,
        )
        if len(arrays) != len(image_buffers):
            raise RuntimeError(
                f"trainingsample returned {len(arrays)} images for {len(image_buffers)} inputs"
            )
        return [Image.fromarray(np.asarray(array)) for array in arrays]

    @staticmethod
    def resize_images(
        images: Iterable[Image.Image], target_sizes: Sequence[tuple[int, int]]
    ) -> list[Image.Image]:
        """Resize PIL images as one Rust-backed batch, with a Pillow fallback."""
        image_list = list(images)
        if len(image_list) != len(target_sizes):
            raise ValueError("images and target_sizes must have the same length")
        if not image_list:
            return []

        try:
            arrays = [np.asarray(image.convert("RGB")) for image in image_list]
            resized = tsr.batch_resize_images(arrays, list(target_sizes))
            if len(resized) != len(image_list):
                raise RuntimeError(
                    f"trainingsample returned {len(resized)} images for {len(image_list)} inputs"
                )
            return [Image.fromarray(np.asarray(array)) for array in resized]
        except Exception:
            logger.debug("trainingsample resize failed; falling back to Pillow", exc_info=True)
            return [
                image.resize(size, Image.Resampling.LANCZOS)
                for image, size in zip(image_list, target_sizes, strict=True)
            ]

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
