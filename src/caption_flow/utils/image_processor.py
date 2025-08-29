"""Image preprocessing utilities."""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List, Any, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image
from ..models import ProcessingItem


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, num_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    @staticmethod
    def prepare_for_inference(item: ProcessingItem) -> Image.Image:
        """
        Prepare image for inference.

        Args:
            image: PIL Image to prepare

        Returns:
            Prepared PIL Image
        """
        # We used to do a lot more hand-holding here with transparency, but oh well.

        if item.image is not None:
            return item.image

        return Image.open(BytesIO(item.image_data))

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
