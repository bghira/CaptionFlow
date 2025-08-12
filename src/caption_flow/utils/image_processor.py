"""Image preprocessing utilities."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Any

import numpy as np
from PIL import Image


class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, num_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    async def process_batch(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Process a batch of images in parallel."""
        loop = asyncio.get_event_loop()

        tasks = []
        for path in image_paths:
            task = loop.run_in_executor(self.executor, self._process_image, path)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    @staticmethod
    def _process_image(path: Path) -> np.ndarray:
        """Process a single image."""
        img = Image.open(path)

        # Resize to standard size
        img = img.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array
        arr = np.array(img, dtype=np.float32)

        # Normalize
        arr = arr / 255.0

        return arr

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
