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


logger = logging.getLogger(__name__)


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

    @staticmethod
    def process_image_data(img_data: Union[str, bytes, Image.Image]) -> Optional[bytes]:
        """
        Process various types of image data into bytes.

        Args:
            img_data: Can be a URL string, bytes, or PIL Image

        Returns:
            Image data as bytes, or None if processing failed
        """
        try:
            if isinstance(img_data, str):
                # It's a URL - download the image
                try:
                    # Download with timeout
                    response = requests.get(
                        img_data,
                        timeout=30,
                        headers={"User-Agent": "Mozilla/5.0 (captionflow-dataset-loader)"},
                    )
                    response.raise_for_status()
                    image_data = response.content

                    # Verify it's an image by trying to open it
                    img = Image.open(BytesIO(image_data))
                    img.verify()  # Verify it's a valid image

                    return image_data

                except Exception as e:
                    logger.error(f"Failed to download image from {img_data}: {e}")
                    return None

            elif hasattr(img_data, "__class__") and "Image" in str(img_data.__class__):
                # It's a PIL Image object
                import io

                # Save as PNG bytes
                img_bytes = io.BytesIO()
                # Convert to RGB
                img_data = img_data.convert("RGB")
                img_data.save(img_bytes, format="PNG")
                return img_bytes.getvalue()

            elif isinstance(img_data, bytes):
                # Already bytes - validate it's an image
                try:
                    img = Image.open(BytesIO(img_data))
                    img.verify()
                    return img_data
                except Exception as e:
                    logger.error(f"Invalid image data: {e}")
                    return None

            else:
                logger.warning(f"Unknown image data type: {type(img_data)}")
                return None

        except Exception as e:
            logger.error(f"Error processing image data: {e}", exc_info=True)
            return None

    @staticmethod
    def prepare_for_inference(image: Image.Image) -> Image.Image:
        """
        Prepare image for inference, handling transparency and mostly black/white images.

        Args:
            image: PIL Image to prepare

        Returns:
            Prepared PIL Image
        """
        # Convert to RGBA to handle transparency
        img_rgba = image.convert("RGBA")
        rgb_img = img_rgba.convert("RGB")
        np_img = np.array(rgb_img)

        # Calculate percentage of pixels that are (0,0,0) or (255,255,255)
        total_pixels = np_img.shape[0] * np_img.shape[1]
        black_pixels = np.all(np_img == [0, 0, 0], axis=-1).sum()
        white_pixels = np.all(np_img == [255, 255, 255], axis=-1).sum()
        black_pct = black_pixels / total_pixels
        white_pct = white_pixels / total_pixels

        threshold = 0.90  # 90% threshold

        is_mostly_black = black_pct >= threshold
        is_mostly_white = white_pct >= threshold

        if is_mostly_black or is_mostly_white:
            # Replace background with opposite color for better contrast
            bg_color = (255, 255, 255) if is_mostly_black else (0, 0, 0)
            background = Image.new("RGB", img_rgba.size, bg_color)
            # Use alpha channel as mask if present
            if img_rgba.mode == "RGBA":
                background.paste(img_rgba.convert("RGB"), mask=img_rgba.split()[3])
            else:
                background.paste(img_rgba.convert("RGB"))

            color_type = "black" if is_mostly_black else "white"
            pct = black_pct if is_mostly_black else white_pct
            logger.debug(
                f"Image is {pct*100:.1f}% {color_type}; background replaced with {bg_color}"
            )

            return background
        else:
            return rgb_img

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
