"""Tests for Rust-backed image preprocessing."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from caption_flow.utils.image_processor import ImageProcessor


def encoded_image(size=(20, 10), image_format="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=image_format)
    return buffer.getvalue()


def test_decode_image_data_uses_trainingsample_and_falls_back_to_pillow():
    data = encoded_image()
    decoded = ImageProcessor.decode_image_data(data)
    assert decoded.mode == "RGB"
    assert decoded.size == (20, 10)

    with patch(
        "caption_flow.utils.image_processor.tsr.imdecode_py",
        side_effect=RuntimeError("unsupported"),
    ):
        fallback = ImageProcessor.decode_image_data(data)
    assert fallback.mode == "RGB"
    assert fallback.size == (20, 10)


def test_constrained_size_preserves_aspect_ratio():
    assert ImageProcessor.constrained_size((80, 40), 100) == (80, 40)
    assert ImageProcessor.constrained_size((200, 100), 100) == (100, 50)
    assert ImageProcessor.constrained_size((1, 1000), 1) == (1, 1)


def test_preprocess_encoded_batch_validates_and_converts_results():
    assert ImageProcessor.preprocess_encoded_batch([], []) == []
    with pytest.raises(ValueError, match="same length"):
        ImageProcessor.preprocess_encoded_batch([b"one"], [])

    processor = Mock()
    processor.batch_preprocess_pipeline.return_value = [np.zeros((5, 10, 3), dtype=np.uint8)]
    with patch(
        "caption_flow.utils.image_processor.tsr.PyBatchProcessor.with_config",
        return_value=processor,
    ) as factory:
        [image] = ImageProcessor.preprocess_encoded_batch([b"one"], [(10, 5)])
    factory.assert_called_once_with(True, 1)
    assert image.size == (10, 5)

    processor.batch_preprocess_pipeline.return_value = []
    with (
        patch(
            "caption_flow.utils.image_processor.tsr.PyBatchProcessor.with_config",
            return_value=processor,
        ),
        pytest.raises(RuntimeError, match="returned 0 images"),
    ):
        ImageProcessor.preprocess_encoded_batch([b"one"], [(10, 5)])


def test_resize_images_uses_batch_and_has_pillow_fallback():
    image = Image.new("RGB", (20, 10), color="green")
    assert ImageProcessor.resize_images([], []) == []
    with pytest.raises(ValueError, match="same length"):
        ImageProcessor.resize_images([image], [])

    resized = ImageProcessor.resize_images([image], [(10, 5)])
    assert resized[0].size == (10, 5)

    with patch(
        "caption_flow.utils.image_processor.tsr.batch_resize_images",
        side_effect=RuntimeError("unsupported"),
    ):
        fallback = ImageProcessor.resize_images([image], [(8, 4)])
    assert fallback[0].size == (8, 4)

    with patch("caption_flow.utils.image_processor.tsr.batch_resize_images", return_value=[]):
        mismatch_fallback = ImageProcessor.resize_images([image], [(6, 3)])
    assert mismatch_fallback[0].size == (6, 3)
