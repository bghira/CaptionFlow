"""Exercise local image responses without starting a background server."""

from unittest.mock import Mock, patch
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from caption_flow.processors.local_filesystem import LocalFilesystemOrchestratorProcessor


@pytest.mark.parametrize("filename", ["image.png", "猫 🌈.png", 'a"b.png'])
def test_image_http_response_handles_filenames(tmp_path, filename):
    path = tmp_path / filename
    Image.new("RGB", (8, 8), "red").save(path)
    processor = LocalFilesystemOrchestratorProcessor()
    processor.all_images = [(path, path.stat().st_size)]
    processor.http_public_address = "localhost"
    loop = Mock()
    loop.create_task.side_effect = lambda coroutine: coroutine.close()
    with (
        patch("caption_flow.processors.local_filesystem.asyncio.new_event_loop", return_value=loop),
        patch("caption_flow.processors.local_filesystem.threading.Thread"),
    ):
        processor._start_http_server()

    with TestClient(processor.http_app) as client:
        response = client.get("/image/0")

    assert response.status_code == 200
    assert response.content == path.read_bytes()
    assert response.headers["content-type"] == "image/png"
    disposition = response.headers["content-disposition"]
    if filename == "image.png":
        assert disposition == 'inline; filename="image.png"'
    else:
        assert disposition == f"inline; filename*=utf-8''{quote(filename)}"
