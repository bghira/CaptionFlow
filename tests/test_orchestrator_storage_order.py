"""Restore scheduling only after durable captions have been loaded."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from caption_flow.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_processor_initializes_after_storage():
    events = []
    storage = Mock()
    storage.initialize = AsyncMock(side_effect=lambda: events.append("storage"))
    storage.get_all_processed_job_ids.return_value = set()
    processor = Mock()
    processor.initialize.side_effect = lambda *args: events.append("processor")
    with (
        patch("caption_flow.orchestrator.StorageManager", return_value=storage),
        patch("caption_flow.orchestrator.WebDatasetOrchestratorProcessor", return_value=processor),
    ):
        orchestrator = Orchestrator({"dataset": {"processor_type": "webdataset"}})
    processor.initialize.assert_not_called()
    orchestrator._heartbeat_loop = AsyncMock()
    orchestrator._checkpoint_loop = AsyncMock()
    orchestrator._stats_update_loop = AsyncMock()
    with patch("caption_flow.orchestrator.websockets.serve", side_effect=RuntimeError("test stop")):
        with pytest.raises(RuntimeError, match="test stop"):
            await orchestrator.start()
    await asyncio.sleep(0)
    assert events == ["storage", "processor"]
