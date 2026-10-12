"""Completion cannot outrun result delivery or storage acceptance."""

import asyncio
import json
import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from caption_flow.orchestrator import Orchestrator
from caption_flow.processors.base import WorkResult
from caption_flow.processors.webdataset import WebDatasetOrchestratorProcessor
from caption_flow.utils.chunk_tracker import ChunkTracker
from caption_flow.workers.caption import CaptionWorker


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [True, False])
async def test_acceptance_precedes_progress_even_for_duplicates(saved):
    events = []
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.processor = Mock()
    orchestrator.processor.handle_result.side_effect = lambda result: events.append("progress")
    orchestrator.storage = Mock()

    async def save(caption):
        events.append("storage")
        return saved

    orchestrator.storage.save_caption = AsyncMock(side_effect=save)
    orchestrator.storage.get_contributor = AsyncMock(return_value=None)
    await orchestrator._handle_results_submission("worker_1", result_message())
    assert events == ["storage", "progress"]
    orchestrator.processor.mark_failed.assert_not_called()


def result_message():
    return dict(
        unit_id="00000:chunk:0",
        job_id="00000:chunk:0:idx:0",
        sample_id="image",
        dataset="dataset",
        outputs={"captions": ["A tree."]},
        metadata={"_item_index": 0},
    )


@pytest.mark.asyncio
async def test_storage_error_does_not_acknowledge_progress():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.processor = Mock()
    orchestrator.storage = Mock()
    orchestrator.storage.save_caption = AsyncMock(side_effect=ValueError("failed write"))
    await orchestrator._handle_results_submission("worker_1", result_message())
    orchestrator.processor.handle_result.assert_not_called()
    orchestrator.processor.mark_failed.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [{"outputs": {}}, {"error": "Generation failed"}])
async def test_failed_results_never_create_empty_completed_rows(failure):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.processor = Mock()
    orchestrator.storage = Mock()
    orchestrator.storage.save_caption = AsyncMock()
    await orchestrator._handle_results_submission("worker_1", {**result_message(), **failure})
    orchestrator.storage.save_caption.assert_not_called()
    orchestrator.processor.handle_result.assert_not_called()
    orchestrator.processor.mark_failed.assert_called_once()


@pytest.mark.asyncio
async def test_rejected_completion_does_not_announce_success():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.processor = Mock()
    orchestrator.processor.mark_completed.return_value = False
    orchestrator._send_activity = AsyncMock()
    await orchestrator._process_worker_message("worker", {"type": "work_complete", "unit_id": "u"})
    orchestrator._send_activity.assert_not_called()


def test_premature_completion_keeps_missing_results_retryable(tmp_path):
    processor = WebDatasetOrchestratorProcessor()
    tracker = processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    unit = "00000:chunk:0"
    tracker.add_chunk(unit, "00000", "shard.tar", 0, 2)
    processor.work_units[unit] = Mock()
    processor.assigned_units["worker"].add(unit)
    tracker.mark_items_processed(unit, 0, 0)
    assert processor.mark_completed(unit, "worker") is False
    assert not tracker.is_chunk_completed(unit)
    assert tracker.chunks[unit].get_unprocessed_ranges() == [(1, 1)]
    assert processor.retry_queue.is_retry(unit)
    processor.handle_result(
        WorkResult(
            unit_id=unit,
            source_id="00000",
            chunk_id=unit,
            sample_id="1",
            outputs={"captions": ["A tree."]},
            metadata={"_item_index": 1},
        )
    )
    assert tracker.is_chunk_completed(unit)
    assert unit not in processor.work_units
    assert not processor.retry_queue.is_retry(unit)
    assert processor.mark_completed(unit, "worker") is True
    processor.mark_failed(unit, "worker", "Late failure")
    assert tracker.is_chunk_completed(unit)
    assert not processor.retry_queue.is_retry(unit)


@pytest.mark.asyncio
async def test_result_sender_preserves_control_order():
    worker = CaptionWorker.__new__(CaptionWorker)
    worker.running = True
    worker.connected = threading.Event()
    worker.connected.set()
    worker.result_queue = Queue()
    worker.items_processed = 1
    worker.dataset_path = "dataset"
    item = SimpleNamespace(
        unit_id="u", chunk_id="c", item_key="i", job_id="j", image=None, image_data=b"", metadata={}
    )
    worker.result_queue.put(
        dict(item=item, outputs={"captions": ["A tree."]}, processing_time_ms=1)
    )
    worker.result_queue.put({"control": {"type": "work_complete", "unit_id": "u"}})
    messages = []

    async def send(message):
        messages.append(json.loads(message))
        if len(messages) == 2:
            worker.running = False

    worker.websocket = SimpleNamespace(send=send)
    await asyncio.wait_for(worker._result_sender(), timeout=2)
    assert [message["type"] for message in messages] == ["submit_results", "work_complete"]


@pytest.mark.parametrize("failed", [0, 1])
def test_work_unit_queues_its_outcome_after_results(failed):
    worker = CaptionWorker.__new__(CaptionWorker)
    worker.processor = Mock()
    worker.processor.process_unit.return_value = []
    worker.vllm_config = {}
    worker.should_stop_processing = threading.Event()
    worker.connected = threading.Event()
    worker.connected.set()
    worker.websocket = Mock()
    worker.result_queue = Queue()
    worker.result_queue.put({"earlier_result": True})
    unit = SimpleNamespace(unit_id="u", unit_size=failed)
    worker._process_work_unit(unit)
    assert worker.result_queue.get_nowait() == {"earlier_result": True}
    message = worker.result_queue.get_nowait()["control"]
    assert message["type"] == ("work_failed" if failed else "work_complete")
    worker.websocket.send.assert_not_called()
