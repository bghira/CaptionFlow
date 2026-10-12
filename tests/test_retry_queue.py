"""Failed chunks must not occupy the entire fresh-work buffer forever."""

from collections import deque
from unittest.mock import Mock

import pytest

from caption_flow.processors.webdataset import WebDatasetOrchestratorProcessor
from caption_flow.utils.retry_queue import RetryQueue


def test_retry_delay_grows_is_capped_and_completion_cancels():
    clock = Mock(return_value=100.0)
    queue = RetryQueue(5, 12, clock)
    for now, delay in [(100, 5), (105, 10), (115, 12), (127, 12)]:
        clock.return_value = now
        queue.defer("failed")
        assert queue.deadlines["failed"] == now + delay
        assert queue.ready() == []
        clock.return_value = now + delay
        assert queue.ready() == ["failed"]
        assert queue.is_retry("failed")
    queue.defer("failed")
    queue.complete("failed")
    clock.return_value = 1000
    assert queue.ready() == []
    assert not queue.is_retry("failed")
    assert len(queue) == 0


def test_repeated_failure_does_not_promote_stale_deadline():
    clock = Mock(return_value=0)
    queue = RetryQueue(clock=clock)
    queue.defer("a")
    queue.defer("a")
    clock.return_value = 5
    assert queue.ready() == []
    clock.return_value = 10
    assert queue.ready() == ["a"]
    assert queue.ready() == []


@pytest.mark.parametrize(
    "initial,maximum", [(0, 5), (-1, 5), (10, 5), (5, float("inf")), (float("nan"), 5)]
)
def test_invalid_delays_are_rejected(initial, maximum):
    with pytest.raises(ValueError, match="retry delays"):
        RetryQueue(initial, maximum)


def test_eight_failed_chunks_leave_capacity_for_fresh_work():
    processor = WebDatasetOrchestratorProcessor()
    clock = Mock(return_value=0)
    processor.retry_queue = RetryQueue(clock=clock)
    for index in range(8):
        unit_id = f"shard:chunk:{index}"
        processor.work_units[unit_id] = Mock()
        processor.assigned_units["worker"].add(unit_id)
        processor.mark_failed(unit_id, "worker", "HTTP 400")
    assert processor._fresh_buffer_size() == 0
    assert len(processor.retry_queue) == 8
    assert not processor.pending_units
    processor.pending_units.append("fresh")
    clock.return_value = 5
    processor._promote_ready_retries()
    assert processor.pending_units[0] == "fresh"
    assert len(processor.pending_units) == 9
    assert processor._fresh_buffer_size() == 1
    processor._promote_ready_retries()
    assert len(processor.pending_units) == 9


def test_promote_does_not_duplicate_assigned_or_pending_units():
    processor = WebDatasetOrchestratorProcessor()
    clock = Mock(return_value=0)
    processor.retry_queue = RetryQueue(clock=clock)
    for unit_id in ["assigned", "pending", "deleted"]:
        processor.retry_queue.defer(unit_id)
    processor.work_units = {"assigned": Mock(), "pending": Mock()}
    processor.assigned_units["worker"].add("assigned")
    processor.pending_units = deque(["pending"])
    clock.return_value = 10
    processor._promote_ready_retries()
    assert list(processor.pending_units) == ["pending"]


def test_completed_unit_restarts_backoff_without_reviving_stale_deadlines():
    clock = Mock(return_value=0)
    queue = RetryQueue(initial=5, maximum=20, clock=clock)
    queue.defer("a")
    clock.return_value = 1
    queue.defer("a")
    queue.complete("a")
    queue.complete("a")
    clock.return_value = 2
    queue.defer("a")
    assert queue.deadlines["a"] == 7
    assert queue.attempts["a"] == 1
    clock.return_value = 5
    assert queue.ready() == []
    clock.return_value = 7
    assert queue.ready() == ["a"]
    clock.return_value = 11
    assert queue.ready() == []


def test_ready_returns_all_due_units_once_without_promoting_future_retries():
    clock = Mock(return_value=0)
    queue = RetryQueue(initial=5, maximum=5, clock=clock)
    for name in ["a", "b", "cancelled"]:
        queue.defer(name)
    queue.complete("cancelled")
    clock.return_value = 3
    queue.defer("later")
    clock.return_value = 5
    assert set(queue.ready()) == {"a", "b"}
    assert queue.ready() == []
    assert len(queue) == 1
    clock.return_value = 8
    assert queue.ready() == ["later"]
    assert len(queue) == 0


def test_backoff_remains_bounded_after_many_failures():
    clock = Mock(return_value=0)
    queue = RetryQueue(initial=5, maximum=300, clock=clock)
    for _ in range(100):
        queue.defer("a")
    assert queue.attempts["a"] == 100
    assert queue.deadlines["a"] == 300
    clock.return_value = 300
    assert queue.ready() == ["a"]
