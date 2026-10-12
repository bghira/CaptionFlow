"""Indexed shard lookahead must not change reads or make prefetch mandatory."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from caption_flow.processors.base import ProcessorConfig, WorkUnit
from caption_flow.processors.webdataset import WebDatasetWorkerProcessor


@pytest.fixture
def processor(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("caption_flow.processors.webdataset.time.monotonic", lambda: clock.now)
    result = WebDatasetWorkerProcessor()
    result.webshart_prefetch_shards = 2
    result.dataset = Mock(num_shards=5)
    result.dataset.get_shard_info.side_effect = lambda index: {"name": f"shard-{index}"}
    result.loader = Mock()
    result.loader.get_shard_cache_status.return_value = {"is_cached": False}
    return result, clock


def test_prefetch_uses_assignment_index_without_moving_loader_cursor(processor):
    worker, _ = processor
    worker._prefetch_upcoming_shards(1)
    assert [call.args for call in worker.loader.prepare_shard_by_name.call_args_list] == [
        ("shard-2",),
        ("shard-3",),
    ]
    worker.loader.shard.assert_not_called()
    worker.loader.load_sample.assert_not_called()


def test_prefetch_stops_at_dataset_end(processor):
    worker, _ = processor
    worker._prefetch_upcoming_shards(3)
    worker.loader.prepare_shard_by_name.assert_called_once_with("shard-4")
    worker._prefetch_upcoming_shards(4)
    assert worker.loader.prepare_shard_by_name.call_count == 1


def test_cached_shards_are_not_downloaded_again(processor):
    worker, _ = processor
    worker.loader.get_shard_cache_status.return_value = {"is_cached": True}
    worker._prefetch_upcoming_shards(1)
    worker.loader.prepare_shard_by_name.assert_not_called()


def test_prefetch_retries_are_throttled_across_units(processor):
    worker, clock = processor
    worker._prefetch_upcoming_shards(0)
    worker._prefetch_upcoming_shards(0)
    clock.now = 29.9
    worker._prefetch_upcoming_shards(0)
    assert worker.loader.prepare_shard_by_name.call_count == 2
    clock.now = 30.0
    worker._prefetch_upcoming_shards(0)
    assert worker.loader.prepare_shard_by_name.call_count == 4


@pytest.mark.parametrize(
    "mode", ["disabled", "ranges", "missing_index", "missing_dataset", "older_loader"]
)
def test_prefetch_is_opt_in_and_compatible(processor, mode):
    worker, _ = processor
    loader = worker.loader
    index = 0
    if mode == "disabled":
        worker.webshart_prefetch_shards = 0
    elif mode == "ranges":
        worker.remote_range_reads = True
    elif mode == "missing_index":
        index = None
    elif mode == "missing_dataset":
        worker.dataset = None
    else:
        worker.loader = SimpleNamespace(load_sample=Mock())
    worker._prefetch_upcoming_shards(index)
    loader.prepare_shard_by_name.assert_not_called()


@pytest.mark.parametrize(
    "method", ["get_shard_info", "get_shard_cache_status", "prepare_shard_by_name"]
)
def test_prefetch_failure_is_nonfatal_and_does_not_log_signed_urls(processor, caplog, method):
    worker, _ = processor
    owner = worker.dataset if method == "get_shard_info" else worker.loader
    getattr(owner, method).side_effect = RuntimeError("https://example.test/?secret=test")
    worker._prefetch_upcoming_shards(0)
    assert "Webshart prefetch failed" in caplog.text
    assert "secret=test" not in caplog.text


def test_prefetch_attempt_history_is_bounded(processor):
    worker, _ = processor
    worker.dataset.num_shards = 100
    for index in range(50):
        worker._prefetch_upcoming_shards(index)
    assert len(worker._prefetch_attempts) <= 8


def test_current_sample_is_read_even_when_prefetch_fails(processor):
    worker, _ = processor
    worker.decode_images = False
    worker.loader.get_shard_cache_status.side_effect = lambda name: {"is_cached": name == "shard-0"}
    worker.loader.prepare_shard_by_name.side_effect = RuntimeError("download failed")
    worker.loader.load_sample.return_value = SimpleNamespace(
        data=b"image", path="sample.jpg", size=5, metadata={}
    )
    unit = WorkUnit(
        unit_id="shard-0:chunk:0",
        chunk_id="shard-0:chunk:0",
        source_id="shard-0",
        unit_size=1,
        data={
            "shard_name": "shard-0",
            "shard_idx": 0,
            "start_index": 0,
            "unprocessed_ranges": [(0, 0)],
        },
        metadata={"chunk_index": 0},
    )
    rows = list(worker.process_unit(unit, {}))
    assert len(rows) == 1 and rows[0]["item_index"] == 0
    worker.loader.load_sample.assert_called_once_with(0, 0)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2", None])
def test_invalid_prefetch_configuration_is_rejected(value):
    worker = WebDatasetWorkerProcessor()
    with pytest.raises(ValueError, match="nonnegative integer"):
        worker.initialize(
            ProcessorConfig(processor_type="webdataset", config={"webshart_prefetch_shards": value})
        )


def test_prefetch_defaults_to_disabled():
    assert WebDatasetWorkerProcessor().webshart_prefetch_shards == 0
