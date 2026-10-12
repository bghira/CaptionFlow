"""A failed shard download must not turn into one remote request per image."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from caption_flow.processors.base import ProcessorConfig, WorkUnit
from caption_flow.processors.webdataset import WebDatasetWorkerProcessor


@pytest.fixture
def clock(monkeypatch):
    time = SimpleNamespace(now=0.0)
    monkeypatch.setattr("caption_flow.processors.webdataset.time.monotonic", lambda: time.now)
    monkeypatch.setattr(
        "caption_flow.processors.webdataset.time.sleep",
        lambda seconds: setattr(time, "now", time.now + seconds),
    )
    return time


def test_cached_shard_does_not_start_download(clock):
    processor = WebDatasetWorkerProcessor()
    processor.loader = Mock()
    processor.loader.get_shard_cache_status.return_value = {"is_cached": True}
    processor._wait_for_cached_shard("shard")
    processor.loader.prepare_shard_by_name.assert_not_called()
    assert clock.now == 0


def test_waits_for_download_before_reading_a_sample(clock):
    processor = WebDatasetWorkerProcessor()
    processor.decode_images = False
    loader = processor.loader = Mock()
    loader.get_shard_cache_status.side_effect = [{"is_cached": False}, {"is_cached": True}]

    def read(*args):
        assert clock.now > 0
        return SimpleNamespace(data=b"image", path="sample.jpg", size=5, metadata={})

    loader.load_sample.side_effect = read
    unit = WorkUnit(
        unit_id="shard:chunk:0",
        chunk_id="shard:chunk:0",
        source_id="shard",
        unit_size=1,
        data={
            "shard_name": "shard",
            "shard_idx": 0,
            "start_index": 0,
            "unprocessed_ranges": [(0, 0)],
        },
        metadata={"chunk_index": 0},
    )
    rows = list(processor.process_unit(unit, {}))
    assert len(rows) == 1 and rows[0]["item_index"] == 0
    loader.prepare_shard_by_name.assert_called_once_with("shard")


def test_timeout_leaves_every_sample_unread_and_available_for_retry(clock):
    processor = WebDatasetWorkerProcessor()
    processor.webshart_cache_wait_timeout = 1
    processor.loader = Mock()
    processor.loader.get_shard_cache_status.return_value = {"is_cached": False}
    unit = WorkUnit(
        unit_id="shard:chunk:0",
        chunk_id="shard:chunk:0",
        source_id="shard",
        unit_size=64,
        data={
            "shard_name": "shard",
            "shard_idx": 0,
            "start_index": 0,
            "unprocessed_ranges": [(0, 63)],
        },
        metadata={"chunk_index": 0},
    )
    with pytest.raises(TimeoutError, match="cached shard shard"):
        list(processor.process_unit(unit, {}))
    processor.loader.load_sample.assert_not_called()
    assert unit.data["unprocessed_ranges"] == [(0, 63)]
    assert clock.now == 1


def test_cache_download_retries_are_spaced_not_per_sample(clock):
    processor = WebDatasetWorkerProcessor()
    processor.webshart_cache_wait_timeout = 1
    processor.webshart_cache_retry_interval = 0.4
    processor.loader = Mock()
    processor.loader.get_shard_cache_status.side_effect = lambda name: {
        "is_cached": clock.now >= 0.9
    }
    processor._wait_for_cached_shard("shard")
    assert processor.loader.prepare_shard_by_name.call_count == 3


def test_explicit_remote_ranges_do_not_force_whole_shard_download(clock):
    processor = WebDatasetWorkerProcessor()
    processor.remote_range_reads = True
    processor.loader = Mock()
    processor._wait_for_cached_shard("shard")
    processor.loader.get_shard_cache_status.assert_not_called()


def test_older_webshart_remains_compatible(clock):
    processor = WebDatasetWorkerProcessor()
    processor.loader = SimpleNamespace(load_sample=Mock())
    processor._wait_for_cached_shard("shard")
    assert clock.now == 0


@pytest.mark.parametrize("field", ["webshart_cache_wait_timeout", "webshart_cache_retry_interval"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_cache_wait_configuration_is_rejected(field, value):
    processor = WebDatasetWorkerProcessor()
    with pytest.raises(ValueError, match="finite positive"):
        processor.initialize(ProcessorConfig(processor_type="webdataset", config={field: value}))
