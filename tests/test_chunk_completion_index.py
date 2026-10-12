"""A bounded in-memory chunk cache must not forget completed work."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from caption_flow.processors.webdataset import WebDatasetOrchestratorProcessor
from caption_flow.utils.chunk_tracker import ChunkTracker


def test_completed_index_survives_eviction_and_reload(tmp_path):
    tracker = ChunkTracker(tmp_path / "chunks.json", max_completed_chunks_in_memory=1)
    with tracker.bulk_update():
        for index in range(4):
            name = f"shard:chunk:{index}"
            tracker.add_chunk(name, "shard", "shard.tar", index * 10, 10)
            tracker.mark_items_processed(name, index * 10, index * 10 + 9)
        tracker._limit_completed_chunks_in_memory()
    assert len(tracker.chunks) == 1
    assert len(tracker.completed_chunk_ids) == 4
    restored = ChunkTracker(tmp_path / "chunks.json")
    for index in range(4):
        name = f"shard:chunk:{index}"
        assert restored.is_chunk_completed(name)
        assert not restored.add_chunk(name, "shard", "shard.tar", index * 10, 10)
    assert restored.get_stats()["total_completed"] == 4


def test_archive_keeps_identity_and_explicit_failure_clears_it(tmp_path):
    tracker = ChunkTracker(tmp_path / "chunks.json")
    tracker.add_chunk("old", "shard", "shard.tar", 0, 10)
    tracker.mark_completed("old")
    tracker.chunks["old"].completed_at = datetime.now(timezone.utc) - timedelta(days=2)
    tracker._archive_old_completed_chunks()
    assert "old" not in tracker.chunks
    assert tracker.is_chunk_completed("old")
    tracker.add_chunk("retry", "shard", "shard.tar", 10, 10)
    tracker.mark_completed("retry")
    tracker.mark_failed("retry")
    assert not tracker.is_chunk_completed("retry")


def test_bulk_updates_save_once_even_when_nested(tmp_path):
    tracker = ChunkTracker(tmp_path / "chunks.json", save_batch_size=1)
    tracker._do_save = Mock(return_value=True)
    with tracker.bulk_update():
        tracker.add_chunk("a", "shard", "shard.tar", 0, 10)
        with tracker.bulk_update():
            tracker.mark_items_processed("a", 0, 9)
        tracker._do_save.assert_not_called()
    tracker._do_save.assert_called_once()


def test_assignment_discards_work_for_evicted_completed_chunks(tmp_path):
    processor = WebDatasetOrchestratorProcessor()
    processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    processor.chunk_tracker.completed_chunk_ids.add("done")
    processor.work_units["done"] = Mock()
    processor.pending_units.append("done")
    assert processor.get_work_units(1, "worker") == []
    assert "done" not in processor.work_units


def test_storage_rebuild_removes_acknowledged_but_unflushed_completion(tmp_path):
    tracker = ChunkTracker(tmp_path / "chunks.json")
    tracker.add_chunk("unflushed", "shard", "shard.tar", 0, 10)
    tracker.mark_completed("unflushed")
    tracker.reset_for_storage_rebuild()
    assert not tracker.is_chunk_completed("unflushed")
    assert not tracker.chunks
    assert tracker.get_stats()["total_completed"] == 0
