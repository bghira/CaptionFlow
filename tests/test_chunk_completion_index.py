"""A bounded in-memory chunk cache must not forget completed work."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

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


def test_storage_reconciliation_does_not_recreate_evicted_completed_chunks(tmp_path):
    processor = WebDatasetOrchestratorProcessor()
    tracker = processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    chunk_id = "shard:chunk:0"
    tracker.add_chunk(chunk_id, "shard", "shard.tar", 0, 2)
    tracker.mark_items_processed(chunk_id, 0, 1)
    del tracker.chunks[chunk_id]
    processor.dataset = Mock()

    processor.update_from_storage({f"{chunk_id}:idx:0", f"{chunk_id}:idx:1"})

    assert chunk_id not in tracker.chunks
    assert tracker.is_chunk_completed(chunk_id)
    processor.dataset.get_shard_info.assert_not_called()
    assert tracker.get_stats()["total_completed"] == 1
    assert ChunkTracker(tmp_path / "chunks.json").is_chunk_completed(chunk_id)


def test_assignment_repairs_fully_processed_pending_chunk_and_cancels_retry(tmp_path):
    processor = WebDatasetOrchestratorProcessor()
    tracker = processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    chunk_id = "shard:chunk:1"
    tracker.add_chunk(chunk_id, "shard", "shard.tar", 2, 2)
    # A legacy checkpoint can contain every processed range but a stale status.
    tracker.chunks[chunk_id].processed_ranges = [(0, 1)]
    tracker.chunks[chunk_id].processed_count = 2
    processor.work_units[chunk_id] = Mock()
    processor.pending_units.append(chunk_id)
    processor.retry_queue.defer(chunk_id)

    assert processor.get_work_units(1, "worker") == []

    assert tracker.is_chunk_completed(chunk_id)
    assert chunk_id not in processor.work_units
    assert not processor.pending_units
    assert not processor.assigned_units["worker"]
    assert not processor.retry_queue.is_retry(chunk_id)
    assert len(processor.retry_queue) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("with_item_index", [False, True])
async def test_lance_sync_remembers_completion_after_eviction_and_reload(tmp_path, with_item_index):
    tracker = ChunkTracker(tmp_path / "chunks.json", max_completed_chunks_in_memory=0)
    tracker.add_chunk("shard:chunk:2", "shard", "shard.tar", 10, 4)
    tracker.mark_items_processed("shard:chunk:2", 11, 11)
    tracker.add_chunk("partial", "shard", "shard.tar", 20, 4)
    tracker.add_chunk("done", "shard", "shard.tar", 30, 1)
    tracker.mark_completed("done")
    rows = []
    for chunk_id, indices in [
        ("shard:chunk:2", [9, 10, 12, 13, 14]),
        ("partial", [20, 22]),
        ("done", [30]),
        ("unknown", [40]),
    ]:
        for index in indices:
            row = {
                "job_id": f"{chunk_id}:idx:{index}",
                "chunk_id": chunk_id,
                "item_key": f"image_{index}",
            }
            if with_item_index:
                row["item_index"] = index
            rows.append(row)
    storage = Mock(captions_path=tmp_path / "captions.lance")
    lance.write_dataset(pa.Table.from_pylist(rows), storage.captions_path)

    await tracker.sync_with_storage(storage)
    # Repeat sync to check that completion counters and ranges stay idempotent.
    await tracker.sync_with_storage(storage)

    assert tracker.is_chunk_completed("shard:chunk:2")
    assert tracker.chunks["shard:chunk:2"].processed_count == 4
    assert tracker.chunks["partial"].get_unprocessed_ranges() == [(1, 1), (3, 3)]
    assert not tracker.is_chunk_completed("partial")
    assert "unknown" not in tracker.chunks
    tracker._limit_completed_chunks_in_memory()
    tracker.flush()
    restored = ChunkTracker(tmp_path / "chunks.json")
    assert set(restored.chunks) == {"partial"}
    assert restored.completed_chunk_ids == {"done", "shard:chunk:2"}
    assert restored.get_stats()["total_completed"] == 2


def test_legacy_checkpoint_rebuilds_completion_identity_without_trusting_count(tmp_path):
    path = tmp_path / "chunks.json"
    tracker = ChunkTracker(path)
    tracker.add_chunk("done", "shard", "shard.tar", 0, 2)
    tracker.mark_completed("done")
    tracker.add_chunk("pending", "shard", "shard.tar", 2, 2)
    tracker.flush()
    legacy = json.loads(path.read_text())
    del legacy["completed_chunk_ids"]
    legacy["completed_count"] = 99
    path.write_text(json.dumps(legacy))

    restored = ChunkTracker(path)

    assert restored.completed_chunk_ids == {"done"}
    assert not restored.is_chunk_completed("pending")
    assert restored.get_stats()["total_completed"] == 1


def test_bulk_update_flushes_progress_and_recovers_depth_after_exception(tmp_path):
    path = tmp_path / "chunks.json"
    tracker = ChunkTracker(path, save_batch_size=1)
    with pytest.raises(RuntimeError, match="interrupted reconciliation"):
        with tracker.bulk_update(), tracker.bulk_update():
            tracker.add_chunk("a", "shard", "shard.tar", 0, 2)
            tracker.mark_items_processed("a", 0, 0)
            raise RuntimeError("interrupted reconciliation")
    assert ChunkTracker(path).chunks["a"].get_unprocessed_ranges() == [(1, 1)]
    # A failed batch must not suppress automatic checkpoints forever.
    tracker.mark_items_processed("a", 1, 1)
    assert ChunkTracker(path).is_chunk_completed("a")
