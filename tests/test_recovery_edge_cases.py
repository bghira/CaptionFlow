"""Recovery must preserve unfinished work and contributor identity."""

from datetime import datetime

import pytest

from caption_flow.models import Caption, Contributor
from caption_flow.processors.base import WorkResult, WorkUnit
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor
from caption_flow.processors.local_filesystem import LocalFilesystemOrchestratorProcessor
from caption_flow.storage import StorageManager
from caption_flow.utils.chunk_tracker import ChunkTracker


def local_processor(tmp_path):
    processor = LocalFilesystemOrchestratorProcessor()
    processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    unit_id = "local:chunk:0"
    processor.chunk_tracker.add_chunk(unit_id, "local", "images", 0, 2)
    processor.work_units[unit_id] = WorkUnit(
        unit_id=unit_id,
        chunk_id=unit_id,
        source_id="local",
        unit_size=2,
        data={"start_index": 0, "chunk_size": 2, "unprocessed_ranges": [(0, 1)]},
    )
    processor.pending_units.append(unit_id)
    assert processor.get_work_units(1, "worker")
    return processor, unit_id


def test_local_completion_requeues_missing_caption(tmp_path):
    processor, unit_id = local_processor(tmp_path)
    processor.chunk_tracker.mark_items_processed(unit_id, 0, 0)

    assert processor.mark_completed(unit_id, "worker") is False
    retry = processor.get_work_units(1, "replacement")
    assert len(retry) == 1
    assert retry[0].data["unprocessed_ranges"] == [(1, 1)]
    assert not processor.chunk_tracker.is_chunk_completed(unit_id)


def test_repeated_local_failure_cannot_assign_same_unit_twice(tmp_path):
    processor, unit_id = local_processor(tmp_path)
    processor.mark_failed(unit_id, "worker", "save failed")
    processor.mark_failed(unit_id, "worker", "unit failed")

    assert len(processor.get_work_units(1, "replacement")) == 1
    assert processor.get_work_units(1, "another") == []


def test_stale_local_failure_does_not_requeue_new_owners_work(tmp_path):
    processor, unit_id = local_processor(tmp_path)
    processor.mark_failed(unit_id, "worker", "save failed")
    assert len(processor.get_work_units(1, "replacement")) == 1
    processor.mark_failed(unit_id, "worker", "late failure")
    assert processor.get_work_units(1, "another") == []


def test_local_completion_after_all_results_stays_complete(tmp_path):
    processor, unit_id = local_processor(tmp_path)
    processor.handle_result(
        WorkResult(
            unit_id=unit_id,
            source_id="local",
            chunk_id=unit_id,
            sample_id="batch",
            outputs={"captions": ["First image", "Second image"]},
            metadata={"item_indices": [0, 1]},
        )
    )
    assert processor.mark_completed(unit_id, "worker") is True
    processor.mark_failed(unit_id, "worker", "late failure")
    assert processor.chunk_tracker.is_chunk_completed(unit_id)
    assert processor.get_work_units(1, "replacement") == []


def test_huggingface_disconnect_requeues_nearly_complete_chunk(tmp_path):
    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor.chunk_size = 100
    processor.total_items = 100
    processor.dataset_name = "test"
    processor.shard_info = {
        0: {"filename": "data.parquet", "start_offset": 0, "end_offset": 99, "num_rows": 100}
    }
    processor.chunk_tracker = ChunkTracker(tmp_path / "chunks.json")
    unit_id = "data:chunk:0"
    processor.chunk_tracker.add_chunk(unit_id, "data", "data.parquet", 0, 100)
    processor.chunk_tracker.mark_assigned(unit_id, "worker")
    processor.chunk_tracker.mark_items_processed(unit_id, 0, 94)
    processor.assigned_units["worker"].add(unit_id)
    try:
        processor.release_assignments("worker")
        retries = processor.get_work_units(1, "replacement")
        assert len(retries) == 1
        assert retries[0].data["unprocessed_ranges"] == [(95, 99)]
    finally:
        processor.queue_handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contributor_id,expected",
    [("new", ("New name", 0, 3)), ("active", ("Active name", 1, 0))],
)
async def test_restart_preserves_contributor_profiles(tmp_path, contributor_id, expected):
    storage = StorageManager(tmp_path)
    await storage.initialize()
    await storage.save_contributor(Contributor("active", "Active name", trust_level=0))
    await storage.save_contributor(Contributor("new", "New name", trust_level=3))
    await storage.save_caption(
        Caption(
            job_id="local:chunk:0:idx:0",
            dataset="test",
            shard="local",
            item_key="0",
            contributor_id="active",
            timestamp=datetime.now(),
            captions=["A tree"],
            outputs={"captions": ["A tree"]},
        )
    )
    await storage.close()

    restarted = StorageManager(tmp_path)
    await restarted.initialize()
    try:
        contributor = await restarted.get_contributor(contributor_id)
        assert contributor is not None
        assert (contributor.name, contributor.total_captions, contributor.trust_level) == expected
    finally:
        await restarted.close()
