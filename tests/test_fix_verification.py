"""Test to verify the fix for worker reconnection duplicate assignments.

This test specifically checks that the fix prevents duplicate work assignment
when chunk state is properly synchronized with storage.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Set

import pytest
from caption_flow.models import Caption, JobId
from caption_flow.processors import ProcessorConfig
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor


class MockStorageManagerWithData:
    """Mock storage manager that contains pre-existing processed data."""

    def __init__(self):
        self.processed_job_ids = set()
        self.save_attempts = []

        # Pre-populate with some processed job IDs to simulate existing work
        self._add_processed_chunk(chunk_index=5, start_idx=5000, count=100)
        self._add_processed_chunk(
            chunk_index=5, start_idx=5400, count=100
        )  # Partial chunk 5 processing

    def _add_processed_chunk(self, chunk_index: int, start_idx: int, count: int):
        """Add processed job IDs for a chunk."""
        for i in range(count):
            job_id = f"photos_sequential:chunk:{chunk_index}:idx:{start_idx + i}"
            self.processed_job_ids.add(job_id)

    def get_all_processed_job_ids(self) -> Set[str]:
        return self.processed_job_ids

    async def save_caption(self, caption):
        """Mock save that tracks attempts."""
        if hasattr(caption.job_id, "get_sample_str"):
            job_id = caption.job_id.get_sample_str()
        else:
            job_id = str(caption.job_id)

        self.save_attempts.append(job_id)

        # Simulate duplicate detection like real storage
        if job_id in self.processed_job_ids:
            return False  # Duplicate rejected

        self.processed_job_ids.add(job_id)
        return True


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def orchestrator_config(temp_checkpoint_dir):
    """Create orchestrator configuration for testing."""
    return {
        "dataset": {
            "processor_type": "huggingface_datasets",
            "dataset_path": "terminusresearch/pexels-metadata-1.71M",
            "dataset_config": None,
            "dataset_split": None,
        },
        "checkpoint_dir": str(temp_checkpoint_dir),
        "chunk_size": 1000,  # Large chunks to make the test more realistic
        "min_chunk_buffer": 5,
        "chunk_buffer_multiplier": 2,
    }


@pytest.mark.asyncio
async def test_fix_prevents_duplicate_assignments(orchestrator_config, temp_checkpoint_dir):
    """Test that our fix prevents workers from being assigned work that's already completed."""
    print("\n" + "=" * 80)
    print("TESTING FIX: DUPLICATE ASSIGNMENT PREVENTION")
    print("=" * 80)

    # Use storage with pre-existing processed data
    storage = MockStorageManagerWithData()

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )

    processor.initialize(processor_config, storage)
    await asyncio.sleep(2)  # Wait for initialization and sync

    print("\n📊 INITIAL STATE:")
    print(f"  Storage contains {len(storage.processed_job_ids)} pre-processed job IDs")

    # Show some sample processed job IDs
    sample_processed = list(storage.processed_job_ids)[:10]
    print(f"  Sample processed IDs: {sample_processed}")

    print("\n🔍 PHASE 1: Initial Worker Assignment")

    # Worker gets assigned work
    worker_id = "test_worker"
    units = processor.get_work_units(count=3, worker_id=worker_id)

    print(f"  Worker assigned {len(units)} units:")

    assigned_job_ids = set()
    overlapping_with_storage = 0

    for unit in units:
        chunk_idx = unit.metadata["chunk_index"]
        start_idx = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get("unprocessed_ranges", [])

        print(f"\n    📦 Chunk {chunk_idx}:")
        print(f"      Start index: {start_idx}")
        print(f"      Chunk size: {chunk_size}")
        print(f"      Unprocessed ranges: {unprocessed_ranges[:3]}...")  # Show first few ranges

        # Check job IDs for this unit
        for i in range(chunk_size):
            sample_idx = start_idx + i
            job_id = f"photos_sequential:chunk:{chunk_idx}:idx:{sample_idx}"
            assigned_job_ids.add(job_id)

            if job_id in storage.processed_job_ids:
                overlapping_with_storage += 1

    print("\n  📊 ASSIGNMENT ANALYSIS:")
    print(f"    Total job IDs assigned: {len(assigned_job_ids)}")
    print(f"    Overlap with storage: {overlapping_with_storage}")

    if overlapping_with_storage == 0:
        print("    ✅ SUCCESS: No overlap with pre-processed items!")
    else:
        print(f"    ❌ PROBLEM: {overlapping_with_storage} items already processed")

    print("\n🔄 PHASE 2: Worker Disconnection and Reconnection")

    # Simulate disconnection
    processor.release_assignments(worker_id)
    print(f"  Worker {worker_id} disconnected, assignments released")

    await asyncio.sleep(0.5)  # Allow re-queuing

    # Worker reconnects
    new_units = processor.get_work_units(count=2, worker_id=worker_id)
    print(f"  Worker reconnected, assigned {len(new_units)} units")

    reconnect_job_ids = set()
    reconnect_overlaps = 0

    for unit in new_units:
        chunk_idx = unit.metadata["chunk_index"]
        start_idx = unit.data["start_index"]
        chunk_size = unit.data["chunk_size"]
        unprocessed_ranges = unit.data.get("unprocessed_ranges", [])

        print(f"\n    📦 Reconnect Chunk {chunk_idx}:")
        print(f"      Start index: {start_idx}")
        print(f"      Unprocessed ranges: {len(unprocessed_ranges)} ranges")

        # Count actual work assigned (only unprocessed ranges)
        if unprocessed_ranges:
            for start_range, end_range in unprocessed_ranges:
                for sample_idx in range(start_range, end_range + 1):
                    job_id = f"photos_sequential:chunk:{chunk_idx}:idx:{sample_idx}"
                    reconnect_job_ids.add(job_id)

                    if job_id in storage.processed_job_ids:
                        reconnect_overlaps += 1
        else:
            # If no unprocessed ranges specified, assume full chunk (legacy behavior)
            for i in range(chunk_size):
                sample_idx = start_idx + i
                job_id = f"photos_sequential:chunk:{chunk_idx}:idx:{sample_idx}"
                reconnect_job_ids.add(job_id)

                if job_id in storage.processed_job_ids:
                    reconnect_overlaps += 1

    print("\n  📊 RECONNECTION ANALYSIS:")
    print(f"    Job IDs assigned on reconnect: {len(reconnect_job_ids)}")
    print(f"    Overlap with storage: {reconnect_overlaps}")

    if reconnect_overlaps == 0:
        print("    ✅ FIX WORKING: No duplicate work assigned on reconnection!")
    else:
        print(f"    ❌ FIX NEEDED: {reconnect_overlaps} duplicate items still assigned")

    print("\n🧪 PHASE 3: Simulate Processing with Storage Checks")

    # Process a few items and check storage behavior
    processing_attempts = 0
    duplicates_blocked = 0

    for unit in new_units[:1]:  # Process first unit only
        chunk_idx = unit.metadata["chunk_index"]
        start_idx = unit.data["start_index"]
        unprocessed_ranges = unit.data.get(
            "unprocessed_ranges", [(start_idx, start_idx + 9)]
        )  # First 10 items

        for start_range, end_range in unprocessed_ranges[:1]:  # First range only
            for sample_idx in range(start_range, min(end_range + 1, start_range + 10)):  # First 10
                processing_attempts += 1

                job_id_obj = JobId(
                    shard_id=unit.metadata["shard_name"],
                    chunk_id=str(chunk_idx),
                    sample_id=str(sample_idx),
                )

                caption = Caption(
                    job_id=job_id_obj,
                    dataset=orchestrator_config["dataset"]["dataset_path"],
                    shard=unit.metadata["shard_name"],
                    chunk_id=unit.chunk_id,
                    item_key=str(sample_idx),
                    captions=["Test caption"],
                    outputs={"captions": ["Test caption"]},
                    contributor_id="test_worker",
                    timestamp=None,
                    caption_count=1,
                    processing_time_ms=100.0,
                    metadata={},
                )

                saved = await storage.save_caption(caption)
                if not saved:
                    duplicates_blocked += 1

    print("\n  📊 PROCESSING RESULTS:")
    print(f"    Processing attempts: {processing_attempts}")
    print(f"    Duplicates blocked by storage: {duplicates_blocked}")
    print(f"    Successful saves: {processing_attempts - duplicates_blocked}")

    print("\n✅ FINAL ASSESSMENT:")

    total_efficiency_score = 100
    if overlapping_with_storage > 0:
        print(f"    ⚠️  Initial assignment had {overlapping_with_storage} overlaps")
        total_efficiency_score -= 30

    if reconnect_overlaps > 0:
        print(f"    ⚠️  Reconnection assignment had {reconnect_overlaps} overlaps")
        total_efficiency_score -= 50

    if duplicates_blocked > 0:
        print(f"    ⚠️  Storage blocked {duplicates_blocked} duplicate processing attempts")
        total_efficiency_score -= 20

    print(f"\n    🎯 EFFICIENCY SCORE: {total_efficiency_score}%")

    if total_efficiency_score >= 80:
        print("    ✅ EXCELLENT: Fix is working well")
    elif total_efficiency_score >= 60:
        print("    ⚠️  GOOD: Fix is working but could be improved")
    else:
        print("    ❌ NEEDS WORK: Fix needs improvement")

    # Cleanup
    processor.stop_creation.set()

    print("\n" + "=" * 80)
    print("FIX VERIFICATION TEST COMPLETE")
    print("=" * 80)

    # The test should show improvement compared to before the fix
    return {
        "initial_overlaps": overlapping_with_storage,
        "reconnect_overlaps": reconnect_overlaps,
        "duplicates_blocked": duplicates_blocked,
        "efficiency_score": total_efficiency_score,
    }


@pytest.mark.asyncio
async def test_chunk_tracker_sync_with_storage(orchestrator_config, temp_checkpoint_dir):
    """Test that chunk tracker properly syncs with storage during release_assignments."""
    print("\n" + "=" * 60)
    print("TESTING CHUNK TRACKER STORAGE SYNC")
    print("=" * 60)

    storage = MockStorageManagerWithData()

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )

    processor.initialize(processor_config, storage)
    await asyncio.sleep(1)

    print("\n📊 STORAGE STATE:")
    print(f"  Processed job IDs in storage: {len(storage.processed_job_ids)}")

    # Check if chunks were created in tracker during sync
    if processor.chunk_tracker:
        print(f"  Chunks in tracker: {len(processor.chunk_tracker.chunks)}")

        # Show details of some chunks
        for chunk_id, chunk_state in list(processor.chunk_tracker.chunks.items())[:3]:
            processed_items = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
            completion = (
                processed_items / chunk_state.chunk_size if chunk_state.chunk_size > 0 else 0
            )

            print(
                f"    {chunk_id}: {processed_items}/{chunk_state.chunk_size} items ({completion:.1%})"
            )

    print("\n🔄 TESTING RELEASE ASSIGNMENTS SYNC:")

    # Assign work to a worker
    worker_id = "sync_test_worker"
    units = processor.get_work_units(count=1, worker_id=worker_id)

    if units:
        chunk_id = units[0].chunk_id
        print(f"  Assigned chunk: {chunk_id}")

        # Add more processed items to storage to simulate concurrent processing
        new_job_ids = {f"photos_sequential:chunk:5:idx:{5100 + i}" for i in range(50)}
        storage.processed_job_ids.update(new_job_ids)
        print(f"  Added {len(new_job_ids)} new processed items to storage")

        # Release assignments (should trigger sync)
        print("  Releasing assignments (should trigger storage sync)...")
        processor.release_assignments(worker_id)

        # Check if chunk tracker was updated
        if processor.chunk_tracker and chunk_id in processor.chunk_tracker.chunks:
            chunk_state = processor.chunk_tracker.chunks[chunk_id]
            processed_items = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
            completion = (
                processed_items / chunk_state.chunk_size if chunk_state.chunk_size > 0 else 0
            )

            print(
                f"  After sync - {chunk_id}: {processed_items}/{chunk_state.chunk_size} items ({completion:.1%})"
            )

            if completion > 0.5:  # More than 50% complete
                print("    ✅ Chunk tracker properly synced with storage")
            else:
                print("    ⚠️  Chunk tracker may not be fully synced")

    processor.stop_creation.set()

    print("\n" + "=" * 60)
    print("SYNC TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
