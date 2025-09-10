"""Test range-level work distribution for partial chunks.

This test verifies that workers are assigned only specific unprocessed ranges
within chunks, enabling clean distribution of partial chunks without overlap.
"""

import asyncio
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Tuple
from unittest.mock import Mock, patch

import pytest
from caption_flow.models import Caption, JobId
from caption_flow.processors import ProcessorConfig
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor


class RangeTrackingStorageManager:
    """Storage manager that tracks processed ranges for testing."""

    def __init__(self):
        self.processed_job_ids = set()
        self.save_attempts = []
        self.processed_ranges_by_chunk = defaultdict(list)

        # Pre-populate with some processed ranges to create gaps
        self._add_processed_range("photos_sequential:chunk:5", 5000, 5050)  # First 50 items
        self._add_processed_range("photos_sequential:chunk:5", 5200, 5250)  # Middle 50 items
        self._add_processed_range("photos_sequential:chunk:5", 5800, 5999)  # Last 200 items

        # Chunk 6 is partially processed
        self._add_processed_range("photos_sequential:chunk:6", 6000, 6100)  # First 100 items
        self._add_processed_range("photos_sequential:chunk:6", 6500, 6600)  # Middle section

    def _add_processed_range(self, chunk_prefix: str, start_idx: int, end_idx: int):
        """Add a processed range to storage."""
        chunk_index = chunk_prefix.split(":")[-1]
        for i in range(start_idx, end_idx + 1):
            job_id = f"{chunk_prefix}:idx:{i}"
            self.processed_job_ids.add(job_id)

        self.processed_ranges_by_chunk[chunk_prefix].append((start_idx, end_idx))

    def get_all_processed_job_ids(self) -> Set[str]:
        return self.processed_job_ids

    async def save_caption(self, caption):
        """Mock save that tracks attempts."""
        if hasattr(caption.job_id, "get_sample_str"):
            job_id = caption.job_id.get_sample_str()
        else:
            job_id = str(caption.job_id)

        self.save_attempts.append(job_id)

        # Simulate duplicate detection
        if job_id in self.processed_job_ids:
            return False  # Duplicate rejected

        self.processed_job_ids.add(job_id)
        return True

    def get_expected_unprocessed_ranges(
        self, chunk_id: str, chunk_start: int, chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Calculate expected unprocessed ranges for a chunk."""
        processed_ranges = self.processed_ranges_by_chunk.get(chunk_id, [])

        if not processed_ranges:
            # No processed ranges, entire chunk is unprocessed
            return [(chunk_start, chunk_start + chunk_size - 1)]

        # Create set of processed indices
        processed_indices = set()
        for start, end in processed_ranges:
            processed_indices.update(range(start, end + 1))

        # Find unprocessed ranges
        unprocessed_ranges = []
        current_start = None

        for i in range(chunk_start, chunk_start + chunk_size):
            if i not in processed_indices:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    unprocessed_ranges.append((current_start, i - 1))
                    current_start = None

        # Don't forget the last range
        if current_start is not None:
            unprocessed_ranges.append((current_start, chunk_start + chunk_size - 1))

        return unprocessed_ranges


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
        "chunk_size": 1000,  # 1000 items per chunk
        "min_chunk_buffer": 5,
        "chunk_buffer_multiplier": 2,
    }


@pytest.mark.asyncio
async def test_range_level_work_distribution(orchestrator_config, temp_checkpoint_dir):
    """Test that work units contain only unprocessed ranges and multiple workers
    can cleanly process different parts of the same chunk.
    """
    print("\n" + "=" * 100)
    print("TESTING RANGE-LEVEL WORK DISTRIBUTION")
    print("=" * 100)

    storage = RangeTrackingStorageManager()

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )

    # Mock the initialization to avoid real dataset loading - manually set required attributes
    processor.dataset_name = orchestrator_config["dataset"]["dataset_path"]
    processor.config = "default"
    processor.split = "train"
    processor.chunk_size = orchestrator_config["chunk_size"]
    processor.min_buffer = orchestrator_config["min_chunk_buffer"]
    processor.buffer_multiplier = orchestrator_config["chunk_buffer_multiplier"]
    processor.storage = storage
    processor.data_files = {}

    # Initialize the chunk tracker manually
    from caption_flow.utils.chunk_tracker import ChunkTracker

    processor.chunk_tracker = ChunkTracker(temp_checkpoint_dir / "chunks.json")

    # Initialize threading components but prevent background work
    import threading
    from collections import deque, defaultdict

    processor.lock = threading.Lock()
    processor.work_units = {}
    processor.pending_units = deque()
    processor.assigned_units = defaultdict(set)
    processor.stop_creation = threading.Event()
    processor.unit_creation_thread = None

    # Set the stop event to prevent any background thread from starting
    processor.stop_creation.set()

    # Update from storage to restore any existing state
    processor.update_from_storage(storage.processed_job_ids)

    # Manually create work units for the test based on the chunk tracker state
    # This simulates what the background thread would do
    from caption_flow.processors.base import WorkUnit
    from caption_flow.models import JobId

    for chunk_id, chunk_state in processor.chunk_tracker.chunks.items():
        if chunk_state.status != "completed":
            unprocessed_ranges = chunk_state.get_unprocessed_ranges()
            if unprocessed_ranges:
                # Convert to absolute ranges
                absolute_ranges = [
                    (r[0] + chunk_state.start_index, r[1] + chunk_state.start_index)
                    for r in unprocessed_ranges
                ]

                # Extract chunk index from chunk_id (e.g., "photos_sequential:chunk:5" -> 5)
                chunk_parts = chunk_id.split(":")
                chunk_index = int(chunk_parts[-1])
                shard_name = ":".join(chunk_parts[:-2])  # Remove ":chunk:N"

                # Calculate work size
                actual_work_size = sum(end - start + 1 for start, end in absolute_ranges)

                work_unit = WorkUnit(
                    unit_id=chunk_id,
                    chunk_id=chunk_id,
                    source_id=shard_name,
                    unit_size=actual_work_size,
                    data={
                        "unprocessed_ranges": absolute_ranges,
                        "actual_work_size": actual_work_size,
                        "range_based": True,
                        "shard_name": shard_name,
                        "start_index": chunk_state.start_index,
                        "chunk_size": chunk_state.chunk_size,
                    },
                    metadata={
                        "chunk_index": chunk_index,
                        "shard_name": shard_name,
                    },
                )

                processor.work_units[chunk_id] = work_unit
                processor.pending_units.append(chunk_id)

    # Add some additional full chunks for testing multi-worker distribution
    for chunk_idx in [7, 8, 9]:  # Add chunks 7, 8, 9
        chunk_id = f"photos_sequential:chunk:{chunk_idx}"
        chunk_start = chunk_idx * 1000

        # Add to chunk tracker
        processor.chunk_tracker.add_chunk(
            chunk_id, "photos_sequential", "dummy_shard.parquet", chunk_start, 1000
        )

        work_unit = WorkUnit(
            unit_id=chunk_id,
            chunk_id=chunk_id,
            source_id="photos_sequential",
            unit_size=1000,
            data={
                "unprocessed_ranges": [(chunk_start, chunk_start + 999)],
                "actual_work_size": 1000,
                "range_based": True,
                "shard_name": "photos_sequential",
                "start_index": chunk_start,
                "chunk_size": 1000,
            },
            metadata={
                "chunk_index": chunk_idx,
                "shard_name": "photos_sequential",
            },
        )

        processor.work_units[chunk_id] = work_unit
        processor.pending_units.append(chunk_id)

    # Patch the _create_work_units_from_chunk method to just return the pre-created units
    def mock_create_work_units_from_chunk(chunk_index):
        chunk_id = f"photos_sequential:chunk:{chunk_index}"
        if chunk_id in processor.work_units:
            return [processor.work_units[chunk_id]]
        return []

    processor._create_work_units_from_chunk = mock_create_work_units_from_chunk

    print("\n📊 INITIAL STORAGE STATE:")
    print(f"  Total processed job IDs: {len(storage.processed_job_ids)}")
    print(f"  Work units available: {len(processor.work_units)}")
    print(f"  Pending units: {len(processor.pending_units)}")

    # Show processed ranges for each chunk
    for chunk_id, ranges in storage.processed_ranges_by_chunk.items():
        total_processed = sum(end - start + 1 for start, end in ranges)
        print(f"  {chunk_id}: {len(ranges)} ranges, {total_processed} items")
        for start, end in ranges:
            print(f"    [{start}-{end}] ({end - start + 1} items)")

    print("\n🔍 PHASE 1: Test Range-Based Work Unit Creation")

    # Request work units from partially processed chunks
    worker1 = "range_worker_1"
    worker2 = "range_worker_2"

    units_w1 = processor.get_work_units(count=3, worker_id=worker1)
    units_w2 = processor.get_work_units(count=3, worker_id=worker2)

    print("\n  Worker assignments:")
    print(f"    {worker1}: {len(units_w1)} units")
    print(f"    {worker2}: {len(units_w2)} units")

    # Analyze each work unit in detail
    all_assigned_ranges = []
    all_assigned_job_ids = set()

    for worker_id, units in [(worker1, units_w1), (worker2, units_w2)]:
        print(f"\n  📦 {worker_id} units:")

        for unit in units:
            chunk_idx = unit.metadata["chunk_index"]
            chunk_id = unit.chunk_id
            unprocessed_ranges = unit.data["unprocessed_ranges"]
            actual_work_size = unit.data.get("actual_work_size", unit.unit_size)
            range_based = unit.data.get("range_based", False)

            print(f"\n    Unit: {unit.unit_id}")
            print(f"      Chunk: {chunk_idx}")
            print(f"      Range-based: {range_based}")
            print(f"      Work size: {actual_work_size} items")
            print(f"      Unprocessed ranges: {unprocessed_ranges}")

            # Verify ranges don't overlap with storage
            overlap_count = 0
            range_job_ids = set()

            for start, end in unprocessed_ranges:
                for sample_idx in range(start, end + 1):
                    job_id = f"photos_sequential:chunk:{chunk_idx}:idx:{sample_idx}"
                    range_job_ids.add(job_id)

                    if job_id in storage.processed_job_ids:
                        overlap_count += 1

            all_assigned_job_ids.update(range_job_ids)
            all_assigned_ranges.extend(
                [(worker_id, start, end) for start, end in unprocessed_ranges]
            )

            if overlap_count == 0:
                print("      ✅ No overlap with processed items")
            else:
                print(f"      ❌ {overlap_count} items overlap with storage")

            # Compare with expected unprocessed ranges
            chunk_start = chunk_idx * 1000
            expected_ranges = storage.get_expected_unprocessed_ranges(
                f"photos_sequential:chunk:{chunk_idx}", chunk_start, 1000
            )

            if chunk_idx in [5, 6]:  # Chunks with known processed items
                print(f"      Expected unprocessed: {expected_ranges}")

                # Check if assigned ranges are subset of expected
                assigned_indices = set()
                for start, end in unprocessed_ranges:
                    assigned_indices.update(range(start, end + 1))

                expected_indices = set()
                for start, end in expected_ranges:
                    expected_indices.update(range(start, end + 1))

                if assigned_indices.issubset(expected_indices):
                    print("      ✅ Assigned ranges are valid subset of unprocessed")
                else:
                    invalid = assigned_indices - expected_indices
                    print(f"      ❌ {len(invalid)} invalid indices assigned")

    print("\n🔍 PHASE 2: Test No Overlap Between Workers")

    # Check for overlaps between workers
    worker_ranges = defaultdict(list)

    for worker_id, start, end in all_assigned_ranges:
        worker_ranges[worker_id].append((start, end))

    overlap_ranges = []
    for w1, ranges1 in worker_ranges.items():
        for w2, ranges2 in worker_ranges.items():
            if w1 >= w2:  # Avoid duplicate checks
                continue

            # Check for overlaps between w1 and w2 ranges
            for start1, end1 in ranges1:
                for start2, end2 in ranges2:
                    # Check if ranges overlap
                    if not (end1 < start2 or end2 < start1):
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        overlap_ranges.append(
                            {
                                "worker1": w1,
                                "worker2": w2,
                                "range1": (start1, end1),
                                "range2": (start2, end2),
                                "overlap": (overlap_start, overlap_end),
                            }
                        )

    print("\n  📊 WORKER OVERLAP ANALYSIS:")
    print("    Worker range assignments:")
    for worker_id, ranges in worker_ranges.items():
        total_items = sum(end - start + 1 for start, end in ranges)
        print(f"      {worker_id}: {len(ranges)} ranges, {total_items} items")

    if len(overlap_ranges) == 0:
        print("    ✅ NO OVERLAP: Workers assigned non-overlapping ranges")
    else:
        print(f"    ❌ OVERLAP DETECTED: {len(overlap_ranges)} overlapping ranges")
        for overlap in overlap_ranges[:3]:  # Show first 3
            print(
                f"      {overlap['worker1']} {overlap['range1']} overlaps {overlap['worker2']} {overlap['range2']}"
            )

    print("\n🔍 PHASE 3: Test Chunk Splitting for Large Gaps")

    # Look for work units that were split from chunks with large gaps
    split_units = []
    for worker_id, units in [(worker1, units_w1), (worker2, units_w2)]:
        for unit in units:
            if unit.data.get("is_split_unit", False):
                split_units.append(unit)

    print("\n  📊 CHUNK SPLITTING ANALYSIS:")
    print(f"    Split units created: {len(split_units)}")

    for unit in split_units:
        ranges = unit.data["unprocessed_ranges"]
        range_count = unit.metadata.get("range_count", 0)
        total_span = ranges[-1][1] - ranges[0][0] + 1 if ranges else 0
        actual_work = sum(end - start + 1 for start, end in ranges)
        efficiency = actual_work / total_span if total_span > 0 else 0

        print(f"    Unit {unit.unit_id}:")
        print(f"      Ranges: {range_count}")
        print(f"      Work efficiency: {efficiency:.1%} (work/span)")
        print(f"      Ranges: {ranges}")

    print("\n🧪 PHASE 4: Simulate Concurrent Processing")

    # Process some items from each worker and verify no conflicts
    processing_results = defaultdict(list)

    for worker_id, units in [(worker1, units_w1[:1]), (worker2, units_w2[:1])]:  # First unit only
        for unit in units:
            unprocessed_ranges = unit.data["unprocessed_ranges"]
            chunk_idx = unit.metadata["chunk_index"]

            # Process first 10 items from first range
            if unprocessed_ranges:
                start, end = unprocessed_ranges[0]
                items_to_process = min(10, end - start + 1)

                for i in range(items_to_process):
                    sample_idx = start + i
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
                        captions=[f"Caption by {worker_id}"],
                        outputs={"captions": [f"Caption by {worker_id}"]},
                        contributor_id=worker_id,
                        timestamp=None,
                        caption_count=1,
                        processing_time_ms=100.0,
                        metadata={"worker_id": worker_id},
                    )

                    saved = await storage.save_caption(caption)
                    processing_results[worker_id].append(
                        {
                            "job_id": job_id_obj.get_sample_str(),
                            "sample_idx": sample_idx,
                            "saved": saved,
                        }
                    )

    print("\n  📊 PROCESSING RESULTS:")
    total_attempts = sum(len(results) for results in processing_results.values())
    total_saved = sum(
        sum(1 for r in results if r["saved"]) for results in processing_results.values()
    )
    total_duplicates = total_attempts - total_saved

    print(f"    Total processing attempts: {total_attempts}")
    print(f"    Successfully saved: {total_saved}")
    print(f"    Duplicates blocked: {total_duplicates}")

    for worker_id, results in processing_results.items():
        saved_count = sum(1 for r in results if r["saved"])
        print(f"    {worker_id}: {saved_count}/{len(results)} saved")

    print("\n✅ FINAL ASSESSMENT:")

    # Calculate overall score
    score = 100
    issues = []

    # Check for storage overlaps
    total_storage_overlaps = sum(
        1 for job_id in all_assigned_job_ids if job_id in storage.processed_job_ids
    )
    if total_storage_overlaps > 0:
        score -= 30
        issues.append(f"{total_storage_overlaps} items overlap with storage")

    # Check for worker overlaps
    if len(overlap_ranges) > 0:
        score -= 40
        issues.append(f"{len(overlap_ranges)} worker overlaps")

    # Check for processing duplicates
    if total_duplicates > 0:
        score -= 20
        issues.append(f"{total_duplicates} processing duplicates")

    print(f"    🎯 RANGE DISTRIBUTION SCORE: {score}%")

    if score >= 90:
        print("    ✅ EXCELLENT: Range-level distribution working perfectly")
    elif score >= 70:
        print("    ⚠️  GOOD: Range-level distribution mostly working")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("    ❌ NEEDS WORK: Range-level distribution has issues")
        for issue in issues:
            print(f"      - {issue}")

    # Cleanup
    processor.stop_creation.set()
    if processor.unit_creation_thread:
        processor.unit_creation_thread.join(timeout=0.1)

    print("\n" + "=" * 100)
    print("RANGE-LEVEL DISTRIBUTION TEST COMPLETE")
    print("=" * 100)

    return {
        "storage_overlaps": total_storage_overlaps,
        "worker_overlaps": len(overlap_ranges),
        "processing_duplicates": total_duplicates,
        "split_units": len(split_units),
        "score": score,
    }


@pytest.mark.asyncio
async def test_chunk_gap_detection_and_splitting(orchestrator_config, temp_checkpoint_dir):
    """Test that chunks with large gaps between processed ranges are properly split
    into multiple work units for efficient distribution.
    """
    print("\n" + "=" * 80)
    print("TESTING CHUNK GAP DETECTION AND SPLITTING")
    print("=" * 80)

    storage = RangeTrackingStorageManager()

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )

    # Mock the initialization to avoid real dataset loading - manually set required attributes
    processor.dataset_name = orchestrator_config["dataset"]["dataset_path"]
    processor.config = "default"
    processor.split = "train"
    processor.chunk_size = orchestrator_config["chunk_size"]
    processor.min_buffer = orchestrator_config["min_chunk_buffer"]
    processor.buffer_multiplier = orchestrator_config["chunk_buffer_multiplier"]
    processor.storage = storage
    processor.data_files = {}

    # Initialize the chunk tracker manually
    from caption_flow.utils.chunk_tracker import ChunkTracker

    processor.chunk_tracker = ChunkTracker(temp_checkpoint_dir / "chunks.json")

    # Initialize threading components but prevent background work
    import threading
    from collections import deque, defaultdict

    processor.lock = threading.Lock()
    processor.work_units = {}
    processor.pending_units = deque()
    processor.assigned_units = defaultdict(set)
    processor.stop_creation = threading.Event()
    processor.unit_creation_thread = None

    # Set the stop event to prevent any background thread from starting
    processor.stop_creation.set()

    # Update from storage to restore any existing state
    processor.update_from_storage(storage.processed_job_ids)

    # Manually create work units for the test based on the chunk tracker state
    # This simulates what the background thread would do
    from caption_flow.processors.base import WorkUnit
    from caption_flow.models import JobId

    for chunk_id, chunk_state in processor.chunk_tracker.chunks.items():
        if chunk_state.status != "completed":
            unprocessed_ranges = chunk_state.get_unprocessed_ranges()
            if unprocessed_ranges:
                # Convert to absolute ranges
                absolute_ranges = [
                    (r[0] + chunk_state.start_index, r[1] + chunk_state.start_index)
                    for r in unprocessed_ranges
                ]

                # Extract chunk index from chunk_id (e.g., "photos_sequential:chunk:5" -> 5)
                chunk_parts = chunk_id.split(":")
                chunk_index = int(chunk_parts[-1])
                shard_name = ":".join(chunk_parts[:-2])  # Remove ":chunk:N"

                # Calculate work size
                actual_work_size = sum(end - start + 1 for start, end in absolute_ranges)

                work_unit = WorkUnit(
                    unit_id=chunk_id,
                    chunk_id=chunk_id,
                    source_id=shard_name,
                    unit_size=actual_work_size,
                    data={
                        "unprocessed_ranges": absolute_ranges,
                        "actual_work_size": actual_work_size,
                        "range_based": True,
                        "shard_name": shard_name,
                        "start_index": chunk_state.start_index,
                        "chunk_size": chunk_state.chunk_size,
                    },
                    metadata={
                        "chunk_index": chunk_index,
                        "shard_name": shard_name,
                    },
                )

                processor.work_units[chunk_id] = work_unit
                processor.pending_units.append(chunk_id)

    # Add some additional full chunks for testing gap detection
    for chunk_idx in [7, 8, 9]:  # Add chunks 7, 8, 9
        chunk_id = f"photos_sequential:chunk:{chunk_idx}"
        chunk_start = chunk_idx * 1000

        # Add to chunk tracker
        processor.chunk_tracker.add_chunk(
            chunk_id, "photos_sequential", "dummy_shard.parquet", chunk_start, 1000
        )

        work_unit = WorkUnit(
            unit_id=chunk_id,
            chunk_id=chunk_id,
            source_id="photos_sequential",
            unit_size=1000,
            data={
                "unprocessed_ranges": [(chunk_start, chunk_start + 999)],
                "actual_work_size": 1000,
                "range_based": True,
                "shard_name": "photos_sequential",
                "start_index": chunk_start,
                "chunk_size": 1000,
            },
            metadata={
                "chunk_index": chunk_idx,
                "shard_name": "photos_sequential",
            },
        )

        processor.work_units[chunk_id] = work_unit
        processor.pending_units.append(chunk_id)

    # Patch the _create_work_units_from_chunk method to just return the pre-created units
    def mock_create_work_units_from_chunk(chunk_index):
        chunk_id = f"photos_sequential:chunk:{chunk_index}"
        if chunk_id in processor.work_units:
            return [processor.work_units[chunk_id]]
        return []

    processor._create_work_units_from_chunk = mock_create_work_units_from_chunk

    print("\n🔍 Testing gap detection logic:")

    # Test the chunk splitting logic directly
    test_worker = "gap_test_worker"
    units = processor.get_work_units(count=5, worker_id=test_worker)

    gap_analysis = []

    for unit in units:
        unprocessed_ranges = unit.data["unprocessed_ranges"]
        is_split = unit.data.get("is_split_unit", False)
        chunk_idx = unit.metadata["chunk_index"]

        if len(unprocessed_ranges) > 1:
            # Calculate gap metrics
            total_span = unprocessed_ranges[-1][1] - unprocessed_ranges[0][0] + 1
            total_work = sum(end - start + 1 for start, end in unprocessed_ranges)
            gap_ratio = (total_span - total_work) / total_span if total_span > 0 else 0

            gap_analysis.append(
                {
                    "unit_id": unit.unit_id,
                    "chunk_index": chunk_idx,
                    "ranges": unprocessed_ranges,
                    "range_count": len(unprocessed_ranges),
                    "total_span": total_span,
                    "total_work": total_work,
                    "gap_ratio": gap_ratio,
                    "is_split": is_split,
                }
            )

    print("\n  📊 GAP ANALYSIS RESULTS:")
    print(f"    Units with multiple ranges: {len(gap_analysis)}")

    for analysis in gap_analysis:
        print(f"\n    Unit: {analysis['unit_id']}")
        print(f"      Chunk: {analysis['chunk_index']}")
        print(f"      Ranges: {analysis['range_count']}")
        print(f"      Gap ratio: {analysis['gap_ratio']:.1%}")
        print(f"      Total work: {analysis['total_work']} items")
        print(f"      Was split: {analysis['is_split']}")
        print(f"      Ranges: {analysis['ranges']}")

        if analysis["gap_ratio"] > 0.5:
            if analysis["is_split"]:
                print("      ✅ High gap ratio correctly triggered splitting")
            else:
                print("      ⚠️  High gap ratio but not split")
        else:
            print("      ✅ Low gap ratio, no splitting needed")

    processor.stop_creation.set()
    if processor.unit_creation_thread:
        processor.unit_creation_thread.join(timeout=0.1)

    print("\n" + "=" * 80)
    print("GAP DETECTION TEST COMPLETE")
    print("=" * 80)

    return gap_analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
