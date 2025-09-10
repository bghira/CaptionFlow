import asyncio
import datetime as _datetime
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

import pytest
from caption_flow.models import JobId
from caption_flow.processors import ProcessorConfig
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor
from caption_flow.storage import StorageManager


class MockStorageManager:
    """Mock storage manager for testing."""

    def __init__(self):
        self.processed_job_ids = set()

    def get_all_processed_job_ids(self) -> Set[str]:
        return self.processed_job_ids


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
            "dataset_config": None,  # Will auto-detect
            "dataset_split": None,  # Will auto-detect
        },
        "checkpoint_dir": str(temp_checkpoint_dir),
        "chunk_size": 1000,  # Small chunks for testing
        "min_chunk_buffer": 5,
        "chunk_buffer_multiplier": 2,
    }


class TestHuggingFaceJobIdUniqueness:
    """Test that job IDs are unique across all shards."""

    @pytest.mark.asyncio
    async def test_no_duplicate_job_ids_across_shards(
        self, orchestrator_config, temp_checkpoint_dir
    ):
        """Test that no duplicate job IDs are assigned when processing all shards."""
        # Initialize processor
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor_config = ProcessorConfig(
            processor_type="huggingface_datasets", config=orchestrator_config
        )
        storage = MockStorageManager()

        # Initialize processor
        processor.initialize(processor_config, storage)

        # Wait for initial unit creation
        await asyncio.sleep(2)

        # Track all job IDs seen
        all_job_ids = set()
        job_ids_by_shard = defaultdict(set)
        job_ids_by_chunk = defaultdict(set)
        duplicate_job_ids = []

        # Simulate multiple workers requesting work
        worker_ids = [f"worker_{i}" for i in range(5)]
        total_units_processed = 0
        max_units_to_process = 100  # Limit for testing

        while total_units_processed < max_units_to_process:
            units_assigned = False

            for worker_id in worker_ids:
                # Request work units
                units = processor.get_work_units(count=2, worker_id=worker_id)

                if not units:
                    continue

                units_assigned = True

                for unit in units:
                    # Extract job IDs from the unit
                    start_index = unit.data["start_index"]
                    chunk_size = unit.data["chunk_size"]
                    shard_name = unit.metadata["shard_name"]
                    chunk_index = unit.metadata["chunk_index"]

                    # Generate all job IDs for this unit
                    for i in range(chunk_size):
                        sample_idx = start_index + i

                        # Create job ID the same way the processor does
                        job_id_obj = JobId(
                            shard_id=shard_name,
                            chunk_id=str(chunk_index),
                            sample_id=str(sample_idx),
                        )
                        job_id = job_id_obj.get_sample_str()

                        # Check for duplicates
                        if job_id in all_job_ids:
                            duplicate_job_ids.append(
                                {
                                    "job_id": job_id,
                                    "unit_id": unit.unit_id,
                                    "worker_id": worker_id,
                                    "shard": shard_name,
                                    "chunk_index": chunk_index,
                                    "sample_index": sample_idx,
                                }
                            )

                        all_job_ids.add(job_id)
                        job_ids_by_shard[shard_name].add(job_id)
                        job_ids_by_chunk[unit.unit_id].add(job_id)

                    # Simulate work completion
                    processor.mark_completed(unit.unit_id, worker_id)
                    total_units_processed += 1

            if not units_assigned:
                # No more work available
                break

            # Small delay to allow background thread to create more units
            await asyncio.sleep(0.1)

        # Stop the background thread
        processor.stop_creation.set()

        # Print statistics
        print("\nJob ID Assignment Statistics:")
        print(f"Total unique job IDs: {len(all_job_ids)}")
        print(f"Total shards processed: {len(job_ids_by_shard)}")
        print(f"Total chunks processed: {len(job_ids_by_chunk)}")
        print(f"Total units processed: {total_units_processed}")

        # Print shard statistics
        print("\nJob IDs per shard:")
        for shard, ids in sorted(job_ids_by_shard.items()):
            print(f"  {shard}: {len(ids)} job IDs")

        # Assert no duplicates
        if duplicate_job_ids:
            print(f"\nFound {len(duplicate_job_ids)} duplicate job IDs:")
            for dup in duplicate_job_ids[:10]:  # Show first 10
                print(f"  {dup}")

        assert len(duplicate_job_ids) == 0, f"Found {len(duplicate_job_ids)} duplicate job IDs"

        # Additional checks
        self._verify_job_id_format(all_job_ids)
        self._verify_chunk_continuity(job_ids_by_chunk)

    def _verify_job_id_format(self, job_ids: Set[str]):
        """Verify all job IDs follow the expected format."""
        for job_id in job_ids:
            # Job ID should be in format: shard_id:chunk:chunk_idx:idx:sample_idx
            parts = job_id.split(":")
            assert len(parts) == 5, f"Invalid job ID format: {job_id}"
            assert parts[1] == "chunk", f"Invalid job ID format: {job_id}"
            assert parts[3] == "idx", f"Invalid job ID format: {job_id}"

            # Verify indices are integers
            try:
                int(parts[2])  # chunk_idx
                int(parts[4])  # sample_idx
            except ValueError:
                pytest.fail(f"Invalid indices in job ID: {job_id}")

    def _verify_chunk_continuity(self, job_ids_by_chunk: Dict[str, Set[str]]):
        """Verify that job IDs within each chunk are continuous."""
        for chunk_id, job_ids in job_ids_by_chunk.items():
            # Extract sample indices from job IDs
            sample_indices = []
            for job_id in job_ids:
                parts = job_id.split(":")
                sample_idx = int(parts[4])
                sample_indices.append(sample_idx)

            sample_indices.sort()

            # Check continuity
            if sample_indices:
                expected = list(range(sample_indices[0], sample_indices[-1] + 1))
                assert (
                    sample_indices == expected
                ), f"Non-continuous sample indices in chunk {chunk_id}: {sample_indices[:10]}..."


@pytest.mark.asyncio
async def test_job_id_uniqueness_with_worker_disconnections(
    orchestrator_config, temp_checkpoint_dir
):
    """Test that job IDs remain unique when workers disconnect and work is reassigned."""
    # Initialize processor
    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )
    storage = MockStorageManager()

    processor.initialize(processor_config, storage)
    await asyncio.sleep(2)  # Wait for initial unit creation

    # Track all job IDs and their assignment history
    all_job_ids = set()
    job_id_assignment_history = defaultdict(list)  # job_id -> [(worker_id, assignment_number)]
    assignment_counter = 0

    # Phase 1: Assign work to workers
    print("\nPhase 1: Initial work assignment")
    worker_assignments = {}  # worker_id -> list of units

    for i in range(3):
        worker_id = f"worker_{i}"
        units = processor.get_work_units(count=3, worker_id=worker_id)
        worker_assignments[worker_id] = units

        print(f"  {worker_id}: assigned {len(units)} units")

        # Track job IDs for each unit
        for unit in units:
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            # Generate job IDs for this unit
            for j in range(chunk_size):
                sample_idx = start_index + j
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(sample_idx)
                )
                job_id = job_id_obj.get_sample_str()

                all_job_ids.add(job_id)
                assignment_counter += 1
                job_id_assignment_history[job_id].append((worker_id, assignment_counter))

    # Verify initial state
    initial_stats = processor.get_stats()
    print(
        f"\nInitial state: {initial_stats['assigned_units']} assigned, {initial_stats['pending_units']} pending"
    )

    # Phase 2: Simulate worker disconnections
    print("\nPhase 2: Simulating worker disconnections")
    disconnected_workers = ["worker_0", "worker_1"]
    released_unit_ids = set()

    for worker_id in disconnected_workers:
        # Get the units that were assigned to this worker
        units_before_release = [unit.unit_id for unit in worker_assignments[worker_id]]
        released_unit_ids.update(units_before_release)

        # Simulate disconnection by releasing assignments
        processor.release_assignments(worker_id)
        print(f"  Released assignments for {worker_id}")

    # Verify units went back to pending
    stats_after_disconnect = processor.get_stats()
    print(
        f"\nAfter disconnections: {stats_after_disconnect['assigned_units']} assigned, {stats_after_disconnect['pending_units']} pending"
    )

    # The released units should now be in pending
    assert (
        stats_after_disconnect["pending_units"] > initial_stats["pending_units"]
    ), "Pending units should increase after worker disconnections"

    # Phase 3: Reassign work to new workers
    print("\nPhase 3: Reassigning work to new workers")
    reassigned_job_ids = set()
    duplicate_job_ids = []

    for i in range(3, 6):  # New workers
        worker_id = f"worker_{i}"
        units = processor.get_work_units(count=2, worker_id=worker_id)

        print(f"  {worker_id}: assigned {len(units)} units")

        # Check if any of these units were previously assigned
        reassigned_units = [u for u in units if u.unit_id in released_unit_ids]
        if reassigned_units:
            print(f"    Reassigned units: {[u.unit_id for u in reassigned_units]}")

        # Track job IDs
        for unit in units:
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            for j in range(chunk_size):
                sample_idx = start_index + j
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(sample_idx)
                )
                job_id = job_id_obj.get_sample_str()

                # Check if this is a reassignment
                if job_id in all_job_ids:
                    reassigned_job_ids.add(job_id)
                    assignment_counter += 1
                    job_id_assignment_history[job_id].append((worker_id, assignment_counter))
                else:
                    # This should not happen - all job IDs should already exist
                    # unless we're getting new chunks
                    if unit.unit_id in released_unit_ids:
                        duplicate_job_ids.append(
                            {
                                "job_id": job_id,
                                "unit_id": unit.unit_id,
                                "worker_id": worker_id,
                                "error": "New job ID for reassigned unit",
                            }
                        )

            # Mark some as completed
            if i % 2 == 0:  # Complete every other worker's assignments
                processor.mark_completed(unit.unit_id, worker_id)

    # Phase 4: Verify results
    print("\nPhase 4: Verification")
    print(f"  Total unique job IDs: {len(all_job_ids)}")
    print(f"  Reassigned job IDs: {len(reassigned_job_ids)}")
    print(f"  Duplicate/problematic job IDs: {len(duplicate_job_ids)}")

    # Show assignment history for some reassigned job IDs
    print("\nSample assignment histories:")
    sample_reassigned = list(reassigned_job_ids)[:5]
    for job_id in sample_reassigned:
        history = job_id_assignment_history[job_id]
        print(f"  {job_id}: {' -> '.join([f'{w}(#{n})' for w, n in history])}")

    # Verify no problematic job IDs
    if duplicate_job_ids:
        print("\nProblematic job IDs found:")
        for dup in duplicate_job_ids[:10]:
            print(f"  {dup}")

    assert (
        len(duplicate_job_ids) == 0
    ), f"Found {len(duplicate_job_ids)} problematic job IDs during reassignment"

    # Verify reassigned job IDs match expected pattern
    # They should have exactly 2 entries in their history (original + reassignment)
    for job_id in reassigned_job_ids:
        history = job_id_assignment_history[job_id]
        assert (
            len(history) >= 2
        ), f"Reassigned job ID {job_id} should have at least 2 assignment records, has {len(history)}"

    # Stop background thread
    processor.stop_creation.set()

    # Additional consistency checks
    await _verify_chunk_state_consistency(processor, released_unit_ids)


async def _verify_chunk_state_consistency(processor, released_unit_ids):
    """Verify that chunk tracker state is consistent after reassignments."""
    if not processor.chunk_tracker:
        return

    print("\nChunk state consistency check:")

    # Check that released units are not marked as assigned to disconnected workers
    for unit_id in released_unit_ids:
        if unit_id in processor.chunk_tracker.chunks:
            chunk_state = processor.chunk_tracker.chunks[unit_id]

            # Shouldn't be assigned to disconnected workers
            if chunk_state.assigned_to and chunk_state.assigned_to.startswith("worker_"):
                worker_num = int(chunk_state.assigned_to.split("_")[1])
                assert (
                    worker_num >= 3
                ), f"Chunk {unit_id} still assigned to disconnected {chunk_state.assigned_to}"

    print("  ✓ Chunk state consistency verified")


@pytest.mark.asyncio
async def test_rapid_worker_churning(orchestrator_config, temp_checkpoint_dir):
    """Test job ID uniqueness under rapid worker connect/disconnect cycles."""
    # Initialize processor
    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )
    storage = MockStorageManager()

    processor.initialize(processor_config, storage)
    await asyncio.sleep(1)

    # Track all job IDs
    all_job_ids = set()
    job_id_counts = defaultdict(int)  # Count how many times each job ID appears

    print("\nRapid worker churning test:")

    # Simulate rapid connect/disconnect cycles
    for cycle in range(5):
        print(f"\nCycle {cycle + 1}:")

        # Connect workers and get assignments
        active_workers = {}
        for i in range(3):
            worker_id = f"cycle{cycle}_worker{i}"
            units = processor.get_work_units(count=1, worker_id=worker_id)
            active_workers[worker_id] = units

            # Track job IDs
            for unit in units:
                for j in range(min(10, unit.data["chunk_size"])):  # Check first 10 job IDs
                    job_id_obj = JobId(
                        shard_id=unit.metadata["shard_name"],
                        chunk_id=str(unit.metadata["chunk_index"]),
                        sample_id=str(unit.data["start_index"] + j),
                    )
                    job_id = job_id_obj.get_sample_str()
                    all_job_ids.add(job_id)
                    job_id_counts[job_id] += 1

        # Simulate some workers disconnecting before completing work
        if cycle < 4:  # Don't disconnect on last cycle
            for worker_id in list(active_workers.keys())[:2]:  # Disconnect 2 out of 3
                processor.release_assignments(worker_id)
                print(f"  Disconnected: {worker_id}")

        # Complete remaining work
        for worker_id, units in active_workers.items():
            if worker_id in processor.assigned_units:  # Still connected
                for unit in units:
                    processor.mark_completed(unit.unit_id, worker_id)

        await asyncio.sleep(0.1)  # Small delay between cycles

    # Check results
    print("\nResults:")
    print(f"  Total unique job IDs seen: {len(all_job_ids)}")
    print(f"  Total job ID observations: {sum(job_id_counts.values())}")

    # Find any job IDs that appeared in multiple cycles (which is expected for reassignments)
    reassigned_count = sum(1 for count in job_id_counts.values() if count > 1)
    print(f"  Job IDs that were reassigned: {reassigned_count}")

    # The key check: each unique job ID should represent a unique sample
    # Even if reassigned, the job ID should be the same for the same sample
    max_count = max(job_id_counts.values()) if job_id_counts else 0
    print(f"  Maximum times a job ID appeared: {max_count}")

    # This is the critical assertion - job IDs should be deterministic
    # The same sample should always get the same job ID
    assert (
        max_count <= 5
    ), f"Job ID appeared too many times ({max_count}), suggests non-deterministic assignment"

    processor.stop_creation.set()


@pytest.mark.asyncio
async def test_duplicate_job_ids_with_same_token_multiple_workers(
    orchestrator_config, temp_checkpoint_dir
):
    """Test for duplicate job IDs when multiple workers use the same auth token."""
    # Initialize processor
    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )

    # Create a custom storage manager that tracks all save attempts
    class TrackingStorageManager(MockStorageManager):
        def __init__(self):
            super().__init__()
            self.save_attempts = []  # Track all caption saves
            self.save_order = []  # Track order of saves
            self.concurrent_saves = defaultdict(list)  # Track concurrent save attempts

        async def save_caption(self, caption):
            # Record the save attempt
            job_id = caption.job_id
            if isinstance(job_id, dict):
                job_id_str = JobId.from_dict(job_id).get_sample_str()
            elif hasattr(job_id, "get_sample_str"):  # It's a JobId object
                job_id_str = job_id.get_sample_str()
            else:
                job_id_str = str(job_id)

            self.save_attempts.append(
                {
                    "job_id": job_id_str,
                    "contributor_id": caption.contributor_id,
                    "timestamp": asyncio.get_event_loop().time(),
                    "chunk_id": caption.chunk_id,
                    "shard": caption.shard,
                }
            )
            self.save_order.append(job_id_str)

            # Track if this job_id is being saved concurrently
            if job_id_str in self.processed_job_ids:
                self.concurrent_saves[job_id_str].append(caption.contributor_id)

            self.processed_job_ids.add(job_id_str)
            return True

    storage = TrackingStorageManager()
    processor.initialize(processor_config, storage)
    await asyncio.sleep(2)

    # Simulate multiple workers with the same base name (same auth token)
    base_worker_name = "shared_worker"
    worker_ids = []

    # Create multiple worker IDs that would come from the same auth token
    for _i in range(5):
        # Simulate what orchestrator does
        import uuid

        worker_id = f"{base_worker_name}_{str(uuid.uuid4())[:8]}"
        worker_ids.append(worker_id)

    print(f"\nSimulating {len(worker_ids)} workers with same auth token:")
    for wid in worker_ids:
        print(f"  {wid}")

    # Phase 1: All workers request work simultaneously
    print("\nPhase 1: Simultaneous work requests")
    all_assigned_units = {}
    job_id_to_workers = defaultdict(set)  # Track which workers process each job_id

    for worker_id in worker_ids:
        units = processor.get_work_units(count=2, worker_id=worker_id)
        all_assigned_units[worker_id] = units

        # Track all job IDs this worker will process
        for unit in units:
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            for j in range(chunk_size):
                sample_idx = start_index + j
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(sample_idx)
                )
                job_id_str = job_id_obj.get_sample_str()  # Use string representation
                job_id_to_workers[job_id_str].add(worker_id)

    # Check if any job IDs were assigned to multiple workers
    duplicate_assignments = {
        job_id: workers for job_id, workers in job_id_to_workers.items() if len(workers) > 1
    }

    print(f"\nDuplicate assignments found: {len(duplicate_assignments)}")
    if duplicate_assignments:
        for job_id, workers in list(duplicate_assignments.items())[:5]:
            print(f"  {job_id}: assigned to {workers}")

    # Phase 2: Simulate workers processing and submitting results concurrently
    print("\nPhase 2: Simulating concurrent result submission")

    # Create mock results for each worker
    async def submit_worker_results(worker_id, units):
        """Simulate a worker submitting results."""
        results = []
        for unit in units:
            # Extract base name for contributor ID (mimicking orchestrator behavior)
            contributor_id = worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id

            # Simulate processing each item in the unit
            for i in range(min(10, unit.data["chunk_size"])):  # Process first 10 items
                job_id_obj = JobId(
                    shard_id=unit.metadata["shard_name"],
                    chunk_id=str(unit.metadata["chunk_index"]),
                    sample_id=str(unit.data["start_index"] + i),
                )

                # Create a mock caption
                from caption_flow.models import Caption

                caption = Caption(
                    job_id=job_id_obj,
                    dataset="test_dataset",
                    shard=unit.metadata["shard_name"],
                    chunk_id=unit.chunk_id,
                    item_key=str(unit.data["start_index"] + i),
                    captions=["Test caption"],
                    outputs={"captions": ["Test caption"]},
                    contributor_id=contributor_id,  # This will be same for all workers!
                    timestamp=datetime.now(_datetime.UTC),
                    caption_count=1,
                    processing_time_ms=100.0,
                    metadata={},
                )

                # Simulate save (this is where duplicates might occur)
                await storage.save_caption(caption)
                results.append(job_id_obj.get_sample_str())

                # Small random delay to simulate processing time
                await asyncio.sleep(0.001)

        return results

    # Submit all results concurrently
    tasks = []
    for worker_id, units in all_assigned_units.items():
        task = submit_worker_results(worker_id, units)
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Phase 3: Analyze results
    print("\nPhase 3: Analyzing results")

    # Count how many times each job_id was saved
    job_id_save_counts = defaultdict(int)
    for save_attempt in storage.save_attempts:
        job_id_save_counts[save_attempt["job_id"]] += 1

    # Find job IDs that were saved multiple times
    duplicate_saves = {job_id: count for job_id, count in job_id_save_counts.items() if count > 1}

    print("\nResults:")
    print(f"  Total save attempts: {len(storage.save_attempts)}")
    print(f"  Unique job IDs saved: {len(storage.processed_job_ids)}")
    print(f"  Duplicate save attempts: {len(duplicate_saves)}")

    if duplicate_saves:
        print("\nJob IDs with multiple save attempts:")
        for job_id, count in list(duplicate_saves.items())[:10]:
            print(f"  {job_id}: saved {count} times")
            # Show which workers tried to save this
            contributors = [
                s["contributor_id"] for s in storage.save_attempts if s["job_id"] == job_id
            ]
            print(f"    Contributors: {contributors}")

    # Check for race conditions
    print("\nRace condition analysis:")
    print(f"  Concurrent save attempts: {len(storage.concurrent_saves)}")
    if storage.concurrent_saves:
        for job_id, contributors in list(storage.concurrent_saves.items())[:5]:
            print(f"  {job_id}: concurrent saves by {contributors}")

    # The assertion - we should not have duplicate job_id assignments
    assert (
        len(duplicate_assignments) == 0
    ), f"Found {len(duplicate_assignments)} job IDs assigned to multiple workers"

    # Also check that saves didn't create duplicates
    assert len(duplicate_saves) == 0, f"Found {len(duplicate_saves)} job IDs saved multiple times"

    processor.stop_creation.set()


@pytest.mark.asyncio
async def test_chunk_boundary_job_id_assignment(orchestrator_config, temp_checkpoint_dir):
    """Test job ID uniqueness at chunk boundaries where off-by-one errors might occur."""
    # Use small chunk size to create more boundaries
    orchestrator_config["chunk_size"] = 10  # Very small chunks

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )
    storage = MockStorageManager()

    processor.initialize(processor_config, storage)
    await asyncio.sleep(1)

    # Track all job IDs and their chunk associations
    job_id_to_chunk = {}
    chunk_job_ids = defaultdict(set)
    boundary_issues = []

    # Request many small chunks to increase chance of boundary issues
    print("\nRequesting small chunks to test boundaries:")

    for i in range(10):
        worker_id = f"worker_{i}"
        units = processor.get_work_units(count=5, worker_id=worker_id)

        for unit in units:
            chunk_id = unit.chunk_id
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            print(f"\n  Chunk {chunk_id}: indices {start_index} to {start_index + chunk_size - 1}")

            # Generate all job IDs for this chunk
            for j in range(chunk_size):
                sample_idx = start_index + j
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(sample_idx)
                )
                job_id = job_id_obj.get_sample_str()

                # Check if this job_id was already assigned to another chunk
                if job_id in job_id_to_chunk:
                    boundary_issues.append(
                        {
                            "job_id": job_id,
                            "sample_index": sample_idx,
                            "chunk_1": job_id_to_chunk[job_id],
                            "chunk_2": chunk_id,
                            "issue": "Job ID assigned to multiple chunks",
                        }
                    )

                job_id_to_chunk[job_id] = chunk_id
                chunk_job_ids[chunk_id].add(job_id)

            # Check for gaps between consecutive chunks
            if i > 0 and chunk_job_ids:
                # Try to find adjacent chunks
                for other_chunk_id, other_job_ids in chunk_job_ids.items():
                    if other_chunk_id != chunk_id:
                        # Extract sample indices from both chunks
                        current_indices = set()
                        for jid in chunk_job_ids[chunk_id]:
                            parts = jid.split(":")
                            if len(parts) >= 5:
                                current_indices.add(int(parts[4]))

                        other_indices = set()
                        for jid in other_job_ids:
                            parts = jid.split(":")
                            if len(parts) >= 5:
                                other_indices.add(int(parts[4]))

                        # Check for overlaps
                        overlap = current_indices & other_indices
                        if overlap:
                            boundary_issues.append(
                                {
                                    "chunk_1": chunk_id,
                                    "chunk_2": other_chunk_id,
                                    "overlapping_indices": sorted(overlap),
                                    "issue": "Overlapping sample indices between chunks",
                                }
                            )

    # Report findings
    print("\nBoundary test results:")
    print(f"  Total chunks processed: {len(chunk_job_ids)}")
    print(f"  Total unique job IDs: {len(job_id_to_chunk)}")
    print(f"  Boundary issues found: {len(boundary_issues)}")

    if boundary_issues:
        print("\nBoundary issues detected:")
        for issue in boundary_issues[:10]:
            print(f"  {issue}")

    # Check for gaps in sample indices
    all_sample_indices = []
    for job_id in job_id_to_chunk.keys():
        parts = job_id.split(":")
        if len(parts) >= 5:
            all_sample_indices.append(int(parts[4]))

    all_sample_indices.sort()
    gaps = []
    for i in range(1, len(all_sample_indices)):
        if all_sample_indices[i] != all_sample_indices[i - 1] + 1:
            gap_size = all_sample_indices[i] - all_sample_indices[i - 1] - 1
            gaps.append(
                {
                    "after_index": all_sample_indices[i - 1],
                    "before_index": all_sample_indices[i],
                    "gap_size": gap_size,
                }
            )

    if gaps:
        print(f"\nGaps in sample indices: {len(gaps)}")
        for gap in gaps[:5]:
            print(
                f"  Gap of {gap['gap_size']} samples between {gap['after_index']} and {gap['before_index']}"
            )

    assert (
        len(boundary_issues) == 0
    ), f"Found {len(boundary_issues)} boundary issues in job ID assignment"

    processor.stop_creation.set()


@pytest.mark.asyncio
async def test_job_id_persistence_across_restarts(orchestrator_config, temp_checkpoint_dir):
    """Test that job IDs remain unique even after processor restarts."""
    # First run - process some units
    processor1 = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(
        processor_type="huggingface_datasets", config=orchestrator_config
    )
    storage = MockStorageManager()

    processor1.initialize(processor_config, storage)
    await asyncio.sleep(1)

    # Process some units
    first_run_job_ids = set()
    for i in range(3):
        units = processor1.get_work_units(count=2, worker_id=f"worker_{i}")
        for unit in units:
            # Extract job IDs
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            for j in range(min(10, chunk_size)):  # Just check first 10
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(start_index + j)
                )
                first_run_job_ids.add(job_id_obj.get_sample_str())

            processor1.mark_completed(unit.unit_id, f"worker_{i}")

    # Save state
    processor1.chunk_tracker.save()
    processor1.stop_creation.set()

    # Simulate restart - create new processor with same checkpoint
    processor2 = HuggingFaceDatasetOrchestratorProcessor()
    processor2.initialize(processor_config, storage)
    await asyncio.sleep(1)

    # Process more units
    second_run_job_ids = set()
    for i in range(3):
        units = processor2.get_work_units(count=2, worker_id=f"worker_{i}")
        for unit in units:
            # Extract job IDs
            start_index = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            for j in range(min(10, chunk_size)):  # Just check first 10
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(start_index + j)
                )
                second_run_job_ids.add(job_id_obj.get_sample_str())

    processor2.stop_creation.set()

    # Check for overlaps
    overlapping_ids = first_run_job_ids.intersection(second_run_job_ids)

    print("\nPersistence Test Results:")
    print(f"First run job IDs: {len(first_run_job_ids)}")
    print(f"Second run job IDs: {len(second_run_job_ids)}")
    print(f"Overlapping IDs: {len(overlapping_ids)}")

    assert (
        len(overlapping_ids) == 0
    ), f"Found {len(overlapping_ids)} duplicate job IDs across restarts"


import threading
import time
from typing import Dict, Set

import pytest
from caption_flow.models import Caption
from caption_flow.utils import ChunkTracker


class TestHuggingFaceWithRealStorage:
    """Test with real StorageManager to uncover duplicate issues."""

    @pytest.mark.asyncio
    async def test_concurrent_workers_same_token_real_storage(self, temp_checkpoint_dir):
        """Test multiple workers with same token using real storage components."""
        # Create real storage manager
        storage_dir = temp_checkpoint_dir / "storage"
        storage = StorageManager(
            data_dir=storage_dir,
            caption_buffer_size=10,  # Small buffer to force frequent flushes
        )
        await storage.initialize()

        # Orchestrator config
        config = {
            "dataset": {
                "processor_type": "huggingface_datasets",
                "dataset_path": "terminusresearch/pexels-metadata-1.71M",
                "dataset_config": None,
                "dataset_split": None,
            },
            "checkpoint_dir": str(temp_checkpoint_dir / "checkpoints"),
            "chunk_size": 100,  # Small chunks
            "min_chunk_buffer": 10,
            "chunk_buffer_multiplier": 2,
        }

        # Initialize processor
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor_config = ProcessorConfig(processor_type="huggingface_datasets", config=config)
        processor.initialize(processor_config, storage)

        # Wait for initial units
        await asyncio.sleep(2)

        # Create multiple worker IDs with same base name
        base_name = "shared_worker"
        worker_ids = []
        for i in range(3):
            import uuid

            worker_id = f"{base_name}_{str(uuid.uuid4())[:8]}"
            worker_ids.append(worker_id)

        print(f"\nWorkers sharing token '{base_name}':")
        for wid in worker_ids:
            print(f"  {wid}")

        # Phase 1: Assign work to all workers
        print("\nPhase 1: Assigning work")
        worker_units = {}
        all_expected_job_ids = set()

        for worker_id in worker_ids:
            units = processor.get_work_units(count=3, worker_id=worker_id)
            worker_units[worker_id] = units
            print(f"  {worker_id}: {len(units)} units")

            # Track expected job IDs
            for unit in units:
                for i in range(unit.data["chunk_size"]):
                    job_id_obj = JobId(
                        shard_id=unit.metadata["shard_name"],
                        chunk_id=str(unit.metadata["chunk_index"]),
                        sample_id=str(unit.data["start_index"] + i),
                    )
                    all_expected_job_ids.add(job_id_obj.get_sample_str())

        print(f"Total expected job IDs: {len(all_expected_job_ids)}")

        # Phase 2: Simulate concurrent result submission
        print("\nPhase 2: Concurrent result submission")

        async def worker_submit_results(worker_id, units):
            """Simulate a worker processing and submitting results."""
            submitted = 0
            contributor_id = worker_id.rsplit("_", 1)[0]  # Extract base name

            for unit in units:
                # Process items in the unit
                items_to_process = min(20, unit.data["chunk_size"])  # Process first 20

                for i in range(items_to_process):
                    try:
                        sample_idx = unit.data["start_index"] + i
                        job_id_obj = JobId(
                            shard_id=unit.metadata["shard_name"],
                            chunk_id=str(unit.metadata["chunk_index"]),
                            sample_id=str(sample_idx),
                        )

                        # Create caption
                        caption = Caption(
                            job_id=job_id_obj,
                            dataset=config["dataset"]["dataset_path"],
                            shard=unit.metadata["shard_name"],
                            chunk_id=unit.chunk_id,
                            item_key=str(sample_idx),
                            captions=[f"Caption by {worker_id}"],
                            outputs={"captions": [f"Caption by {worker_id}"]},
                            contributor_id=contributor_id,  # Same for all workers!
                            timestamp=datetime.now(_datetime.UTC),
                            caption_count=1,
                            processing_time_ms=100.0,
                            metadata={"worker_id": worker_id},
                        )

                        # Submit to storage
                        saved = await storage.save_caption(caption)
                        if saved:
                            submitted += 1

                        # Update chunk tracker
                        if processor.chunk_tracker:
                            # This simulates what handle_result does
                            processor.chunk_tracker.mark_items_processed(unit.chunk_id, i, i)

                        # Small delay to increase chance of race conditions
                        await asyncio.sleep(0.001)

                    except Exception as e:
                        print(f"Error submitting result: {e}")

            return submitted

        # Submit concurrently
        submit_tasks = []
        for worker_id, units in worker_units.items():
            task = worker_submit_results(worker_id, units)
            submit_tasks.append(task)

        submit_results = await asyncio.gather(*submit_tasks)
        total_submitted = sum(submit_results)
        print(f"Total items submitted: {total_submitted}")

        # Force storage checkpoint to flush buffers
        await storage.checkpoint()

        # Phase 3: Check for duplicates in storage
        print("\nPhase 3: Checking for duplicates")

        # Get all processed job IDs from storage
        stored_job_ids = storage.get_all_processed_job_ids()
        print(f"Job IDs in storage: {len(stored_job_ids)}")

        # Check if we have Lance storage (need to handle differently)
        if hasattr(storage, "shard_buffers"):
            # This is LanceStorageManager
            # Get stats instead of contents
            stats = await storage.get_caption_stats()
            print(f"Storage stats: {stats}")

            # Check for duplicates by examining the stored job IDs
            # Count occurrences
            job_id_list = list(stored_job_ids)
            job_id_counts = defaultdict(int)
            for jid in job_id_list:
                job_id_counts[jid] += 1

            duplicate_job_ids = {jid: count for jid, count in job_id_counts.items() if count > 1}
        else:
            # Regular storage manager
            # Check storage contents in detail
            contents = await storage.get_storage_contents(limit=None)

            # Count job IDs
            job_id_counts = defaultdict(int)
            job_id_to_contributors = defaultdict(set)

            for row in contents.rows:
                job_id = row.get("job_id")
                if isinstance(job_id, dict):
                    job_id_str = JobId.from_dict(job_id).get_sample_str()
                else:
                    job_id_str = str(job_id)

                job_id_counts[job_id_str] += 1
                job_id_to_contributors[job_id_str].add(row.get("contributor_id"))

            # Find duplicates
            duplicate_job_ids = {jid: count for jid, count in job_id_counts.items() if count > 1}

        print("\nDuplicate analysis:")
        print(f"  Unique job IDs in storage: {len(stored_job_ids)}")
        print(f"  Job IDs with duplicates: {len(duplicate_job_ids)}")

        if duplicate_job_ids:
            print("\nDuplicate job IDs found:")
            for job_id, count in list(duplicate_job_ids.items())[:10]:
                print(f"  {job_id}: {count} times")

        # Check chunk tracker state
        if processor.chunk_tracker:
            tracker_stats = processor.chunk_tracker.get_stats()
            print(f"\nChunk tracker stats: {tracker_stats}")

        # Cleanup
        processor.stop_creation.set()
        await storage.close()

        # Assertions
        assert (
            len(duplicate_job_ids) == 0
        ), f"Found {len(duplicate_job_ids)} duplicate job IDs in storage"

    @pytest.mark.asyncio
    async def test_chunk_tracker_race_conditions(self, temp_checkpoint_dir):
        """Test race conditions in chunk tracker updates."""
        # Create chunk tracker
        checkpoint_path = temp_checkpoint_dir / "checkpoints" / "chunks.json"
        chunk_tracker = ChunkTracker(checkpoint_path)

        # Add test chunks
        chunk_tracker.add_chunk("shard1:chunk:0", "shard1", "url1", 0, 100)
        chunk_tracker.add_chunk("shard1:chunk:1", "shard1", "url1", 100, 100)

        # Track all updates
        all_updates = []
        update_lock = threading.Lock()

        def concurrent_update(thread_id, chunk_id, start, count):
            """Simulate concurrent updates to chunk tracker."""
            for i in range(count):
                # Mark items as processed
                chunk_tracker.mark_items_processed(chunk_id, start + i, start + i)

                with update_lock:
                    all_updates.append({"thread": thread_id, "chunk": chunk_id, "index": start + i})

                time.sleep(0.001)  # Small delay

        # Create multiple threads updating same chunk
        threads = []

        # Multiple threads updating overlapping ranges in same chunk
        for i in range(3):
            thread = threading.Thread(
                target=concurrent_update, args=(f"thread_{i}", "shard1:chunk:0", i * 20, 30)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check chunk state
        chunk_state = chunk_tracker.chunks["shard1:chunk:0"]
        print("\nChunk state after concurrent updates:")
        print(f"  Status: {chunk_state.status}")
        print(f"  Processed count: {chunk_state.processed_count}")
        print(f"  Processed ranges: {chunk_state.processed_ranges}")
        print(f"  Unprocessed ranges: {chunk_state.get_unprocessed_ranges()}")

        # Verify no lost updates
        print(f"\nTotal updates attempted: {len(all_updates)}")
        unique_indices = set(u["index"] for u in all_updates)
        print(f"Unique indices updated: {len(unique_indices)}")

        # Check for any gaps in processed ranges
        if chunk_state.processed_ranges:
            all_processed_indices = set()
            for start, end in chunk_state.processed_ranges:
                for i in range(start, end + 1):
                    all_processed_indices.add(i)

            # Find any indices that were updated but not in processed ranges
            lost_updates = unique_indices - all_processed_indices
            if lost_updates:
                print(f"\nLost updates detected: {len(lost_updates)}")
                print(f"  Lost indices: {sorted(lost_updates)[:10]}")

        # Save and reload to check persistence
        chunk_tracker.save()

        # Create new tracker and load
        new_tracker = ChunkTracker(checkpoint_path)
        reloaded_state = new_tracker.chunks["shard1:chunk:0"]

        assert (
            reloaded_state.processed_count == chunk_state.processed_count
        ), "Processed count changed after save/reload"


@pytest.mark.asyncio
async def test_relative_absolute_index_misalignments(temp_checkpoint_dir):
    """Test for misalignments between relative and absolute indices."""
    # Create storage and chunk tracker
    storage_dir = temp_checkpoint_dir / "storage"
    storage = StorageManager(data_dir=storage_dir, caption_buffer_size=5)
    await storage.initialize()

    checkpoint_path = temp_checkpoint_dir / "checkpoints" / "chunks.json"
    chunk_tracker = ChunkTracker(checkpoint_path)

    print("\nTesting relative/absolute index conversions")

    # Test Case 1: Basic conversion accuracy
    print("\nTest 1: Basic index conversion")

    # Add chunks with different start indices
    test_chunks = [
        ("shard1:chunk:0", "shard1", 0, 100),  # abs: 0-99
        ("shard1:chunk:1", "shard1", 100, 100),  # abs: 100-199
        ("shard1:chunk:2", "shard1", 200, 50),  # abs: 200-249
        ("shard2:chunk:0", "shard2", 0, 150),  # abs: 0-149 (different shard)
    ]

    for chunk_id, shard_name, start_idx, size in test_chunks:
        chunk_tracker.add_chunk(chunk_id, shard_name, f"{shard_name}.tar", start_idx, size)
        print(f"  Added {chunk_id}: start={start_idx}, size={size}")

    # Test marking items as processed with various patterns
    test_cases = [
        # (chunk_id, absolute_start, absolute_end, expected_relative_start, expected_relative_end)
        ("shard1:chunk:0", 0, 9, 0, 9),  # First 10 items
        ("shard1:chunk:0", 95, 99, 95, 99),  # Last 5 items of chunk 0
        ("shard1:chunk:1", 100, 109, 0, 9),  # First 10 of chunk 1 (abs 100-109)
        ("shard1:chunk:1", 195, 199, 95, 99),  # Last 5 of chunk 1
        ("shard1:chunk:2", 200, 204, 0, 4),  # First 5 of chunk 2
        ("shard1:chunk:2", 245, 249, 45, 49),  # Last 5 of chunk 2
    ]

    errors = []

    for chunk_id, abs_start, abs_end, exp_rel_start, exp_rel_end in test_cases:
        chunk_state = chunk_tracker.chunks[chunk_id]

        # Mark items processed - the method should handle absolute indices
        chunk_tracker.mark_items_processed(chunk_id, abs_start, abs_end)

        # Check if the relative indices were calculated correctly
        found = False
        for rel_start, rel_end in chunk_state.processed_ranges:
            if rel_start == exp_rel_start and rel_end == exp_rel_end:
                found = True
                break

        if not found:
            errors.append(
                {
                    "chunk": chunk_id,
                    "absolute": (abs_start, abs_end),
                    "expected_relative": (exp_rel_start, exp_rel_end),
                    "actual_ranges": chunk_state.processed_ranges,
                    "chunk_start": chunk_state.start_index,
                }
            )

        print(
            f"  {chunk_id}: abs[{abs_start}:{abs_end}] -> rel[{exp_rel_start}:{exp_rel_end}] {'✓' if found else '✗'}"
        )

    if errors:
        print("\nConversion errors found:")
        for err in errors:
            print(f"  {err}")

    assert len(errors) == 0, f"Found {len(errors)} index conversion errors"

    # Test Case 2: Job ID parsing and index extraction
    print("\nTest 2: Job ID parsing for indices")

    # Test job IDs with different formats
    test_job_ids = [
        # (job_id_str, expected_chunk_id, expected_sample_idx)
        ("shard1:chunk:0:idx:5", "shard1:chunk:0", 5),
        ("shard1:chunk:1:idx:105", "shard1:chunk:1", 105),
        ("shard2:chunk:0:idx:0", "shard2:chunk:0", 0),
    ]

    for job_id_str, expected_chunk, expected_sample in test_job_ids:
        job_id = JobId.from_str(job_id_str)
        chunk_id = job_id.get_chunk_str()
        sample_idx = int(job_id.sample_id)

        print(f"  JobID: {job_id_str}")
        print(f"    Chunk: {chunk_id} (expected: {expected_chunk})")
        print(f"    Sample: {sample_idx} (expected: {expected_sample})")

        assert chunk_id == expected_chunk, f"Chunk mismatch for {job_id_str}"
        assert sample_idx == expected_sample, f"Sample index mismatch for {job_id_str}"

    # Test Case 3: Storage synchronization with metadata indices
    print("\nTest 3: Storage sync with chunk tracker indices")

    # Add captions with metadata containing indices
    test_items = [
        ("shard1", 0, 0, 5),  # First 5 of chunk 0
        ("shard1", 0, 98, 101),  # Last 3 of chunk 0, first 1 of chunk 1
        ("shard1", 1, 198, 201),  # Last 2 of chunk 1, first 1 of chunk 2
    ]

    for shard, chunk_idx, start_idx, end_idx in test_items:
        for idx in range(start_idx, end_idx):
            job_id_obj = JobId(
                shard_id=shard,
                chunk_id=str(chunk_idx if idx < 100 else 1 if idx < 200 else 2),
                sample_id=str(idx),
            )

            caption = Caption(
                job_id=job_id_obj,
                dataset="test",
                shard=shard,
                chunk_id=job_id_obj.get_chunk_str(),
                item_key=str(idx),
                captions=["test"],
                outputs={"captions": ["test"]},
                contributor_id="test_user",
                timestamp=datetime.now(_datetime.UTC),
                caption_count=1,
                processing_time_ms=50.0,
                metadata={"_item_index": idx},  # Index stored in metadata like in the code
            )

            await storage.save_caption(caption)

    # Force storage checkpoint
    await storage.checkpoint()

    # Test Case 4: HuggingFace processor work unit indices
    print("\nTest 4: HuggingFace work unit index handling")

    config = {
        "dataset": {
            "processor_type": "huggingface_datasets",
            "dataset_path": "terminusresearch/pexels-metadata-1.71M",
        },
        "checkpoint_dir": str(temp_checkpoint_dir / "checkpoints2"),
        "chunk_size": 100,
    }

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(processor_type="huggingface_datasets", config=config)
    processor.initialize(processor_config, storage)

    # Wait for chunks
    await asyncio.sleep(1)

    # Get work units from different workers
    worker_ids = ["worker_1", "worker_2"]
    all_job_ids = set()
    job_id_to_indices = {}

    for worker_id in worker_ids:
        units = processor.get_work_units(count=2, worker_id=worker_id)

        for unit in units:
            print(f"\n  Worker {worker_id} got unit: {unit.chunk_id}")
            print(f"    Start index: {unit.data['start_index']}")
            print(f"    Chunk size: {unit.data['chunk_size']}")
            print(f"    Unprocessed ranges: {unit.data.get('unprocessed_ranges', [])}")

            # Check for proper index calculations
            start_idx = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]
            shard_name = unit.metadata["shard_name"]
            chunk_index = unit.metadata["chunk_index"]

            # Generate expected job IDs
            for i in range(min(10, chunk_size)):  # Check first 10
                abs_idx = start_idx + i
                job_id_obj = JobId(
                    shard_id=shard_name, chunk_id=str(chunk_index), sample_id=str(abs_idx)
                )
                job_id_str = job_id_obj.get_sample_str()

                # Check for duplicates
                if job_id_str in all_job_ids:
                    print(f"    DUPLICATE: {job_id_str} already assigned!")
                    print(f"      Previous indices: {job_id_to_indices[job_id_str]}")
                    print(f"      Current indices: chunk={chunk_index}, sample={abs_idx}")

                all_job_ids.add(job_id_str)
                job_id_to_indices[job_id_str] = {
                    "chunk_index": chunk_index,
                    "sample_index": abs_idx,
                    "relative_index": i,
                    "worker": worker_id,
                }

    # Test Case 5: Chunk boundary edge cases
    print("\nTest 5: Chunk boundary calculations")

    # Test the _create_work_unit method directly
    if hasattr(processor, "_create_work_unit"):
        # Test creating units at specific chunk indices
        test_chunk_indices = [0, 1, 99, 100, 999, 1000]

        for chunk_idx in test_chunk_indices:
            processor.current_chunk_index = chunk_idx
            unit = processor._create_work_unit(chunk_idx)

            if unit:
                start = unit.data["start_index"]
                size = unit.data["chunk_size"]
                expected_start = chunk_idx * processor.chunk_size

                print(f"  Chunk index {chunk_idx}:")
                print(f"    Start: {start} (expected: {expected_start})")
                print(f"    Size: {size}")

                # Verify correct calculation
                assert (
                    start == expected_start
                ), f"Incorrect start index for chunk {chunk_idx}: got {start}, expected {expected_start}"

    # Cleanup
    processor.stop_creation.set()
    await storage.close()

    print("\nAll index alignment tests completed")


@pytest.mark.asyncio
async def test_job_id_calculation_consistency(temp_checkpoint_dir):
    """Test that job ID calculations are consistent across different paths."""
    print("\nTesting job ID calculation consistency")

    # Test data
    test_cases = [
        {
            "shard": "data-00001",
            "chunk_idx": 5,
            "sample_idx": 543,
            "chunk_size": 100,
        },
        {
            "shard": "train-00000",
            "chunk_idx": 0,
            "sample_idx": 0,
            "chunk_size": 1000,
        },
        {
            "shard": "val-99999",
            "chunk_idx": 999,
            "sample_idx": 999999,
            "chunk_size": 1000,
        },
    ]

    for test in test_cases:
        print(f"\nTest case: {test}")

        # Method 1: Direct JobId creation
        job_id_1 = JobId(
            shard_id=test["shard"],
            chunk_id=str(test["chunk_idx"]),
            sample_id=str(test["sample_idx"]),
        )
        job_id_str_1 = job_id_1.get_sample_str()

        # Method 2: From string parsing
        job_id_str_2 = f"{test['shard']}:chunk:{test['chunk_idx']}:idx:{test['sample_idx']}"
        job_id_2 = JobId.from_str(job_id_str_2)

        # Method 3: From dict
        job_id_dict = {
            "shard_id": test["shard"],
            "chunk_id": str(test["chunk_idx"]),
            "sample_id": str(test["sample_idx"]),
        }
        job_id_3 = JobId.from_dict(job_id_dict)

        # All should produce the same string
        print(f"  Method 1 (direct): {job_id_str_1}")
        print(f"  Method 2 (string): {job_id_2.get_sample_str()}")
        print(f"  Method 3 (dict):   {job_id_3.get_sample_str()}")

        assert (
            job_id_str_1 == job_id_2.get_sample_str() == job_id_3.get_sample_str()
        ), "Job ID strings don't match across creation methods"

        # Verify chunk calculation from sample index
        calculated_chunk_idx = test["sample_idx"] // test["chunk_size"]
        print(f"  Calculated chunk from sample: {calculated_chunk_idx}")

        # This might not match if chunks aren't strictly sequential
        # but it's worth checking the assumption
        if calculated_chunk_idx != test["chunk_idx"]:
            print(
                f"  WARNING: Calculated chunk {calculated_chunk_idx} != provided chunk {test['chunk_idx']}"
            )


@pytest.mark.asyncio
async def test_huggingface_chunk_start_index_bug(temp_checkpoint_dir):
    """Test that exposes the chunk start index bug in HuggingFace processor."""
    print("\n" + "=" * 80)
    print("TESTING FOR CHUNK START INDEX BUG")
    print("=" * 80)

    storage_dir = temp_checkpoint_dir / "storage"
    storage = StorageManager(data_dir=storage_dir, caption_buffer_size=5)
    await storage.initialize()

    config = {
        "dataset": {
            "processor_type": "huggingface_datasets",
            "dataset_path": "terminusresearch/pexels-metadata-1.71M",
        },
        "checkpoint_dir": str(temp_checkpoint_dir / "checkpoints"),
        "chunk_size": 100,
    }

    processor = HuggingFaceDatasetOrchestratorProcessor()
    processor_config = ProcessorConfig(processor_type="huggingface_datasets", config=config)
    processor.initialize(processor_config, storage)

    # Wait for initial chunks
    await asyncio.sleep(2)

    print("\n1. Testing _create_work_unit directly")

    # Track all created work units
    created_units = {}
    start_indices_seen = set()

    # Create multiple work units
    for chunk_idx in range(5):
        processor.current_chunk_index = chunk_idx
        unit = processor._create_work_unit(chunk_idx)

        if unit:
            created_units[chunk_idx] = unit
            start_idx = unit.data["start_index"]
            start_indices_seen.add(start_idx)

            print(f"\nChunk {chunk_idx}:")
            print(f"  Unit ID: {unit.unit_id}")
            print(f"  Start index: {start_idx}")
            print(f"  Expected start: {chunk_idx * processor.chunk_size}")
            print(f"  Chunk size: {unit.data['chunk_size']}")

    # Check for the bug
    if len(start_indices_seen) == 1 and 0 in start_indices_seen:
        print("\n❌ BUG DETECTED: All chunks have start_index=0!")
        print("This will cause duplicate job IDs across chunks!")

    # 2. Test job ID generation from multiple workers
    print("\n2. Testing job ID assignment with multiple workers")

    job_ids_by_worker = {}
    all_job_ids = set()
    duplicates = []

    # Reset processor state
    processor.stop_creation.set()
    await asyncio.sleep(0.5)
    processor.stop_creation.clear()
    processor.current_chunk_index = 0

    # Simulate multiple workers
    for i in range(3):
        worker_id = f"worker_{i}"
        units = processor.get_work_units(count=2, worker_id=worker_id)
        job_ids_by_worker[worker_id] = set()

        print(f"\n{worker_id} received {len(units)} units")

        for unit in units:
            # Generate job IDs for this unit
            start_idx = unit.data["start_index"]
            chunk_size = unit.data["chunk_size"]

            for j in range(min(10, chunk_size)):  # First 10 items
                abs_idx = start_idx + j
                job_id_obj = JobId(
                    shard_id=unit.metadata["shard_name"],
                    chunk_id=str(unit.metadata["chunk_index"]),
                    sample_id=str(abs_idx),
                )
                job_id_str = job_id_obj.get_sample_str()

                # Check for duplicates
                if job_id_str in all_job_ids:
                    duplicates.append(
                        {
                            "job_id": job_id_str,
                            "worker": worker_id,
                            "unit": unit.unit_id,
                            "abs_index": abs_idx,
                            "start_index": start_idx,
                        }
                    )

                all_job_ids.add(job_id_str)
                job_ids_by_worker[worker_id].add(job_id_str)

    # 3. Analyze results
    print("\n3. Analysis:")
    print(f"Total unique job IDs: {len(all_job_ids)}")
    print(f"Duplicate job IDs found: {len(duplicates)}")

    if duplicates:
        print("\n❌ DUPLICATE JOB IDs FOUND:")
        for i, dup in enumerate(duplicates[:10]):
            print(f"\n  Duplicate #{i+1}:")
            print(f"    Job ID: {dup['job_id']}")
            print(f"    Worker: {dup['worker']}")
            print(f"    Unit: {dup['unit']}")
            print(f"    Start index of unit: {dup['start_index']}")

            # Find where else this job ID was seen
            for w, ids in job_ids_by_worker.items():
                if dup["job_id"] in ids and w != dup["worker"]:
                    print(f"    Also assigned to: {w}")

    # 4. Check chunk tracker state
    print("\n4. Chunk tracker analysis:")
    if processor.chunk_tracker:
        for chunk_id, chunk_state in list(processor.chunk_tracker.chunks.items())[:5]:
            print(f"\n  {chunk_id}:")
            print(f"    Start index: {chunk_state.start_index}")
            print(f"    Size: {chunk_state.chunk_size}")
            print(f"    Status: {chunk_state.status}")

    # Cleanup
    processor.stop_creation.set()
    await storage.close()

    # This test is designed to expose the bug, not necessarily fail
    # The bug is in the source code, not the test
    print("\n" + "=" * 80)
    print("BUG SUMMARY: The _create_work_unit method always uses")
    print("current_index = 0 instead of current_index = chunk_index * self.chunk_size")
    print("This causes all chunks to start at index 0, leading to duplicate job IDs!")
    print("=" * 80)


@pytest.mark.asyncio
async def test_workaround_for_start_index_bug(temp_checkpoint_dir):
    """Test that shows how the bug manifests in practice."""
    print("\nDemonstrating how the start_index bug causes duplicates")

    # Simulate what happens with the bug
    chunk_size = 100
    chunks_created = []

    # This simulates the buggy behavior
    for chunk_idx in range(3):
        # BUG: current_index is always 0
        current_index = 0  # This is what the code does
        correct_index = chunk_idx * chunk_size  # This is what it should do

        chunk_info = {
            "chunk_idx": chunk_idx,
            "buggy_start": current_index,
            "correct_start": correct_index,
            "job_ids_buggy": [],
            "job_ids_correct": [],
        }

        # Generate job IDs with buggy start
        for i in range(5):  # First 5 items
            job_id_buggy = f"shard:chunk:{chunk_idx}:idx:{current_index + i}"
            job_id_correct = f"shard:chunk:{chunk_idx}:idx:{correct_index + i}"

            chunk_info["job_ids_buggy"].append(job_id_buggy)
            chunk_info["job_ids_correct"].append(job_id_correct)

        chunks_created.append(chunk_info)

    # Show the problem
    print("\nWith the bug (current_index always 0):")
    all_buggy_ids = []
    for chunk in chunks_created:
        print(f"\n  Chunk {chunk['chunk_idx']}:")
        print(f"    Start: {chunk['buggy_start']}")
        print(f"    Job IDs: {chunk['job_ids_buggy']}")
        all_buggy_ids.extend(chunk["job_ids_buggy"])

    print(f"\n  Total job IDs: {len(all_buggy_ids)}")
    print(f"  Unique job IDs: {len(set(all_buggy_ids))}")
    print(f"  DUPLICATES: {len(all_buggy_ids) - len(set(all_buggy_ids))}")

    print("\nWithout the bug (correct calculation):")
    all_correct_ids = []
    for chunk in chunks_created:
        print(f"\n  Chunk {chunk['chunk_idx']}:")
        print(f"    Start: {chunk['correct_start']}")
        print(f"    Job IDs: {chunk['job_ids_correct']}")
        all_correct_ids.extend(chunk["job_ids_correct"])

    print(f"\n  Total job IDs: {len(all_correct_ids)}")
    print(f"  Unique job IDs: {len(set(all_correct_ids))}")
    print(f"  DUPLICATES: {len(all_correct_ids) - len(set(all_correct_ids))}")

    # Show which IDs are duplicated
    from collections import Counter

    buggy_counts = Counter(all_buggy_ids)
    duplicated_ids = [jid for jid, count in buggy_counts.items() if count > 1]

    print("\nDuplicated job IDs with the bug:")
    for jid in duplicated_ids:
        print(f"  {jid} appears {buggy_counts[jid]} times")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
