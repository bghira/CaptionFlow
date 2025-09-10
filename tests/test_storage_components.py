import asyncio
import datetime as _datetime
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest
from caption_flow.models import Caption, JobId
from caption_flow.processors import ProcessorConfig
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor
from caption_flow.storage import StorageManager
from caption_flow.utils import ChunkTracker


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestHuggingFaceWithRealStorage:
    """Test with real StorageManager to uncover duplicate issues."""

    @pytest.mark.asyncio
    async def _setup_storage_and_processor(self, temp_checkpoint_dir):
        """Setup storage manager and processor for testing."""
        storage_dir = temp_checkpoint_dir / "storage"
        storage = StorageManager(data_dir=storage_dir, caption_buffer_size=10)
        await storage.initialize()

        config = {
            "dataset": {
                "processor_type": "huggingface_datasets",
                "dataset_path": "terminusresearch/pexels-metadata-1.71M",
                "dataset_config": None,
                "dataset_split": None,
            },
            "checkpoint_dir": str(temp_checkpoint_dir / "checkpoints"),
            "chunk_size": 100,
            "min_chunk_buffer": 10,
            "chunk_buffer_multiplier": 2,
        }

        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor_config = ProcessorConfig(processor_type="huggingface_datasets", config=config)
        processor.initialize(processor_config, storage)

        await asyncio.sleep(2)  # Wait for initial units
        return storage, processor, config

    def _create_worker_ids(self, base_name="shared_worker", count=3):
        """Create multiple worker IDs with same base name."""
        import uuid

        worker_ids = []
        for _ in range(count):
            worker_id = f"{base_name}_{str(uuid.uuid4())[:8]}"
            worker_ids.append(worker_id)

        print(f"\nWorkers sharing token '{base_name}':")
        for wid in worker_ids:
            print(f"  {wid}")

        return worker_ids

    def _assign_work_to_workers(self, processor, worker_ids):
        """Assign work units to all workers."""
        print("\nPhase 1: Assigning work")
        worker_units = {}
        all_expected_job_ids = set()

        for worker_id in worker_ids:
            units = processor.get_work_units(count=3, worker_id=worker_id)
            worker_units[worker_id] = units
            print(f"  {worker_id}: {len(units)} units")

            for unit in units:
                for i in range(unit.data["chunk_size"]):
                    job_id_obj = JobId(
                        shard_id=unit.metadata["shard_name"],
                        chunk_id=str(unit.metadata["chunk_index"]),
                        sample_id=str(unit.data["start_index"] + i),
                    )
                    all_expected_job_ids.add(job_id_obj.get_sample_str())

        print(f"Total expected job IDs: {len(all_expected_job_ids)}")
        return worker_units, all_expected_job_ids

    async def _simulate_concurrent_submission(self, storage, processor, worker_units, config):
        """Simulate concurrent result submission from multiple workers."""
        print("\nPhase 2: Concurrent result submission")

        async def worker_submit_results(worker_id, units):
            submitted = 0
            contributor_id = worker_id.rsplit("_", 1)[0]

            for unit in units:
                items_to_process = min(20, unit.data["chunk_size"])

                for i in range(items_to_process):
                    try:
                        sample_idx = unit.data["start_index"] + i
                        job_id_obj = JobId(
                            shard_id=unit.metadata["shard_name"],
                            chunk_id=str(unit.metadata["chunk_index"]),
                            sample_id=str(sample_idx),
                        )

                        caption = Caption(
                            job_id=job_id_obj,
                            dataset=config["dataset"]["dataset_path"],
                            shard=unit.metadata["shard_name"],
                            chunk_id=unit.chunk_id,
                            item_key=str(sample_idx),
                            captions=[f"Caption by {worker_id}"],
                            outputs={"captions": [f"Caption by {worker_id}"]},
                            contributor_id=contributor_id,
                            timestamp=datetime.now(_datetime.UTC),
                            caption_count=1,
                            processing_time_ms=100.0,
                            metadata={"worker_id": worker_id},
                        )

                        await storage.save_caption(caption)
                        submitted += 1

                        if processor.chunk_tracker:
                            processor.chunk_tracker.mark_items_processed(unit.chunk_id, i, i)

                        await asyncio.sleep(0.001)

                    except Exception as e:
                        print(f"Error submitting result: {e}")

            return submitted

        submit_tasks = [
            worker_submit_results(worker_id, units) for worker_id, units in worker_units.items()
        ]

        submit_results = await asyncio.gather(*submit_tasks)
        total_submitted = sum(submit_results)
        print(f"Total items submitted: {total_submitted}")

        await storage.checkpoint()
        return total_submitted

    async def _check_for_duplicates(self, storage, processor):
        """Check storage for duplicate job IDs."""
        print("\nPhase 3: Checking for duplicates")

        stored_job_ids = storage.get_all_processed_job_ids()
        print(f"Job IDs in storage: {len(stored_job_ids)}")

        contents = await storage.get_storage_contents(limit=1000)

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

        duplicate_job_ids = {jid: count for jid, count in job_id_counts.items() if count > 1}

        print("\nDuplicate analysis:")
        print(f"  Unique job IDs in storage: {len(job_id_counts)}")
        print(f"  Job IDs with duplicates: {len(duplicate_job_ids)}")

        if duplicate_job_ids:
            print("\nDuplicate job IDs found:")
            for job_id, count in list(duplicate_job_ids.items())[:10]:
                print(f"  {job_id}: {count} times")
                print(f"    Contributors: {job_id_to_contributors[job_id]}")

        if processor.chunk_tracker:
            tracker_stats = processor.chunk_tracker.get_stats()
            print(f"\nChunk tracker stats: {tracker_stats}")

        return duplicate_job_ids

    @pytest.mark.asyncio
    async def test_concurrent_workers_same_token_real_storage(self, temp_checkpoint_dir):
        """Test multiple workers with same token using real storage components."""
        storage, processor, config = await self._setup_storage_and_processor(temp_checkpoint_dir)

        worker_ids = self._create_worker_ids()
        worker_units, all_expected_job_ids = self._assign_work_to_workers(processor, worker_ids)

        await self._simulate_concurrent_submission(storage, processor, worker_units, config)
        duplicate_job_ids = await self._check_for_duplicates(storage, processor)

        processor.stop_creation.set()
        await storage.close()

        assert (
            len(duplicate_job_ids) == 0
        ), f"Found {len(duplicate_job_ids)} duplicate job IDs in storage"

    @pytest.mark.asyncio
    async def test_storage_chunk_tracker_sync_issues(self, temp_checkpoint_dir):
        """Test synchronization issues between storage and chunk tracker."""
        # Create storage
        storage_dir = temp_checkpoint_dir / "storage"
        storage = StorageManager(data_dir=storage_dir, caption_buffer_size=5)  # Very small buffer
        await storage.initialize()

        # Create chunk tracker
        checkpoint_path = temp_checkpoint_dir / "checkpoints" / "chunks.json"
        chunk_tracker = ChunkTracker(checkpoint_path)

        # Add some chunks
        chunk_tracker.add_chunk("shard1:chunk:0", "shard1", "url1", 0, 100)
        chunk_tracker.add_chunk("shard1:chunk:1", "shard1", "url1", 100, 100)

        # Simulate concurrent updates
        print("\nSimulating concurrent updates to storage and chunk tracker")

        async def update_storage_and_tracker(worker_id, chunk_id, start_idx, count):
            """Simulate a worker updating both storage and tracker."""
            results = []

            for i in range(count):
                sample_idx = start_idx + i
                job_id_obj = JobId(
                    shard_id="shard1", chunk_id=chunk_id.split(":")[-1], sample_id=str(sample_idx)
                )

                # Save to storage
                caption = Caption(
                    job_id=job_id_obj,
                    dataset="test",
                    shard="shard1",
                    chunk_id=chunk_id,
                    item_key=str(sample_idx),
                    captions=["test"],
                    outputs={"captions": ["test"]},
                    contributor_id=worker_id,
                    timestamp=datetime.now(_datetime.UTC),
                    caption_count=1,
                    processing_time_ms=50.0,
                    metadata={},
                )

                await storage.save_caption(caption)

                # Update chunk tracker (with potential race condition)
                chunk_tracker.mark_items_processed(chunk_id, i, i)

                results.append(job_id_obj.get_sample_str())

                # Introduce some delay
                await asyncio.sleep(0.001)

            return results

        # Run concurrent updates
        tasks = []

        # Multiple workers updating same chunk
        for i in range(3):
            task = update_storage_and_tracker(
                f"worker_{i}",
                "shard1:chunk:0",
                i * 10,
                10,  # Different ranges
            )
            tasks.append(task)

        # Workers updating different chunks
        for i in range(2):
            task = update_storage_and_tracker(f"worker_{i+3}", "shard1:chunk:1", i * 20, 20)
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Force checkpoint
        await storage.checkpoint()
        chunk_tracker.save()

        # Check consistency
        print("\nChecking consistency between storage and chunk tracker")

        # Get all job IDs from storage
        storage_job_ids = storage.get_all_processed_job_ids()

        # Get processed ranges from chunk tracker
        chunk_states = {}
        for chunk_id, state in chunk_tracker.chunks.items():
            chunk_states[chunk_id] = {
                "status": state.status,
                "processed_count": state.processed_count,
                "processed_ranges": state.processed_ranges,
                "unprocessed_ranges": state.get_unprocessed_ranges(),
            }

        print(f"Storage has {len(storage_job_ids)} job IDs")
        print(f"Chunk states: {chunk_states}")

        # Verify no duplicates in storage
        contents = await storage.get_storage_contents()
        job_id_counts = defaultdict(int)

        for row in contents.rows:
            job_id = row.get("job_id")
            if isinstance(job_id, dict):
                job_id_str = JobId.from_dict(job_id).get_sample_str()
            else:
                job_id_str = str(job_id)
            job_id_counts[job_id_str] += 1

        duplicates = {jid: count for jid, count in job_id_counts.items() if count > 1}

        if duplicates:
            print(f"\nFound {len(duplicates)} duplicates in storage!")
            for jid, count in list(duplicates.items())[:5]:
                print(f"  {jid}: {count} times")

        await storage.close()

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicates in storage"
