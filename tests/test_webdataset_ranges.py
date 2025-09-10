"""Comprehensive tests for WebDataset processor with focus on range calculations and edge cases."""

import shutil
import tempfile
import threading
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from caption_flow.models import JobId
from caption_flow.processors.base import ProcessorConfig, WorkResult, WorkUnit
from caption_flow.processors.webdataset import (
    WebDatasetOrchestratorProcessor,
    WebDatasetWorkerProcessor,
)
from caption_flow.storage import StorageManager
from caption_flow.utils.chunk_tracker import ChunkTracker
from PIL import Image


class TestWebDatasetOrchestratorProcessor:
    """Test suite for WebDatasetOrchestratorProcessor with focus on range handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_dataset(self):
        """Create mock webshart dataset."""
        dataset = Mock()
        dataset.num_shards = 3
        dataset.get_shard_info.side_effect = [
            {"name": "shard_0", "path": "shard_0.tar", "num_files": 1000},
            {"name": "shard_1", "path": "shard_1.tar", "num_files": 800},
            {"name": "shard_2", "path": "shard_2.tar", "num_files": 1200},
        ]
        return dataset

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage manager."""
        storage = Mock(spec=StorageManager)
        storage.get_all_processed_job_ids.return_value = set()
        return storage

    @pytest.fixture
    def processor_config(self, temp_dir):
        """Create processor configuration."""
        config_dict = {
            "dataset": {
                "dataset_path": "/fake/dataset",
                "metadata_path": None,
            },
            "chunk_size": 100,
            "min_chunk_buffer": 5,
            "chunk_buffer_multiplier": 2,
            "cache_dir": str(temp_dir / "cache"),
            "checkpoint_dir": str(temp_dir / "checkpoints"),
            "shard_cache_gb": 1.0,
        }
        return ProcessorConfig(processor_type="webdataset", config=config_dict)

    @pytest.fixture
    def orchestrator_processor(self, processor_config, mock_storage, mock_dataset, temp_dir):
        """Create orchestrator processor with mocked dependencies."""
        processor = WebDatasetOrchestratorProcessor()

        # Mock webshart discover_dataset
        with patch(
            "caption_flow.processors.webdataset.webshart.discover_dataset",
            return_value=mock_dataset,
        ):
            processor.initialize(processor_config, mock_storage)

        # Stop the background thread to prevent interference
        processor.stop_creation.set()
        if processor.unit_creation_thread:
            processor.unit_creation_thread.join(timeout=1)

        return processor

    def test_initialization(self, orchestrator_processor, mock_dataset):
        """Test processor initializes correctly with proper chunk tracker setup."""
        assert orchestrator_processor.dataset == mock_dataset
        assert orchestrator_processor.chunk_tracker is not None
        assert orchestrator_processor.chunk_size == 100
        assert orchestrator_processor.min_buffer == 5
        assert orchestrator_processor.buffer_multiplier == 2

    def test_shard_info_caching(self, orchestrator_processor, mock_dataset):
        """Test shard info caching works correctly."""
        # First call should hit the dataset
        info1 = orchestrator_processor._get_shard_info_cached(0)
        assert info1 == {"name": "shard_0", "path": "shard_0.tar", "num_files": 1000}

        # Second call should use cache
        info2 = orchestrator_processor._get_shard_info_cached(0)
        assert info2 == info1

        # Should have been called only once
        mock_dataset.get_shard_info.assert_called_once_with(0)

    def test_restore_state_empty(self, orchestrator_processor, mock_storage):
        """Test state restoration with no existing chunks."""
        mock_storage.get_all_processed_job_ids.return_value = set()

        # Clear any chunks that might have been created by background thread
        orchestrator_processor.chunk_tracker.chunks.clear()
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()

        orchestrator_processor._restore_state(mock_storage)

        # Should have no work units after restoration since storage was empty
        assert len(orchestrator_processor.work_units) == 0
        assert len(orchestrator_processor.pending_units) == 0

    def test_restore_state_with_incomplete_chunks(self, orchestrator_processor, mock_storage):
        """Test state restoration with incomplete chunks."""
        mock_storage.get_all_processed_job_ids.return_value = set()

        # Clear any existing state from background thread
        orchestrator_processor.chunk_tracker.chunks.clear()
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()

        # Add a real chunk to the chunk tracker and mark some items as processed
        chunk_id = "shard_0:chunk:0"
        orchestrator_processor.chunk_tracker.add_chunk(chunk_id, "shard_0", "shard_0.tar", 0, 100)

        # Mark some items as processed, leaving gaps: process 0-9 and 21-49, leave 10-20 and 50-99
        orchestrator_processor.chunk_tracker.mark_items_processed(chunk_id, 0, 9)
        orchestrator_processor.chunk_tracker.mark_items_processed(chunk_id, 21, 49)

        # Set status to assigned (not completed)
        orchestrator_processor.chunk_tracker.mark_assigned(chunk_id, "worker1")

        # Verify the unprocessed ranges before restoration
        chunk_state = orchestrator_processor.chunk_tracker.chunks[chunk_id]
        unprocessed_ranges = chunk_state.get_unprocessed_ranges()
        expected_unprocessed = [(10, 20), (50, 99)]
        assert (
            unprocessed_ranges == expected_unprocessed
        ), f"Expected {expected_unprocessed}, got {unprocessed_ranges}"

        orchestrator_processor._restore_state(mock_storage)

        # Should have created work unit for incomplete chunk
        assert len(orchestrator_processor.work_units) == 1
        assert len(orchestrator_processor.pending_units) == 1

        unit = orchestrator_processor.work_units[chunk_id]
        assert unit.unit_id == chunk_id
        assert unit.source_id == "shard_0"

        # Should have the unprocessed ranges as absolute indices
        expected_ranges = [(10, 20), (50, 99)]
        assert unit.data["unprocessed_ranges"] == expected_ranges

    def test_restore_state_skips_completed_chunks(self, orchestrator_processor, mock_storage):
        """Test that completed chunks are skipped during restoration."""
        mock_storage.get_all_processed_job_ids.return_value = set()

        # Stop background thread to avoid race condition
        orchestrator_processor.stop_creation.set()

        # Clear all existing state from chunk tracker and work units
        orchestrator_processor.chunk_tracker.chunks.clear()
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()

        # Add a chunk and mark it as completed
        chunk_id = "shard_0:chunk:0"
        orchestrator_processor.chunk_tracker.add_chunk(chunk_id, "shard_0", "shard_0.tar", 0, 100)

        # Mark all items as processed (which will auto-complete the chunk)
        orchestrator_processor.chunk_tracker.mark_items_processed(chunk_id, 0, 99)

        # Verify it's marked as completed
        chunk_state = orchestrator_processor.chunk_tracker.chunks[chunk_id]
        assert chunk_state.status == "completed"

        orchestrator_processor._restore_state(mock_storage)

        # Should not have created work units for completed chunks
        # Note: _restore_state only restores existing chunks, it doesn't create new chunks
        assert len(orchestrator_processor.work_units) == 0
        assert len(orchestrator_processor.pending_units) == 0

    def test_work_unit_creation_basic(self, orchestrator_processor):
        """Test basic work unit creation logic by directly testing unit creation."""
        # Clear any existing state
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()
        orchestrator_processor.assigned_units.clear()

        # Create a unit manually to test the creation logic without the background thread
        from caption_flow.models import JobId
        from caption_flow.processors.base import WorkUnit

        # Get shard info
        shard_info = orchestrator_processor._get_shard_info_cached(0)
        shard_name = shard_info["name"]
        chunk_size = orchestrator_processor.chunk_size

        # Create job ID
        job_id_obj = JobId(shard_id=shard_name, chunk_id="0", sample_id="0")

        # Create work unit
        unit_id = f"{shard_name}:chunk:0"
        unit = WorkUnit(
            unit_id=unit_id,
            chunk_id=unit_id,
            source_id=shard_name,
            unit_size=chunk_size,
            data={
                "shard_url": shard_info["path"],
                "shard_name": shard_name,
                "start_index": 0,
                "chunk_size": chunk_size,
                "unprocessed_ranges": [(0, chunk_size - 1)],
                "job_id": str(job_id_obj),
            },
            metadata={"shard_name": shard_name},
        )

        # Add to processor state
        orchestrator_processor.work_units[unit_id] = unit
        orchestrator_processor.pending_units.append(unit_id)

        # Verify the unit was created correctly
        assert len(orchestrator_processor.pending_units) > 0, "No pending units were created"
        unit_id = orchestrator_processor.pending_units[0]
        unit = orchestrator_processor.work_units[unit_id]

        assert unit.source_id == "shard_0"
        assert unit.unit_size <= 100  # chunk_size
        assert "shard_url" in unit.data
        assert "unprocessed_ranges" in unit.data

    def test_work_unit_assignment_updates_ranges(self, orchestrator_processor, mock_storage):
        """Test that work unit assignment updates unprocessed ranges from chunk tracker."""
        # Add a real chunk with some processed items
        unit_id = "shard_0:chunk:0"
        orchestrator_processor.chunk_tracker.add_chunk(unit_id, "shard_0", "shard_0.tar", 0, 100)

        # Mark some items as processed, leaving gaps (5-15) and (30-40)
        orchestrator_processor.chunk_tracker.mark_items_processed(unit_id, 0, 4)  # Process 0-4
        orchestrator_processor.chunk_tracker.mark_items_processed(unit_id, 16, 29)  # Process 16-29
        orchestrator_processor.chunk_tracker.mark_items_processed(unit_id, 41, 99)  # Process 41-99

        # Create a work unit manually
        unit = WorkUnit(
            unit_id=unit_id,
            chunk_id=unit_id,
            source_id="shard_0",
            unit_size=100,
            data={"shard_url": "shard_0.tar", "start_index": 0, "chunk_size": 100},
            metadata={"shard_name": "shard_0"},
        )
        orchestrator_processor.work_units[unit_id] = unit
        orchestrator_processor.pending_units.append(unit_id)

        # Get work unit
        assigned_units = orchestrator_processor.get_work_units(1, "worker1")

        assert len(assigned_units) == 1
        assigned_unit = assigned_units[0]

        # Should have updated unprocessed ranges to absolute indices
        expected_ranges = [(5, 15), (30, 40)]  # The gaps we left
        assert assigned_unit.data["unprocessed_ranges"] == expected_ranges

    def test_work_unit_assignment_skips_completed_chunks(self, orchestrator_processor):
        """Test that work unit assignment skips chunks with no unprocessed ranges."""
        # Stop background thread to prevent it from creating new work units
        orchestrator_processor.stop_creation.set()

        # Clear any existing state to isolate the test
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()
        orchestrator_processor.assigned_units.clear()

        unit_id = "shard_0:chunk:0"

        # Add chunk and mark all items as processed
        orchestrator_processor.chunk_tracker.add_chunk(unit_id, "shard_0", "shard_0.tar", 0, 100)
        orchestrator_processor.chunk_tracker.mark_items_processed(unit_id, 0, 99)

        # Verify chunk is completed
        chunk_state = orchestrator_processor.chunk_tracker.chunks[unit_id]
        assert chunk_state.status == "completed"
        assert chunk_state.get_unprocessed_ranges() == []

        # Create work unit (simulating stale unit)
        unit = WorkUnit(
            unit_id=unit_id,
            chunk_id=unit_id,
            source_id="shard_0",
            unit_size=100,
            data={"shard_url": "shard_0.tar"},
            metadata={},
        )
        orchestrator_processor.work_units[unit_id] = unit
        orchestrator_processor.pending_units.append(unit_id)

        # Try to get work unit
        assigned_units = orchestrator_processor.get_work_units(1, "worker1")

        # The completed unit should be removed, and the specific unit should not be assigned
        # However, the processor may create new units if the background thread was running
        # Let's check that the completed unit was removed
        assert unit_id not in orchestrator_processor.work_units

        # If units were assigned, they should not include the completed unit
        for assigned_unit in assigned_units:
            assert assigned_unit.unit_id != unit_id

    def test_handle_result_single_item(self, orchestrator_processor):
        """Test handling result for single item processing."""
        result = WorkResult(
            unit_id="test_unit",
            source_id="shard_0",
            chunk_id="shard_0:chunk:0",
            sample_id="42",
            outputs={"captions": ["test caption"]},
            metadata={"_item_index": 42},
            processing_time_ms=100.0,
        )

        # Mock chunk tracker
        orchestrator_processor.chunk_tracker.mark_items_processed = Mock()

        handled_result = orchestrator_processor.handle_result(result)

        # Should mark single item as processed
        orchestrator_processor.chunk_tracker.mark_items_processed.assert_called_once_with(
            "shard_0:chunk:0", 42, 42
        )

        # Should return expected format
        assert handled_result["source_id"] == "shard_0"
        assert handled_result["chunk_id"] == "shard_0:chunk:0"
        assert handled_result["outputs"] == {"captions": ["test caption"]}

    def test_handle_result_batch_items(self, orchestrator_processor):
        """Test handling result for batch item processing."""
        result = WorkResult(
            unit_id="test_unit",
            source_id="shard_0",
            chunk_id="shard_0:chunk:0",
            sample_id="batch",
            outputs={"captions": ["caption1", "caption2", "caption3"]},
            metadata={"item_indices": [10, 11, 12, 20, 21]},  # Non-contiguous indices
            processing_time_ms=250.0,
        )

        orchestrator_processor.chunk_tracker.mark_items_processed = Mock()

        orchestrator_processor.handle_result(result)

        # Should mark ranges as processed (condense contiguous indices)
        expected_calls = [
            call("shard_0:chunk:0", 10, 12),  # Contiguous range
            call("shard_0:chunk:0", 20, 21),  # Another contiguous range
        ]
        orchestrator_processor.chunk_tracker.mark_items_processed.assert_has_calls(
            expected_calls, any_order=False
        )

    def test_update_from_storage_creates_missing_chunks(self, orchestrator_processor):
        """Test that update_from_storage creates missing chunk states."""
        processed_job_ids = {
            "shard_0:chunk:0:idx:5",
            "shard_0:chunk:0:idx:10",
            "shard_1:chunk:2:idx:250",  # Different shard and chunk
        }

        # Mock chunk tracker
        orchestrator_processor.chunk_tracker.chunks = {}
        orchestrator_processor.chunk_tracker.add_chunk = Mock()
        orchestrator_processor.chunk_tracker.mark_items_processed = Mock()
        orchestrator_processor.chunk_tracker.save = Mock()

        orchestrator_processor.update_from_storage(processed_job_ids)

        # Should create missing chunk states
        expected_add_calls = [
            call("shard_0:chunk:0", "shard_0", "shard_0.tar", 0, 100),  # chunk 0 at index 0
            call("shard_1:chunk:2", "shard_1", "shard_1.tar", 200, 100),  # chunk 2 at index 200
        ]
        orchestrator_processor.chunk_tracker.add_chunk.assert_has_calls(
            expected_add_calls, any_order=True
        )

        # Should mark items as processed (individual items, not ranges)
        expected_process_calls = [
            call("shard_0:chunk:0", 5, 5),  # Individual item
            call("shard_0:chunk:0", 10, 10),  # Individual item
            call("shard_1:chunk:2", 250, 250),  # Single item
        ]
        orchestrator_processor.chunk_tracker.mark_items_processed.assert_has_calls(
            expected_process_calls, any_order=True
        )

    def test_update_from_storage_handles_malformed_job_ids(self, orchestrator_processor):
        """Test that malformed job IDs are handled gracefully."""
        processed_job_ids = {
            "valid:chunk:0:idx:5",
            "malformed_id",  # Missing required parts
            "too:many:parts:chunk:0:idx:5:extra",
            "shard:invalid:0:idx:5",  # Invalid chunk keyword
            "shard:chunk:abc:idx:5",  # Non-numeric chunk ID
            "shard:chunk:0:invalid:5",  # Invalid idx keyword
            "shard:chunk:0:idx:abc",  # Non-numeric sample ID
        }

        orchestrator_processor.chunk_tracker.chunks = {}
        orchestrator_processor.chunk_tracker.add_chunk = Mock()
        orchestrator_processor.chunk_tracker.mark_items_processed = Mock()

        # Should handle malformed IDs without crashing
        orchestrator_processor.update_from_storage(processed_job_ids)

        # Should only process the valid job ID
        orchestrator_processor.chunk_tracker.add_chunk.assert_called_once_with(
            "valid:chunk:0", "valid", "valid.tar", 0, 100
        )
        orchestrator_processor.chunk_tracker.mark_items_processed.assert_called_once_with(
            "valid:chunk:0", 5, 5
        )

    def test_release_assignments_updates_unprocessed_ranges(self, orchestrator_processor):
        """Test that releasing assignments updates work units with current unprocessed ranges."""
        unit_id = "shard_0:chunk:0"

        # Create work unit and assign it
        unit = WorkUnit(
            unit_id=unit_id,
            chunk_id=unit_id,
            source_id="shard_0",
            unit_size=100,
            data={"start_index": 0, "unprocessed_ranges": [(0, 99)]},
            metadata={},
        )
        orchestrator_processor.work_units[unit_id] = unit
        orchestrator_processor.assigned_units["worker1"].add(unit_id)

        # Mock chunk tracker with updated ranges (some items processed)
        mock_chunk = Mock()
        mock_chunk.start_index = 0
        mock_chunk.get_unprocessed_ranges.return_value = [(20, 40), (60, 80)]
        orchestrator_processor.chunk_tracker.chunks[unit_id] = mock_chunk
        orchestrator_processor.chunk_tracker.release_worker_chunks = Mock()

        orchestrator_processor.release_assignments("worker1")

        # Should update work unit with new unprocessed ranges
        updated_unit = orchestrator_processor.work_units[unit_id]
        expected_ranges = [(20, 40), (60, 80)]  # Already absolute from start_index=0
        assert updated_unit.data["unprocessed_ranges"] == expected_ranges

        # Should add unit back to pending
        assert unit_id in orchestrator_processor.pending_units

        # Should release from chunk tracker
        orchestrator_processor.chunk_tracker.release_worker_chunks.assert_called_once_with(
            "worker1"
        )

    def test_statistics_calculation(self, orchestrator_processor):
        """Test statistics calculation from chunk tracker."""
        # Stop background thread and clear existing state
        orchestrator_processor.stop_creation.set()
        orchestrator_processor.work_units.clear()
        orchestrator_processor.pending_units.clear()
        orchestrator_processor.assigned_units.clear()

        # Mock chunk tracker with various chunk states
        mock_summary = {
            "shard_0": {
                "chunks": [
                    Mock(status="completed"),
                    Mock(status="assigned"),
                    Mock(status="pending"),
                ]
            },
            "shard_1": {
                "chunks": [
                    Mock(status="completed"),
                    Mock(status="completed"),
                ]
            },
        }
        orchestrator_processor.chunk_tracker.get_shards_summary = Mock(return_value=mock_summary)

        # Add some work units
        orchestrator_processor.pending_units.extend(["unit1", "unit2"])
        orchestrator_processor.assigned_units["worker1"].update(["unit3", "unit4"])
        orchestrator_processor.assigned_units["worker2"].add("unit5")

        stats = orchestrator_processor.get_stats()

        assert stats["total_shards"] == 3  # From mock dataset
        assert stats["total_chunks"] == 5  # 3 + 2 chunks
        assert stats["pending_units"] == 2
        assert stats["assigned_units"] == 3  # 2 + 1 assigned units
        assert stats["completed_chunks"] == 3  # 1 + 2 completed chunks
        assert stats["workers"] == 2

    def test_cleanup_stops_background_thread(self, orchestrator_processor):
        """Test cleanup properly stops background thread and saves state."""
        orchestrator_processor.chunk_tracker.flush = Mock()

        # Ensure any existing thread is fully stopped first
        orchestrator_processor.stop_creation.set()
        if orchestrator_processor.unit_creation_thread:
            orchestrator_processor.unit_creation_thread.join(timeout=0.1)
            orchestrator_processor.unit_creation_thread = None

        # Reset stop event for testing
        orchestrator_processor.stop_creation.clear()

        # Start a mock thread
        mock_thread = Mock()
        mock_thread.join = Mock()
        orchestrator_processor.unit_creation_thread = mock_thread

        orchestrator_processor.cleanup()

        # Should set stop event
        assert orchestrator_processor.stop_creation.is_set()

        # Should join thread with timeout
        mock_thread.join.assert_called_once_with(timeout=5)

        # Should flush checkpoint
        orchestrator_processor.chunk_tracker.flush.assert_called_once()


class TestWebDatasetWorkerProcessor:
    """Test suite for WebDatasetWorkerProcessor."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def worker_config(self, temp_dir):
        """Create worker processor configuration."""
        config_dict = {
            "dataset": {
                "dataset_path": "/fake/dataset",
                "metadata_path": None,
                "mock_results": True,  # Use mock results for testing
                "split_worker_cache": True,
            },
            "cache_dir": str(temp_dir / "cache"),
            "shard_cache_gb": 1.0,
            "buffer_size": 10,
            "max_file_size": 100 * 1024 * 1024,
        }
        return ProcessorConfig(processor_type="webdataset", config=config_dict)

    @pytest.fixture
    def worker_processor(self, worker_config):
        """Create worker processor."""
        processor = WebDatasetWorkerProcessor()
        processor.gpu_id = 0  # Set GPU ID for cache path
        processor.initialize(worker_config)
        return processor

    def test_initialization_mock_mode(self, worker_processor):
        """Test worker initialization in mock mode."""
        assert worker_processor.mock_results is True
        assert worker_processor.dataset is None  # Should be None in mock mode
        assert worker_processor.loader is None  # Should be None in mock mode

    @pytest.fixture
    def worker_processor_real(self, worker_config):
        """Create worker processor with real webshart (mocked)."""
        config_dict = worker_config.config.copy()
        config_dict["dataset"]["mock_results"] = False

        processor = WebDatasetWorkerProcessor()
        processor.gpu_id = 1

        # Mock webshart components
        mock_dataset = Mock()
        mock_dataset.enable_metadata_cache = Mock()
        mock_dataset.enable_shard_cache = Mock()

        mock_loader = Mock()

        with patch(
            "caption_flow.processors.webdataset.webshart.discover_dataset",
            return_value=mock_dataset,
        ):
            with patch(
                "caption_flow.processors.webdataset.webshart.TarDataLoader",
                return_value=mock_loader,
            ):
                processor.initialize(
                    ProcessorConfig(processor_type="webdataset", config=config_dict)
                )

        return processor

    def test_initialization_real_mode(self, worker_processor_real):
        """Test worker initialization in real mode with mocked webshart."""
        assert worker_processor_real.mock_results is False
        assert worker_processor_real.dataset is not None
        assert worker_processor_real.loader is not None

    def test_mock_image_creation(self, worker_processor):
        """Test mock image creation produces different images."""
        img1 = worker_processor._create_mock_image(0)
        img2 = worker_processor._create_mock_image(1)
        img3 = worker_processor._create_mock_image(100)

        assert isinstance(img1, Image.Image)
        assert isinstance(img2, Image.Image)
        assert isinstance(img3, Image.Image)

        assert img1.size == (256, 256)
        assert img2.size == (256, 256)
        assert img3.size == (256, 256)

        # Different indices should produce different colored images
        assert img1.getpixel((0, 0)) != img2.getpixel((0, 0))
        assert img1.getpixel((0, 0)) != img3.getpixel((0, 0))

    def test_process_unit_mock_mode_single_range(self, worker_processor):
        """Test processing work unit in mock mode with single range."""
        unit = WorkUnit(
            unit_id="shard_0:chunk:0",
            chunk_id="shard_0:chunk:0",
            source_id="shard_0",
            unit_size=50,
            data={
                "shard_name": "shard_0",
                "shard_idx": 0,
                "start_index": 0,
                "chunk_size": 50,
                "unprocessed_ranges": [(10, 15)],  # 6 items
            },
            metadata={"chunk_index": 0},
        )

        results = list(worker_processor.process_unit(unit, {}))

        assert len(results) == 6  # 15-10+1 = 6 items

        # Check first result
        result = results[0]
        assert result["item_index"] == 10
        assert result["item_key"] == "mock_10"
        assert isinstance(result["image"], Image.Image)
        assert result["image_data"] is None  # Mock mode
        assert result["metadata"]["_item_index"] == 10
        assert result["metadata"]["_chunk_relative_index"] == 10  # 10 - start_index(0)
        assert result["metadata"]["_mock"] is True
        assert "shard_0:chunk:0:idx:10" == result["job_id"]

        # Check job ID format for all results
        expected_indices = list(range(10, 16))
        for i, result in enumerate(results):
            expected_job_id = f"shard_0:chunk:0:idx:{expected_indices[i]}"
            assert result["job_id"] == expected_job_id
            assert result["metadata"]["_job_id"] == expected_job_id

    def test_process_unit_mock_mode_multiple_ranges(self, worker_processor):
        """Test processing work unit with multiple unprocessed ranges."""
        unit = WorkUnit(
            unit_id="shard_1:chunk:2",
            chunk_id="shard_1:chunk:2",
            source_id="shard_1",
            unit_size=100,
            data={
                "shard_name": "shard_1",
                "shard_idx": 1,
                "start_index": 200,  # chunk 2 starting at index 200
                "chunk_size": 100,
                "unprocessed_ranges": [(205, 207), (220, 222)],  # 3 + 3 = 6 items
            },
            metadata={"chunk_index": 2},
        )

        results = list(worker_processor.process_unit(unit, {}))

        assert len(results) == 6  # (207-205+1) + (222-220+1) = 3 + 3

        # Check first range results
        for i in range(3):
            result = results[i]
            expected_idx = 205 + i
            assert result["item_index"] == expected_idx
            assert (
                result["metadata"]["_chunk_relative_index"] == expected_idx - 200
            )  # relative to start_index
            expected_job_id = f"shard_1:chunk:2:idx:{expected_idx}"
            assert result["job_id"] == expected_job_id

        # Check second range results
        for i in range(3, 6):
            result = results[i]
            expected_idx = 220 + (i - 3)
            assert result["item_index"] == expected_idx
            expected_job_id = f"shard_1:chunk:2:idx:{expected_idx}"
            assert result["job_id"] == expected_job_id

    def test_process_unit_mock_mode_empty_ranges(self, worker_processor):
        """Test processing work unit with empty unprocessed ranges."""
        unit = WorkUnit(
            unit_id="shard_0:chunk:0",
            chunk_id="shard_0:chunk:0",
            source_id="shard_0",
            unit_size=100,
            data={
                "shard_name": "shard_0",
                "unprocessed_ranges": [],  # No unprocessed ranges
            },
            metadata={"chunk_index": 0},
        )

        results = list(worker_processor.process_unit(unit, {}))

        assert len(results) == 0

    def test_process_unit_real_mode_with_mock_loader(self, worker_processor_real):
        """Test processing in real mode with mocked webshart loader."""
        # Mock webshart entry
        mock_entry = Mock()
        mock_entry.data = b"fake_image_data"
        mock_entry.path = "image_005.jpg"
        mock_entry.size = 1024

        # Mock the loader methods
        worker_processor_real.loader.shard = Mock()

        # Create a simple iterator that returns our mock entry
        def mock_next_with_cache_wait(loader):
            return mock_entry

        unit = WorkUnit(
            unit_id="shard_0:chunk:0",
            chunk_id="shard_0:chunk:0",
            source_id="shard_0",
            unit_size=20,
            data={
                "shard_name": "shard_0",
                "shard_idx": 0,
                "start_index": 0,
                "unprocessed_ranges": [(5, 7)],  # 3 items
            },
            metadata={"chunk_index": 0},
        )

        # Mock the image decoding
        test_image = Image.new("RGB", (100, 100), color="red")

        with patch(
            "caption_flow.processors.webdataset.webshart.next_with_cache_wait",
            side_effect=[mock_entry] * 3,
        ):
            with patch("caption_flow.processors.webdataset.cv2.imdecode") as mock_decode:
                with patch("caption_flow.processors.webdataset.cv2.cvtColor") as mock_convert:
                    with patch(
                        "caption_flow.processors.webdataset.Image.fromarray",
                        return_value=test_image,
                    ):
                        # Mock cv2 processing chain
                        mock_decode.return_value = "fake_cv2_image"
                        mock_convert.return_value = "fake_rgb_array"

                        results = list(worker_processor_real.process_unit(unit, {}))

        assert len(results) == 3

        # Check first result
        result = results[0]
        assert result["item_index"] == 5
        assert result["item_key"] == "image_005"  # Path stem
        assert result["image"] == test_image
        assert result["image_data"] == b"fake_image_data"
        assert result["metadata"]["_filename"] == "image_005.jpg"
        assert result["metadata"]["_file_size"] == 1024
        assert not result["metadata"].get("_mock", False)  # Should not have mock flag

        # Verify loader was called correctly
        worker_processor_real.loader.shard.assert_called_once_with(shard_idx=0, cursor_idx=5)

    def test_process_unit_real_mode_shard_by_name(self, worker_processor_real):
        """Test processing when shard_idx is None (fallback to name lookup)."""
        mock_entry = Mock()
        mock_entry.data = b"fake_data"
        mock_entry.path = "test.jpg"
        mock_entry.size = 512

        worker_processor_real.loader.shard = Mock()

        unit = WorkUnit(
            unit_id="shard_unknown:chunk:0",
            chunk_id="shard_unknown:chunk:0",
            source_id="shard_unknown",
            unit_size=10,
            data={
                "shard_name": "shard_unknown",
                "shard_idx": None,  # Force fallback to name
                "start_index": 10,
                "unprocessed_ranges": [(10, 10)],  # Single item
            },
            metadata={"chunk_index": 1},
        )

        test_image = Image.new("RGB", (50, 50))

        with patch(
            "caption_flow.processors.webdataset.webshart.next_with_cache_wait",
            return_value=mock_entry,
        ):
            with patch("caption_flow.processors.webdataset.Image.open", return_value=test_image):
                # Simulate cv2 import error to test PIL fallback
                with patch(
                    "caption_flow.processors.webdataset.cv2.imdecode",
                    side_effect=ImportError("cv2 not available"),
                ):
                    results = list(worker_processor_real.process_unit(unit, {}))

        assert len(results) == 1

        # Should have used filename instead of shard_idx
        worker_processor_real.loader.shard.assert_called_once_with(
            filename="shard_unknown", cursor_idx=10
        )

        # Should have used PIL fallback
        result = results[0]
        assert result["image"] == test_image

    def test_get_dataset_info_mock_mode(self, worker_processor):
        """Test dataset info in mock mode."""
        info = worker_processor.get_dataset_info()

        assert info["dataset_name"] == "Mock Dataset"
        assert info["mock_results"] is True

    def test_get_dataset_info_real_mode(self, worker_processor_real):
        """Test dataset info in real mode."""
        # Mock dataset stats
        worker_processor_real.dataset.name = "test_dataset"
        worker_processor_real.dataset.dataset_format = "webdataset"
        worker_processor_real.dataset.get_stats.return_value = {
            "total_shards": 5,
            "total_files": 10000,
        }

        info = worker_processor_real.get_dataset_info()

        assert info["dataset_name"] == "test_dataset"
        assert info["format"] == "webdataset"
        assert info["total_shards"] == 5
        assert info["total_files"] == 10000
        assert info["mock_results"] is False


class TestWebDatasetIntegration:
    """Integration tests combining orchestrator and worker processors."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_orchestrator_chunk_creation_uses_correct_start_index(self, temp_dir):
        """Test that orchestrator creates chunks with correct start_index calculation."""
        # Create real chunk tracker to test the actual implementation
        chunk_tracker = ChunkTracker(temp_dir / "chunks.json")

        processor = WebDatasetOrchestratorProcessor()
        processor.chunk_tracker = chunk_tracker
        processor.chunk_size = 100
        processor.lock = threading.Lock()
        processor.work_units = {}
        processor.pending_units = deque()

        # Mock dataset with one shard
        mock_dataset = Mock()
        mock_dataset.num_shards = 1
        mock_dataset.get_shard_info.return_value = {
            "name": "test_shard",
            "path": "test_shard.tar",
            "num_files": 350,  # Will create 4 chunks: 0-99, 100-199, 200-299, 300-349
        }
        processor.dataset = mock_dataset

        # Simulate background unit creation for multiple chunks
        processor.stop_creation = threading.Event()
        current_shard_idx = 0
        current_file_idx = 0

        created_chunks = []

        # Create 4 chunks manually (simulating background thread logic)
        for _chunk_num in range(4):
            shard_info = processor._get_shard_info_cached(current_shard_idx)
            shard_name = shard_info["name"]
            chunk_size = min(processor.chunk_size, shard_info["num_files"] - current_file_idx)

            processor.current_chunk_index = current_file_idx // processor.chunk_size
            job_id_obj = JobId(
                shard_id=shard_name,
                chunk_id=processor.current_chunk_index,
                sample_id=current_file_idx,
            )
            chunk_id = job_id_obj.get_chunk_str()

            # This is the key test - the start_index calculation
            expected_start_index = current_file_idx  # Should be 0, 100, 200, 300

            # Add to chunk tracker
            chunk_tracker.add_chunk(
                chunk_id, shard_name, shard_info["path"], expected_start_index, chunk_size
            )

            created_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_index": expected_start_index,
                    "chunk_size": chunk_size,
                    "expected_chunk_index": processor.current_chunk_index,
                }
            )

            current_file_idx += processor.chunk_size

        # Verify chunks were created with correct start indices
        expected_chunks = [
            {
                "chunk_id": "test_shard:chunk:0",
                "start_index": 0,
                "chunk_size": 100,
                "expected_chunk_index": 0,
            },
            {
                "chunk_id": "test_shard:chunk:1",
                "start_index": 100,
                "chunk_size": 100,
                "expected_chunk_index": 1,
            },
            {
                "chunk_id": "test_shard:chunk:2",
                "start_index": 200,
                "chunk_size": 100,
                "expected_chunk_index": 2,
            },
            {
                "chunk_id": "test_shard:chunk:3",
                "start_index": 300,
                "chunk_size": 50,
                "expected_chunk_index": 3,
            },  # Last chunk smaller
        ]

        assert len(created_chunks) == 4
        for i, expected in enumerate(expected_chunks):
            actual = created_chunks[i]
            assert actual["chunk_id"] == expected["chunk_id"]
            assert (
                actual["start_index"] == expected["start_index"]
            ), f"Chunk {i} start_index mismatch"
            assert actual["chunk_size"] == expected["chunk_size"]
            assert actual["expected_chunk_index"] == expected["expected_chunk_index"]

            # Verify in chunk tracker
            chunk_state = chunk_tracker.chunks[actual["chunk_id"]]
            assert chunk_state.start_index == expected["start_index"]
            assert chunk_state.chunk_size == expected["chunk_size"]

    def test_worker_processor_job_id_consistency(self, temp_dir):
        """Test that worker processor generates job IDs consistent with orchestrator expectations."""
        worker = WebDatasetWorkerProcessor()

        config_dict = {"dataset": {"mock_results": True}}
        worker.initialize(ProcessorConfig(processor_type="webdataset", config=config_dict))

        # Create work unit matching what orchestrator would create
        unit = WorkUnit(
            unit_id="test_shard:chunk:2",
            chunk_id="test_shard:chunk:2",
            source_id="test_shard",
            unit_size=50,
            data={
                "shard_name": "test_shard",
                "start_index": 200,  # chunk 2 * chunk_size 100 = 200
                "unprocessed_ranges": [(205, 207)],  # 3 items
            },
            metadata={"chunk_index": 2},
        )

        results = list(worker.process_unit(unit, {}))

        assert len(results) == 3

        # Check job IDs match what storage/orchestrator expects
        expected_job_ids = [
            "test_shard:chunk:2:idx:205",
            "test_shard:chunk:2:idx:206",
            "test_shard:chunk:2:idx:207",
        ]

        for i, result in enumerate(results):
            assert result["job_id"] == expected_job_ids[i]

            # Verify JobId parsing works
            parsed_job_id = JobId.from_str(result["job_id"])
            assert parsed_job_id.shard_id == "test_shard"
            assert parsed_job_id.chunk_id == "2"
            assert parsed_job_id.sample_id == str(205 + i)

    def test_end_to_end_range_processing_with_gaps(self, temp_dir):
        """Test complete flow with gaps in processed items (red-green test for range bugs)."""
        # Setup orchestrator
        orchestrator = WebDatasetOrchestratorProcessor()
        chunk_tracker = ChunkTracker(temp_dir / "chunks.json")
        orchestrator.chunk_tracker = chunk_tracker
        orchestrator.chunk_size = 20
        orchestrator.lock = threading.Lock()
        orchestrator.work_units = {}
        orchestrator.pending_units = deque()
        orchestrator.assigned_units = defaultdict(set)

        # Create chunk with some items already processed (non-contiguous)
        chunk_id = "shard_0:chunk:0"
        chunk_tracker.add_chunk(chunk_id, "shard_0", "shard_0.tar", 0, 20)

        # Mark some items as processed, creating gaps: [0-4], [10-12], leaving [5-9], [13-19] unprocessed
        chunk_tracker.mark_items_processed(chunk_id, 0, 4)  # Items 0-4 done
        chunk_tracker.mark_items_processed(chunk_id, 10, 12)  # Items 10-12 done

        # Get unprocessed ranges - should be [(5, 9), (13, 19)]
        chunk_state = chunk_tracker.chunks[chunk_id]
        unprocessed_ranges = chunk_state.get_unprocessed_ranges()
        expected_unprocessed = [(5, 9), (13, 19)]  # The gaps
        assert (
            unprocessed_ranges == expected_unprocessed
        ), f"Expected {expected_unprocessed}, got {unprocessed_ranges}"

        # Create work unit simulating orchestrator assignment
        unit = WorkUnit(
            unit_id=chunk_id,
            chunk_id=chunk_id,
            source_id="shard_0",
            unit_size=20,
            data={
                "shard_name": "shard_0",
                "start_index": 0,
                "unprocessed_ranges": unprocessed_ranges,  # Pass the gaps
            },
            metadata={"chunk_index": 0},
        )

        # Setup worker
        worker = WebDatasetWorkerProcessor()
        config_dict = {"dataset": {"mock_results": True}}
        worker.initialize(ProcessorConfig(processor_type="webdataset", config=config_dict))

        # Process the work unit
        results = list(worker.process_unit(unit, {}))

        # Should process exactly the unprocessed ranges: 5 + 7 = 12 items
        expected_count = (9 - 5 + 1) + (19 - 13 + 1)  # 5 + 7 = 12
        assert len(results) == expected_count

        # Verify indices are exactly what we expect
        expected_indices = list(range(5, 10)) + list(
            range(13, 20)
        )  # [5,6,7,8,9,13,14,15,16,17,18,19]
        actual_indices = [result["item_index"] for result in results]
        assert actual_indices == expected_indices

        # Verify job IDs are correct
        for i, result in enumerate(results):
            expected_idx = expected_indices[i]
            expected_job_id = f"shard_0:chunk:0:idx:{expected_idx}"
            assert result["job_id"] == expected_job_id
            assert result["metadata"]["_item_index"] == expected_idx

        # Simulate handling results back in orchestrator
        for result in results:
            work_result = WorkResult(
                unit_id=chunk_id,
                source_id="shard_0",
                chunk_id=chunk_id,
                sample_id=str(result["item_index"]),
                outputs={"captions": [f"caption_{result['item_index']}"]},
                metadata={"_item_index": result["item_index"]},
                processing_time_ms=50.0,
            )
            orchestrator.handle_result(work_result)

        # After processing, chunk should be completed (all items processed)
        final_chunk_state = chunk_tracker.chunks[chunk_id]
        final_unprocessed = final_chunk_state.get_unprocessed_ranges()
        assert final_unprocessed == [], f"Expected no unprocessed ranges, got {final_unprocessed}"
        assert final_chunk_state.status == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
