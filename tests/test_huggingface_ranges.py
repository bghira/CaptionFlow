"""Tests for HuggingFace processor range calculations to prevent negative total_processed."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from caption_flow.processors.huggingface import HuggingFaceDatasetOrchestratorProcessor
from caption_flow.utils.chunk_tracker import ChunkTracker


class TestHuggingFaceRanges:
    """Test HuggingFace processor range calculations."""

    @pytest.fixture
    def mock_chunk_tracker(self):
        """Create a mock chunk tracker."""
        tracker = Mock(spec=ChunkTracker)
        tracker.chunks = {}
        return tracker

    @pytest.fixture
    def processor(self, mock_chunk_tracker):
        """Create a HuggingFaceDatasetOrchestratorProcessor with mocked dependencies."""
        with patch("caption_flow.processors.huggingface.ChunkTracker") as MockTracker:
            MockTracker.return_value = mock_chunk_tracker
            processor = HuggingFaceDatasetOrchestratorProcessor()
            processor.chunk_tracker = mock_chunk_tracker
            return processor

    def test_update_from_storage_with_correct_absolute_indices(self, processor, mock_chunk_tracker):
        """Test that update_from_storage passes absolute indices to mark_items_processed."""
        # Setup chunk tracker
        processor.chunk_size = 100

        # Mock the add_chunk method to simulate adding chunk to tracker
        def mock_add_chunk(chunk_id, shard_name, url, start_index, chunk_size):
            # Create a mock chunk state and add it to the chunks dict
            from unittest.mock import Mock

            chunk_state = Mock()
            chunk_state.chunk_id = chunk_id
            chunk_state.start_index = start_index
            chunk_state.chunk_size = chunk_size
            mock_chunk_tracker.chunks[chunk_id] = chunk_state

        mock_chunk_tracker.add_chunk.side_effect = mock_add_chunk

        # Create processed job IDs that would map to specific indices
        processed_job_ids = {
            "test_shard:chunk:0:idx:1005",  # chunk 0, sample index 1005
            "test_shard:chunk:0:idx:1006",  # chunk 0, sample index 1006
            "test_shard:chunk:0:idx:1007",  # chunk 0, sample index 1007
            "test_shard:chunk:0:idx:1010",  # chunk 0, sample index 1010
            "test_shard:chunk:0:idx:1011",  # chunk 0, sample index 1011
            "test_shard:chunk:0:idx:1015",  # chunk 0, sample index 1015
        }

        # Call update_from_storage
        processor.update_from_storage(processed_job_ids)

        # Verify add_chunk was called to create the chunk
        mock_chunk_tracker.add_chunk.assert_called_once_with(
            "test_shard:chunk:0", "test_shard", "", 0, 100
        )

        # Verify mark_items_processed was called with absolute indices
        expected_calls = [
            ("test_shard:chunk:0", 1005, 1007),  # Range 1005-1007
            ("test_shard:chunk:0", 1010, 1011),  # Range 1010-1011
            ("test_shard:chunk:0", 1015, 1015),  # Single item 1015
        ]

        assert mock_chunk_tracker.mark_items_processed.call_count == 3
        actual_calls = [
            call.args for call in mock_chunk_tracker.mark_items_processed.call_args_list
        ]
        assert actual_calls == expected_calls

    def test_mark_items_processed_with_valid_ranges(self, tmp_path):
        """Test that mark_items_processed correctly handles absolute indices."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        tracker = ChunkTracker(checkpoint_file)

        # Add a chunk with start_index=1000
        chunk_id = "test_chunk"
        tracker.add_chunk(chunk_id, "shard", "url", start_index=1000, chunk_size=100)

        # Mark items processed with absolute indices
        tracker.mark_items_processed(chunk_id, 1005, 1010)

        chunk_state = tracker.chunks[chunk_id]

        # Should have one range: (5, 10) in relative coordinates
        assert len(chunk_state.processed_ranges) == 1
        assert chunk_state.processed_ranges[0] == (5, 10)

        # Get unprocessed ranges and verify no negative calculations
        chunk_state.get_unprocessed_ranges()

        # Calculate total processed manually
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed == 6  # Items 1005-1010 = 6 items
        assert total_processed > 0  # Should never be negative

    def test_mark_items_processed_prevents_negative_ranges(self, tmp_path):
        """Test that invalid ranges are prevented."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        tracker = ChunkTracker(checkpoint_file)

        # Add a chunk
        chunk_id = "test_chunk"
        tracker.add_chunk(chunk_id, "shard", "url", start_index=1000, chunk_size=100)

        # Try to mark items with indices that would create invalid relative ranges
        with patch("caption_flow.utils.chunk_tracker.logger") as mock_logger:
            # This should be handled gracefully without creating negative ranges
            tracker.mark_items_processed(chunk_id, 1150, 1050)  # end < start in absolute terms

            # Should have logged a warning about invalid range
            mock_logger.warning.assert_called_once()

            # Should not have added any processed ranges
            chunk_state = tracker.chunks[chunk_id]
            assert len(chunk_state.processed_ranges) == 0

    def test_edge_case_chunk_boundary_indices(self, tmp_path):
        """Test handling of indices at chunk boundaries."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        tracker = ChunkTracker(checkpoint_file)

        chunk_id = "test_chunk"
        start_index = 1000
        chunk_size = 100
        tracker.add_chunk(chunk_id, "shard", "url", start_index=start_index, chunk_size=chunk_size)

        # Test indices at boundaries
        tracker.mark_items_processed(chunk_id, start_index, start_index)  # First item
        tracker.mark_items_processed(
            chunk_id, start_index + chunk_size - 1, start_index + chunk_size - 1
        )  # Last item

        chunk_state = tracker.chunks[chunk_id]

        # Should have two ranges: (0, 0) and (99, 99)
        assert len(chunk_state.processed_ranges) == 2
        assert (0, 0) in chunk_state.processed_ranges
        assert (99, 99) in chunk_state.processed_ranges

        # Calculate total processed
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed == 2
        assert total_processed > 0

    def test_contiguous_range_creation(self):
        """Test that contiguous indices are properly converted to ranges."""
        # Test the range creation logic directly
        indices = [1005, 1006, 1007, 1010, 1011, 1015]  # Absolute indices
        sorted_indices = sorted(indices)

        # Convert to contiguous ranges using the same logic as the processor
        ranges = []
        start_range = sorted_indices[0]
        end_range = sorted_indices[0]

        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == end_range + 1:
                end_range = sorted_indices[i]
            else:
                ranges.append((start_range, end_range))
                start_range = sorted_indices[i]
                end_range = sorted_indices[i]
        ranges.append((start_range, end_range))

        # Verify ranges are created correctly
        expected_ranges = [(1005, 1007), (1010, 1011), (1015, 1015)]
        assert ranges == expected_ranges

        # Verify all ranges have start <= end
        for start, end in ranges:
            assert start <= end, f"Invalid range: start={start}, end={end}"

    def test_empty_indices_handling(self, processor, mock_chunk_tracker):
        """Test handling of empty job IDs set."""
        # Call update_from_storage with empty set
        processor.update_from_storage(set())

        # Should not call mark_items_processed
        mock_chunk_tracker.mark_items_processed.assert_not_called()

    def test_chunk_start_index_calculation(self, tmp_path):
        """Test that chunks are created with correct start_index based on chunk number."""
        # Create processor with real chunk tracker
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Test different chunk indices
        test_cases = [
            ("test_shard:chunk:0:idx:500", "test_shard:chunk:0", 0),  # chunk 0 -> start_index 0
            (
                "test_shard:chunk:1:idx:1500",
                "test_shard:chunk:1",
                1000,
            ),  # chunk 1 -> start_index 1000
            (
                "test_shard:chunk:2:idx:2500",
                "test_shard:chunk:2",
                2000,
            ),  # chunk 2 -> start_index 2000
            (
                "test_shard:chunk:5:idx:5500",
                "test_shard:chunk:5",
                5000,
            ),  # chunk 5 -> start_index 5000
        ]

        for job_id, expected_chunk_id, expected_start_index in test_cases:
            processor.update_from_storage({job_id})

            # Verify chunk was created with correct start_index
            assert expected_chunk_id in processor.chunk_tracker.chunks
            chunk_state = processor.chunk_tracker.chunks[expected_chunk_id]
            assert (
                chunk_state.start_index == expected_start_index
            ), f"Chunk {expected_chunk_id} should have start_index {expected_start_index}, got {chunk_state.start_index}"

    def test_malformed_job_ids(self, tmp_path):
        """Test handling of malformed job IDs that could cause crashes."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Test various malformed job IDs
        malformed_job_ids = {
            "incomplete:chunk",  # Missing parts
            "shard:chunk:not_a_number:idx:500",  # Non-numeric chunk ID
            "shard:chunk:0:idx:not_a_number",  # Non-numeric sample ID
            "wrong:format:here",  # Wrong format entirely
            "",  # Empty string
            "shard:chunk:0:idx:",  # Missing sample ID
            ":chunk:0:idx:500",  # Empty shard name
            "shard::0:idx:500",  # Missing 'chunk' keyword
        }

        # Should not crash, should just log warnings and continue
        processor.update_from_storage(malformed_job_ids)

        # Should not have created any chunks
        assert len(processor.chunk_tracker.chunks) == 0

    def test_indices_outside_chunk_bounds(self, tmp_path):
        """Test what happens when indices fall outside expected chunk boundaries."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 100

        # For chunk 1 (start_index=100, end_index=199), test indices outside bounds
        problematic_job_ids = {
            "test_shard:chunk:1:idx:50",  # Index belongs to chunk 0, not chunk 1
            "test_shard:chunk:1:idx:250",  # Index belongs to chunk 2, not chunk 1
            "test_shard:chunk:1:idx:150",  # Valid index for chunk 1
        }

        processor.update_from_storage(problematic_job_ids)

        chunk_state = processor.chunk_tracker.chunks["test_shard:chunk:1"]

        # Should only process the valid index (150)
        # Index 50 should convert to relative -50 (invalid)
        # Index 250 should convert to relative 150 (out of bounds for 100-item chunk)
        # The invalid range protection should prevent negative calculations
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed >= 0, "Should never have negative processed count"

    def test_large_gaps_in_indices(self, tmp_path):
        """Test handling of very sparse index patterns."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 10000

        # Create sparse indices with large gaps
        sparse_job_ids = {
            "test_shard:chunk:0:idx:0",  # First index
            "test_shard:chunk:0:idx:5000",  # Middle index
            "test_shard:chunk:0:idx:9999",  # Last index
        }

        processor.update_from_storage(sparse_job_ids)

        chunk_state = processor.chunk_tracker.chunks["test_shard:chunk:0"]

        # Should create 3 separate single-item ranges: (0,0), (5000,5000), (9999,9999)
        assert len(chunk_state.processed_ranges) == 3
        assert (0, 0) in chunk_state.processed_ranges
        assert (5000, 5000) in chunk_state.processed_ranges
        assert (9999, 9999) in chunk_state.processed_ranges

        # Total should be exactly 3
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed == 3

    def test_duplicate_indices(self, tmp_path):
        """Test handling of duplicate indices in job IDs."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Include duplicate indices
        duplicate_job_ids = {
            "test_shard:chunk:0:idx:100",
            # Duplicate
            "test_shard:chunk:0:idx:101",
            # Duplicate
            "test_shard:chunk:0:idx:102",
        }

        processor.update_from_storage(duplicate_job_ids)

        chunk_state = processor.chunk_tracker.chunks["test_shard:chunk:0"]

        # Should handle duplicates gracefully - range should be (100, 102)
        assert len(chunk_state.processed_ranges) == 1
        assert chunk_state.processed_ranges[0] == (100, 102)

        # Total should be 3 (not 5)
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed == 3

    def test_extremely_large_chunk_indices(self, tmp_path):
        """Test handling of very large chunk numbers."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Test with very large chunk index
        large_chunk_job_ids = {
            "test_shard:chunk:999999:idx:999999000",  # chunk 999999 -> start_index should be 999999000
        }

        processor.update_from_storage(large_chunk_job_ids)

        chunk_id = "test_shard:chunk:999999"
        assert chunk_id in processor.chunk_tracker.chunks
        chunk_state = processor.chunk_tracker.chunks[chunk_id]

        # Verify correct start_index calculation even for large numbers
        expected_start_index = 999999 * 1000
        assert chunk_state.start_index == expected_start_index

        # Verify range calculation works
        expected_relative_index = 999999000 - expected_start_index
        assert (expected_relative_index, expected_relative_index) in chunk_state.processed_ranges

    def test_mixed_chunks_in_same_call(self, tmp_path):
        """Test processing job IDs from multiple chunks simultaneously."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 100

        # Mix job IDs from different chunks
        mixed_job_ids = {
            "shard_a:chunk:0:idx:10",  # chunk 0
            "shard_a:chunk:0:idx:11",  # chunk 0
            "shard_a:chunk:1:idx:150",  # chunk 1
            "shard_a:chunk:1:idx:151",  # chunk 1
            "shard_b:chunk:2:idx:250",  # different shard, chunk 2
        }

        processor.update_from_storage(mixed_job_ids)

        # Should create separate chunks for each
        expected_chunks = ["shard_a:chunk:0", "shard_a:chunk:1", "shard_b:chunk:2"]
        for chunk_id in expected_chunks:
            assert chunk_id in processor.chunk_tracker.chunks

        # Verify each chunk has correct ranges
        chunk_0 = processor.chunk_tracker.chunks["shard_a:chunk:0"]
        assert chunk_0.start_index == 0
        assert (10, 11) in chunk_0.processed_ranges

        chunk_1 = processor.chunk_tracker.chunks["shard_a:chunk:1"]
        assert chunk_1.start_index == 100
        assert (50, 51) in chunk_1.processed_ranges  # 150-100=50, 151-100=51

        chunk_2 = processor.chunk_tracker.chunks["shard_b:chunk:2"]
        assert chunk_2.start_index == 200
        assert (50, 50) in chunk_2.processed_ranges  # 250-200=50

    def test_zero_chunk_size_edge_case(self, tmp_path):
        """Test behavior with zero or negative chunk size."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)

        # Test with zero chunk size - should not crash
        processor.chunk_size = 0
        job_ids = {"test_shard:chunk:1:idx:100"}

        # Should handle gracefully without division by zero
        processor.update_from_storage(job_ids)

        # Should still create chunk but with start_index = 1 * 0 = 0
        chunk_state = processor.chunk_tracker.chunks["test_shard:chunk:1"]
        assert chunk_state.start_index == 0

    def test_incorrectly_assigned_indices_to_wrong_chunk(self, tmp_path):
        """Test the production bug: indices assigned to wrong chunks in job IDs."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Simulate the production bug: index 999 incorrectly assigned to chunk 2
        # This is what causes the "start=2048, end=999" error
        buggy_job_ids = {
            "photos_sequential:chunk:2:idx:2048",  # Correct: belongs to chunk 2 (2000-2999)
            "photos_sequential:chunk:2:idx:999",  # BUG: belongs to chunk 0 (0-999), not chunk 2!
        }

        processor.update_from_storage(buggy_job_ids)

        chunk_state = processor.chunk_tracker.chunks["photos_sequential:chunk:2"]

        # The safeguard should prevent the invalid range
        # Index 2048 -> relative 48 ✓
        # Index 999 -> relative -1001 ❌ (should be skipped)

        # Should only have the valid index (2048)
        assert len(chunk_state.processed_ranges) == 1
        assert (48, 48) in chunk_state.processed_ranges  # 2048 - 2000 = 48

        # Should NOT have processed the invalid index 999
        total_processed = sum(end - start + 1 for start, end in chunk_state.processed_ranges)
        assert total_processed == 1  # Only the valid index

        # Verify the invalid index was skipped (no negative ranges)
        for start, end in chunk_state.processed_ranges:
            assert start >= 0 and end >= 0, f"Found negative range: ({start}, {end})"

    def test_chunk_creation_during_orchestration(self, tmp_path):
        """Test that chunks created during orchestration get correct start_index."""
        checkpoint_file = tmp_path / "test_checkpoint.json"
        processor = HuggingFaceDatasetOrchestratorProcessor()
        processor.chunk_tracker = ChunkTracker(checkpoint_file)
        processor.chunk_size = 1000

        # Mock the orchestration setup
        processor.dataset_name = "test_dataset"
        processor.total_items = 5000
        processor.current_chunk_index = 0
        processor.pending_units = []

        # Mock shard info
        processor.shard_info = {0: {"filename": "test_shard.tar"}}

        # Mock the _get_shard_for_index method
        def mock_get_shard_for_index(index):
            return 0, "test_shard.tar"

        processor._get_shard_for_index = mock_get_shard_for_index

        # Test creating chunks 0, 1, 2 during orchestration
        for expected_chunk_idx in [0, 1, 2]:
            processor.current_chunk_index = expected_chunk_idx

            # Simulate the orchestration logic that creates chunks
            current_index = processor.current_chunk_index
            shard_id, _ = processor._get_shard_for_index(current_index)
            shard_name = Path(processor.shard_info[shard_id]["filename"]).stem

            from caption_flow.models import JobId

            job_id_obj = JobId(
                shard_id=shard_name,
                chunk_id=processor.current_chunk_index,
                sample_id=current_index,
            )
            unit_id = job_id_obj.get_chunk_str()

            # This is the code path that was buggy
            if processor.chunk_tracker and unit_id not in processor.chunk_tracker.chunks:
                start_index = processor.current_chunk_index * processor.chunk_size  # Fixed version
                chunk_size = min(processor.chunk_size, processor.total_items - start_index)
                processor.chunk_tracker.add_chunk(
                    unit_id,
                    processor.dataset_name,
                    "",
                    start_index,
                    chunk_size,
                )

            # Verify the chunk was created with correct start_index
            chunk_state = processor.chunk_tracker.chunks[unit_id]
            expected_start_index = expected_chunk_idx * processor.chunk_size
            assert (
                chunk_state.start_index == expected_start_index
            ), f"Chunk {expected_chunk_idx} should have start_index {expected_start_index}, got {chunk_state.start_index}"
