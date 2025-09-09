"""Tests for HuggingFace processor range calculations to prevent negative total_processed."""

import pytest
from unittest.mock import Mock, patch
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
