"""Tests for the DatasetViewer module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from caption_flow.viewer import DatasetViewer, SelectableListItem


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with sample parquet data."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)

    # Create sample data
    sample_data = {
        "shard": ["shard1", "shard1", "shard2", "shard2"],
        "job_id": [0, 1, 2, 3],
        "item_index": [0, 1, 0, 1],
        "filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "captions": ["A cat sitting", "A dog running", "A bird flying", "A fish swimming"],
        "url": [
            "http://example.com/img1.jpg",
            "http://example.com/img2.jpg",
            "http://example.com/img3.jpg",
            "http://example.com/img4.jpg",
        ],
    }

    df = pd.DataFrame(sample_data)
    df.to_parquet(data_dir / "captions.parquet")

    yield data_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_dir_no_shard():
    """Create a temporary data directory with sample data but no shard column."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)

    # Create sample data without shard column
    sample_data = {
        "job_id": [0, 1, 2, 3],
        "filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "captions": ["A cat sitting", "A dog running", "A bird flying", "A fish swimming"],
        "url": [
            "http://example.com/img1.jpg",
            "http://example.com/img2.jpg",
            "http://example.com/img3.jpg",
            "http://example.com/img4.jpg",
        ],
    }

    df = pd.DataFrame(sample_data)
    df.to_parquet(data_dir / "captions.parquet")

    yield data_dir
    shutil.rmtree(temp_dir)


class TestSelectableListItem:
    """Test SelectableListItem widget."""

    def test_init(self):
        """Test SelectableListItem initialization."""
        content = "Test item"
        on_select = Mock()
        item = SelectableListItem(content, on_select)

        assert item.content == content
        assert item.on_select == on_select
        assert item.selectable()

    def test_keypress_enter(self):
        """Test keypress handling for enter key."""
        content = "Test item"
        on_select = Mock()
        item = SelectableListItem(content, on_select)

        result = item.keypress((20,), "enter")

        assert result is None
        on_select.assert_called_once()

    def test_keypress_space(self):
        """Test keypress handling for space key."""
        content = "Test item"
        on_select = Mock()
        item = SelectableListItem(content, on_select)

        result = item.keypress((20,), " ")

        assert result is None
        on_select.assert_called_once()

    def test_keypress_other(self):
        """Test keypress handling for other keys."""
        content = "Test item"
        on_select = Mock()
        item = SelectableListItem(content, on_select)

        result = item.keypress((20,), "q")

        assert result == "q"
        on_select.assert_not_called()

    def test_no_callback(self):
        """Test keypress with no callback."""
        content = "Test item"
        item = SelectableListItem(content)

        result = item.keypress((20,), "enter")

        assert result == "enter"


class TestDatasetViewerInit:
    """Test DatasetViewer initialization."""

    def test_init_success(self, temp_data_dir):
        """Test successful viewer initialization."""
        viewer = DatasetViewer(temp_data_dir)

        assert viewer.data_dir == temp_data_dir
        assert viewer.captions_path == temp_data_dir / "captions.parquet"
        assert viewer.df is None
        assert viewer.shards == []
        assert viewer.current_shard_idx == 0
        assert viewer.current_item_idx == 0
        assert viewer.current_shard_items == []
        assert not viewer.disable_images

    def test_init_no_captions_file(self):
        """Test initialization when captions file doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)

        try:
            with pytest.raises(FileNotFoundError):
                DatasetViewer(data_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_palette_defined(self, temp_data_dir):
        """Test that color palette is properly defined."""
        viewer = DatasetViewer(temp_data_dir)

        assert hasattr(viewer, "palette")
        assert isinstance(viewer.palette, list)
        assert len(viewer.palette) > 0

        # Check some expected palette entries
        palette_names = [item[0] for item in viewer.palette]
        expected_names = ["normal", "selected", "header", "footer", "title", "error"]
        for name in expected_names:
            assert name in palette_names


class TestDatasetViewerLoadData:
    """Test DatasetViewer data loading functionality."""

    def test_load_data_with_shards(self, temp_data_dir):
        """Test loading data with shard column."""
        viewer = DatasetViewer(temp_data_dir)
        viewer.load_data()

        assert viewer.df is not None
        assert len(viewer.df) == 4
        assert viewer.shards == ["shard1", "shard2"]
        assert viewer.current_shard_idx == 0
        assert len(viewer.current_shard_items) == 2  # shard1 has 2 items

    def test_load_data_no_shards(self, temp_data_dir_no_shard):
        """Test loading data without shard column."""
        viewer = DatasetViewer(temp_data_dir_no_shard)
        viewer.load_data()

        assert viewer.df is not None
        assert len(viewer.df) == 4
        assert viewer.shards == ["all"]
        assert "shard" in viewer.df.columns
        assert all(viewer.df["shard"] == "all")
        assert len(viewer.current_shard_items) == 4  # all items in one shard

    def test_load_shard_valid_index(self, temp_data_dir):
        """Test loading a specific shard with valid index."""
        viewer = DatasetViewer(temp_data_dir)
        viewer.load_data()

        # Load second shard
        viewer._load_shard(1)

        assert viewer.current_shard_idx == 1
        assert len(viewer.current_shard_items) == 2  # shard2 has 2 items
        assert viewer.current_item_idx == 0
        assert viewer.current_image_url is None

    def test_load_shard_invalid_index(self, temp_data_dir):
        """Test loading shard with invalid index."""
        viewer = DatasetViewer(temp_data_dir)
        viewer.load_data()

        original_shard_idx = viewer.current_shard_idx
        original_items = viewer.current_shard_items.copy()

        # Try to load invalid shard
        viewer._load_shard(99)

        # Should not change anything
        assert viewer.current_shard_idx == original_shard_idx
        assert viewer.current_shard_items == original_items

    def test_load_shard_sorts_by_item_index(self, temp_data_dir):
        """Test that items are sorted by item_index when available."""
        viewer = DatasetViewer(temp_data_dir)
        viewer.load_data()

        # Check that items in first shard are sorted by item_index
        item_indices = [item["item_index"] for item in viewer.current_shard_items]
        assert item_indices == sorted(item_indices)

    def test_load_shard_sorts_by_job_id_fallback(self):
        """Test that items are sorted by job_id when item_index not available."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)

        try:
            # Create data without item_index but with job_id
            sample_data = {
                "shard": ["shard1", "shard1"],
                "job_id": [5, 2],
                "filename": ["img1.jpg", "img2.jpg"],
                "captions": ["Caption 1", "Caption 2"],
            }

            df = pd.DataFrame(sample_data)
            df.to_parquet(data_dir / "captions.parquet")

            viewer = DatasetViewer(data_dir)
            viewer.load_data()

            # Check that items are sorted by job_id
            job_ids = [item["job_id"] for item in viewer.current_shard_items]
            assert job_ids == [2, 5]  # Should be sorted

        finally:
            shutil.rmtree(temp_dir)


class TestDatasetViewerUI:
    """Test DatasetViewer UI creation (mocked)."""

    def test_ui_components_exist(self, temp_data_dir):
        """Test that UI components are properly defined."""
        viewer = DatasetViewer(temp_data_dir)

        # Test that create_ui method exists
        assert hasattr(viewer, "create_ui")
        assert callable(viewer.create_ui)


class TestDatasetViewerImageHandling:
    """Test DatasetViewer image handling functionality."""

    def test_init_temp_files_list(self, temp_data_dir):
        """Test that temp files list is initialized."""
        viewer = DatasetViewer(temp_data_dir)
        assert viewer.temp_files == []
        assert viewer.current_image_url is None

    def test_session_initialization(self, temp_data_dir):
        """Test that session is None initially."""
        viewer = DatasetViewer(temp_data_dir)
        assert viewer.session is None


class TestDatasetViewerUtilityMethods:
    """Test DatasetViewer utility methods."""

    def test_widget_storage(self, temp_data_dir):
        """Test that widget references are stored properly."""
        viewer = DatasetViewer(temp_data_dir)

        # Test initial widget state
        assert viewer.shards_list is None
        assert viewer.items_list is None
        assert viewer.caption_box is None
        assert viewer.image_box is None
        assert viewer.image_widget is None
        assert viewer.shards_box is None
        assert viewer.items_box is None

    def test_disable_images_flag(self, temp_data_dir):
        """Test disable_images flag functionality."""
        viewer = DatasetViewer(temp_data_dir)

        # Should default to False
        assert not viewer.disable_images

        # Can be set to True
        viewer.disable_images = True
        assert viewer.disable_images


class TestDatasetViewerIntegration:
    """Integration tests for DatasetViewer."""

    def test_full_initialization_cycle(self, temp_data_dir):
        """Test full initialization and data loading cycle."""
        viewer = DatasetViewer(temp_data_dir)

        # Should initialize successfully
        assert viewer.data_dir == temp_data_dir

        # Should load data successfully
        viewer.load_data()
        assert viewer.df is not None
        assert len(viewer.shards) > 0
        assert len(viewer.current_shard_items) > 0

        # Should be able to switch shards
        if len(viewer.shards) > 1:
            viewer.current_shard_items.copy()
            viewer._load_shard(1)
            # Items should be different (unless shards have identical data)
            assert viewer.current_shard_idx == 1

    def test_data_consistency(self, temp_data_dir):
        """Test that data remains consistent throughout operations."""
        viewer = DatasetViewer(temp_data_dir)
        viewer.load_data()

        original_df_len = len(viewer.df)
        original_shards_len = len(viewer.shards)

        # Switch between shards
        for i in range(len(viewer.shards)):
            viewer._load_shard(i)

            # Core data should remain unchanged
            assert len(viewer.df) == original_df_len
            assert len(viewer.shards) == original_shards_len
            assert 0 <= viewer.current_shard_idx < len(viewer.shards)
            assert viewer.current_item_idx == 0  # Reset for each shard

    def test_path_handling(self, temp_data_dir):
        """Test proper Path handling."""
        # Test with string path
        viewer1 = DatasetViewer(str(temp_data_dir))
        assert isinstance(viewer1.data_dir, Path)

        # Test with Path object
        viewer2 = DatasetViewer(temp_data_dir)
        assert isinstance(viewer2.data_dir, Path)

        # Both should be equivalent
        assert viewer1.data_dir == viewer2.data_dir


if __name__ == "__main__":
    pytest.main([__file__])
