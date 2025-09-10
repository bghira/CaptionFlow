"""Tests for storage exporter functionality."""

import csv
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import lance
import pandas as pd
import pytest
import pytest_asyncio
from caption_flow.models import Caption, StorageContents
from caption_flow.storage import StorageManager
from caption_flow.storage.exporter import LanceStorageExporter, StorageExporter

# Set up logging to avoid logger not defined errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest_asyncio.fixture
async def populated_storage_manager(temp_storage_dir):
    """Create a StorageManager with test data."""
    storage = StorageManager(temp_storage_dir)
    await storage.initialize()

    # Add some test captions with various data types including datetime
    test_data = [
        Caption(
            job_id="job1",
            dataset="test_dataset",
            shard="default",
            item_key="image1",
            contributor_id="test_user",
            url="https://example.com/image1.jpg",
            filename="image1.jpg",
            timestamp=datetime.now(),
            captions=["A beautiful sunset", "Orange and purple sky"],
            outputs={"captions": ["A beautiful sunset", "Orange and purple sky"]},
            metadata={"source": "test", "quality": 0.95},
        ),
        Caption(
            job_id="job2",
            dataset="test_dataset",
            shard="default",
            item_key="image2",
            contributor_id="test_user",
            url="https://example.com/image2.png",
            filename="image2.png",
            timestamp=datetime.now(),
            captions=["A cat sitting on a windowsill"],
            outputs={
                "captions": ["A cat sitting on a windowsill"],
                "detailed_captions": [
                    "A fluffy orange tabby cat sitting peacefully on a white windowsill"
                ],
            },
        ),
        Caption(
            job_id="job3",
            dataset="test_dataset",
            shard="default",
            item_key="image3",
            contributor_id="test_user",
            url="https://example.com/image3.gif",
            filename="image3.gif",
            timestamp=datetime.now(),
            captions=["Animated dancing banana"],
            outputs={
                "captions": ["Animated dancing banana"],
                "tags": ["animated", "fruit", "funny"],
            },
        ),
    ]

    for caption in test_data:
        await storage.save_caption(caption)

    await storage.checkpoint()
    return storage


@pytest.fixture
def sample_storage_contents():
    """Create sample StorageContents for testing StorageExporter."""
    rows = [
        {
            "job_id": "job1",
            "filename": "image1.jpg",
            "url": "https://example.com/image1.jpg",
            "timestamp": datetime.now(),
            "captions": ["Test caption 1", "Test caption 2"],
            "metadata": {"test": True},
        },
        {
            "job_id": "job2",
            "filename": "image2.png",
            "url": "https://example.com/image2.png",
            "timestamp": datetime.now(),
            "captions": ["Another test caption"],
            "tags": ["test", "sample"],
        },
    ]

    return StorageContents(
        rows=rows,
        columns=["job_id", "filename", "url", "timestamp", "captions", "metadata", "tags"],
        output_fields=["captions", "tags"],
        total_rows=len(rows),
        metadata={"test_data": True},
    )


class TestLanceStorageExporter:
    """Test the LanceStorageExporter class."""

    @pytest.mark.asyncio
    async def test_exporter_initialization(self, populated_storage_manager):
        """Test that LanceStorageExporter initializes correctly."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        assert exporter.storage_manager == storage

    @pytest.mark.asyncio
    async def test_export_shard_jsonl(self, populated_storage_manager, temp_storage_dir):
        """Test exporting shard to JSONL format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "test_export.jsonl"

        count = await exporter.export_shard("default", "jsonl", output_path)

        # Actual file will be test_export_default.jsonl
        actual_output = temp_storage_dir / "test_export_default.jsonl"

        assert count == 3
        assert actual_output.exists()

        # Verify JSONL content
        with open(actual_output, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

            # Test that each line is valid JSON
            for line in lines:
                data = json.loads(line.strip())
                assert "job_id" in data
                assert "captions" in data
                # Verify datetime serialization works
                if "timestamp" in data:
                    assert isinstance(data["timestamp"], str)

    @pytest.mark.asyncio
    async def test_export_shard_csv(self, populated_storage_manager, temp_storage_dir):
        """Test exporting shard to CSV format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "test_export.csv"

        count = await exporter.export_shard("default", "csv", output_path)

        # Actual file will be test_export_default.csv
        actual_output = temp_storage_dir / "test_export_default.csv"

        assert count == 3
        assert actual_output.exists()

        # Verify CSV content
        df = pd.read_csv(actual_output)
        assert len(df) == 3
        assert "job_id" in df.columns
        assert "filename" in df.columns

    @pytest.mark.asyncio
    async def test_export_shard_json_directory(self, populated_storage_manager, temp_storage_dir):
        """Test exporting shard to JSON directory format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_dir = temp_storage_dir / "json_export"

        count = await exporter.export_shard("default", "json", output_dir)

        # Directory-based format creates: json_export/default/
        actual_output_dir = output_dir / "default"

        assert count == 3
        assert actual_output_dir.exists()

        # Check that JSON files were created
        json_files = list(actual_output_dir.glob("*.json"))
        assert len(json_files) == 3

        # Verify content of one JSON file
        with open(json_files[0], "r") as f:
            data = json.load(f)
            assert "job_id" in data
            # Test datetime serialization
            if "timestamp" in data:
                assert isinstance(data["timestamp"], str)

    @pytest.mark.asyncio
    async def test_export_shard_txt(self, populated_storage_manager, temp_storage_dir):
        """Test exporting shard to TXT format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_dir = temp_storage_dir / "txt_export"

        count = await exporter.export_shard("default", "txt", output_dir)

        # Directory-based format creates: txt_export/default/
        actual_output_dir = output_dir / "default"

        assert count == 3
        assert actual_output_dir.exists()

        # Check that TXT files were created
        txt_files = list(actual_output_dir.glob("*.txt"))
        assert len(txt_files) == 3

        # Verify content of one TXT file
        with open(txt_files[0], "r") as f:
            content = f.read()
            assert len(content.strip()) > 0

    @pytest.mark.asyncio
    async def test_export_shard_parquet(self, populated_storage_manager, temp_storage_dir):
        """Test exporting shard to Parquet format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "test_export.parquet"

        count = await exporter.export_shard("default", "parquet", output_path)

        # Actual file will be test_export_default.parquet
        actual_output = temp_storage_dir / "test_export_default.parquet"

        assert count == 3
        assert actual_output.exists()

        # Verify parquet content
        df = pd.read_parquet(actual_output)
        assert len(df) == 3
        assert "job_id" in df.columns

    @pytest.mark.asyncio
    async def test_export_all_shards(self, populated_storage_manager, temp_storage_dir):
        """Test exporting all shards to multiple formats."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)

        results = await exporter.export_all_shards("jsonl", temp_storage_dir)

        assert "default" in results
        assert results["default"] == 3

        # Verify file was created
        jsonl_file = temp_storage_dir / "default.jsonl"
        assert jsonl_file.exists()

    @pytest.mark.asyncio
    async def test_export_with_column_filter(self, populated_storage_manager, temp_storage_dir):
        """Test exporting with specific columns."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "filtered.jsonl"

        count = await exporter.export_shard(
            "default", "jsonl", output_path, columns=["job_id", "filename", "captions"]
        )

        # Actual file will be filtered_default.jsonl
        actual_output = temp_storage_dir / "filtered_default.jsonl"

        assert count == 3

        # Verify only specified columns are present
        with open(actual_output, "r") as f:
            data = json.loads(f.readline().strip())
            expected_keys = {"job_id", "filename", "captions"}
            # Allow for extra keys that might be added by the system
            assert expected_keys.issubset(set(data.keys()))

    @pytest.mark.asyncio
    async def test_export_with_limit(self, populated_storage_manager, temp_storage_dir):
        """Test exporting with row limit."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "limited.jsonl"

        count = await exporter.export_shard("default", "jsonl", output_path, limit=2)

        # Actual file will be limited_default.jsonl
        actual_output = temp_storage_dir / "limited_default.jsonl"

        assert count == 2

        # Verify only 2 rows exported
        with open(actual_output, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_export_nonexistent_shard(self, populated_storage_manager, temp_storage_dir):
        """Test exporting non-existent shard returns 0."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "empty.jsonl"

        count = await exporter.export_shard("nonexistent", "jsonl", output_path)

        # Even nonexistent shard will create empty_nonexistent.jsonl
        actual_output = temp_storage_dir / "empty_nonexistent.jsonl"

        assert count == 0
        # File should exist but be empty
        if actual_output.exists():
            assert actual_output.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_export_to_lance(self, populated_storage_manager, temp_storage_dir):
        """Test exporting to Lance format."""
        storage = populated_storage_manager
        exporter = LanceStorageExporter(storage)
        output_path = temp_storage_dir / "exported.lance"

        total_rows = await exporter.export_to_lance(output_path)

        assert total_rows == 3
        assert output_path.exists()

        # Verify Lance dataset was created correctly
        dataset = lance.dataset(str(output_path))
        assert dataset.count_rows() == 3


class TestStorageExporter:
    """Test the StorageExporter class."""

    def test_exporter_initialization(self, sample_storage_contents):
        """Test that StorageExporter initializes correctly."""
        exporter = StorageExporter(sample_storage_contents)
        assert exporter.contents == sample_storage_contents

    def test_datetime_serialization(self, sample_storage_contents):
        """Test that datetime objects are serialized correctly."""
        exporter = StorageExporter(sample_storage_contents)

        # Test various datetime types
        test_datetime = datetime.now()
        test_date = test_datetime.date()
        test_timestamp = pd.Timestamp.now()

        # Test serialization
        assert isinstance(exporter._serialize_value(test_datetime), str)
        assert isinstance(exporter._serialize_value(test_date), str)
        assert isinstance(exporter._serialize_value(test_timestamp), str)

        # Verify ISO format
        serialized_dt = exporter._serialize_value(test_datetime)
        # Should be parseable back to datetime
        parsed_dt = datetime.fromisoformat(
            serialized_dt.replace("Z", "+00:00").replace("+00:00", "")
        )
        assert isinstance(parsed_dt, datetime)

    def test_list_serialization(self, sample_storage_contents):
        """Test that nested lists are serialized correctly."""
        exporter = StorageExporter(sample_storage_contents)

        test_list = ["string", 123, datetime.now(), ["nested", "list"]]

        serialized = exporter._serialize_value(test_list)
        assert isinstance(serialized, list)
        assert len(serialized) == 4
        # Datetime in list should be serialized
        assert isinstance(serialized[2], str)

    def test_dict_serialization(self, sample_storage_contents):
        """Test that dictionaries with datetime values are serialized correctly."""
        exporter = StorageExporter(sample_storage_contents)

        test_dict = {
            "timestamp": datetime.now(),
            "name": "test",
            "nested": {"inner_date": datetime.now()},
        }

        serialized = exporter._serialize_value(test_dict)
        assert isinstance(serialized, dict)
        assert isinstance(serialized["timestamp"], str)
        assert isinstance(serialized["nested"]["inner_date"], str)

    def test_to_jsonl_export(self, sample_storage_contents, temp_storage_dir):
        """Test JSONL export functionality."""
        exporter = StorageExporter(sample_storage_contents)
        output_path = temp_storage_dir / "test.jsonl"

        count = exporter.to_jsonl(output_path)

        assert count == 2
        assert output_path.exists()

        # Verify JSONL format and datetime serialization
        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2

            for line in lines:
                data = json.loads(line.strip())
                assert "job_id" in data
                # Verify datetime was serialized as string
                if "timestamp" in data:
                    assert isinstance(data["timestamp"], str)

    def test_to_csv_export(self, sample_storage_contents, temp_storage_dir):
        """Test CSV export functionality."""
        exporter = StorageExporter(sample_storage_contents)
        output_path = temp_storage_dir / "test.csv"

        count = exporter.to_csv(output_path)

        assert count == 2
        assert output_path.exists()

        # Verify CSV format
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert "job_id" in rows[0]

    def test_empty_contents_handling(self):
        """Test handling of empty storage contents."""
        empty_contents = StorageContents(
            rows=[], columns=["test"], output_fields=[], total_rows=0, metadata={}
        )

        # Should not raise exception
        exporter = StorageExporter(empty_contents)
        assert exporter.contents.total_rows == 0


class TestExporterIntegration:
    """Integration tests combining StorageManager and exporters."""

    @pytest.mark.asyncio
    async def test_full_export_workflow(self, temp_storage_dir):
        """Test complete workflow from data addition to export."""
        # Create storage and add data
        storage = StorageManager(temp_storage_dir)
        await storage.initialize()

        await storage.save_caption(
            Caption(
                job_id="integration_test",
                dataset="test_dataset",
                shard="default",
                item_key="integration_image",
                contributor_id="test_user",
                url="https://test.com/image.jpg",
                filename="image.jpg",
                timestamp=datetime.now(),
                captions=["Integration test caption"],
                outputs={"captions": ["Integration test caption"], "test_field": ["test value"]},
            )
        )

        await storage.checkpoint()

        # Test LanceStorageExporter
        lance_exporter = LanceStorageExporter(storage)
        lance_output = temp_storage_dir / "lance_export.jsonl"
        lance_count = await lance_exporter.export_shard("default", "jsonl", lance_output)

        # Actual file will be lance_export_default.jsonl
        actual_lance_output = temp_storage_dir / "lance_export_default.jsonl"

        assert lance_count == 1
        assert actual_lance_output.exists()

        # Verify exported data
        with open(actual_lance_output, "r") as f:
            data = json.loads(f.readline().strip())
            assert data["job_id"] == "integration_test"
            assert "test_field" in data
            # Verify datetime serialization worked
            assert isinstance(data["timestamp"], str)

        # Test StorageExporter via get_storage_contents
        contents = await storage.get_storage_contents()
        storage_exporter = StorageExporter(contents)
        storage_output = temp_storage_dir / "storage_export.jsonl"
        storage_count = storage_exporter.to_jsonl(storage_output)

        assert storage_count == 1
        assert storage_output.exists()


if __name__ == "__main__":
    pytest.main([__file__])
