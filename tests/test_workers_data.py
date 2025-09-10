"""Comprehensive tests for DataWorker module."""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from queue import Queue, Empty
from threading import Event
import io

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from caption_flow.workers.data import DataWorker, DataSample


@pytest.fixture
def sample_config():
    """Sample DataWorker configuration."""
    return {
        "token": "test-token",
        "name": "test-data-worker",
        "data_source": "test_data.jsonl",
        "source_type": "auto",
        "batch_size": 5,
        "server": "ws://localhost:8765",
    }


@pytest.fixture
def sample_storage_config():
    """Sample storage configuration."""
    return {
        "forward_to_orchestrator": True,
        "local": {"enabled": True, "path": "./test_data"},
        "s3": {
            "enabled": True,
            "bucket": "test-bucket",
            "access_key": "test-key",
            "secret_key": "test-secret",
            "region": "us-east-1",
        },
    }


@pytest.fixture
def sample_data_sample():
    """Sample DataSample for testing."""
    return DataSample(
        sample_id="test_001",
        image_url="https://example.com/image.jpg",
        metadata={"caption": "A test image", "source": "test"},
    )


class TestDataSample:
    """Test DataSample dataclass."""

    def test_init_with_url(self):
        """Test DataSample initialization with URL."""
        sample = DataSample(
            sample_id="test_001",
            image_url="https://example.com/image.jpg",
            metadata={"caption": "test"},
        )

        assert sample.sample_id == "test_001"
        assert sample.image_url == "https://example.com/image.jpg"
        assert sample.image_data is None
        assert sample.metadata == {"caption": "test"}

    def test_init_with_data(self):
        """Test DataSample initialization with image data."""
        image_data = b"fake_image_data"
        sample = DataSample(
            sample_id="test_002",
            image_data=image_data,
            metadata={"caption": "test"},
        )

        assert sample.sample_id == "test_002"
        assert sample.image_url is None
        assert sample.image_data == image_data
        assert sample.metadata == {"caption": "test"}

    def test_init_minimal(self):
        """Test DataSample initialization with minimal data."""
        sample = DataSample(sample_id="test_003")

        assert sample.sample_id == "test_003"
        assert sample.image_url is None
        assert sample.image_data is None
        assert sample.metadata is None


class TestDataWorkerInit:
    """Test DataWorker initialization."""

    def test_init_basic(self, sample_config):
        """Test basic DataWorker initialization."""
        worker = DataWorker(sample_config)

        assert worker.data_source == "test_data.jsonl"
        assert worker.source_type == "auto"
        assert worker.batch_size == 5
        assert worker.storage_config is None
        assert worker.s3_client is None
        assert isinstance(worker.can_send, Event)
        assert isinstance(worker.send_queue, Queue)
        assert worker.send_queue.maxsize == 100

    def test_init_with_defaults(self):
        """Test DataWorker initialization with default values."""
        config = {"token": "test", "name": "test-worker"}
        worker = DataWorker(config)

        assert worker.data_source is None
        assert worker.source_type == "auto"
        assert worker.batch_size == 10  # Default value

    @patch("caption_flow.workers.data.BaseWorker.__init__")
    def test_init_calls_parent(self, mock_parent_init, sample_config):
        """Test that DataWorker calls parent constructor."""
        DataWorker(sample_config)
        mock_parent_init.assert_called_once_with(sample_config)


class TestDataWorkerMetrics:
    """Test DataWorker metrics functionality."""

    def test_init_metrics(self, sample_config):
        """Test metrics initialization."""
        worker = DataWorker(sample_config)
        worker._init_metrics()

        assert worker.samples_sent == 0
        assert worker.samples_stored == 0
        assert worker.samples_failed == 0

    def test_get_heartbeat_data(self, sample_config):
        """Test heartbeat data generation."""
        worker = DataWorker(sample_config)
        worker._init_metrics()
        worker.samples_sent = 10
        worker.samples_stored = 5
        worker.samples_failed = 2

        heartbeat = worker._get_heartbeat_data()

        assert heartbeat["type"] == "heartbeat"
        assert heartbeat["sent"] == 10
        assert heartbeat["stored"] == 5
        assert heartbeat["failed"] == 2
        assert "queue_size" in heartbeat


class TestDataWorkerAuth:
    """Test DataWorker authentication."""

    def test_get_auth_data(self, sample_config):
        """Test authentication data generation."""
        worker = DataWorker(sample_config)
        auth_data = worker._get_auth_data()

        assert auth_data["token"] == "test-token"
        assert auth_data["name"] == "test-data-worker"
        assert auth_data["role"] == "data_worker"


class TestDataWorkerWelcome:
    """Test DataWorker welcome handling."""

    @pytest.mark.asyncio
    async def test_handle_welcome_basic(self, sample_config, sample_storage_config):
        """Test basic welcome message handling."""
        worker = DataWorker(sample_config)
        welcome_data = {"storage_config": sample_storage_config}

        with patch.object(worker, "_setup_s3_client") as mock_setup_s3:
            await worker._handle_welcome(welcome_data)

        assert worker.storage_config == sample_storage_config
        assert worker.can_send.is_set()
        mock_setup_s3.assert_called_once_with(sample_storage_config["s3"])

    @pytest.mark.asyncio
    async def test_handle_welcome_no_s3(self, sample_config):
        """Test welcome message without S3 configuration."""
        worker = DataWorker(sample_config)
        welcome_data = {"storage_config": {"local": {"enabled": True}}}

        with patch.object(worker, "_setup_s3_client") as mock_setup_s3:
            await worker._handle_welcome(welcome_data)

        assert worker.storage_config == {"local": {"enabled": True}}
        mock_setup_s3.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_welcome_no_storage_config(self, sample_config):
        """Test welcome message without storage config."""
        worker = DataWorker(sample_config)
        welcome_data = {}

        await worker._handle_welcome(welcome_data)

        assert worker.storage_config == {}


class TestDataWorkerMessages:
    """Test DataWorker message handling."""

    @pytest.mark.asyncio
    async def test_handle_backpressure(self, sample_config):
        """Test backpressure message handling."""
        worker = DataWorker(sample_config)
        worker.can_send.set()  # Start with ability to send

        await worker._handle_message({"type": "backpressure"})

        assert not worker.can_send.is_set()

    @pytest.mark.asyncio
    async def test_handle_resume(self, sample_config):
        """Test resume message handling."""
        worker = DataWorker(sample_config)
        worker.can_send.clear()  # Start without ability to send

        await worker._handle_message({"type": "resume"})

        assert worker.can_send.is_set()

    @pytest.mark.asyncio
    async def test_handle_unknown_message(self, sample_config):
        """Test unknown message handling."""
        worker = DataWorker(sample_config)
        initial_state = worker.can_send.is_set()

        await worker._handle_message({"type": "unknown"})

        # State should not change
        assert worker.can_send.is_set() == initial_state


class TestDataWorkerTasks:
    """Test DataWorker task creation."""

    @pytest.mark.asyncio
    async def test_create_tasks(self, sample_config):
        """Test task creation."""
        worker = DataWorker(sample_config)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = Mock()
            tasks = await worker._create_tasks()

        assert len(tasks) == 4
        assert mock_create_task.call_count == 4


class TestDataWorkerDisconnect:
    """Test DataWorker disconnection handling."""

    @pytest.mark.asyncio
    async def test_on_disconnect(self, sample_config):
        """Test disconnection handling."""
        worker = DataWorker(sample_config)
        worker.can_send.set()

        # Add some items to queue
        worker.send_queue.put("item1")
        worker.send_queue.put("item2")

        await worker._on_disconnect()

        assert not worker.can_send.is_set()
        assert worker.send_queue.empty()


class TestDataWorkerS3Setup:
    """Test DataWorker S3 configuration."""

    @patch("caption_flow.workers.data.boto3.client")
    def test_setup_s3_client_success(self, mock_boto3_client, sample_config):
        """Test successful S3 client setup."""
        worker = DataWorker(sample_config)
        s3_config = {
            "endpoint_url": "https://s3.amazonaws.com",
            "access_key": "test-key",
            "secret_key": "test-secret",
            "region": "us-east-1",
            "bucket": "test-bucket",
        }

        mock_client = Mock()
        mock_boto3_client.return_value = mock_client

        result = worker._setup_s3_client(s3_config)

        assert result == mock_client
        assert worker.s3_client == mock_client
        assert worker.s3_bucket == "test-bucket"
        mock_boto3_client.assert_called_once()

    def test_setup_s3_client_no_config(self, sample_config):
        """Test S3 client setup with no config."""
        worker = DataWorker(sample_config)
        result = worker._setup_s3_client(None)
        assert result is None

    @patch("caption_flow.workers.data.boto3.client")
    def test_setup_s3_client_failure(self, mock_boto3_client, sample_config):
        """Test S3 client setup failure."""
        worker = DataWorker(sample_config)
        s3_config = {"bucket": "test-bucket"}

        mock_boto3_client.side_effect = Exception("Connection failed")

        result = worker._setup_s3_client(s3_config)

        assert result is None


class TestDataWorkerSourceType:
    """Test DataWorker source type detection."""

    @pytest.mark.asyncio
    async def test_load_data_source_auto_jsonl(self, sample_config):
        """Test auto-detection of JSONL source."""
        config = {**sample_config, "data_source": "test.jsonl", "source_type": "auto"}
        worker = DataWorker(config)

        with patch.object(worker, "_load_jsonl") as mock_load_jsonl:
            mock_load_jsonl.return_value = iter([])

            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)

            mock_load_jsonl.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_source_auto_csv(self, sample_config):
        """Test auto-detection of CSV source."""
        config = {**sample_config, "data_source": "test.csv", "source_type": "auto"}
        worker = DataWorker(config)

        with patch.object(worker, "_load_csv") as mock_load_csv:
            mock_load_csv.return_value = iter([])

            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)

            mock_load_csv.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_source_auto_parquet(self, sample_config):
        """Test auto-detection of Parquet source."""
        config = {**sample_config, "data_source": "test.parquet", "source_type": "auto"}
        worker = DataWorker(config)

        with patch.object(worker, "_load_parquet") as mock_load_parquet:
            mock_load_parquet.return_value = iter([])

            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)

            mock_load_parquet.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_source_auto_huggingface(self, sample_config):
        """Test auto-detection of HuggingFace source."""
        config = {**sample_config, "data_source": "hf://dataset/name", "source_type": "auto"}
        worker = DataWorker(config)

        with patch.object(worker, "_load_huggingface") as mock_load_hf:
            mock_load_hf.return_value = iter([])

            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)

            mock_load_hf.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_source_explicit_type(self, sample_config):
        """Test explicit source type."""
        config = {**sample_config, "source_type": "jsonl"}
        worker = DataWorker(config)

        with patch.object(worker, "_load_jsonl") as mock_load_jsonl:
            mock_load_jsonl.return_value = iter([])

            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)

            mock_load_jsonl.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_source_unknown_type(self, sample_config):
        """Test unknown source type."""
        config = {**sample_config, "source_type": "unknown"}
        worker = DataWorker(config)

        with pytest.raises(ValueError, match="Unknown source type: unknown"):
            samples = []
            async for sample in worker._load_data_source():
                samples.append(sample)


class TestDataWorkerJsonlLoader:
    """Test DataWorker JSONL loading."""

    @pytest.mark.asyncio
    async def test_load_jsonl_success(self, sample_config, tmp_path):
        """Test successful JSONL loading."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_data = [
            {"id": "001", "url": "https://example.com/1.jpg", "caption": "Test 1"},
            {"id": "002", "image_url": "https://example.com/2.jpg", "caption": "Test 2"},
            {"url": "https://example.com/3.jpg", "caption": "Test 3"},  # No ID
        ]

        with open(jsonl_file, "w") as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + "\n")

        config = {**sample_config, "data_source": str(jsonl_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_jsonl():
            samples.append(sample)

        assert len(samples) == 3
        assert samples[0].sample_id == "001"
        assert samples[0].image_url == "https://example.com/1.jpg"
        assert samples[1].sample_id == "002"
        assert samples[1].image_url == "https://example.com/2.jpg"
        assert samples[2].sample_id == "sample_2"  # Auto-generated ID
        assert samples[2].image_url == "https://example.com/3.jpg"

    @pytest.mark.asyncio
    async def test_load_jsonl_with_invalid_lines(self, sample_config, tmp_path):
        """Test JSONL loading with invalid lines."""
        # Create test JSONL file with invalid JSON
        jsonl_file = tmp_path / "test.jsonl"
        content = """{"id": "001", "url": "https://example.com/1.jpg"}
invalid json line
{"id": "002", "url": "https://example.com/2.jpg"}"""

        with open(jsonl_file, "w") as f:
            f.write(content)

        config = {**sample_config, "data_source": str(jsonl_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_jsonl():
            samples.append(sample)

        # Should only load valid lines
        assert len(samples) == 2
        assert samples[0].sample_id == "001"
        assert samples[1].sample_id == "002"


class TestDataWorkerCsvLoader:
    """Test DataWorker CSV loading."""

    @pytest.mark.asyncio
    async def test_load_csv_success(self, sample_config, tmp_path):
        """Test successful CSV loading."""
        # Create test CSV file
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "id": ["001", "002", "003"],
                "image_url": [
                    "https://example.com/1.jpg",
                    "https://example.com/2.jpg",
                    "https://example.com/3.jpg",
                ],
                "caption": ["Test 1", "Test 2", "Test 3"],
            }
        )
        df.to_csv(csv_file, index=False)

        config = {**sample_config, "data_source": str(csv_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_csv():
            samples.append(sample)

        assert len(samples) == 3
        assert samples[0].sample_id == "001"
        assert samples[0].image_url == "https://example.com/1.jpg"
        assert samples[0].metadata["caption"] == "Test 1"

    @pytest.mark.asyncio
    async def test_load_csv_no_url_column(self, sample_config, tmp_path):
        """Test CSV loading without URL column."""
        # Create test CSV file without URL column
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"id": ["001", "002"], "caption": ["Test 1", "Test 2"]})
        df.to_csv(csv_file, index=False)

        config = {**sample_config, "data_source": str(csv_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_csv():
            samples.append(sample)

        assert len(samples) == 2
        assert samples[0].image_url is None

    @pytest.mark.asyncio
    async def test_load_csv_no_id_column(self, sample_config, tmp_path):
        """Test CSV loading without ID column."""
        # Create test CSV file without ID column
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "url": ["https://example.com/1.jpg", "https://example.com/2.jpg"],
                "caption": ["Test 1", "Test 2"],
            }
        )
        df.to_csv(csv_file, index=False)

        config = {**sample_config, "data_source": str(csv_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_csv():
            samples.append(sample)

        assert len(samples) == 2
        assert samples[0].sample_id == "0"  # Uses index
        assert samples[1].sample_id == "1"


class TestDataWorkerParquetLoader:
    """Test DataWorker Parquet loading."""

    @pytest.mark.asyncio
    async def test_load_parquet_success(self, sample_config, tmp_path):
        """Test successful Parquet loading."""
        # Create test Parquet file
        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame(
            {
                "id": ["001", "002", "003"],
                "link": [
                    "https://example.com/1.jpg",
                    "https://example.com/2.jpg",
                    "https://example.com/3.jpg",
                ],
                "caption": ["Test 1", "Test 2", "Test 3"],
            }
        )
        df.to_parquet(parquet_file)

        config = {**sample_config, "data_source": str(parquet_file)}
        worker = DataWorker(config)

        samples = []
        async for sample in worker._load_parquet():
            samples.append(sample)

        assert len(samples) == 3
        assert samples[0].sample_id == "001"
        assert samples[0].image_url == "https://example.com/1.jpg"
        assert samples[0].metadata["caption"] == "Test 1"


class TestDataWorkerHuggingFaceLoader:
    """Test DataWorker HuggingFace loading."""

    @pytest.mark.asyncio
    async def test_load_huggingface_with_url_prefix(self, sample_config):
        """Test HuggingFace loading with hf:// prefix."""
        config = {**sample_config, "data_source": "hf://test/dataset"}
        worker = DataWorker(config)

        mock_dataset = [
            {"id": "001", "url": "https://example.com/1.jpg", "caption": "Test 1"},
            {"id": "002", "image_url": "https://example.com/2.jpg", "caption": "Test 2"},
        ]

        with patch("caption_flow.workers.data.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            samples = []
            async for sample in worker._load_huggingface():
                samples.append(sample)

        mock_load_dataset.assert_called_once_with("test/dataset", split="train", streaming=True)
        assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_load_huggingface_without_prefix(self, sample_config):
        """Test HuggingFace loading without hf:// prefix."""
        config = {**sample_config, "data_source": "test/dataset"}
        worker = DataWorker(config)

        mock_dataset = [{"id": "001", "url": "https://example.com/1.jpg"}]

        with patch("caption_flow.workers.data.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            samples = []
            async for sample in worker._load_huggingface():
                samples.append(sample)

        mock_load_dataset.assert_called_once_with("test/dataset", split="train", streaming=True)

    @pytest.mark.asyncio
    async def test_load_huggingface_with_pil_image(self, sample_config):
        """Test HuggingFace loading with PIL Image."""
        config = {**sample_config, "data_source": "test/dataset"}
        worker = DataWorker(config)

        # Create a mock PIL Image
        mock_image = Mock()
        mock_image.save = Mock()

        mock_dataset = [
            {"id": "001", "image": mock_image, "caption": "Test 1"},
        ]

        with patch("caption_flow.workers.data.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            samples = []
            async for sample in worker._load_huggingface():
                samples.append(sample)

        assert len(samples) == 1
        assert samples[0].image_data is not None
        mock_image.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_huggingface_auto_id(self, sample_config):
        """Test HuggingFace loading with auto-generated IDs."""
        config = {**sample_config, "data_source": "test/dataset"}
        worker = DataWorker(config)

        mock_dataset = [
            {"url": "https://example.com/1.jpg", "caption": "Test 1"},  # No ID
            {"url": "https://example.com/2.jpg", "caption": "Test 2"},  # No ID
        ]

        with patch("caption_flow.workers.data.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            samples = []
            async for sample in worker._load_huggingface():
                samples.append(sample)

        assert len(samples) == 2
        assert samples[0].sample_id == "hf_0"
        assert samples[1].sample_id == "hf_1"


class TestDataWorkerImageDownload:
    """Test DataWorker image downloading."""

    @pytest.mark.asyncio
    async def test_download_image_success(self, sample_config):
        """Test successful image download."""
        worker = DataWorker(sample_config)
        url = "https://example.com/image.jpg"
        expected_data = b"fake_image_data"

        with patch("caption_flow.workers.data.aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=expected_data)

            mock_session_instance = Mock()
            mock_session_instance.get = Mock()
            mock_session_instance.get.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session_instance.get.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await worker._download_image(url)

        assert result == expected_data

    @pytest.mark.asyncio
    async def test_download_image_failure(self, sample_config):
        """Test image download failure."""
        worker = DataWorker(sample_config)
        url = "https://example.com/image.jpg"

        with patch("caption_flow.workers.data.aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 404

            mock_session_instance = Mock()
            mock_session_instance.get = Mock()
            mock_session_instance.get.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session_instance.get.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await worker._download_image(url)

        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_exception(self, sample_config):
        """Test image download with exception."""
        worker = DataWorker(sample_config)
        url = "https://example.com/image.jpg"

        with patch("caption_flow.workers.data.aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = Exception("Connection error")

            result = await worker._download_image(url)

        assert result is None


class TestDataWorkerStorage:
    """Test DataWorker storage functionality."""

    @pytest.mark.asyncio
    async def test_store_sample_local(self, sample_config, sample_data_sample, tmp_path):
        """Test storing sample locally."""
        worker = DataWorker(sample_config)
        worker.storage_config = {"local": {"enabled": True, "path": str(tmp_path)}}

        image_data = b"fake_image_data"

        result = await worker._store_sample(sample_data_sample, image_data)

        assert result is True

        # Check files were created
        image_path = tmp_path / "test_001.jpg"
        meta_path = tmp_path / "test_001.json"

        assert image_path.exists()
        assert meta_path.exists()

        # Check file contents
        assert image_path.read_bytes() == image_data
        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata == sample_data_sample.metadata

    @pytest.mark.asyncio
    async def test_store_sample_s3(self, sample_config, sample_data_sample):
        """Test storing sample to S3."""
        worker = DataWorker(sample_config)
        worker.storage_config = {"s3": {"enabled": True}}

        # Mock S3 client
        mock_s3_client = Mock()
        worker.s3_client = mock_s3_client
        worker.s3_bucket = "test-bucket"

        image_data = b"fake_image_data"

        result = await worker._store_sample(sample_data_sample, image_data)

        assert result is True
        assert mock_s3_client.put_object.call_count == 2  # Image + metadata

    @pytest.mark.asyncio
    async def test_store_sample_no_storage(self, sample_config, sample_data_sample):
        """Test storing sample with no storage enabled."""
        worker = DataWorker(sample_config)
        worker.storage_config = {}

        image_data = b"fake_image_data"

        result = await worker._store_sample(sample_data_sample, image_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_sample_local_error(self, sample_config, sample_data_sample):
        """Test storing sample locally with error."""
        worker = DataWorker(sample_config)
        worker.storage_config = {"local": {"enabled": True, "path": "/invalid/path"}}

        image_data = b"fake_image_data"

        result = await worker._store_sample(sample_data_sample, image_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_sample_s3_error(self, sample_config, sample_data_sample):
        """Test storing sample to S3 with error."""
        worker = DataWorker(sample_config)
        worker.storage_config = {"s3": {"enabled": True}}

        # Mock S3 client with error
        mock_s3_client = Mock()
        mock_s3_client.put_object.side_effect = Exception("S3 error")
        worker.s3_client = mock_s3_client
        worker.s3_bucket = "test-bucket"

        image_data = b"fake_image_data"

        result = await worker._store_sample(sample_data_sample, image_data)

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
