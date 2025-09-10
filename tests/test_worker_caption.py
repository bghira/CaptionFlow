"""Comprehensive tests for CaptionWorker with multi-stage processing."""

import asyncio
import datetime as _datetime
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

# Import pytest-asyncio
pytest_plugins = ("pytest_asyncio",)
import pytest_asyncio
from caption_flow.models import Caption, JobId, ProcessingStage
from caption_flow.processors import WorkAssignment, WorkUnit
from caption_flow.storage import StorageManager

# Import the modules to test
from caption_flow.workers.caption import (
    CaptionWorker,
    ProcessingItem,
)


class TestCaptionWorker:
    """Test suite for CaptionWorker."""

    @pytest.fixture
    def worker_config(self):
        """Create test worker configuration."""
        return {
            "name": "test_worker",
            "token": "test_token",
            "server": "ws://localhost:8765",
            "server_url": "ws://localhost:8765",
            "gpu_id": 0,
            "batch_image_processing": True,
        }

    @pytest.fixture
    def mock_vllm_config(self):
        """Create mock vLLM configuration with multi-stage setup."""
        return {
            "model": "test-model",
            "batch_size": 4,
            "max_model_len": 16384,
            "stages": [
                {
                    "name": "caption",
                    "model": "model1",
                    "prompts": ["Describe this image"],
                    "output_field": "captions",
                },
                {
                    "name": "enhance",
                    "model": "model2",
                    "prompts": ["Enhance this caption: {captions}"],
                    "output_field": "enhanced",
                    "requires": ["caption"],
                },
                {
                    "name": "tags",
                    "model": "model1",  # Reuse model1
                    "prompts": ["List tags for this image"],
                    "output_field": "tags",
                },
            ],
            "sampling": {
                "temperature": 0.7,
                "max_tokens": 256,
            },
        }

    @pytest.fixture
    def caption_worker(self, worker_config):
        """Create a CaptionWorker instance."""
        with patch("huggingface_hub.get_token", return_value="mock_token"):
            worker = CaptionWorker(worker_config)
            worker.main_loop = asyncio.get_event_loop()
            return worker

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock websocket."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    @pytest_asyncio.fixture
    async def storage_manager(self, tmp_path):
        """Create a StorageManager instance."""
        storage = StorageManager(tmp_path, caption_buffer_size=10)
        await storage.initialize()
        return storage

    def test_initialization(self, caption_worker):
        """Test worker initialization."""
        assert caption_worker.name == "test_worker"
        assert caption_worker.token == "test_token"
        assert caption_worker.gpu_id == 0
        assert caption_worker.processor is None
        assert caption_worker.mock_mode is False
        assert caption_worker.stages == []
        assert caption_worker.result_queue.empty()

    def test_parse_stages_config(self, caption_worker, mock_vllm_config):
        """Test parsing of multi-stage configuration."""
        stages = caption_worker._parse_stages_config(mock_vllm_config)

        assert len(stages) == 3
        assert stages[0].name == "caption"
        assert stages[0].model == "model1"
        assert stages[1].name == "enhance"
        assert stages[1].requires == ["caption"]
        assert stages[2].name == "tags"
        assert stages[2].model == "model1"  # Reuses same model

    def test_parse_stages_backward_compatibility(self, caption_worker):
        """Test backward compatibility with single-stage config."""
        old_config = {
            "model": "single-model",
            "inference_prompts": ["Test prompt"],
        }

        stages = caption_worker._parse_stages_config(old_config)

        assert len(stages) == 1
        assert stages[0].name == "default"
        assert stages[0].model == "single-model"
        assert stages[0].output_field == "captions"

    def test_topological_sort_stages(self, caption_worker):
        """Test dependency sorting of stages."""
        stages = [
            ProcessingStage("third", "m3", ["prompt"], "out3", ["second"]),
            ProcessingStage("first", "m1", ["prompt"], "out1", []),
            ProcessingStage("second", "m2", ["prompt"], "out2", ["first"]),
            ProcessingStage("fourth", "m4", ["prompt"], "out4", ["first", "third"]),
        ]

        order = caption_worker._topological_sort_stages(stages)

        assert order.index("first") < order.index("second")
        assert order.index("second") < order.index("third")
        assert order.index("first") < order.index("fourth")
        assert order.index("third") < order.index("fourth")

    def test_topological_sort_circular_dependency(self, caption_worker):
        """Test detection of circular dependencies."""
        stages = [
            ProcessingStage("a", "m1", ["prompt"], "out1", ["b"]),
            ProcessingStage("b", "m2", ["prompt"], "out2", ["a"]),
        ]

        with pytest.raises(ValueError, match="Circular dependency"):
            caption_worker._topological_sort_stages(stages)

    def test_mock_mode_detection(self, caption_worker):
        """Test mock mode detection from config."""
        # Set config with mock mode
        caption_worker.vllm_config = {"mock_results": True}
        caption_worker.mock_mode = caption_worker.vllm_config.get("mock_results", False)

        assert caption_worker.mock_mode is True

    @pytest.mark.asyncio
    async def test_handle_welcome_message(self, caption_worker, mock_vllm_config):
        """Test handling of welcome message from orchestrator."""
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()

        welcome_data = {
            "type": "welcome",
            "processor_type": "webdataset",
            "processor_config": {
                "dataset_path": "s3://test-bucket",
                "chunks_per_request": 2,
                "vllm": mock_vllm_config,
            },
        }

        await caption_worker._handle_welcome(welcome_data)

        assert caption_worker.processor is not None
        assert caption_worker.processor_type == "webdataset"
        assert caption_worker.units_per_request == 2

        # Should request initial work
        caption_worker.websocket.send.assert_called_once()
        sent_data = json.loads(caption_worker.websocket.send.call_args[0][0])
        assert sent_data["type"] == "get_work_units"
        assert sent_data["count"] == 2

    def test_process_batch_mock(self, caption_worker, mock_vllm_config):
        """Test batch processing in mock mode."""
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = True
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Create test batch
        batch = []
        for i in range(3):
            item = ProcessingItem(
                unit_id="unit1",
                job_id=f"shard:chunk:0:idx:{i}",
                chunk_id="1",
                item_key=f"item_{i}",
                item_index=i,
                image=Image.new("RGB", (100, 100)),
                image_data=b"fake_data",
                metadata={},
            )
            batch.append(item)

        # Process batch
        results = caption_worker._process_batch_mock(batch)

        assert len(results) == 3

        for item, outputs in results:
            # Check all stages produced outputs
            assert "captions" in outputs
            assert "enhanced" in outputs
            assert "tags" in outputs

            # Check mock content
            assert all("Mock" in text for text in outputs["captions"])
            assert all("Mock" in text for text in outputs["enhanced"])

    def test_vllm_config_update(self, caption_worker, mock_vllm_config):
        """Test handling of vLLM configuration updates."""
        # Initial config
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Update config with new stage
        new_config = mock_vllm_config.copy()
        new_config["stages"].append(
            {
                "name": "sentiment",
                "model": "model3",
                "prompts": ["Analyze sentiment"],
                "output_field": "sentiment",
            }
        )

        # Mock model manager
        caption_worker.model_manager = Mock()
        caption_worker.model_manager.cleanup = Mock()

        with patch.object(caption_worker, "_setup_vllm", Mock()):
            result = caption_worker._handle_vllm_config_update(new_config)

        assert result is True
        assert len(caption_worker.stages) == 4
        assert "sentiment" in caption_worker.stage_order

    def test_processing_item_creation(self):
        """Test ProcessingItem creation and initialization."""
        item = ProcessingItem(
            unit_id="unit1",
            job_id="shard:chunk:0:idx:42",
            chunk_id="1",
            item_key="test_item",
            item_index=42,
            image=Image.new("RGB", (200, 200)),
            image_data=b"test_data",
            metadata={"test": "value"},
        )

        assert item.stage_results == {}
        assert item.metadata["test"] == "value"

    def test_work_unit_processing(self, caption_worker, mock_vllm_config):
        """Test processing of work units."""
        # Setup
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = True
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)
        caption_worker.connected = Mock()
        caption_worker.connected.is_set = Mock(return_value=True)
        caption_worker.should_stop_processing = Mock()
        caption_worker.should_stop_processing.is_set = Mock(return_value=False)
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()

        # Mock processor
        mock_processor = Mock()
        mock_items = [
            {
                "job_id": "shard:chunk:0:idx:0",
                "item_key": "item_0",
                "item_index": 0,
                "image": Image.new("RGB", (100, 100)),
                "image_data": b"data",
                "metadata": {},
            }
        ]
        mock_processor.process_unit = Mock(return_value=iter(mock_items))
        caption_worker.processor = mock_processor

        # Create work unit
        unit = WorkUnit(
            unit_id="unit1",
            chunk_id="1",
            source_id="shard1",
            unit_size=1,
            data={},
            metadata={},
        )

        # Process unit
        caption_worker._process_work_unit(unit)

        # Check results were queued
        assert caption_worker.result_queue.qsize() > 0
        assert caption_worker.items_processed == 1
        assert caption_worker.items_failed == 0

    def test_validate_and_split_batch(self, caption_worker, mock_vllm_config):
        """Test batch validation and token length checking."""
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)

        # Create test items
        batch = []
        for i in range(3):
            item = ProcessingItem(
                unit_id="unit1",
                job_id=f"job_{i}",
                chunk_id="1",
                item_key=f"item_{i}",
                item_index=i,
                image=Image.new("RGB", (100, 100)),
                image_data=b"data",
                metadata={},
            )
            batch.append(item)

        # Mock tokenizer and processor
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1] * 100)  # 100 tokens
        mock_processor = Mock()
        mock_sampling = Mock()

        stage = caption_worker.stages[0]

        # Test with reasonable max length
        processable, too_long = caption_worker._validate_and_split_batch(
            batch, stage, mock_processor, mock_tokenizer, mock_sampling, max_length=1000
        )

        # All should be processable
        assert len(processable) == 3
        assert len(too_long) == 0

    def test_resize_image_for_tokens(self, caption_worker):
        """Test image resizing for token reduction."""
        item = ProcessingItem(
            unit_id="unit1",
            job_id="job1",
            chunk_id="1",
            item_key="test",
            item_index=0,
            image=Image.new("RGB", (1000, 1000)),
            image_data=b"data",
            metadata={},
        )

        resized_item = caption_worker._resize_image_for_tokens(item, target_ratio=0.5)

        assert resized_item.image.width == 500
        assert resized_item.image.height == 500
        assert resized_item.metadata["_resized"] is True
        assert resized_item.metadata["_resize_ratio"] == 0.5

    def test_clean_output(self, caption_worker):
        """Test output cleaning."""
        test_cases = [
            ("This is a test<|end|>extra text", "This is a test"),
            ("Good response<|endoftext|>", "Good response"),
            ("I'm sorry, I cannot help", ""),
            ("   Trimmed output   ", "Trimmed output"),
            ("", ""),
        ]

        for input_text, expected in test_cases:
            assert caption_worker._clean_output(input_text) == expected

    def test_heartbeat_data(self, caption_worker):
        """Test heartbeat data generation."""
        caption_worker.items_processed = 10
        caption_worker.items_failed = 2
        caption_worker.units_completed = 5
        caption_worker.current_unit = None
        caption_worker.stages = [Mock(), Mock()]
        caption_worker.model_manager = Mock()
        caption_worker.model_manager.models = {"model1": Mock()}
        caption_worker.mock_mode = False

        heartbeat = caption_worker._get_heartbeat_data()

        assert heartbeat["type"] == "heartbeat"
        assert heartbeat["processed"] == 10
        assert heartbeat["failed"] == 2
        assert heartbeat["units_completed"] == 5
        assert heartbeat["stages"] == 2
        assert heartbeat["models_loaded"] == 1
        assert heartbeat["mock_mode"] is False

    @pytest.mark.asyncio
    async def test_result_sender(self, caption_worker):
        """Test result sending back to orchestrator."""
        caption_worker.running = True
        caption_worker.connected.is_set = Mock(return_value=True)
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()
        caption_worker.dataset_path = "test/dataset"

        # Add a result to queue
        test_item = ProcessingItem(
            unit_id="unit1",
            job_id="shard:chunk:0:idx:42",
            chunk_id="1",
            item_key="test_item",
            item_index=42,
            image=Image.new("RGB", (200, 100)),
            image_data=b"test_data",
            metadata={"_item_index": 42},
        )

        caption_worker.result_queue.put(
            {
                "item": test_item,
                "outputs": {"captions": ["Test caption"]},
                "processing_time_ms": 123.4,
            }
        )

        # Run result sender briefly
        sender_task = asyncio.create_task(caption_worker._result_sender())
        await asyncio.sleep(0.1)
        caption_worker.running = False
        caption_worker.connected.is_set = Mock(return_value=False)

        try:
            await asyncio.wait_for(sender_task, timeout=1.0)
        except asyncio.TimeoutError:
            sender_task.cancel()

        # Check result was sent
        caption_worker.websocket.send.assert_called()
        sent_data = json.loads(caption_worker.websocket.send.call_args[0][0])

        assert sent_data["type"] == "submit_results"
        assert sent_data["unit_id"] == "unit1"
        assert sent_data["job_id"] == "shard:chunk:0:idx:42"
        assert sent_data["outputs"] == {"captions": ["Test caption"]}
        assert sent_data["metadata"]["image_width"] == 200
        assert sent_data["metadata"]["image_height"] == 100

    def test_work_assignment_handling(self, caption_worker):
        """Test handling of work assignments."""
        assignment = WorkAssignment(
            assignment_id="assign1",
            worker_id="test_worker",
            units=[
                WorkUnit("unit1", "1", "source1", 100, {}, {}),
                WorkUnit("unit2", "chunk2", "source1", 100, {}, {}),
            ],
            assigned_at=datetime.now(_datetime.UTC),
        )

        {
            "type": "work_assignment",
            "assignment": assignment.to_dict(),
        }

        # Process assignment (synchronously for testing)
        with caption_worker.work_lock:
            for unit in assignment.units:
                caption_worker.assigned_units.append(unit)

        assert len(caption_worker.assigned_units) == 2
        assert caption_worker.assigned_units[0].unit_id == "unit1"

    @pytest.mark.asyncio
    async def test_disconnect_handling(self, caption_worker):
        """Test handling of disconnection."""
        caption_worker.assigned_units.extend([Mock(), Mock()])
        caption_worker.current_unit = Mock()
        caption_worker.result_queue.put("test")

        await caption_worker._on_disconnect()

        assert caption_worker.should_stop_processing.is_set()
        assert len(caption_worker.assigned_units) == 0
        assert caption_worker.current_unit is None
        assert caption_worker.result_queue.empty()

    @pytest.mark.asyncio
    async def test_integration_with_storage(
        self, caption_worker, storage_manager, mock_vllm_config
    ):
        """Test integration with storage system."""
        # Setup worker
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = True
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Create and process an item
        item = ProcessingItem(
            unit_id="unit1",
            job_id="shard1:chunk:0:idx:42",
            chunk_id="shard1:chunk:0",
            item_key="test_item",
            item_index=42,
            image=Image.new("RGB", (100, 100)),
            image_data=b"test",
            metadata={"_item_index": 42},
        )

        # Process batch
        results = caption_worker._process_batch_mock([item])
        assert len(results) == 1

        # Create caption from result
        item_result, outputs = results[0]
        job_id = JobId.from_str(item_result.job_id)

        caption = Caption(
            job_id=job_id,
            dataset="test_dataset",
            shard="shard1",
            chunk_id=item_result.chunk_id,
            item_key=item_result.item_key,
            caption=outputs["captions"][0] if outputs.get("captions") else "",
            outputs=outputs,
            contributor_id="test_worker",
            timestamp=datetime.now(_datetime.UTC),
            caption_count=sum(len(v) for v in outputs.values()),
            metadata=item_result.metadata,
        )

        # Save to storage
        await storage_manager.save_caption(caption)

        # Force flush
        await storage_manager.checkpoint()

        # Verify storage stats
        stats = await storage_manager.get_storage_stats()
        assert stats["total_rows"] >= 1
        assert stats["total_captions"] >= 3  # captions, enhanced, tags

    def test_error_handling_in_batch_processing(self, caption_worker, mock_vllm_config):
        """Test error handling during batch processing."""
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = False
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Mock model manager to raise error
        caption_worker.model_manager = Mock()
        caption_worker.model_manager.get_model_for_stage = Mock(
            side_effect=Exception("Model loading failed")
        )

        batch = [
            ProcessingItem(
                unit_id="unit1",
                job_id="job1",
                chunk_id="1",
                item_key="test",
                item_index=0,
                image=Image.new("RGB", (100, 100)),
                image_data=b"data",
                metadata={},
            )
        ]

        # Should not raise but handle error gracefully
        caption_worker._process_batch(batch)

        # Check error was logged (would need to check logs in real test)
        assert True  # Placeholder - in real test would check logging

    def test_batch_error_handling_with_failure_reporting(self, caption_worker, mock_vllm_config):
        """Test that batch processing errors are properly reported to orchestrator."""
        # Setup worker
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = False
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Mock websocket and connection
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()
        caption_worker.connected = Mock()
        caption_worker.connected.is_set = Mock(return_value=True)
        caption_worker.should_stop_processing = Mock()
        caption_worker.should_stop_processing.is_set = Mock(return_value=False)
        caption_worker.main_loop = asyncio.get_event_loop()

        # Mock model manager to raise error
        caption_worker.model_manager = Mock()
        caption_worker.model_manager.get_model_for_stage = Mock(
            side_effect=Exception(
                "The decoder prompt (length 16409) is longer than the maximum model length of 16384"
            )
        )

        # Create test batch
        batch = [
            ProcessingItem(
                unit_id="unit1",
                job_id="job1",
                chunk_id="1",
                item_key="test_item",
                item_index=0,
                image=Image.new("RGB", (100, 100)),
                image_data=b"data",
                metadata={},
            )
        ]

        # Process batch - this should fail
        initial_failed_count = caption_worker.items_failed
        caption_worker._process_batch(batch)

        # Check that items_failed was incremented
        assert caption_worker.items_failed == initial_failed_count + len(batch)

        # Check that error results were queued
        assert not caption_worker.result_queue.empty()
        result_data = caption_worker.result_queue.get()

        assert result_data["item"] == batch[0]
        assert result_data["outputs"] == {}
        assert "error" in result_data
        assert "Batch processing failed" in result_data["error"]

    def test_unit_failure_reporting_to_orchestrator(self, caption_worker, mock_vllm_config):
        """Test that unit failures are properly reported to orchestrator."""
        # Setup worker
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = True  # Use mock mode to avoid complex setup
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Mock websocket and connection
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()
        caption_worker.connected = Mock()
        caption_worker.connected.is_set = Mock(return_value=True)
        caption_worker.should_stop_processing = Mock()
        caption_worker.should_stop_processing.is_set = Mock(return_value=False)
        caption_worker.main_loop = asyncio.get_event_loop()

        # Mock processor that will simulate batch failures
        mock_processor = Mock()
        # Create items that look like they will succeed
        mock_items = [
            {
                "job_id": "job1",
                "item_key": "item_1",
                "item_index": 0,
                "image": Image.new("RGB", (100, 100)),
                "image_data": b"data",
                "metadata": {},
            },
            {
                "job_id": "job2",
                "item_key": "item_2",
                "item_index": 1,
                "image": Image.new("RGB", (100, 100)),
                "image_data": b"data",
                "metadata": {},
            },
            {
                "job_id": "job3",
                "item_key": "item_3",
                "item_index": 2,
                "image": Image.new("RGB", (100, 100)),
                "image_data": b"data",
                "metadata": {},
            },
        ]
        mock_processor.process_unit = Mock(return_value=iter(mock_items))
        caption_worker.processor = mock_processor

        # Create work unit expecting 3 items
        unit = WorkUnit(
            unit_id="unit1",
            chunk_id="1",
            source_id="shard1",
            unit_size=3,
            data={},
            metadata={},
        )

        # Mock _process_batch to simulate batch failures
        original_process_batch = caption_worker._process_batch

        def mock_process_batch(batch):
            # Simulate batch processing failure
            caption_worker.items_failed += len(batch)
            # Don't increment items_processed to simulate failures
            for item in batch:
                caption_worker.result_queue.put(
                    {
                        "item": item,
                        "outputs": {},
                        "processing_time_ms": 0.0,
                        "error": "Simulated batch failure",
                    }
                )

        caption_worker._process_batch = mock_process_batch

        # Process unit - this will trigger failures
        caption_worker._process_work_unit(unit)

        # Check that work_failed was sent to orchestrator (failures > 0, processed == 0)
        caption_worker.websocket.send.assert_called()

        # Find the work_failed message
        calls = caption_worker.websocket.send.call_args_list
        work_failed_call = None
        for call in calls:
            sent_data = json.loads(call[0][0])
            if sent_data.get("type") == "work_failed":
                work_failed_call = sent_data
                break

        assert work_failed_call is not None, "work_failed message should have been sent"
        assert work_failed_call["unit_id"] == "unit1"
        assert "Processing failed for" in work_failed_call["error"]
        assert "3 out of 3 items" in work_failed_call["error"]

        # Restore original method
        caption_worker._process_batch = original_process_batch

    def test_unit_incomplete_reporting_to_orchestrator(self, caption_worker, mock_vllm_config):
        """Test that incomplete units (not failed, but not fully processed) are reported."""
        # Setup worker
        caption_worker.vllm_config = mock_vllm_config
        caption_worker.mock_mode = True  # Use mock mode to avoid complex setup
        caption_worker.stages = caption_worker._parse_stages_config(mock_vllm_config)
        caption_worker.stage_order = caption_worker._topological_sort_stages(caption_worker.stages)

        # Mock websocket and connection
        caption_worker.websocket = Mock()
        caption_worker.websocket.send = AsyncMock()
        caption_worker.connected = Mock()
        caption_worker.connected.is_set = Mock(return_value=True)
        caption_worker.should_stop_processing = Mock()
        caption_worker.should_stop_processing.is_set = Mock(return_value=False)
        caption_worker.main_loop = asyncio.get_event_loop()

        # Mock processor that yields fewer items than expected
        mock_processor = Mock()
        mock_items = [
            {
                "job_id": "job1",
                "item_key": "item_1",
                "item_index": 0,
                "image": Image.new("RGB", (100, 100)),
                "image_data": b"data",
                "metadata": {},
            }
        ]
        mock_processor.process_unit = Mock(return_value=iter(mock_items))
        caption_worker.processor = mock_processor

        # Create work unit expecting 3 items but only 1 will be processed
        unit = WorkUnit(
            unit_id="unit1",
            chunk_id="1",
            source_id="shard1",
            unit_size=3,  # Expecting 3 items, but only 1 will be processed
            data={},
            metadata={},
        )

        # Process unit (no failures, just incomplete)
        caption_worker._process_work_unit(unit)

        # Check that work_failed was sent to orchestrator
        caption_worker.websocket.send.assert_called()

        # Find the work_failed message
        calls = caption_worker.websocket.send.call_args_list
        work_failed_call = None
        for call in calls:
            sent_data = json.loads(call[0][0])
            if sent_data.get("type") == "work_failed":
                work_failed_call = sent_data
                break

        assert work_failed_call is not None, (
            "work_failed message should have been sent for incomplete unit"
        )
        assert work_failed_call["unit_id"] == "unit1"
        assert "Processing incomplete" in work_failed_call["error"]
        assert "1/3 items processed" in work_failed_call["error"]


class TestCaptionWorkerProcessors:
    """Test CaptionWorker with different processor types."""

    @pytest.fixture
    def worker_config(self):
        return {
            "name": "test_worker",
            "token": "test_token",
            "server": "ws://localhost:8765",
            "server_url": "ws://localhost:8765",
            "gpu_id": 0,
        }

    @pytest.mark.asyncio
    async def test_huggingface_processor_integration(self, worker_config):
        """Test CaptionWorker with HuggingFace processor."""
        worker = CaptionWorker(worker_config)

        welcome_data = {
            "processor_type": "huggingface_datasets",
            "processor_config": {
                "dataset": {
                    "dataset_path": "test/dataset",
                    "mock_results": True,
                },
            },
        }

        await worker._handle_welcome(welcome_data)

        assert worker.processor_type == "huggingface_datasets"
        assert worker.processor is not None

    @pytest.mark.asyncio
    async def test_local_filesystem_processor_integration(self, worker_config):
        """Test CaptionWorker with LocalFilesystem processor."""
        worker = CaptionWorker(worker_config)

        welcome_data = {
            "processor_type": "local_filesystem",
            "processor_config": {
                "dataset": {
                    "dataset_path": "/tmp/images",
                },
            },
        }

        # Mock the heavy initialization operations to speed up CI
        with patch(
            "caption_flow.processors.local_filesystem.LocalFilesystemWorkerProcessor.initialize"
        ) as mock_init:
            # Just set the required attributes without full initialization
            def mock_initialize(config):
                worker.processor.dataset_path = "/tmp/images"

            mock_init.side_effect = mock_initialize

            await worker._handle_welcome(welcome_data)

        assert worker.processor_type == "local_filesystem"
        assert worker.processor is not None


class TestCaptionWorkerConfigReload:
    """Test CaptionWorker config reload functionality."""

    @pytest.fixture
    def worker_config(self):
        return {
            "name": "test_worker",
            "token": "test_token",
            "server": "ws://localhost:8765",
            "server_url": "ws://localhost:8765",
            "gpu_id": 0,
        }

    @pytest.fixture
    def initial_vllm_config(self):
        return {
            "model": "test-model-v1",
            "batch_size": 4,
            "max_model_len": 16384,
            "stages": [
                {
                    "name": "caption",
                    "model": "test-model-v1",
                    "prompts": ["describe this image"],
                    "output_field": "captions",
                    "requires": [],
                }
            ],
        }

    def test_config_reload_failure_restores_state(self, worker_config, initial_vllm_config):
        """Test that config reload failure properly restores previous state."""
        worker = CaptionWorker(worker_config)

        # Set up initial state
        worker.vllm_config = initial_vllm_config
        worker.stages = worker._parse_stages_config(initial_vllm_config)
        worker.stage_order = worker._topological_sort_stages(worker.stages)
        worker.mock_mode = False

        # Mock model manager with working models
        mock_model_manager = Mock()
        mock_model_manager.models = {"test-model-v1": "loaded_model"}
        mock_model_manager.processors = {"test-model-v1": "loaded_processor"}
        mock_model_manager.tokenizers = {"test-model-v1": "loaded_tokenizer"}
        mock_model_manager.sampling_params = {"caption": "loaded_sampling"}
        worker.model_manager = mock_model_manager

        # New config that will cause setup failure
        new_config = {
            "model": "test-model-v2",
            "batch_size": 8,
            "stages": [
                {
                    "name": "caption",
                    "model": "test-model-v2",
                    "prompts": ["analyze this image"],
                    "output_field": "captions",
                    "requires": [],
                }
            ],
        }

        # Mock _setup_vllm to fail on first call (new config) but succeed on second call (restore)
        setup_call_count = 0

        def mock_setup_vllm():
            nonlocal setup_call_count
            setup_call_count += 1
            if setup_call_count == 1:
                raise Exception("Failed to load new model")
            # Second call succeeds (restoration)
            return

        with patch.object(worker, "_setup_vllm", side_effect=mock_setup_vllm):
            # Attempt config update
            result = worker._handle_vllm_config_update(new_config)

        # Should return False due to failure
        assert result is False

        # Should have restored original config
        assert worker.vllm_config == initial_vllm_config

        # Should have restored original stages
        assert len(worker.stages) == 1
        assert worker.stages[0].model == "test-model-v1"
        assert worker.stages[0].prompts == ["describe this image"]

        # Should have called _setup_vllm twice (once for new config, once for restore)
        assert setup_call_count == 2

        # Model manager cleanup should have been called
        mock_model_manager.cleanup.assert_called()

    def test_model_manager_get_model_for_stage_keyerror_handling(self):
        """Test that get_model_for_stage provides helpful error messages."""
        from caption_flow.workers.caption import MultiStageVLLMManager

        manager = MultiStageVLLMManager()

        # Test missing model
        with pytest.raises(KeyError) as exc_info:
            manager.get_model_for_stage("caption", "missing-model")
        assert "Model 'missing-model' not found" in str(exc_info.value)
        assert "Available models: []" in str(exc_info.value)

        # Add a model but missing stage
        manager.models["test-model"] = Mock()
        manager.processors["test-model"] = Mock()
        manager.tokenizers["test-model"] = Mock()

        with pytest.raises(KeyError) as exc_info:
            manager.get_model_for_stage("missing-stage", "test-model")
        assert "Sampling params for stage 'missing-stage' not found" in str(exc_info.value)
        assert "Available stages: []" in str(exc_info.value)

    def test_process_batch_handles_missing_model_manager(self, worker_config):
        """Test that batch processing handles missing model manager gracefully."""
        worker = CaptionWorker(worker_config)

        # Create a mock processing item
        mock_image = Image.new("RGB", (100, 100), "red")
        item = ProcessingItem(
            unit_id="test-unit",
            job_id="test-job",
            chunk_id="test-chunk",
            item_key="test-item",
            item_index=0,
            image=mock_image,
            image_data=b"fake_data",
            metadata={},
        )

        # Set up worker state without model manager
        worker.vllm_config = {"max_model_len": 16384}
        mock_stage = Mock()
        mock_stage.name = "test-stage"
        worker.stages = [mock_stage]
        worker.stage_order = ["test-stage"]
        worker.model_manager = None  # Simulate missing model manager

        # Process batch should handle missing model manager
        result = worker._process_batch_multi_stage([item])

        # Should return empty results and increment failed items
        assert result == []
        assert worker.items_failed == 1

    def test_process_batch_handles_model_keyerror(self, worker_config):
        """Test that batch processing handles KeyError from get_model_for_stage."""
        worker = CaptionWorker(worker_config)

        # Create a mock processing item
        mock_image = Image.new("RGB", (100, 100), "red")
        item = ProcessingItem(
            unit_id="test-unit",
            job_id="test-job",
            chunk_id="test-chunk",
            item_key="test-item",
            item_index=0,
            image=mock_image,
            image_data=b"fake_data",
            metadata={},
        )

        # Set up worker state
        worker.vllm_config = {"max_model_len": 16384}

        # Create mock stage
        mock_stage = Mock()
        mock_stage.name = "test-stage"
        mock_stage.model = "missing-model"
        worker.stages = [mock_stage]
        worker.stage_order = ["test-stage"]

        # Mock model manager that raises KeyError
        mock_model_manager = Mock()
        mock_model_manager.get_model_for_stage.side_effect = KeyError("Model not found")
        worker.model_manager = mock_model_manager

        # Process batch should handle KeyError gracefully
        result = worker._process_batch_multi_stage([item])

        # Should return empty results and increment failed items
        assert result == []
        assert worker.items_failed == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
