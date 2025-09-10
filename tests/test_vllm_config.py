"""Comprehensive tests for vLLM configuration management."""

from unittest.mock import Mock, patch

import pytest
from caption_flow.utils.vllm_config import VLLMConfigChange, VLLMConfigManager


class TestVLLMConfigChange:
    """Test VLLMConfigChange dataclass."""

    def test_default_values(self):
        """Test default values of VLLMConfigChange."""
        change = VLLMConfigChange()

        assert change.requires_reload is False
        assert change.model_changed is False
        assert change.sampling_changed is False
        assert change.prompts_changed is False
        assert change.changed_fields == []

    def test_custom_values(self):
        """Test VLLMConfigChange with custom values."""
        change = VLLMConfigChange(
            requires_reload=True,
            model_changed=True,
            sampling_changed=True,
            changed_fields=["model", "sampling"],
        )

        assert change.requires_reload is True
        assert change.model_changed is True
        assert change.sampling_changed is True
        assert change.prompts_changed is False
        assert change.changed_fields == ["model", "sampling"]


class TestVLLMConfigManagerInit:
    """Test VLLMConfigManager initialization."""

    def test_init(self):
        """Test VLLMConfigManager initialization."""
        manager = VLLMConfigManager()

        assert manager.current_config is None
        assert manager.current_sampling_params is None

    def test_class_constants(self):
        """Test class constants are defined correctly."""
        assert hasattr(VLLMConfigManager, "RELOAD_REQUIRED_FIELDS")
        assert hasattr(VLLMConfigManager, "RUNTIME_UPDATEABLE_FIELDS")

        # Check some expected fields
        assert "model" in VLLMConfigManager.RELOAD_REQUIRED_FIELDS
        assert "tensor_parallel_size" in VLLMConfigManager.RELOAD_REQUIRED_FIELDS
        assert "batch_size" in VLLMConfigManager.RUNTIME_UPDATEABLE_FIELDS
        assert "sampling" in VLLMConfigManager.RUNTIME_UPDATEABLE_FIELDS


class TestVLLMConfigManagerAnalyzeChange:
    """Test VLLMConfigManager config change analysis."""

    def test_analyze_config_change_first_time(self):
        """Test analyzing config change for first time setup."""
        manager = VLLMConfigManager()
        new_config = {"model": "test-model", "temperature": 0.8}

        change = manager.analyze_config_change(None, new_config)

        assert change.requires_reload is True
        assert change.model_changed is True
        assert change.sampling_changed is False
        assert change.prompts_changed is False

    def test_analyze_config_change_no_changes(self):
        """Test analyzing config when nothing changed."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "temperature": 0.8}
        new_config = {"model": "test-model", "temperature": 0.8}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is False
        assert change.model_changed is False
        assert change.sampling_changed is False
        assert change.prompts_changed is False
        assert change.changed_fields == []

    def test_analyze_config_change_model_changed(self):
        """Test analyzing config when model changed."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model", "temperature": 0.8}
        new_config = {"model": "new-model", "temperature": 0.8}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is True
        assert change.model_changed is True
        assert "model" in change.changed_fields

    def test_analyze_config_change_sampling_changed(self):
        """Test analyzing config when sampling parameters changed."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "sampling": {"temperature": 0.7}}
        new_config = {"model": "test-model", "sampling": {"temperature": 0.9}}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is False
        assert change.model_changed is False
        assert change.sampling_changed is True
        assert "sampling" in change.changed_fields

    def test_analyze_config_change_prompts_changed(self):
        """Test analyzing config when inference prompts changed."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "inference_prompts": ["old prompt"]}
        new_config = {"model": "test-model", "inference_prompts": ["new prompt"]}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is False
        assert change.model_changed is False
        assert change.sampling_changed is False
        assert change.prompts_changed is True
        assert "inference_prompts" in change.changed_fields

    def test_analyze_config_change_reload_required_field(self):
        """Test analyzing config when a reload-required field changed."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "tensor_parallel_size": 1}
        new_config = {"model": "test-model", "tensor_parallel_size": 2}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is True
        assert change.model_changed is False
        assert "tensor_parallel_size" in change.changed_fields

    def test_analyze_config_change_multiple_changes(self):
        """Test analyzing config with multiple changes."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model", "sampling": {"temperature": 0.7}, "batch_size": 10}
        new_config = {"model": "new-model", "sampling": {"temperature": 0.9}, "batch_size": 20}

        change = manager.analyze_config_change(old_config, new_config)

        assert change.requires_reload is True  # Due to model change
        assert change.model_changed is True
        assert change.sampling_changed is True
        assert set(change.changed_fields) == {"model", "sampling", "batch_size"}

    def test_analyze_config_change_new_field_added(self):
        """Test analyzing config when new field is added."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model"}
        new_config = {"model": "test-model", "new_field": "value"}

        change = manager.analyze_config_change(old_config, new_config)

        assert "new_field" in change.changed_fields

    def test_analyze_config_change_field_removed(self):
        """Test analyzing config when field is removed."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "old_field": "value"}
        new_config = {"model": "test-model"}

        change = manager.analyze_config_change(old_config, new_config)

        assert "old_field" in change.changed_fields


class TestVLLMConfigManagerReloadCheck:
    """Test VLLMConfigManager reload checking."""

    def test_should_reload_vllm_first_time(self):
        """Test should_reload_vllm for first time."""
        manager = VLLMConfigManager()
        new_config = {"model": "test-model"}

        result = manager.should_reload_vllm(None, new_config)

        assert result is True

    def test_should_reload_vllm_no_changes(self):
        """Test should_reload_vllm with no changes."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "batch_size": 10}
        new_config = {"model": "test-model", "batch_size": 10}

        result = manager.should_reload_vllm(old_config, new_config)

        assert result is False

    def test_should_reload_vllm_runtime_change_only(self):
        """Test should_reload_vllm with only runtime-updateable changes."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "batch_size": 10}
        new_config = {"model": "test-model", "batch_size": 20}

        result = manager.should_reload_vllm(old_config, new_config)

        assert result is False

    def test_should_reload_vllm_reload_required(self):
        """Test should_reload_vllm with reload-required changes."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model", "batch_size": 10}
        new_config = {"model": "new-model", "batch_size": 20}

        result = manager.should_reload_vllm(old_config, new_config)

        assert result is True


class TestVLLMConfigManagerInitParams:
    """Test VLLMConfigManager initialization parameters."""

    def test_get_vllm_init_params_minimal(self):
        """Test getting vLLM init params with minimal config."""
        manager = VLLMConfigManager()
        vllm_config = {"model": "test-model"}

        params = manager.get_vllm_init_params(vllm_config)

        expected = {
            "model": "test-model",
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "max_model_len": 16384,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.92,
            "dtype": "float16",
            "limit_mm_per_prompt": {"image": 1},
            "disable_mm_preprocessor_cache": True,
        }

        assert params == expected

    def test_get_vllm_init_params_custom(self):
        """Test getting vLLM init params with custom config."""
        manager = VLLMConfigManager()
        vllm_config = {
            "model": "custom-model",
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "enforce_eager": False,
            "gpu_memory_utilization": 0.8,
            "dtype": "float32",
            "limit_mm_per_prompt": {"image": 2},
            "disable_mm_preprocessor_cache": False,
        }

        params = manager.get_vllm_init_params(vllm_config)

        expected = {
            "model": "custom-model",
            "trust_remote_code": True,
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "enforce_eager": False,
            "gpu_memory_utilization": 0.8,
            "dtype": "float32",
            "limit_mm_per_prompt": {"image": 2},
            "disable_mm_preprocessor_cache": False,
        }

        assert params == expected


class TestVLLMConfigManagerTokenizerReload:
    """Test VLLMConfigManager tokenizer reload checking."""

    def test_requires_tokenizer_reload_first_time(self):
        """Test tokenizer reload for first time."""
        manager = VLLMConfigManager()
        new_config = {"model": "test-model"}

        result = manager.requires_tokenizer_reload(None, new_config)

        assert result is True

    def test_requires_tokenizer_reload_same_model(self):
        """Test tokenizer reload with same model."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "other": "param"}
        new_config = {"model": "test-model", "other": "different"}

        result = manager.requires_tokenizer_reload(old_config, new_config)

        assert result is False

    def test_requires_tokenizer_reload_different_model(self):
        """Test tokenizer reload with different model."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model"}
        new_config = {"model": "new-model"}

        result = manager.requires_tokenizer_reload(old_config, new_config)

        assert result is True

    def test_requires_tokenizer_reload_model_added(self):
        """Test tokenizer reload when model is added."""
        manager = VLLMConfigManager()
        old_config = {"other": "param"}
        new_config = {"model": "new-model", "other": "param"}

        result = manager.requires_tokenizer_reload(old_config, new_config)

        assert result is True

    def test_requires_tokenizer_reload_model_removed(self):
        """Test tokenizer reload when model is removed."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model"}
        new_config = {"other": "param"}

        result = manager.requires_tokenizer_reload(old_config, new_config)

        assert result is True


class TestVLLMConfigManagerRuntimeUpdate:
    """Test VLLMConfigManager runtime configuration updates."""

    def test_update_runtime_config_reload_required(self):
        """Test runtime update when reload is required."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {"model": "old-model"}
        new_config = {"model": "new-model"}

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is False
        assert params is None

    def test_update_runtime_config_no_changes(self):
        """Test runtime update with no changes."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {"model": "test-model", "batch_size": 10}
        new_config = {"model": "test-model", "batch_size": 10}

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is True
        assert params is None

    @patch.object(VLLMConfigManager, "create_sampling_params")
    def test_update_runtime_config_sampling_changed(self, mock_create_sampling):
        """Test runtime update with sampling changes."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {"model": "test-model", "sampling": {"temperature": 0.7}}
        new_config = {"model": "test-model", "sampling": {"temperature": 0.9}}

        mock_params = Mock()
        mock_create_sampling.return_value = mock_params

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is True
        assert params == mock_params
        mock_create_sampling.assert_called_once_with(new_config)

    def test_update_runtime_config_batch_size_changed(self):
        """Test runtime update with batch size changes."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {"model": "test-model", "batch_size": 10}
        new_config = {"model": "test-model", "batch_size": 20}

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is True
        assert params is None  # batch_size doesn't create new sampling params

    def test_update_runtime_config_prompts_changed(self):
        """Test runtime update with prompts changes."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {"model": "test-model", "inference_prompts": ["old"]}
        new_config = {"model": "test-model", "inference_prompts": ["new"]}

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is True
        assert params is None  # prompts don't create new sampling params

    @patch.object(VLLMConfigManager, "create_sampling_params")
    def test_update_runtime_config_mixed_changes(self, mock_create_sampling):
        """Test runtime update with mixed changes."""
        manager = VLLMConfigManager()
        vllm_instance = Mock()
        old_config = {
            "model": "test-model",
            "sampling": {"temperature": 0.7},
            "batch_size": 10,
            "inference_prompts": ["old"],
        }
        new_config = {
            "model": "test-model",
            "sampling": {"temperature": 0.9},
            "batch_size": 20,
            "inference_prompts": ["new"],
        }

        mock_params = Mock()
        mock_create_sampling.return_value = mock_params

        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

        assert success is True
        assert params == mock_params
        mock_create_sampling.assert_called_once_with(new_config)


class TestVLLMConfigManagerIntegration:
    """Integration tests for VLLMConfigManager."""

    def test_full_workflow_first_time(self):
        """Test full workflow for first time setup."""
        manager = VLLMConfigManager()
        config = {"model": "test-model", "sampling": {"temperature": 0.8}}

        # Check if reload required
        assert manager.should_reload_vllm(None, config) is True

        # Get init params
        init_params = manager.get_vllm_init_params(config)
        assert init_params["model"] == "test-model"

        # Check tokenizer reload
        assert manager.requires_tokenizer_reload(None, config) is True

    def test_full_workflow_runtime_update(self):
        """Test full workflow for runtime update."""
        manager = VLLMConfigManager()
        old_config = {"model": "test-model", "sampling": {"temperature": 0.7}, "batch_size": 10}
        new_config = {"model": "test-model", "sampling": {"temperature": 0.9}, "batch_size": 20}

        # Check if reload required
        assert manager.should_reload_vllm(old_config, new_config) is False

        # Check tokenizer reload
        assert manager.requires_tokenizer_reload(old_config, new_config) is False

        # Apply runtime update
        vllm_instance = Mock()
        with patch.object(manager, "create_sampling_params") as mock_create:
            mock_create.return_value = Mock()
            success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)

            assert success is True
            assert params is not None

    def test_full_workflow_reload_required(self):
        """Test full workflow when reload is required."""
        manager = VLLMConfigManager()
        old_config = {"model": "old-model", "sampling": {"temperature": 0.7}}
        new_config = {"model": "new-model", "sampling": {"temperature": 0.9}}

        # Check if reload required
        assert manager.should_reload_vllm(old_config, new_config) is True

        # Check tokenizer reload
        assert manager.requires_tokenizer_reload(old_config, new_config) is True

        # Runtime update should fail
        vllm_instance = Mock()
        success, params = manager.update_runtime_config(vllm_instance, old_config, new_config)
        assert success is False
        assert params is None

        # Should get new init params
        init_params = manager.get_vllm_init_params(new_config)
        assert init_params["model"] == "new-model"


if __name__ == "__main__":
    pytest.main([__file__])
