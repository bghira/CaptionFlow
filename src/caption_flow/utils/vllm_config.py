"""vLLM configuration management utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfigChange:
    """Represents changes between vLLM configurations."""

    requires_reload: bool = False
    model_changed: bool = False
    sampling_changed: bool = False
    prompts_changed: bool = False
    changed_fields: List[str] = field(default_factory=list)


class VLLMConfigManager:
    """Manages vLLM configuration changes and reloading."""

    # Fields that require full vLLM reload
    RELOAD_REQUIRED_FIELDS = {
        "model",
        "tensor_parallel_size",
        "max_model_len",
        "dtype",
        "gpu_memory_utilization",
        "enforce_eager",
        "limit_mm_per_prompt",
        "disable_mm_preprocessor_cache",
    }

    # Fields that can be updated without reload
    RUNTIME_UPDATEABLE_FIELDS = {
        "batch_size",
        "sampling",
        "inference_prompts",
    }

    def __init__(self):
        self.current_config: Optional[Dict[str, Any]] = None
        self.current_sampling_params = None

    def analyze_config_change(
        self, old_config: Optional[Dict[str, Any]], new_config: Dict[str, Any]
    ) -> VLLMConfigChange:
        """Analyze differences between configs to determine required actions."""
        change = VLLMConfigChange()

        if not old_config:
            # First time setup
            change.requires_reload = True
            change.model_changed = True
            logger.info("Initial vLLM configuration - full load required")
            return change

        # Check each field for changes
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                change.changed_fields.append(key)

                if key in self.RELOAD_REQUIRED_FIELDS:
                    change.requires_reload = True
                    if key == "model":
                        change.model_changed = True
                        logger.info(f"Model changed from {old_value} to {new_value}")
                elif key == "sampling":
                    change.sampling_changed = True
                elif key == "inference_prompts":
                    change.prompts_changed = True

        if change.changed_fields:
            logger.info(f"vLLM config changes detected: {change.changed_fields}")
            if change.requires_reload:
                logger.info("Changes require vLLM reload")
            else:
                logger.info("Changes can be applied without reload")
        else:
            logger.debug("No vLLM config changes detected")

        return change

    def create_sampling_params(self, vllm_config: Dict[str, Any]):
        """Create SamplingParams from config."""
        from vllm import SamplingParams

        sampling_config = vllm_config.get("sampling", {})

        params = SamplingParams(
            temperature=sampling_config.get("temperature", 0.7),
            top_p=sampling_config.get("top_p", 0.95),
            max_tokens=sampling_config.get("max_tokens", 256),
            stop=sampling_config.get("stop", ["<|end|>", "<|endoftext|>", "<|im_end|>"]),
            repetition_penalty=sampling_config.get("repetition_penalty", 1.05),
            skip_special_tokens=sampling_config.get("skip_special_tokens", True),
        )

        self.current_sampling_params = params
        return params

    def should_reload_vllm(
        self, old_config: Optional[Dict[str, Any]], new_config: Dict[str, Any]
    ) -> bool:
        """Quick check if vLLM needs to be reloaded."""
        change = self.analyze_config_change(old_config, new_config)
        return change.requires_reload

    def get_vllm_init_params(self, vllm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vLLM initialization parameters from config."""
        return {
            "model": vllm_config["model"],
            "trust_remote_code": True,
            "tensor_parallel_size": vllm_config.get("tensor_parallel_size", 1),
            "max_model_len": vllm_config.get("max_model_len", 16384),
            "enforce_eager": vllm_config.get("enforce_eager", True),
            "gpu_memory_utilization": vllm_config.get("gpu_memory_utilization", 0.92),
            "dtype": vllm_config.get("dtype", "float16"),
            "limit_mm_per_prompt": vllm_config.get("limit_mm_per_prompt", {"image": 1}),
            "disable_mm_preprocessor_cache": vllm_config.get("disable_mm_preprocessor_cache", True),
        }

    def requires_tokenizer_reload(
        self, old_config: Optional[Dict[str, Any]], new_config: Dict[str, Any]
    ) -> bool:
        """Check if tokenizer/processor need to be reloaded."""
        if not old_config:
            return True

        # Tokenizer/processor depend on the model
        return old_config.get("model") != new_config.get("model")

    def update_runtime_config(
        self, vllm_instance, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> Tuple[bool, Optional[Any]]:
        """Update vLLM configuration at runtime without reload.

        Returns
        -------
            Tuple of (success, new_sampling_params)

        """
        change = self.analyze_config_change(old_config, new_config)

        if change.requires_reload:
            logger.warning("Config changes require reload, cannot update at runtime")
            return False, None

        # Update sampling params if changed
        new_sampling_params = None
        if change.sampling_changed:
            new_sampling_params = self.create_sampling_params(new_config)
            logger.info("Updated sampling parameters")

        # Note: batch_size and prompts are handled by the worker directly
        # as they don't affect the vLLM instance itself

        return True, new_sampling_params
