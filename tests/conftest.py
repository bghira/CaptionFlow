"""Shared test setup for optional runtime dependencies."""

import sys
from types import ModuleType

try:
    import vllm  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # CaptionWorker imports vLLM lazily. Most worker tests patch these two
    # symbols and should not require the large platform-specific runtime.
    vllm_stub = ModuleType("vllm")

    class LLM:
        """Minimal fallback used as a patch target in unit tests."""

    class SamplingParams:
        """Minimal fallback used as a patch target in unit tests."""

    vllm_stub.LLM = LLM
    vllm_stub.SamplingParams = SamplingParams
    sys.modules["vllm"] = vllm_stub
