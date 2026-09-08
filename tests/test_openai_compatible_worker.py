"""Tests for the BYOK OpenAI-compatible caption worker."""

import asyncio
import base64
import io
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from caption_flow.utils.image_processor import ImageProcessor
from caption_flow.workers.caption import ProcessingItem
from caption_flow.workers.openai_compatible import (
    AdaptiveEndpointPool,
    ChatRequest,
    EndpointConfig,
    EndpointRequestError,
    OpenAICompatibleWorker,
    _extract_message_text,
    _parse_duration,
    _retry_delay,
)


def endpoint_config(**overrides):
    values = {
        "name": "primary",
        "base_url": "https://provider.example/v1",
        "api_key": "secret",
        "initial_concurrency": 1,
        "max_concurrency": 4,
        "probe_after_successes": 1,
    }
    values.update(overrides)
    return EndpointConfig(**values)


def test_endpoint_config_reads_secret_from_environment_and_maps_model():
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        config = EndpointConfig.from_dict(
            {
                "name": "zai",
                "base_url": "https://api.example/v4",
                "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                "model_map": {"shared-caption-model": "provider-model"},
            },
            0,
        )

    assert config.api_key == "secret"
    assert config.url == "https://api.example/v4/chat/completions"
    assert config.resolve_model("shared-caption-model") == "provider-model"
    assert "secret" not in repr(config)


def test_endpoint_config_requires_key_environment_variable():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="MISSING_TEST_KEY"):
            EndpointConfig.from_dict(
                {"base_url": "https://provider.example/v1", "api_key_env": "MISSING_TEST_KEY"},
                0,
            )


def test_endpoint_config_validates_limits_and_fallbacks():
    with patch.dict(
        os.environ,
        {
            "CAPTIONFLOW_TEST_API_KEY": "secret",
            "CAPTIONFLOW_TEST_BASE_URL": "https://gateway.example/v2/",
        },
        clear=True,
    ):
        config = EndpointConfig.from_dict(
            {
                "base_url_env": "CAPTIONFLOW_TEST_BASE_URL",
                "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                "requests_per_minute": 0,
                "model_map": {"*": "fallback-model"},
            },
            1,
        )
        assert config.name == "endpoint-2"
        assert config.base_url == "https://gateway.example/v2"
        assert config.requests_per_minute is None
        assert config.resolve_model("unmapped-model") == "fallback-model"

        with pytest.raises(ValueError, match="initial_concurrency"):
            EndpointConfig.from_dict(
                {
                    "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                    "initial_concurrency": 2,
                    "max_concurrency": 1,
                },
                0,
            )
        with pytest.raises(ValueError, match="cannot override"):
            EndpointConfig.from_dict(
                {
                    "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                    "extra_body": {"messages": []},
                },
                0,
            )

    with pytest.raises(ValueError, match="no model configured"):
        endpoint_config().resolve_model(None)


def test_endpoint_errors_classify_retryable_and_throttled_responses():
    assert EndpointRequestError("api", 503, "overloaded", {}).retryable
    assert EndpointRequestError("api", 400, "concurrency exceeded", {}).throttled
    assert not EndpointRequestError("api", 400, "invalid request", {}).retryable


def test_rate_limit_header_parsing():
    assert _parse_duration("250ms") == pytest.approx(0.25)
    assert _parse_duration("2s") == pytest.approx(2.0)
    assert _retry_delay({"retry-after": "3"}, default=1) == pytest.approx(3.0)
    assert _retry_delay({"x-ratelimit-reset-requests": "500ms"}, default=1) == pytest.approx(0.5)


def test_rate_limit_header_parsing_handles_dates_invalid_values_and_defaults():
    with patch("caption_flow.workers.openai_compatible.time.time", return_value=2_000_000_000):
        assert _parse_duration("2000000005") == pytest.approx(5)
    assert _parse_duration("invalid") is None
    assert _parse_duration("invalid-ms") is None
    assert _parse_duration("-2") == 0
    assert _retry_delay({"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"}, 1) == 0
    assert _retry_delay({"retry-after": "not-a-date"}, 2) == 2
    assert _retry_delay({"ratelimit-reset": "4s"}, 1) == 4
    assert _retry_delay({}, 0) == pytest.approx(0.05)


def test_extracts_string_and_structured_message_content():
    assert (
        _extract_message_text({"choices": [{"message": {"content": "caption"}}]}, "test")
        == "caption"
    )
    assert (
        _extract_message_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "first "},
                                {"type": "output_text", "text": "second"},
                            ]
                        }
                    }
                ]
            },
            "test",
        )
        == "first second"
    )
    assert (
        _extract_message_text(
            {"choices": [{"message": {"content": ["plain ", {"type": "ignored"}]}}]},
            "test",
        )
        == "plain"
    )
    assert _extract_message_text({"choices": [{"message": {"content": 42}}]}, "test") == "42"
    with pytest.raises(ValueError, match="no message content"):
        _extract_message_text({}, "test")
    with pytest.raises(ValueError, match="empty message content"):
        _extract_message_text({"choices": [{"message": {"content": []}}]}, "test")


@pytest.mark.asyncio
async def test_pool_lifecycle_guards_and_disabled_endpoint_result():
    with pytest.raises(ValueError, match="At least one"):
        AdaptiveEndpointPool([])

    pool = AdaptiveEndpointPool([endpoint_config(requests_per_minute=60)])
    with pytest.raises(RuntimeError, match="not open"):
        await pool.run_many([ChatRequest("caption", None, "model")])

    await pool.open()
    assert await pool.run_many([]) == []
    await pool.close()
    assert pool.session is None

    pool.session = object()
    state = pool.states[0]
    pool._reserve(state, 10)
    assert state.next_request_at == pytest.approx(11)
    assert pool.snapshot()[0]["inflight"] == 1
    pool._release(state)
    state.disabled_reason = "bad credentials"
    results = await pool.run_many([ChatRequest("caption", None, "model")])
    assert isinstance(results[0], RuntimeError)
    assert "disabled" in str(results[0])


@pytest.mark.asyncio
async def test_pool_disables_endpoint_after_authentication_failure():
    pool = AdaptiveEndpointPool([endpoint_config(max_retries=0)])
    pool.session = object()
    error = EndpointRequestError("primary", 401, "unauthorized", {})
    pool._request_once = AsyncMock(side_effect=error)

    results = await pool.run_many([ChatRequest("caption", None, "model")])

    assert results == [error]
    assert pool.states[0].disabled_reason == "HTTP 401 authentication/authorization failure"


@pytest.mark.asyncio
async def test_pool_probes_then_backs_off_at_provider_concurrency_limit():
    class LimitedPool(AdaptiveEndpointPool):
        def __init__(self):
            super().__init__([endpoint_config()])
            self.active = 0
            self.max_seen = 0
            self.session = object()

        async def _request_once(self, state, request):
            self.active += 1
            self.max_seen = max(self.max_seen, self.active)
            try:
                await asyncio.sleep(0.005)
                if self.active > 2:
                    raise EndpointRequestError(
                        state.config.name,
                        429,
                        "concurrency limit reached",
                        {"retry-after": "0"},
                    )
                return request.prompt
            finally:
                self.active -= 1

    pool = LimitedPool()
    requests = [ChatRequest(str(index), None, "model") for index in range(10)]
    results = await pool.run_many(requests)

    assert results == [str(index) for index in range(10)]
    assert pool.max_seen >= 3
    assert pool.states[0].rate_limited_requests >= 1
    assert pool.states[0].successful_requests == 10


@pytest.mark.asyncio
async def test_rate_limited_request_spills_to_another_endpoint_without_waiting():
    class SpilloverPool(AdaptiveEndpointPool):
        def __init__(self):
            super().__init__(
                [
                    endpoint_config(name="limited"),
                    endpoint_config(name="available"),
                ]
            )
            self.session = object()

        async def _request_once(self, state, request):
            if state.config.name == "limited":
                raise EndpointRequestError(
                    state.config.name,
                    429,
                    "plan request limit reached",
                    {"retry-after": "30"},
                )
            return f"{state.config.name}:{request.prompt}"

    pool = SpilloverPool()
    started = asyncio.get_running_loop().time()
    results = await pool.run_many([ChatRequest("caption", None, "model")])
    elapsed = asyncio.get_running_loop().time() - started

    assert results == ["available:caption"]
    assert elapsed < 1
    assert pool.states[0].rate_limited_requests == 1


@pytest.mark.asyncio
async def test_request_uses_openai_chat_completions_shape():
    class FakeResponse:
        status = 200
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            return {"choices": [{"message": {"content": "a caption"}}]}

    class FakeSession:
        closed = False

        def __init__(self):
            self.call = None

        def post(self, url, **kwargs):
            self.call = (url, kwargs)
            return FakeResponse()

    config = endpoint_config(
        model_map={"shared-model": "provider-model"},
        extra_body={"reasoning_effort": "low"},
    )
    pool = AdaptiveEndpointPool([config])
    session = FakeSession()
    pool.session = session

    text = await pool._request_once(
        pool.states[0],
        ChatRequest(
            prompt="describe",
            image_data_url="data:image/png;base64,AAAA",
            requested_model="shared-model",
            parameters={"max_tokens": 100},
            extra_body={
                "reasoning_effort": "high",
                "response_format": {"type": "json_object"},
            },
            system_prompt="Be precise",
            image_detail="low",
        ),
    )

    assert text == "a caption"
    url, kwargs = session.call
    assert url == "https://provider.example/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["headers"]["User-Agent"].startswith("CaptionFlow/")
    assert kwargs["json"]["model"] == "provider-model"
    assert kwargs["json"]["max_tokens"] == 100
    assert kwargs["json"]["reasoning_effort"] == "high"
    assert kwargs["json"]["response_format"] == {"type": "json_object"}
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "Be precise"}
    assert kwargs["json"]["messages"][1]["content"][0]["image_url"]["detail"] == "low"


@pytest.mark.asyncio
async def test_request_surfaces_provider_error_response():
    class ErrorResponse:
        status = 429
        headers = {"Retry-After": "2"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def text(self):
            return "rate limit reached"

    class ErrorSession:
        closed = False

        def post(self, *args, **kwargs):
            return ErrorResponse()

    pool = AdaptiveEndpointPool([endpoint_config()])
    pool.session = ErrorSession()

    with pytest.raises(EndpointRequestError) as exc_info:
        await pool._request_once(pool.states[0], ChatRequest("describe", None, "model"))

    assert exc_info.value.status == 429
    assert exc_info.value.headers == {"retry-after": "2"}


def test_worker_encodes_image_and_keeps_api_key_out_of_auth_payload():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "name": "community-worker",
        "openai_compatible": {
            "endpoints": [
                {
                    "base_url": "https://provider.example/v1",
                    "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                    "model": "vision-model",
                }
            ]
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    item = ProcessingItem(
        unit_id="unit",
        job_id="shard:chunk:0:idx:0",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=Image.new("RGB", (2, 2), color="red"),
        image_data=b"",
        metadata={},
    )

    assert worker._image_data_url(item).startswith("data:image/jpeg;base64,")
    assert worker._get_auth_data() == {
        "token": "orchestrator-token",
        "name": "community-worker",
    }
    assert "provider-secret" not in str(worker._get_heartbeat_data())


def test_worker_image_encoding_handles_passthrough_conversion_and_missing_data():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {"api_key_env": "CAPTIONFLOW_TEST_API_KEY", "model": "vision"},
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    png_buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(png_buffer, format="PNG")
    passthrough = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=None,
        image_data=png_buffer.getvalue(),
        metadata={},
    )
    assert worker._image_data_url(passthrough).startswith("data:image/jpeg;base64,")

    worker.image_format = "png"
    assert worker._image_data_url(passthrough).startswith("data:image/png;base64,")
    worker.image_format = "jpeg"

    converted = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="image.webp",
        item_index=0,
        image=Image.new("RGBA", (2, 2), color="red"),
        image_data=b"",
        metadata={"image_format": "unsupported"},
    )
    assert worker._image_data_url(converted).startswith("data:image/jpeg;base64,")

    converted.image = None
    with pytest.raises(ValueError, match="has no image data"):
        worker._image_data_url(converted)


def test_worker_resizes_large_images_before_encoding():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "max_image_dimension": 1024,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    jpeg_buffer = io.BytesIO()
    Image.new("RGB", (2400, 1600), color="green").save(jpeg_buffer, format="JPEG")
    item = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="large.jpg",
        item_index=0,
        image=None,
        image_data=jpeg_buffer.getvalue(),
        metadata={},
    )

    encoded = worker._image_data_url(item)
    encoded_bytes = base64.b64decode(encoded.split(",", 1)[1])
    with Image.open(io.BytesIO(encoded_bytes)) as resized:
        assert resized.size == (1024, 683)


def test_worker_disables_image_resize_for_non_positive_dimension():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "max_image_dimension": -1,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    assert worker.max_image_dimension is None


def test_worker_fuses_encoded_image_batch_preprocessing():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "max_image_dimension": 100,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    items = []
    for index, size in enumerate(((200, 100), (100, 200))):
        buffer = io.BytesIO()
        Image.new("RGB", size, color="blue").save(buffer, format="PNG")
        items.append(
            ProcessingItem(
                unit_id="unit",
                job_id=f"job-{index}",
                chunk_id="chunk",
                item_key=f"image-{index}.png",
                item_index=index,
                image=None,
                image_data=buffer.getvalue(),
                metadata={},
            )
        )

    with patch.object(
        ImageProcessor,
        "preprocess_encoded_batch",
        wraps=ImageProcessor.preprocess_encoded_batch,
    ) as preprocess:
        urls = worker._image_data_urls(items)

    preprocess.assert_called_once()
    assert preprocess.call_args.args[1] == [(100, 50), (50, 100)]
    for item, expected_size in zip(items, ((100, 50), (50, 100)), strict=True):
        encoded = base64.b64decode(urls[id(item)].split(",", 1)[1])
        with Image.open(io.BytesIO(encoded)) as image:
            assert image.size == expected_size


def test_worker_batch_preprocessing_falls_back_per_item():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(buffer, format="PNG")
    item = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=None,
        image_data=buffer.getvalue(),
        metadata={},
    )
    with (
        patch.object(ImageProcessor, "preprocess_encoded_batch", side_effect=RuntimeError),
        patch.object(worker, "_image_data_url", return_value="fallback") as fallback,
    ):
        assert worker._image_data_urls([item]) == {id(item): "fallback"}
    fallback.assert_called_once_with(item)


def test_worker_validates_config_and_applies_shared_updates():
    base = {"server": "ws://localhost:8765", "token": "orchestrator-token"}
    with pytest.raises(ValueError, match="must be a mapping"):
        OpenAICompatibleWorker({**base, "openai_compatible": []})

    with pytest.raises(ValueError, match="list of mappings"):
        OpenAICompatibleWorker({**base, "openai_compatible": {"endpoints": ["invalid"]}})

    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        with pytest.raises(ValueError, match="list of non-empty strings"):
            OpenAICompatibleWorker(
                {
                    **base,
                    "openai_compatible": {
                        "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                        "refusal_markers": "i cannot describe",
                    },
                }
            )

        with pytest.raises(ValueError, match="retry_extra_body must be a mapping"):
            OpenAICompatibleWorker(
                {
                    **base,
                    "openai_compatible": {
                        "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                        "retry_extra_body": [],
                    },
                }
            )

        with pytest.raises(ValueError, match="retry_extra_body cannot override: model"):
            OpenAICompatibleWorker(
                {
                    **base,
                    "openai_compatible": {
                        "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                        "retry_extra_body": {"model": "other"},
                    },
                }
            )

    config = {
        **base,
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "batch_size": 3,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(config)

    worker.vllm_config = {}
    worker._apply_local_overrides()
    assert worker._handle_vllm_config_update({})
    assert worker._handle_vllm_config_update(
        {"model": "vision", "inference_prompts": ["Describe"], "mock_results": True}
    )
    assert worker.vllm_config["batch_size"] == 3
    assert worker.mock_mode is True
    assert worker.stage_order == ["default"]

    with pytest.raises(RuntimeError, match="loop is not ready"):
        worker._process_batch_multi_stage([])


def test_worker_configures_refusal_markers_from_shared_or_local_config():
    base = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        worker = OpenAICompatibleWorker(base)

    assert worker._is_refusal_text("I'm sorry, I can't describe this image.")
    assert worker._is_refusal_text("A shirt reading “I can't stay at home.”")

    assert worker._handle_vllm_config_update(
        {
            "model": "vision",
            "inference_prompts": ["Describe"],
            "refusal_markers": [
                " I cannot describe ",
                "i cannot describe",
                "unable to provide a caption",
            ],
        }
    )
    assert worker.refusal_markers == (
        "i cannot describe",
        "unable to provide a caption",
    )
    assert worker._is_refusal_text("I cannot describe this image.")
    assert not worker._is_refusal_text("A shirt reading “I can't stay at home.”")

    local_config = {
        **base,
        "openai_compatible": {
            **base["openai_compatible"],
            "refusal_markers": [],
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        local_worker = OpenAICompatibleWorker(local_config)
    assert local_worker._handle_vllm_config_update(
        {
            "model": "vision",
            "inference_prompts": ["Describe"],
            "refusal_markers": ["i cannot describe"],
        }
    )
    assert local_worker.refusal_markers == ()
    assert not local_worker._is_refusal_text("I cannot describe this image.")

    with pytest.raises(ValueError, match="list of non-empty strings"):
        worker._handle_vllm_config_update(
            {
                "model": "vision",
                "inference_prompts": ["Describe"],
                "refusal_markers": [""],
            }
        )


def test_worker_can_require_valid_json_output():
    config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "validate_json_output": True,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        worker = OpenAICompatibleWorker(config)

    assert worker._validated_output('{"caption": "A red square."}') is not None
    assert worker._validated_output('{"caption": "truncated"') is None
    assert worker._validated_output('{"confidence": NaN}') is None
    assert worker._validated_output("I'm sorry, but I cannot describe it.") is None

    config["openai_compatible"]["validate_json_output"] = "true"
    with (
        patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}),
        pytest.raises(ValueError, match="validate_json_output must be a boolean"),
    ):
        OpenAICompatibleWorker(config)


def test_worker_can_repair_only_invalid_json_escapes():
    config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "validate_json_output": True,
            "repair_invalid_json_escapes": True,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        worker = OpenAICompatibleWorker(config)

    malformed = r'{"text": "C:\users\path and \u12Z4", "quote": "ok\nline"}'
    repaired = worker._validated_output(malformed)
    assert repaired is not None
    assert json.loads(repaired) == {
        "text": r"C:\users\path and \u12Z4",
        "quote": "ok\nline",
    }
    assert worker._validated_output('{"text": "still truncated"') is None
    assert worker._validated_output(r'{"text": "bad\q"') is None

    config["openai_compatible"]["repair_invalid_json_escapes"] = 1
    with (
        patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}),
        pytest.raises(ValueError, match="repair_invalid_json_escapes must be a boolean"),
    ):
        OpenAICompatibleWorker(config)


def test_worker_can_canonicalize_json_and_normalize_yxyx_boxes():
    config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "validate_json_output": True,
            "canonicalize_json_output": True,
            "normalize_yxyx_bboxes": True,
            "deduplicate_json_elements": True,
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}):
        worker = OpenAICompatibleWorker(config)

    output = worker._validated_output(
        '{"description":"caf\\u00e9","elements":['
        '{"bbox":[900,800,100,200]},{"bbox":[900,800,100,200]}]}'
    )
    assert output == '{"description":"café","elements":[{"bbox":[100,200,900,800]}]}'

    for option in (
        "repair_invalid_json_escapes",
        "canonicalize_json_output",
        "normalize_yxyx_bboxes",
        "deduplicate_json_elements",
    ):
        invalid = {
            **config,
            "openai_compatible": {**config["openai_compatible"], option: "true"},
        }
        with (
            patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}),
            pytest.raises(ValueError, match=f"{option} must be a boolean"),
        ):
            OpenAICompatibleWorker(invalid)

        without_validation = {
            **config,
            "openai_compatible": {
                **config["openai_compatible"],
                "validate_json_output": False,
                option: True,
            },
        }
        with (
            patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "secret"}),
            pytest.raises(ValueError, match="validate_json_output must be enabled"),
        ):
            OpenAICompatibleWorker(without_validation)


def test_worker_runs_shared_caption_stage_through_endpoint_pool():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "batch_image_processing": False,
        "openai_compatible": {
            "endpoints": [
                {
                    "base_url": "https://provider.example/v1",
                    "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                    "model": "provider-vision-model",
                }
            ]
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    worker.vllm_config = {
        "model": "shared-model",
        "inference_prompts": ["Describe this image"],
        "sampling": {"max_tokens": 123, "repetition_penalty": 1.05},
    }
    worker.stages = worker._parse_stages_config(worker.vllm_config)
    worker.stage_order = worker._topological_sort_stages(worker.stages)
    worker.endpoint_pool.run_many = AsyncMock(return_value=["A red square."])
    worker.api_loop = asyncio.new_event_loop()
    item = ProcessingItem(
        unit_id="unit",
        job_id="shard:chunk:0:idx:0",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=Image.new("RGB", (2, 2), color="red"),
        image_data=b"",
        metadata={},
    )

    try:
        results = worker._process_batch_multi_stage([item])
    finally:
        worker.api_loop.close()

    assert results == [(item, {"captions": ["A red square."]})]
    request = worker.endpoint_pool.run_many.call_args.args[0][0]
    assert request.requested_model == "shared-model"
    assert request.parameters == {"max_tokens": 123}
    assert request.image_data_url.startswith("data:image/jpeg;base64,")


def test_worker_retries_policy_rejection_with_metadata_and_without_image():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "batch_image_processing": False,
        "openai_compatible": {
            "endpoints": [
                {
                    "base_url": "https://provider.example/v1",
                    "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
                    "model": "provider-vision-model",
                }
            ]
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    worker.vllm_config = {
        "model": "shared-model",
        "inference_prompts": ["Describe this image"],
        "retry_prompt": "Rewrite neutrally: {column:captions}",
        "retry_without_image": True,
    }
    worker.stages = worker._parse_stages_config(worker.vllm_config)
    worker.stage_order = worker._topological_sort_stages(worker.stages)
    policy_error = EndpointRequestError(
        "provider",
        400,
        '{"contentFilter": [{"level": 2}], "error": "unsafe or sensitive content"}',
        {},
    )
    worker.endpoint_pool.run_many = AsyncMock(
        side_effect=[[policy_error], ["A woman sits among fallen leaves in a forest."]]
    )
    worker.api_loop = asyncio.new_event_loop()
    item = ProcessingItem(
        unit_id="unit",
        job_id="shard:chunk:0:idx:6",
        chunk_id="chunk",
        item_key="image.png",
        item_index=6,
        image=Image.new("RGB", (2, 2), color="red"),
        image_data=b"",
        metadata={"captions": "A woman sits in a forest."},
    )

    try:
        results = worker._process_batch_multi_stage([item])
    finally:
        worker.api_loop.close()

    assert results == [(item, {"captions": ["A woman sits among fallen leaves in a forest."]})]
    assert worker.endpoint_pool.run_many.await_count == 2
    first_request = worker.endpoint_pool.run_many.await_args_list[0].args[0][0]
    retry_request = worker.endpoint_pool.run_many.await_args_list[1].args[0][0]
    assert first_request.image_data_url.startswith("data:image/jpeg;base64,")
    assert retry_request.prompt == "Rewrite neutrally: A woman sits in a forest."
    assert retry_request.image_data_url is None


def test_worker_retries_empty_refusal_but_not_unrelated_request_error():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    worker.vllm_config = {
        "model": "vision",
        "inference_prompts": ["Describe"],
        "retry_prompt": "Try a neutral description",
    }
    worker.stages = worker._parse_stages_config(worker.vllm_config)
    worker.stage_order = worker._topological_sort_stages(worker.stages)
    worker.endpoint_pool.run_many = AsyncMock(
        side_effect=[["I'm sorry, I cannot describe this image."], ["A neutral caption."]]
    )
    worker.api_loop = asyncio.new_event_loop()
    item = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=Image.new("RGB", (2, 2), color="red"),
        image_data=b"",
        metadata={},
    )

    try:
        results = worker._process_batch_multi_stage([item])
    finally:
        worker.api_loop.close()

    assert results == [(item, {"captions": ["A neutral caption."]})]
    retry_request = worker.endpoint_pool.run_many.await_args_list[1].args[0][0]
    assert retry_request.image_data_url.startswith("data:image/jpeg;base64,")
    assert worker._is_semantic_caption_failure(ValueError("returned no message content"))
    assert not worker._is_semantic_caption_failure(
        EndpointRequestError("provider", 400, "invalid request", {})
    )


def test_worker_retries_invalid_json_output():
    worker_config = {
        "server": "ws://localhost:8765",
        "token": "orchestrator-token",
        "openai_compatible": {
            "api_key_env": "CAPTIONFLOW_TEST_API_KEY",
            "model": "vision",
            "validate_json_output": True,
            "retry_extra_body": {"response_format": {"type": "json_object"}},
        },
    }
    with patch.dict(os.environ, {"CAPTIONFLOW_TEST_API_KEY": "provider-secret"}):
        worker = OpenAICompatibleWorker(worker_config)

    worker.vllm_config = {
        "model": "vision",
        "inference_prompts": ["Describe as JSON"],
        "retry_prompt": "Retry as valid JSON",
        "retry_sampling": {"max_tokens": 4096, "repetition_penalty": 1.1},
    }
    worker.stages = worker._parse_stages_config(worker.vllm_config)
    worker.stage_order = worker._topological_sort_stages(worker.stages)
    worker.endpoint_pool.run_many = AsyncMock(
        side_effect=[['{"caption": "truncated"'], ['{"caption": "A red square."}']]
    )
    worker.api_loop = asyncio.new_event_loop()
    item = ProcessingItem(
        unit_id="unit",
        job_id="job",
        chunk_id="chunk",
        item_key="image.png",
        item_index=0,
        image=Image.new("RGB", (2, 2), color="red"),
        image_data=b"",
        metadata={},
    )

    try:
        results = worker._process_batch_multi_stage([item])
    finally:
        worker.api_loop.close()

    assert results == [(item, {"captions": ['{"caption": "A red square."}']})]
    assert worker.endpoint_pool.run_many.await_count == 2
    retry_request = worker.endpoint_pool.run_many.await_args_list[1].args[0][0]
    assert retry_request.parameters == {"max_tokens": 4096}
    assert retry_request.extra_body == {"response_format": {"type": "json_object"}}


def test_cli_selects_openai_compatible_worker():
    from caption_flow.cli import main

    runner = CliRunner()
    with patch("caption_flow.workers.openai_compatible.OpenAICompatibleWorker") as worker_class:
        worker = worker_class.return_value
        worker.start = AsyncMock()
        result = runner.invoke(
            main,
            [
                "worker",
                "--server",
                "ws://localhost:8765",
                "--token",
                "test-token",
                "--openai-compatible",
            ],
        )

    assert result.exit_code == 0
    worker_class.assert_called_once()
