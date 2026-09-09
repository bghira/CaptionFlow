"""The same caption contract through the native and HTTP generation adapters."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from caption_flow.utils.image_processor import ImageProcessor
from caption_flow.utils.output_policy import OutputPolicy, validate_response_format
from caption_flow.utils.vllm_config import create_native_sampling_params
from caption_flow.workers.caption import CaptionWorker, MultiStageVLLMManager, ProcessingItem
from caption_flow.workers.openai_compatible import OpenAICompatibleWorker


def item(index=0):
    return ProcessingItem(
        unit_id="unit",
        job_id=f"job-{index}",
        chunk_id="chunk",
        item_key=f"image-{index}",
        item_index=index,
        image=Image.new("RGB", (32, 16)),
        image_data=b"",
        metadata={"source": f"original-{index}"},
    )


@pytest.fixture(params=["native", "api"])
def backend(request, monkeypatch):
    import vllm

    monkeypatch.setattr(vllm, "SamplingParams", SimpleNamespace)
    monkeypatch.setattr(
        vllm,
        "sampling_params",
        SimpleNamespace(StructuredOutputsParams=SimpleNamespace),
        raising=False,
    )
    config = {"server": "ws://localhost:8765", "token": "test", "batch_image_processing": False}
    if request.param == "api":
        monkeypatch.setenv("CAPTIONFLOW_TEST_KEY", "test")
        worker = OpenAICompatibleWorker(
            {
                **config,
                "openai_compatible": {"api_key_env": "CAPTIONFLOW_TEST_KEY"},
            }
        )
        worker.api_loop = asyncio.new_event_loop()
        worker._image_data_urls = Mock(side_effect=lambda batch: {id(i): "image" for i in batch})
    else:
        worker = CaptionWorker(config)
        worker.model_manager = MultiStageVLLMManager(0)
        worker.model_manager.models["vision"] = Mock()
        worker.model_manager.processors["vision"] = Mock()
        worker.model_manager.tokenizers["vision"] = Mock()
        worker._validate_and_split_batch = Mock(side_effect=lambda batch, *args: (batch, []))
        worker._build_vllm_input = Mock(
            side_effect=lambda image, prompt, *args: {"image": image, "prompt": prompt}
        )

    def configure(**settings):
        shared = {"model": "vision", "inference_prompts": ["Describe"], **settings}
        worker.vllm_config = shared
        worker.stages = worker._parse_stages_config(shared)
        worker.stage_order = worker._topological_sort_stages(worker.stages)
        if request.param == "native":
            for stage in worker.stages:
                worker.model_manager.create_sampling_params(stage, shared.get("sampling", {}))

    def respond(*batches):
        if request.param == "api":
            generator = AsyncMock(side_effect=list(batches))
            worker.endpoint_pool.run_many = generator
        else:
            generator = Mock(
                side_effect=[
                    [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in batch]
                    for batch in batches
                ]
            )
            worker.model_manager.models["vision"].generate = generator
        return generator

    yield SimpleNamespace(worker=worker, kind=request.param, configure=configure, respond=respond)
    if request.param == "api":
        worker.api_loop.close()


def test_both_backends_share_the_stage_runner():
    assert (
        OpenAICompatibleWorker._process_batch_multi_stage
        is CaptionWorker._process_batch_multi_stage
    )


def test_json_transformations_and_retry_constraints_work_on_both_backends(backend):
    backend.configure(
        output_processing={
            "validate_json_output": True,
            "canonicalize_json_output": True,
            "normalize_yxyx_bboxes": True,
            "deduplicate_json_elements": True,
        },
        sampling={"max_tokens": 500, "top_k": 20, "min_p": 0.1},
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "caption", "schema": {"type": "object"}},
        },
        retry_prompt="Retry JSON: {column:source}",
        retry_without_image=True,
        retry_sampling={"max_tokens": 1000, "top_k": 10},
        retry_response_format={"type": "json_object"},
    )
    generator = backend.respond(
        ['{"truncated":', '{"elements": [{"bbox":[8,9,1,2]}, {"bbox":[8,9,1,2]}]}'],
        ['{"caption":"recovered"}'],
    )
    first, second = item(), item(1)
    results = backend.worker._process_batch_multi_stage([first, second])
    assert results == [
        (first, {"captions": ['{"caption":"recovered"}']}),
        (second, {"captions": ['{"elements":[{"bbox":[1,2,8,9]}]}']}),
    ]
    assert backend.worker.items_processed == 2
    assert backend.worker.items_failed == 0
    assert generator.call_count == 2
    primary, retry = generator.call_args_list
    assert len(retry.args[0]) == 1
    if backend.kind == "native":
        assert primary.args[1].top_k == 20
        assert primary.args[1].min_p == 0.1
        assert primary.args[1].structured_outputs.json == {"type": "object"}
        assert retry.args[1].max_tokens == 1000
        assert retry.args[1].top_k == 10
        assert retry.args[1].structured_outputs.json_object is True
        assert retry.args[0][0] == {"image": None, "prompt": "Retry JSON: original-0"}
        # Retry overrides must never leak back into subsequent primary batches.
        assert backend.worker.model_manager.sampling_params["default"] is primary.args[1]
    else:
        call = retry.args[0][0]
        assert call.prompt == "Retry JSON: original-0"
        assert call.image_data_url is None
        assert call.parameters == {"max_tokens": 1000}
        assert call.extra_body == {"response_format": {"type": "json_object"}}
        assert primary.args[0][0].extra_body["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("bad_caption", ["", "I cannot describe it.", '{"truncated":'])
def test_failed_items_never_reach_the_next_stage_or_count_as_success(backend, bad_caption):
    backend.configure(
        output_processing={"validate_json_output": True},
        stages=[
            {"name": "first", "prompts": ["JSON"], "output_field": "raw", "retry_prompt": "Retry"},
            {
                "name": "second",
                "prompts": ["Expand {column:first_output_0}"],
                "requires": ["first"],
                "output_field": "caption",
                "output_processing": {"validate_json_output": False},
            },
        ],
    )
    generator = backend.respond([bad_caption, '{"caption":"ok"}'], [bad_caption], ["Expanded"])
    failed, accepted = item(), item(1)
    results = backend.worker._process_batch_multi_stage([failed, accepted])
    assert results == [(accepted, {"raw": ['{"caption":"ok"}'], "caption": ["Expanded"]})]
    assert failed.stage_results == {}
    assert backend.worker.items_processed == 1
    assert backend.worker.items_failed == 1
    requests = generator.call_args_list[2].args[0]
    assert len(requests) == 1
    prompt = requests[0]["prompt"] if backend.kind == "native" else requests[0].prompt
    assert prompt == 'Expand {"caption":"ok"}'
    if backend.kind == "api":
        backend.worker._image_data_urls.assert_called_once()
        assert backend.worker._stage_image_urls == {}


def test_configurable_refusal_policy_preserves_literal_ocr(backend):
    backend.configure(output_processing={"refusal_markers": []})
    text = 'A sign reads "I cannot stay".'
    backend.respond([text])
    image_item = item()
    assert backend.worker._process_batch_multi_stage([image_item]) == [
        (image_item, {"captions": [text]})
    ]
    assert backend.worker._clean_output(text) == text


def test_one_valid_prompt_is_enough_and_retry_is_not_redundant(backend):
    backend.configure(inference_prompts=["First", "Second"], retry_prompt="Retry")
    generator = backend.respond(["", "Accepted"])
    image_item = item()
    assert backend.worker._process_batch_multi_stage([image_item]) == [
        (image_item, {"captions": ["Accepted"]})
    ]
    assert generator.call_count == 1


def test_multiple_outputs_flow_into_the_next_stage_context(backend):
    backend.configure(
        stages=[
            {"name": "first", "prompts": ["First", "Second"], "output_field": "draft"},
            {"name": "second", "prompts": ["Combine {column:draft}"], "requires": ["first"]},
        ]
    )
    generator = backend.respond(["One", "Two"], ["Combined"])
    image_item = item()
    assert backend.worker._process_batch_multi_stage([image_item]) == [
        (image_item, {"draft": ["One", "Two"], "captions": ["Combined"]})
    ]
    calls = generator.call_args_list[1].args[0]
    assert (
        calls[0]["prompt"] if backend.kind == "native" else calls[0].prompt
    ) == "Combine One, Two"


def test_native_generation_rejects_mismatched_response_count(backend):
    if backend.kind != "native":
        pytest.skip("Native batch shape validation")
    backend.configure()
    backend.respond([])
    with pytest.raises(RuntimeError, match="different number of results"):
        backend.worker._process_batch_multi_stage([item()])


def test_native_resize_recovery_retains_limits_and_original_dimensions(monkeypatch):
    worker = CaptionWorker(
        {"server": "ws://localhost:8765", "token": "test", "batch_image_processing": False}
    )
    worker.vllm_config = {"model": "vision", "max_model_len": 512}
    stage = worker._parse_stages_config(worker.vllm_config)[0]
    stage.max_model_len = 1024
    worker.model_manager = Mock()
    worker.model_manager.get_model_for_stage.return_value = (Mock(), Mock(), Mock(), Mock())
    image_item = item()
    ImageProcessor.prepare_for_inference(image_item)
    checks = []

    def validate(batch, *args):
        checks.append(args[-1])
        if batch[0].image.width > 16:
            return [], batch
        return batch, []

    monkeypatch.setattr(worker, "_validate_and_split_batch", validate)
    resized = worker._prepare_stage_batch([image_item], stage)
    assert len(resized) == 1
    assert resized[0].image.width == 16
    assert resized[0].metadata["image_width"] == 32
    assert checks == [1024, 1024, 1024]
    monkeypatch.setattr(worker, "_validate_and_split_batch", lambda batch, *args: ([], batch))
    assert worker._prepare_stage_batch([image_item], stage) == []
    assert "token limit exceeded" in worker.result_queue.get_nowait()["error"]


def test_local_partial_policy_inherits_shared_validation(backend):
    backend.worker.output_processing_overrides = {"canonicalize_json_output": True}
    OutputPolicy.from_config(backend.worker.output_processing_overrides, partial=True)
    backend.configure(output_processing={"validate_json_output": True})
    policy = backend.worker._output_policy(backend.worker.stages[0])
    assert policy.process('{ "caption" : "ok" }') == '{"caption":"ok"}'
    with pytest.raises(ValueError, match="must be enabled"):
        backend.configure()


def test_hot_reload_applies_policy_prompts_and_retry_without_reloading_models(backend):
    backend.configure()
    worker = backend.worker
    worker._setup_vllm = Mock()
    updated = {
        **worker.vllm_config,
        "inference_prompts": ["New prompt"],
        "retry_prompt": "New retry",
        "retry_sampling": {"max_tokens": 1500},
        "output_processing": {"validate_json_output": True},
        "retry_response_format": {"type": "json_object"},
    }
    assert worker._handle_vllm_config_update(updated)
    worker._setup_vllm.assert_not_called()
    assert worker.stages[0].prompts == ["New prompt"]
    assert worker.stages[0].retry_prompt == "New retry"
    assert worker.stages[0].retry_sampling == {"max_tokens": 1500}
    assert worker._output_policy(worker.stages[0]).validate_json_output
    original_config, original_stages = worker.vllm_config, worker.stages
    with pytest.raises(ValueError, match="schema must be a mapping"):
        worker._handle_vllm_config_update({**updated, "response_format": {"type": "json_schema"}})
    assert worker.vllm_config is original_config
    assert worker.stages is original_stages


def test_transport_failure_does_not_trigger_semantic_retry(backend):
    if backend.kind != "api":
        pytest.skip("Provider transport classification is specific to HTTP")
    backend.configure(retry_prompt="Retry")
    generator = backend.respond([RuntimeError("transport failed")])
    assert backend.worker._process_batch_multi_stage([item()]) == []
    assert generator.call_count == 1
    assert backend.worker.items_failed == 1


@pytest.mark.parametrize("modern", [True, False])
def test_native_constraints_support_both_vllm_apis(monkeypatch, modern):
    import vllm

    constraint_class = "StructuredOutputsParams" if modern else "GuidedDecodingParams"
    monkeypatch.setattr(
        vllm,
        "sampling_params",
        SimpleNamespace(**{constraint_class: SimpleNamespace}),
        raising=False,
    )
    monkeypatch.setattr(vllm, "SamplingParams", SimpleNamespace)
    result = create_native_sampling_params(
        {"top_k": 12, "seed": 42, "max_tokens": 30}, {"type": "json_object"}
    )
    assert result.top_k == 12
    assert result.seed == 42
    assert result.max_tokens == 30
    assert getattr(result, "structured_outputs" if modern else "guided_decoding").json_object
    plain = create_native_sampling_params({}, {"type": "text"})
    assert not hasattr(plain, "structured_outputs")
    assert not hasattr(plain, "guided_decoding")
    with pytest.raises(ValueError, match="not both"):
        create_native_sampling_params({"guided_decoding": {"json_object": True}}, {"type": "text"})


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"type": "unknown"},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"schema": []}},
    ],
)
def test_response_format_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="response_format"):
        validate_response_format(value)


@pytest.mark.parametrize(
    "text",
    ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}', '{"unfinished":', "", None],
)
def test_shared_policy_rejects_unusable_json(text):
    assert OutputPolicy(validate_json_output=True).process(text) is None


def test_escape_repair_preserves_existing_valid_backslashes_and_unicode():
    policy = OutputPolicy(validate_json_output=True, repair_invalid_json_escapes=True)
    output = policy.process(r'{"valid":"\\\\","unicode":"\u00e9","bad":"\q"}')
    assert json.loads(output) == {"valid": "\\\\", "unicode": "é", "bad": r"\q"}
    assert policy.process(r'{"truncated":"\q"') is None


def test_recursive_normalization_deduplicates_after_children_and_preserves_other_arrays():
    policy = OutputPolicy(
        validate_json_output=True, normalize_yxyx_bboxes=True, deduplicate_json_elements=True
    )
    value = {
        "elements": [{"elements": [1, 1]}, {"elements": [1]}],
        "other": [1, 1],
        "nested": [{"bbox": [True, 1, 2, 3]}, {"bbox": [4, 3, 2, 1]}],
    }
    assert json.loads(policy.process(json.dumps(value))) == {
        "elements": [{"elements": [1]}],
        "other": [1, 1],
        "nested": [{"bbox": [True, 1, 2, 3]}, {"bbox": [2, 1, 4, 3]}],
    }


def test_output_policy_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown output_processing"):
        OutputPolicy.from_config({"unknown": True})


def test_native_decode_can_be_reused_and_original_dimensions_survive_resize(monkeypatch):
    image_item = item()
    image_item.image = None
    image_item.image_data = b"encoded"
    decode = Mock(return_value=Image.new("RGB", (640, 480)))
    monkeypatch.setattr(ImageProcessor, "decode_image_data", decode)
    first = ImageProcessor.prepare_for_inference(image_item)
    assert ImageProcessor.prepare_for_inference(image_item) is first
    decode.assert_called_once_with(b"encoded")
    assert image_item.image_data == b""
    image_item.image = first.resize((320, 240))
    ImageProcessor.prepare_for_inference(image_item)
    assert image_item.metadata["image_width"] == 640
    assert image_item.metadata["image_height"] == 480
    image_item.metadata["image_width"] = None
    ImageProcessor.prepare_for_inference(image_item)
    assert image_item.metadata["image_width"] == 320


def test_native_text_only_retry_uses_chat_template_without_multimodal_payload():
    worker = CaptionWorker(
        {"server": "ws://localhost:8765", "token": "test", "batch_image_processing": False}
    )
    processor = Mock()
    processor.apply_chat_template.return_value = "formatted"
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=[1, 2, 3]))
    assert worker._build_vllm_input(None, "Retry", processor, tokenizer) == {
        "prompt_token_ids": [1, 2, 3]
    }
    assert processor.apply_chat_template.call_args.args[0] == [
        {"role": "user", "content": [{"type": "text", "text": "Retry"}]}
    ]
