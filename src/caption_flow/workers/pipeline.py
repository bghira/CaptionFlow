"""Backend-independent stage execution and semantic caption recovery."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from ..models import ProcessingStage, StageResult
from ..utils.output_policy import OutputPolicy
from ..utils.prompt_template import PromptTemplateManager

logger = logging.getLogger(__name__)


class CaptionRejectedError(ValueError):
    """A backend rejected content before returning a caption; semantic retry is allowed."""


@dataclass(frozen=True)
class StageRequest:
    item: Any
    prompt: str


def stage_context(item: Any) -> dict:
    context = dict(item.metadata)
    for name, result in item.stage_results.items():
        for index, output in enumerate(result.outputs):
            context[f"{name}_output_{index}"] = output
        context[result.output_field] = (
            result.outputs[0] if len(result.outputs) == 1 else result.outputs
        )
    return context


class CaptionBackend(Protocol):
    def _prepare_stage_batch(self, batch: list, stage: ProcessingStage) -> list: ...
    def _generate_stage(
        self, stage: ProcessingStage, requests: list[StageRequest], retry: bool
    ) -> list: ...
    def _output_policy(self, stage: ProcessingStage) -> OutputPolicy: ...


def _collect_outputs(
    backend: CaptionBackend,
    stage: ProcessingStage,
    policy: OutputPolicy,
    requests: list[StageRequest],
    *,
    retry: bool,
) -> tuple[dict, set]:
    outputs = defaultdict(list)
    retryable = set()
    if not requests:
        return outputs, retryable
    responses = backend._generate_stage(stage, requests, retry)
    for request, response in zip(requests, responses, strict=True):
        key = id(request.item)
        if isinstance(response, Exception):
            logger.error(
                "Generation failed for %s in %s: %s", request.item.item_key, stage.name, response
            )
            if isinstance(response, CaptionRejectedError):
                retryable.add(key)
            continue
        cleaned = policy.process(response)
        if cleaned is None:
            retryable.add(key)
        else:
            outputs[key].append(cleaned)
    return outputs, retryable


def run_caption_pipeline(
    backend: CaptionBackend, batch: list, stages: list[ProcessingStage], order: list[str]
) -> tuple[list, int]:
    """Run the same acceptance, retry and stage-progression rules for every backend."""
    active = list(batch)
    failed = 0
    stage_map = {stage.name: stage for stage in stages}
    for name in order:
        if not active:
            break
        stage = stage_map[name]
        prepared = backend._prepare_stage_batch(active, stage)
        failed += len(active) - len(prepared)
        active = prepared
        if not active:
            break
        policy = backend._output_policy(stage)
        templates = PromptTemplateManager(stage.prompts)
        requests = [
            StageRequest(item, prompt)
            for item in active
            for prompt in templates.format_all(stage_context(item))
        ]
        outputs, retryable = _collect_outputs(backend, stage, policy, requests, retry=False)
        if stage.retry_prompt:
            retries = [
                StageRequest(
                    item,
                    PromptTemplateManager([stage.retry_prompt]).format_all(stage_context(item))[0],
                )
                for item in active
                if id(item) in retryable and not outputs[id(item)]
            ]
            recovered, _ = _collect_outputs(backend, stage, policy, retries, retry=True)
            outputs.update(recovered)

        successful = []
        for item in active:
            if outputs[id(item)]:
                item.stage_results[name] = StageResult(name, stage.output_field, outputs[id(item)])
                successful.append(item)
            else:
                item.stage_results.pop(name, None)
                failed += 1
        active = successful

    results = []
    for item in active:
        fields = defaultdict(list)
        for name in order:
            result = item.stage_results[name]
            fields[result.output_field].extend(result.outputs)
        results.append((item, dict(fields)))
    return results, failed
