"""BYOK worker backend for OpenAI-compatible multimodal chat endpoints.

All provider configuration and credentials live in the worker process. The
orchestrator continues to distribute only work and captioning-stage settings.
"""

from __future__ import annotations

import asyncio
import base64
import email.utils
import io
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp
from PIL import Image

from .. import __version__
from ..models import ProcessingStage, StageResult
from ..utils.image_processor import ImageProcessor
from ..utils.prompt_template import PromptTemplateManager
from .caption import CaptionWorker, ProcessingItem

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class EndpointRequestError(RuntimeError):
    """An HTTP error returned by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        status: int,
        message: str,
        headers: Dict[str, str],
    ):
        super().__init__(f"{endpoint} returned HTTP {status}: {message[:500]}")
        self.endpoint = endpoint
        self.status = status
        self.message = message
        self.headers = headers

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 425, 429, 500, 502, 503, 504} or self.throttled

    @property
    def throttled(self) -> bool:
        message = self.message.lower()
        return self.status == 429 or (
            self.status in {400, 409, 503}
            and any(
                marker in message
                for marker in (
                    "concurren",
                    "rate limit",
                    "rate_limit",
                    "too many request",
                    "capacity",
                    "overloaded",
                )
            )
        )


@dataclass(frozen=True)
class EndpointConfig:
    """Local configuration for one provider account or API endpoint."""

    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: Optional[str] = None
    model_map: Dict[str, str] = field(default_factory=dict)
    initial_concurrency: int = 1
    max_concurrency: int = 16
    requests_per_minute: Optional[float] = None
    timeout_seconds: float = 120.0
    max_retries: int = 6
    probe_after_successes: int = 8
    headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    chat_completions_path: str = "chat/completions"
    verify_ssl: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int) -> "EndpointConfig":
        name = str(raw.get("name") or f"endpoint-{index + 1}")
        base_url = raw.get("base_url")
        base_url_env = raw.get("base_url_env")
        if not base_url and base_url_env:
            base_url = os.environ.get(str(base_url_env))
        if not base_url:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        key_env = str(raw.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(f"Endpoint '{name}' requires API key environment variable {key_env}")

        initial = int(raw.get("initial_concurrency", 1))
        maximum = int(raw.get("max_concurrency", 16))
        if initial < 1 or maximum < 1 or initial > maximum:
            raise ValueError(
                f"Endpoint '{name}' requires 1 <= initial_concurrency <= max_concurrency"
            )

        rpm = raw.get("requests_per_minute")
        if rpm is not None:
            rpm = float(rpm)
            if rpm <= 0:
                rpm = None

        reserved = {"model", "messages"}
        extra_body = dict(raw.get("extra_body", {}))
        overlap = reserved.intersection(extra_body)
        if overlap:
            raise ValueError(
                f"Endpoint '{name}' extra_body cannot override: {', '.join(sorted(overlap))}"
            )

        return cls(
            name=name,
            base_url=str(base_url).rstrip("/"),
            api_key=api_key,
            model=raw.get("model") or os.environ.get("OPENAI_MODEL"),
            model_map={str(k): str(v) for k, v in raw.get("model_map", {}).items()},
            initial_concurrency=initial,
            max_concurrency=maximum,
            requests_per_minute=rpm,
            timeout_seconds=float(raw.get("timeout_seconds", 120)),
            max_retries=int(raw.get("max_retries", 6)),
            probe_after_successes=max(1, int(raw.get("probe_after_successes", 8))),
            headers={str(k): str(v) for k, v in raw.get("headers", {}).items()},
            extra_body=extra_body,
            chat_completions_path=str(raw.get("chat_completions_path", "chat/completions")).lstrip(
                "/"
            ),
            verify_ssl=bool(raw.get("verify_ssl", True)),
        )

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.chat_completions_path}"

    def resolve_model(self, requested_model: Optional[str]) -> str:
        if requested_model and requested_model in self.model_map:
            return self.model_map[requested_model]
        if "*" in self.model_map:
            return self.model_map["*"]
        resolved = self.model or requested_model
        if not resolved:
            raise ValueError(f"Endpoint '{self.name}' has no model configured")
        return resolved


@dataclass
class EndpointState:
    """Adaptive state for one endpoint; it is never sent to the orchestrator."""

    config: EndpointConfig
    concurrency: int
    inflight: int = 0
    success_streak: int = 0
    cooldown_until: float = 0.0
    next_request_at: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    failed_requests: int = 0
    latency_ewma_ms: Optional[float] = None
    disabled_reason: Optional[str] = None


@dataclass(frozen=True)
class ChatRequest:
    """Provider-neutral input for a single chat completion call."""

    prompt: str
    image_data_url: Optional[str]
    requested_model: Optional[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    image_detail: Optional[str] = None


@dataclass
class PendingRequest:
    index: int
    request: ChatRequest
    attempts: int = 0
    ready_at: float = 0.0


class AdaptiveEndpointPool:
    """Pool OpenAI-compatible endpoints with AIMD concurrency discovery.

    Each endpoint starts conservatively. Sustained successes add one request
    slot; explicit rate/concurrency responses halve the active limit and apply
    the provider's retry delay when available.
    """

    def __init__(self, configs: Iterable[EndpointConfig]):
        self.states = [
            EndpointState(config=config, concurrency=config.initial_concurrency)
            for config in configs
        ]
        if not self.states:
            raise ValueError("At least one OpenAI-compatible endpoint is required")
        self.session: Optional[aiohttp.ClientSession] = None
        self._round_robin = 0
        self._snapshot_lock = Lock()

    async def open(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=sum(state.config.max_concurrency for state in self.states),
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(connector=connector)

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._snapshot_lock:
            return [
                {
                    "name": state.config.name,
                    "concurrency": state.concurrency,
                    "max_concurrency": state.config.max_concurrency,
                    "inflight": state.inflight,
                    "requests": state.total_requests,
                    "successful": state.successful_requests,
                    "rate_limited": state.rate_limited_requests,
                    "failed": state.failed_requests,
                    "latency_ewma_ms": state.latency_ewma_ms,
                    "disabled": state.disabled_reason is not None,
                }
                for state in self.states
            ]

    def _available_states(self, now: float) -> List[EndpointState]:
        return [
            state
            for state in self.states
            if not state.disabled_reason
            and state.inflight < state.concurrency
            and state.cooldown_until <= now
            and state.next_request_at <= now
        ]

    def _choose_state(self, now: float) -> Optional[EndpointState]:
        available = self._available_states(now)
        if not available:
            return None

        # Prefer the lowest utilization, then rotate ties so equally sized
        # accounts all contribute capacity.
        ordered = self.states[self._round_robin :] + self.states[: self._round_robin]
        candidates = sorted(
            available,
            key=lambda state: (
                state.inflight / max(1, state.concurrency),
                ordered.index(state),
            ),
        )
        selected = candidates[0]
        self._round_robin = (self.states.index(selected) + 1) % len(self.states)
        return selected

    def _reserve(self, state: EndpointState, now: float) -> None:
        with self._snapshot_lock:
            state.inflight += 1
            state.total_requests += 1
            if state.config.requests_per_minute:
                interval = 60.0 / state.config.requests_per_minute
                state.next_request_at = max(now, state.next_request_at) + interval

    def _release(self, state: EndpointState) -> None:
        with self._snapshot_lock:
            state.inflight = max(0, state.inflight - 1)

    def _record_success(self, state: EndpointState, elapsed_ms: float) -> None:
        with self._snapshot_lock:
            state.successful_requests += 1
            state.success_streak += 1
            if state.latency_ewma_ms is None:
                state.latency_ewma_ms = elapsed_ms
            else:
                state.latency_ewma_ms = state.latency_ewma_ms * 0.8 + elapsed_ms * 0.2

            threshold = state.config.probe_after_successes * state.concurrency
            if (
                state.success_streak >= threshold
                and state.concurrency < state.config.max_concurrency
            ):
                state.concurrency += 1
                state.success_streak = 0
                logger.info(
                    "Endpoint %s raised discovered concurrency to %d",
                    state.config.name,
                    state.concurrency,
                )

    def _record_failure(self, state: EndpointState, error: Exception, now: float) -> float:
        retry_delay = min(30.0, 0.5 * (2 ** min(state.failed_requests, 6)))
        with self._snapshot_lock:
            state.failed_requests += 1
            state.success_streak = 0

            if isinstance(error, EndpointRequestError):
                retry_delay = _retry_delay(error.headers, default=retry_delay)
                if error.status in {401, 403}:
                    state.disabled_reason = (
                        f"HTTP {error.status} authentication/authorization failure"
                    )
                if error.throttled:
                    state.rate_limited_requests += 1
                    old_limit = state.concurrency
                    state.concurrency = max(1, math.ceil(state.concurrency / 2))
                    state.cooldown_until = max(state.cooldown_until, now + retry_delay)
                    logger.warning(
                        "Endpoint %s throttled; concurrency %d -> %d, retrying in %.2fs",
                        state.config.name,
                        old_limit,
                        state.concurrency,
                        retry_delay,
                    )
        return retry_delay

    def _next_wakeup(self, pending: deque[PendingRequest], now: float) -> float:
        times = [request.ready_at for request in pending if request.ready_at > now]
        for state in self.states:
            if state.disabled_reason:
                continue
            if state.cooldown_until > now:
                times.append(state.cooldown_until)
            if state.next_request_at > now:
                times.append(state.next_request_at)
        return min(times, default=now + 0.25)

    async def run_many(self, requests: List[ChatRequest]) -> List[Any]:  # noqa: C901
        """Execute calls in order, returning text or an exception per call."""
        if not self.session:
            raise RuntimeError("Endpoint pool is not open")
        if not requests:
            return []

        pending = deque(
            PendingRequest(index=i, request=request) for i, request in enumerate(requests)
        )
        running: Dict[asyncio.Task, Tuple[PendingRequest, EndpointState, float]] = {}
        results: List[Any] = [None] * len(requests)

        while pending or running:
            now = asyncio.get_running_loop().time()

            # Fill every currently discovered endpoint slot.
            made_progress = True
            while pending and made_progress:
                made_progress = False
                for _ in range(len(pending)):
                    item = pending.popleft()
                    if item.ready_at > now:
                        pending.append(item)
                        continue
                    state = self._choose_state(now)
                    if state is None:
                        pending.appendleft(item)
                        break
                    item.attempts += 1
                    self._reserve(state, now)
                    task = asyncio.create_task(self._request_once(state, item.request))
                    running[task] = (item, state, now)
                    made_progress = True

            if not running:
                enabled = [state for state in self.states if not state.disabled_reason]
                if not enabled:
                    error = RuntimeError("All OpenAI-compatible endpoints are disabled")
                    while pending:
                        results[pending.popleft().index] = error
                    break
                wake_at = self._next_wakeup(pending, now)
                await asyncio.sleep(max(0.01, min(1.0, wake_at - now)))
                continue

            wake_at = self._next_wakeup(pending, now)
            timeout = max(0.01, min(1.0, wake_at - now)) if pending else None
            done, _ = await asyncio.wait(
                running,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                item, state, started_at = running.pop(task)
                self._release(state)
                elapsed_ms = (asyncio.get_running_loop().time() - started_at) * 1000
                try:
                    text = task.result()
                    self._record_success(state, elapsed_ms)
                    results[item.index] = text
                except Exception as error:
                    now = asyncio.get_running_loop().time()
                    retry_delay = self._record_failure(state, error, now)
                    retryable = not isinstance(error, EndpointRequestError) or error.retryable
                    max_retries = max(state.config.max_retries for state in self.states)
                    enabled = any(not candidate.disabled_reason for candidate in self.states)
                    if retryable and enabled and item.attempts <= max_retries:
                        alternative_ready = any(
                            candidate is not state
                            and not candidate.disabled_reason
                            and candidate.cooldown_until <= now
                            and candidate.next_request_at <= now
                            for candidate in self.states
                        )
                        # A provider's Retry-After applies to that provider, not
                        # to another independently configured endpoint.
                        item.ready_at = now if alternative_ready else now + retry_delay
                        pending.append(item)
                    else:
                        results[item.index] = error

        return results

    async def _request_once(self, state: EndpointState, request: ChatRequest) -> str:
        if not self.session:
            raise RuntimeError("Endpoint pool is not open")

        config = state.config
        model = config.resolve_model(request.requested_model)
        content: List[Dict[str, Any]] = []
        if request.image_data_url:
            image_url: Dict[str, Any] = {"url": request.image_data_url}
            if request.image_detail:
                image_url["detail"] = request.image_detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": request.prompt})

        messages: List[Dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content})

        body = {
            "model": model,
            "messages": messages,
            **request.parameters,
            **config.extra_body,
        }
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"CaptionFlow/{__version__}",
            **config.headers,
        }
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        async with self.session.post(
            config.url,
            headers=headers,
            json=body,
            timeout=timeout,
            ssl=config.verify_ssl,
        ) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            if response.status < 200 or response.status >= 300:
                body_text = await response.text()
                raise EndpointRequestError(
                    config.name,
                    response.status,
                    body_text,
                    response_headers,
                )
            payload = await response.json(content_type=None)
            return _extract_message_text(payload, config.name)


def _extract_message_text(payload: Dict[str, Any], endpoint_name: str) -> str:
    """Extract text from common Chat Completions response variants."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Endpoint '{endpoint_name}' returned no message content") from error

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                pieces.append(str(part.get("text", "")))
        text = "".join(pieces)
    else:
        text = str(content or "")

    if not text.strip():
        raise ValueError(f"Endpoint '{endpoint_name}' returned empty message content")
    return text.strip()


def _parse_duration(value: str) -> Optional[float]:
    value = value.strip().lower()
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            try:
                return max(0.0, float(value[: -len(suffix)]) * multiplier)
            except ValueError:
                return None
    try:
        number = float(value)
    except ValueError:
        return None
    # Large values are normally Unix timestamps rather than seconds.
    if number > 1_000_000_000:
        return max(0.0, number - time.time())
    return max(0.0, number)


def _retry_delay(headers: Dict[str, str], default: float) -> float:
    retry_after = headers.get("retry-after")
    if retry_after:
        parsed = _parse_duration(retry_after)
        if parsed is not None:
            return min(3600.0, parsed)
        try:
            when = email.utils.parsedate_to_datetime(retry_after)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return min(3600.0, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            pass

    for name in ("x-ratelimit-reset-requests", "ratelimit-reset"):
        if name in headers:
            parsed = _parse_duration(headers[name])
            if parsed is not None:
                return min(3600.0, parsed)
    return max(0.05, default)


class OpenAICompatibleWorker(CaptionWorker):
    """CaptionFlow worker backed by a local pool of remote API endpoints."""

    _SUPPORTED_SAMPLING_KEYS = {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    }
    _REFUSAL_MARKERS = (
        "i'm sorry",
        "i’m sorry",
        "i cannot",
        "i can't",
        "i can’t",
        "unable to provide",
        "unable to describe",
        "cannot provide",
        "can't provide",
        "can’t provide",
        "cannot assist",
        "can't assist",
        "can’t assist",
    )
    _POLICY_ERROR_MARKERS = (
        "contentfilter",
        "content filter",
        "content policy",
        "content_policy",
        "moderation",
        "policy violation",
        "safety system",
        "unsafe or sensitive",
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raw_config = config.get("openai_compatible")
        if raw_config is True:
            raw_config = {}
        if not isinstance(raw_config, dict):
            raise ValueError("openai_compatible worker configuration must be a mapping")
        self.api_config = raw_config

        raw_endpoints = raw_config.get("endpoints")
        if raw_endpoints is None:
            raw_endpoints = [raw_config]
        if not isinstance(raw_endpoints, list) or not all(
            isinstance(endpoint, dict) for endpoint in raw_endpoints
        ):
            raise ValueError("openai_compatible.endpoints must be a list of mappings")

        endpoints = [
            EndpointConfig.from_dict(endpoint, i) for i, endpoint in enumerate(raw_endpoints)
        ]
        self.endpoint_pool = AdaptiveEndpointPool(endpoints)
        self.api_loop: Optional[asyncio.AbstractEventLoop] = None
        self.system_prompt = raw_config.get("system_prompt")
        self.include_image = bool(raw_config.get("include_image", True))
        self.image_detail = raw_config.get("image_detail", "auto")
        self.image_format = str(raw_config.get("image_format", "jpeg")).lower()
        self.image_quality = int(raw_config.get("image_quality", 90))
        configured_dimension = int(raw_config.get("max_image_dimension", 0) or 0)
        self.max_image_dimension = configured_dimension if configured_dimension > 0 else None

    async def _pre_start(self):
        """Fetch shared stage settings, then start the API processing thread."""
        logger.info("Connecting to orchestrator for captioning-stage configuration...")
        configured = False
        while not configured and self.running:
            try:
                await self._initial_connect_for_config()
                configured = True
            except Exception as error:
                logger.error("Failed to get configuration: %s", error)
                await asyncio.sleep(5)

        self._apply_local_overrides()
        self.mock_mode = bool(self.vllm_config.get("mock_results", False))
        if self.mock_mode:
            logger.info("Mock mode enabled; remote endpoints will not be called")
        else:
            logger.info(
                "Configured %d OpenAI-compatible endpoint(s)", len(self.endpoint_pool.states)
            )
        Thread(target=self._processing_thread, name="captionflow-api-worker", daemon=True).start()

    def _apply_local_overrides(self) -> None:
        if not self.vllm_config:
            return
        self.vllm_config = dict(self.vllm_config)
        batch_size = self.api_config.get("batch_size", self.config.get("batch_size"))
        if batch_size is None:
            batch_size = sum(state.config.max_concurrency for state in self.endpoint_pool.states)
        self.vllm_config["batch_size"] = max(1, int(batch_size))

    def _handle_vllm_config_update(self, new_config: Dict[str, Any]) -> bool:
        """Apply shared prompt/stage changes without touching local credentials."""
        if not new_config:
            return True
        self.vllm_config = dict(new_config)
        self._apply_local_overrides()
        self.mock_mode = bool(self.vllm_config.get("mock_results", False))
        self.stages = self._parse_stages_config(self.vllm_config)
        self.stage_order = self._topological_sort_stages(self.stages)
        return True

    def _processing_thread(self):
        """Own a persistent HTTP event loop/session inside the worker thread."""
        loop = asyncio.new_event_loop()
        self.api_loop = loop
        asyncio.set_event_loop(loop)
        try:
            if not self.mock_mode:
                loop.run_until_complete(self.endpoint_pool.open())
            super()._processing_thread()
        finally:
            if not self.mock_mode:
                loop.run_until_complete(self.endpoint_pool.close())
            loop.close()
            self.api_loop = None

    def _process_batch_multi_stage(  # noqa: C901
        self, batch: List[ProcessingItem], max_attempts: int = 3
    ) -> List[Tuple[ProcessingItem, Dict]]:
        del max_attempts  # Retries are controlled per endpoint.
        if not self.api_loop:
            raise RuntimeError("OpenAI-compatible endpoint loop is not ready")

        active_batch = list(batch)
        image_urls: Dict[int, Optional[str]] = {}
        if self.include_image:
            image_urls = self._image_data_urls(active_batch)

        for stage_name in self.stage_order:
            stage = next(stage for stage in self.stages if stage.name == stage_name)
            requests: List[ChatRequest] = []
            owners: List[ProcessingItem] = []
            sampling = self._sampling_for_stage(stage)

            for item in active_batch:
                context = self._stage_context(item)

                for prompt in PromptTemplateManager(stage.prompts).format_all(context):
                    requests.append(
                        ChatRequest(
                            prompt=prompt,
                            image_data_url=image_urls.get(id(item)),
                            requested_model=stage.model,
                            parameters=sampling,
                            system_prompt=self.system_prompt,
                            image_detail=self.image_detail,
                        )
                    )
                    owners.append(item)

            api_started = time.monotonic()
            responses = self.api_loop.run_until_complete(self.endpoint_pool.run_many(requests))
            outputs_by_item: Dict[int, List[str]] = defaultdict(list)
            retryable_item_ids = set()
            for owner, response in zip(owners, responses, strict=True):
                if isinstance(response, Exception):
                    logger.error(
                        "API request failed for item %s in stage %s: %s",
                        owner.item_key,
                        stage_name,
                        response,
                    )
                    if self._is_semantic_caption_failure(response):
                        retryable_item_ids.add(id(owner))
                    continue
                cleaned = self._clean_output(response)
                if cleaned and not self._is_refusal_text(cleaned):
                    outputs_by_item[id(owner)].append(cleaned)
                else:
                    retryable_item_ids.add(id(owner))

            retry_items = [
                item
                for item in active_batch
                if id(item) in retryable_item_ids
                and not outputs_by_item.get(id(item))
                and stage.retry_prompt
            ]
            if retry_items:
                retry_requests = []
                for item in retry_items:
                    context = self._stage_context(item)
                    retry_prompt = PromptTemplateManager([stage.retry_prompt]).format_all(context)[
                        0
                    ]
                    retry_requests.append(
                        ChatRequest(
                            prompt=retry_prompt,
                            image_data_url=(
                                None if stage.retry_without_image else image_urls.get(id(item))
                            ),
                            requested_model=stage.model,
                            parameters=sampling,
                            system_prompt=self.system_prompt,
                            image_detail=self.image_detail,
                        )
                    )

                logger.info(
                    "Retrying %d empty or refused caption(s) in stage %s%s",
                    len(retry_items),
                    stage_name,
                    " without images" if stage.retry_without_image else "",
                )
                retry_responses = self.api_loop.run_until_complete(
                    self.endpoint_pool.run_many(retry_requests)
                )
                for item, response in zip(retry_items, retry_responses, strict=True):
                    if isinstance(response, Exception):
                        logger.error(
                            "Fallback API request failed for item %s in stage %s: %s",
                            item.item_key,
                            stage_name,
                            response,
                        )
                        continue
                    cleaned = self._clean_output(response)
                    if cleaned and not self._is_refusal_text(cleaned):
                        outputs_by_item[id(item)].append(cleaned)
                    else:
                        logger.error(
                            "Fallback returned no usable output for %s in stage %s",
                            item.item_key,
                            stage_name,
                        )

            logger.info(
                "API stage %s completed %d request(s) in %.3f seconds",
                stage_name,
                len(requests) + len(retry_items),
                time.monotonic() - api_started,
            )

            next_batch = []
            for item in active_batch:
                outputs = outputs_by_item.get(id(item), [])
                if outputs:
                    item.stage_results[stage_name] = StageResult(
                        stage_name=stage_name,
                        output_field=stage.output_field,
                        outputs=outputs,
                    )
                    next_batch.append(item)
                else:
                    logger.error("No outputs for %s in stage %s", item.item_key, stage_name)
                    self.items_failed += 1
            active_batch = next_batch
            if not active_batch:
                break

        results = []
        for item in active_batch:
            outputs_by_field: Dict[str, List[str]] = defaultdict(list)
            for stage_result in item.stage_results.values():
                outputs_by_field[stage_result.output_field].extend(stage_result.outputs)
            results.append((item, dict(outputs_by_field)))
            self.items_processed += 1
        return results

    @classmethod
    def _is_refusal_text(cls, text: str) -> bool:
        normalized = text.strip().lower()
        return any(marker in normalized for marker in cls._REFUSAL_MARKERS)

    @classmethod
    def _is_semantic_caption_failure(cls, error: Exception) -> bool:
        if isinstance(error, EndpointRequestError):
            message = error.message.lower()
            return any(marker in message for marker in cls._POLICY_ERROR_MARKERS)
        if isinstance(error, ValueError):
            message = str(error).lower()
            return "message content" in message
        return False

    @staticmethod
    def _stage_context(item: ProcessingItem) -> Dict[str, Any]:
        context = dict(item.metadata)
        for previous_name, result in item.stage_results.items():
            for index, output in enumerate(result.outputs):
                context[f"{previous_name}_output_{index}"] = output
            context[result.output_field] = (
                result.outputs[0] if len(result.outputs) == 1 else result.outputs
            )
        return context

    def _sampling_for_stage(self, stage: ProcessingStage) -> Dict[str, Any]:
        sampling = dict(self.vllm_config.get("sampling", {}))
        if stage.sampling:
            sampling.update(stage.sampling)
        return {
            key: value
            for key, value in sampling.items()
            if key in self._SUPPORTED_SAMPLING_KEYS and value is not None
        }

    def _image_data_url(self, item: ProcessingItem) -> str:  # noqa: C901
        source_format = (
            (item.image.format if item.image else None)
            or item.metadata.get("image_format")
            or item.metadata.get("_image_format")
        )
        if not source_format and item.image_data:
            try:
                with Image.open(io.BytesIO(item.image_data)) as source:
                    source_format = source.format
            except (OSError, ValueError):
                pass

        target_format = "jpeg" if self.image_format in {"jpg", "jpeg"} else self.image_format
        if target_format not in {"jpeg", "png", "gif", "webp"}:
            target_format = "jpeg"
        normalized_source = str(source_format or "").lower()
        if normalized_source in {"jpg", "jpeg"}:
            normalized_source = "jpeg"

        image = item.image
        needs_resize = False
        if self.max_image_dimension:
            if image is not None:
                needs_resize = max(image.size) > self.max_image_dimension
            elif item.image_data:
                with Image.open(io.BytesIO(item.image_data)) as source:
                    needs_resize = max(source.size) > self.max_image_dimension

        if item.image_data and normalized_source == target_format and not needs_resize:
            data = item.image_data
        elif image is not None or item.image_data:
            output = io.BytesIO()
            save_format = "JPEG" if target_format == "jpeg" else target_format.upper()
            if image is None:
                image = ImageProcessor.decode_image_data(item.image_data)
            if self.max_image_dimension and max(image.size) > self.max_image_dimension:
                target_size = ImageProcessor.constrained_size(image.size, self.max_image_dimension)
                image = ImageProcessor.resize_images([image], [target_size])[0]
            save_kwargs: Dict[str, Any] = {}
            if save_format == "JPEG":
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                save_kwargs["quality"] = self.image_quality
            image.save(output, format=save_format, **save_kwargs)
            data = output.getvalue()
        else:
            raise ValueError(f"Item {item.item_key} has no image data")

        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/{target_format};base64,{encoded}"

    def _image_data_urls(self, items: Iterable[ProcessingItem]) -> Dict[int, Optional[str]]:
        """Encode a request batch, fusing byte decode and resize when possible."""
        item_list = list(items)
        results: Dict[int, Optional[str]] = {}
        target_format = "jpeg" if self.image_format in {"jpg", "jpeg"} else self.image_format
        fast_items: List[ProcessingItem] = []
        fast_sizes: List[tuple[int, int]] = []

        if target_format == "jpeg":
            for item in item_list:
                if item.image is not None or not item.image_data:
                    continue
                try:
                    with Image.open(io.BytesIO(item.image_data)) as source:
                        source_format = str(source.format or "").lower()
                        source_size = source.size
                    normalized_source = (
                        "jpeg" if source_format in {"jpg", "jpeg"} else source_format
                    )
                    target_size = source_size
                    if self.max_image_dimension:
                        target_size = ImageProcessor.constrained_size(
                            source_size, self.max_image_dimension
                        )
                    if normalized_source == "jpeg" and target_size == source_size:
                        encoded = base64.b64encode(item.image_data).decode("ascii")
                        results[id(item)] = f"data:image/jpeg;base64,{encoded}"
                    else:
                        fast_items.append(item)
                        fast_sizes.append(target_size)
                except (OSError, ValueError):
                    continue

        if fast_items:
            try:
                images = ImageProcessor.preprocess_encoded_batch(
                    [item.image_data for item in fast_items], fast_sizes
                )
                for item, image in zip(fast_items, images, strict=True):
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=self.image_quality)
                    encoded = base64.b64encode(output.getvalue()).decode("ascii")
                    results[id(item)] = f"data:image/jpeg;base64,{encoded}"
            except Exception:
                logger.warning(
                    "Rust batch image preprocessing failed; falling back to individual images",
                    exc_info=True,
                )

        for item in item_list:
            if id(item) not in results:
                results[id(item)] = self._image_data_url(item)
        return results

    def _get_heartbeat_data(self) -> Dict[str, Any]:
        data = super()._get_heartbeat_data()
        data.update(
            {
                "backend": "openai_compatible",
                "models_loaded": 0,
                "endpoint_pool": self.endpoint_pool.snapshot(),
            }
        )
        return data
