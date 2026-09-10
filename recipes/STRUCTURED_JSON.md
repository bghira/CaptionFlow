# Structured JSON captions

Direct vLLM and OpenAI-compatible workers use the same caption pipeline:
prompt formatting, output validation and transformations, semantic retries,
and stage success/failure accounting. Choose either backend without losing
JSON captioning or recovery behavior.

## Shared inference configuration

Put caption policy and decoding constraints in the orchestrator's `inference`
configuration (the legacy `vllm` spelling also works):

```yaml
orchestrator:
  inference:
    model: "vision-model"
    inference_prompts:
      - >-
        Describe the visible image as one JSON object matching the requested
        schema. Use only visible evidence, without speculation. Return JSON only.
    sampling:
      temperature: 0.3
      max_tokens: 2048

    output_processing:
      validate_json_output: true
      canonicalize_json_output: true
      # Optional repairs; boxes must use [ymin, xmin, ymax, xmax] order.
      repair_invalid_json_escapes: true
      normalize_yxyx_bboxes: true
      deduplicate_json_elements: true
      # Optional: override refusal substrings, or [] to disable detection.
      refusal_markers: ["i cannot describe", "unable to provide a caption"]

    response_format:
      type: "json_schema"
      json_schema:
        name: "caption"
        strict: true
        schema:
          type: "object"
          additionalProperties: false
          required: ["description"]
          properties:
            description: {type: "string"}

    retry_prompt: >-
      Try again. Return exactly one valid JSON object with a concise,
      visually grounded description and no surrounding prose.
    retry_sampling:
      temperature: 0.1
      max_tokens: 1024
    retry_response_format:
      type: "json_object"
```

An empty caption, configured refusal match, invalid JSON, or provider content
rejection can receive one semantic retry. Unrelated HTTP/transport failures
use the endpoint's request retry policy, not the semantic prompt. Only items
with an accepted output proceed to the next stage; failed items are not
reported as successful empty captions. With multiple prompts, one accepted
output is sufficient, and that item is not retried.

`output_processing`, `response_format`, `retry_response_format`, `retry_prompt`,
`retry_sampling`, and `retry_without_image` can also be set per stage. Stage
output policy merges over inference defaults. Set `retry_without_image: true`
only when deliberately retrying from text/metadata; it omits the image on both
backends. Prompt, output-policy, and retry changes apply on config reload
without reloading native model weights.

`validate_json_output` checks JSON syntax and rejects non-finite numbers. It
does **not** validate against JSON Schema. `response_format` requests constrained
decoding from the backend; support for specific schemas depends on the model,
provider, and vLLM version. Supported envelope types are `text`, `json_object`,
and `json_schema`; a retry inherits the primary constraint unless overridden.
Use `retry_response_format: {type: text}` to request unconstrained retry decoding.

The optional transforms require `validate_json_output: true`:

- `repair_invalid_json_escapes` escapes only invalid backslashes, preserving
  existing valid escape pairs. It cannot repair truncated JSON.
- `canonicalize_json_output` emits compact JSON with literal Unicode.
- `normalize_yxyx_bboxes` sorts inverted coordinate pairs. It does not infer
  boxes, clamp their range, or assess localization quality.
- `deduplicate_json_elements` removes exact duplicates from arrays named
  `elements`, including nested arrays of that name.

## Direct vLLM worker

Use a normal GPU worker config and launch with `caption-flow worker --vllm`.
The worker translates `response_format` into vLLM's native structured-output
parameters, including the older guided-decoding API. It retains native tensor
parallelism, memory/cache configuration, token checks, and image resize recovery.

Native `sampling` and `retry_sampling` are passed through to `SamplingParams`,
including vLLM-specific controls such as `top_k`, `min_p`, `seed`, and
`repetition_penalty`. Retry settings overlay primary settings without mutating
the cached primary parameters. Use fields supported by your installed vLLM.
Do not also configure native `structured_outputs`/`guided_decoding` when using
the shared `response_format` envelope.

## OpenAI-compatible worker

Launch with `caption-flow worker --openai-compatible`. Endpoint credentials,
request concurrency, image encoding, and provider-specific request extensions
remain local:

```yaml
worker:
  server: "ws://orchestrator.example:8765"
  token: "replace-with-captionflow-worker-token"
  openai_compatible:
    endpoints:
      - name: "local-vllm"
        base_url: "http://inference.example:8000/v1"
        api_key_env: "VLLM_API_KEY"
        model: "vision-model"
        initial_concurrency: 16
        max_concurrency: 16
```

This adapter forwards shared `response_format` and standard OpenAI sampling
fields to chat completions. Non-standard provider fields belong in endpoint
`extra_body`; retry-only extensions belong in `openai_compatible.retry_extra_body`.
Shared response formats override endpoint defaults; local `retry_extra_body`
overrides the shared retry body. Neither extension can replace `model` or
`messages`.

## Compatibility and local overrides

Either worker can override shared policy using `worker.output_processing`.
Precedence is inference defaults, stage policy, then local worker policy.
The legacy `inference.refusal_markers` is still accepted, with
`inference.output_processing.refusal_markers` taking precedence.

Existing JSON/refusal flags under `worker.openai_compatible` remain accepted
as compatibility aliases for local output policy. New configs should use
shared inference policy; explicit `worker.output_processing` overrides those
legacy aliases. Existing endpoint `extra_body.response_format` and
`retry_extra_body.response_format` continue working for API workers.
