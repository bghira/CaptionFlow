# Structured JSON captions

CaptionFlow's OpenAI-compatible worker can request provider-constrained JSON,
reject malformed responses, and retry failed items with a smaller schema or
different sampling settings. This works for Ideogram-style captions and other
JSON caption formats without coupling CaptionFlow to one schema.

## Configure the worker

Provider-specific request fields remain local to the worker. Put the primary
structured-output constraint in an endpoint's `extra_body` and enable local
validation on the worker:

```yaml
worker:
  server: "ws://orchestrator.example:8765"
  token: "replace-with-captionflow-worker-token"

  openai_compatible:
    validate_json_output: true
    canonicalize_json_output: true

    # Optional repairs for common model failures. Bounding boxes are expected
    # in [ymin, xmin, ymax, xmax] order.
    repair_invalid_json_escapes: true
    normalize_yxyx_bboxes: true
    deduplicate_json_elements: true

    endpoints:
      - name: "local-vllm"
        base_url: "http://inference.example:8000/v1"
        api_key_env: "VLLM_API_KEY"
        model: "vision-model"
        initial_concurrency: 16
        max_concurrency: 16
        extra_body:
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
```

`validate_json_output` checks strict JSON syntax, including rejecting
non-standard `NaN` and `Infinity` values. It does not itself implement JSON
Schema validation; use the endpoint's `response_format` when the provider
supports constrained decoding. The optional transforms require
`validate_json_output: true`.

`repair_invalid_json_escapes` only escapes backslashes that cannot begin a
valid JSON escape. `normalize_yxyx_bboxes` corrects inverted coordinate pairs;
it does not infer boxes, clamp their range, or assess localization quality.
`deduplicate_json_elements` removes exact duplicate objects from arrays named
`elements`.

## Configure recovery

The orchestrator owns the shared prompt and standard sampling settings. A
response that is empty, refused, rejected by a provider safety filter, or
invalid JSON can receive one semantic retry:

```yaml
orchestrator:
  inference:
    inference_prompts:
      - >-
        Describe the image as one JSON object matching the requested schema.
        Return JSON only.
    sampling:
      temperature: 0.3
      max_tokens: 2048
    retry_prompt: >-
      Try again. Return exactly one valid JSON object with a concise,
      visually grounded description and no surrounding prose.
    retry_sampling:
      temperature: 0.1
      max_tokens: 1024
```

`retry_sampling` accepts the standard OpenAI sampling fields supported by the
worker. It can be configured globally under `inference` or per stage. Changes
received from the orchestrator apply without restarting the worker.

Some providers need a different constrained-decoding body for the retry. Set
that locally so provider details and credentials never pass through the
orchestrator:

```yaml
worker:
  openai_compatible:
    retry_extra_body:
      response_format:
        type: "json_object"
```

Retry request fields override the endpoint's primary `extra_body`, except for
`model` and `messages`, which CaptionFlow reserves. Non-standard sampling such
as vLLM's `repetition_penalty` can also be placed in `retry_extra_body`.
