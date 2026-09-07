# GLM-5.3-Flash with vLLM

This recipe serves GLM-5.3-Flash from local weights with vLLM and connects it
to CaptionFlow through the OpenAI-compatible worker. No GLM-specific
CaptionFlow code is required once the vLLM Chat Completions endpoint is
available.

## Tested configuration

- 4 NVIDIA H100 80 GB GPUs
- `wtdcode/GLM-5.3-Flash-AWQ-W4A16`, revision
  `abd7b07719111f137e1de8a0c1b7e01c11b74d1a`
- `vllm/vllm-openai:glm53-flash`
- vLLM `0.1.dev20051+g487ecf187`
- tensor parallelism across all four GPUs
- an 8,192-token model context
- up to 64 concurrent vLLM sequences

The tested checkpoint occupies approximately 177.7 GiB on disk and about
41.7 GiB of model memory per GPU with tensor parallel size 4. It quantizes the
routed MoE experts to W4A16 while retaining the vision encoder and several
sensitive components in BF16.

The official FP8 checkpoint did not fit this 4x80 GB setup: its estimated
runtime requirement was approximately 386 GiB, exceeding the 320 GB of total
VRAM. Use the W4A16 checkpoint for this hardware configuration. This recipe
does not claim that W4A16 and FP8 have identical output quality.

## Download the checkpoint

Allow at least 180 GiB of free local storage:

```bash
export GLM_MODEL_DIR=/models/GLM-5.3-Flash-AWQ-W4A16

hf download wtdcode/GLM-5.3-Flash-AWQ-W4A16 \
  --revision abd7b07719111f137e1de8a0c1b7e01c11b74d1a \
  --local-dir "$GLM_MODEL_DIR"
```

Pinning the revision prevents a later checkpoint update from silently changing
the deployment.

## Start vLLM

Use a vLLM build that recognizes `Glm5NextForConditionalGeneration`. The
tested image reports vLLM version `0.1.dev20051+g487ecf187`; an older stable
image may reject the architecture.

Inside that image or environment, start the server with:

```bash
vllm serve "$GLM_MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.94 \
  --reasoning-parser glm45 \
  --no-enable-flashinfer-autotune
```

Bind to `0.0.0.0` only when the endpoint must be reachable from another host,
and protect it with appropriate network controls or authentication.

The first startup can take roughly five minutes while vLLM loads the weights,
compiles kernels, profiles memory, and captures CUDA graphs. The tested run
loaded the weights in about 80 seconds and completed the full engine startup in
about five minutes.

`--no-enable-flashinfer-autotune` skips an additional startup autotuning phase.
It is part of the known-good invocation, but it is not a CaptionFlow
requirement. Benchmark before removing it or changing the vLLM kernel setup.

Confirm that the server is ready:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

The model list should contain `glm-5.3-flash`.

## Configure the CaptionFlow worker

Install CaptionFlow's lightweight API worker dependencies in the environment
that will run the worker:

```bash
pip install "caption-flow[openai]"
```

Create a worker configuration such as `worker.glm-5.3-flash.yaml`:

```yaml
worker:
  server: "ws://orchestrator.example:8765"
  token: "replace-with-captionflow-worker-token"
  name: "vllm-glm-5.3-flash"
  when_finished: "stay_connected"
  batch_image_processing: false

  openai_compatible:
    batch_size: 64
    include_image: true
    image_detail: "auto"
    image_format: "jpeg"
    image_quality: 90
    max_image_dimension: 1024
    system_prompt: >-
      You are a precise, minimalist image captioner. Write short factual
      descriptions using only visually supported details. Return only the
      requested caption.

    endpoints:
      - name: "local-vllm-glm-5.3-flash"
        base_url: "http://127.0.0.1:8000/v1"
        api_key_env: "VLLM_API_KEY"
        model: "glm-5.3-flash"
        initial_concurrency: 64
        max_concurrency: 64
        probe_after_successes: 8
        timeout_seconds: 300
        max_retries: 3
        extra_body:
          chat_template_kwargs:
            reasoning_effort: "low"
```

The OpenAI client requires a non-empty API key even when the local vLLM server
does not enforce authentication. Give it a non-secret placeholder unless the
server was started with a real API key:

```bash
export VLLM_API_KEY=local-vllm
caption-flow worker \
  --config worker.glm-5.3-flash.yaml \
  --openai-compatible
```

The worker token must match one of the orchestrator's `auth.worker_tokens`.
When vLLM and the CaptionFlow worker run in different containers, replace
`127.0.0.1` with a private, reachable vLLM address.

## Configure caption generation

The orchestrator owns the prompt and sampling configuration. This is the
configuration used for concise captions:

```yaml
orchestrator:
  inference:
    model: "glm-5.3-flash"
    batch_size: 8
    sampling:
      temperature: 0.3
      top_p: 0.9
      max_tokens: 3000
    inference_prompts:
      - >-
        Write one concise, minimalist standalone caption for this photograph in
        one or two short sentences, preferably 20 to 45 words. Identify only
        the main subject, action, and essential setting or visual context. Omit
        exhaustive detail, interpretation, and speculation. Do not use labels
        or bullet points. Return only the caption.
```

The high token ceiling prevents reasoning from being truncated. The prompt and
`reasoning_effort: low` still keep the returned caption short, while
`--reasoning-parser glm45` separates model reasoning from the final content.

Hosted providers may reject an image in a safety prefilter before GLM sees the
captioning prompt. When the source dataset contains a usable text description,
the OpenAI-compatible worker can make one text-only fallback attempt:

```yaml
orchestrator:
  inference:
    retry_prompt: >-
      Rewrite the following source description as one concise, neutral,
      standalone caption. Preserve only concrete visual details, omit
      speculation and sensitive details, and return only the caption.
      Source description: {column:captions}
    retry_without_image: true
```

The fallback runs only when every primary output for an item is empty, a known
refusal, or a provider content-policy error. Ordinary authentication, network,
and malformed-request failures remain visible instead of being converted into
captions. Replace `captions` in `{column:captions}` with the source metadata
column available in the dataset.

## Operational notes

- Keep the served model name and the endpoint model name identical. The tested
  alias is exactly `glm-5.3-flash`.
- `batch_size: 64` and `initial_concurrency: 64` keep the local endpoint full.
  On the tested 4x H100 system, 128 matched images took 20.1 seconds at this
  setting versus 71.4 seconds when the endpoint began at concurrency 8. Use a
  conservative initial value for rate-limited hosted providers.
- `max_image_dimension: 1024` bounds vision-token use and request size. Without
  normalization, 40 of 128 high-resolution test images exceeded the shared
  8,192-token context window.
- Reduce `initial_concurrency`, `max_concurrency`, or `max_num_seqs` if the
  workload encounters memory pressure. Increase them only after measuring
  throughput and latency on representative images.
- A warning that no MLA prefill backend supports the model was non-fatal in the
  tested vLLM build. Sparse MLA used its supported top-k path and captioning
  continued normally.
- Do not judge quantization quality from unmatched worker outputs. A useful
  comparison requires the same images, prompt, sampling settings, and decoding
  path on W4A16 and a higher-precision checkpoint.
