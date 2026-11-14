# CaptionFlow

<!-- [![Tests](https://github.com/bghira/CaptionFlow/workflows/tests/badge.svg)](https://github.com/bghira/CaptionFlow/actions/workflows/tests.yml) -->
[![codecov](https://codecov.io/github/bghira/CaptionFlow/graph/badge.svg?token=PRAQPNGYAS)](https://codecov.io/github/bghira/CaptionFlow)
[![PyPI version](https://badge.fury.io/py/caption-flow.svg)](https://badge.fury.io/py/caption-flow)

scalable, fault-tolerant image captioning with **vLLM or contributed API workers**.

a fast websocket-based orchestrator paired with lightweight gpu workers achieves exceptional performance for batched requests through vLLM.

CaptionFlow is also integrated in [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner), where it powers an end-to-end caption-to-training workflow through the SimpleTuner WebUI. Use CaptionFlow directly when you want a standalone distributed captioning system, or use it through SimpleTuner when you want dataset captioning, caption review/export, and model training managed as one suite.

* **orchestrator**: hands out work in chunked shards, collects captions, checkpoints progress, and keeps simple stats.
* **workers (vLLM or BYOK API)**: connect to the orchestrator, stream in image samples, and generate 1..N captions per image using prompts supplied by the orchestrator.
* **config-driven**: all components read YAML config; flags can override.

> no conda. just `venv` + `pip`.

---

## install from pypi

```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install --upgrade pip
pip install "caption-flow[vllm]"
```

The Linux GPU worker extra targets vLLM 0.33.x with PyTorch 2.14.0,
torchvision 0.29.0, and torchaudio 2.11.0 (as required by vLLM), plus
Transformers 5.16.1–5.20.x and Pillow 13. To upgrade an existing worker from a checkout,
run `pip install --upgrade -e ".[vllm]"` in its environment and restart it.

The tested worker stack uses NumPy 2.3.x on Python 3.12 and 2.4.x
on Python 3.13, and FastAPI below 0.137. These bounds are shared with the
base install so adding the worker extra remains compatible. CI resolves
the full Linux worker and development dependencies on both Python versions.
Lance 8.x, DuckDB 1.x, and Hugging Face Hub 1.x remain bounded to their
existing major versions pending separate migration testing.

An OpenAI-compatible API worker does not need the GPU dependencies:

```bash
pip install "caption-flow[openai]"
```

For an orchestrator or monitor-only install, use `pip install -e .`.
`.[captioning]` is an alias for `.[vllm]` for integrations such as
SimpleTuner. Terminal image previews remain optional because the current
`term-image` release requires an older Pillow major than CaptionFlow uses.

On Apple Silicon with macOS 14 or newer, install the MPS-compatible PyTorch
chain and image utilities with:

```bash
pip install -e ".[apple]"
```

For native vLLM Metal captioning, use **macOS 15+, Apple Silicon, and Python
3.12**. The current Metal wheels are published on
[GitHub releases](https://github.com/vllm-project/vllm-metal/releases), while
PyPI still carries the obsolete 0.1.0 plugin. From this checkout, install the
matching stable vLLM 0.33.0 core and Metal 0.33.0 wheels into your environment:

```bash
python -m pip install --upgrade -r requirements.metal.txt
python -m pip install -e ".[dev]"
python -m pip check
python examples/metal_smoke.py
```

The smoke test downloads `mlx-community/Qwen3-VL-2B-Instruct-4bit` on first
use and runs two images through CaptionFlow's native caption pipeline. It
checks that the Metal plugin and MLX GPU are active, and that the captions
identify the images' shapes and different colors. Pass `--model /path/to/model`
to use a local checkpoint. The test does not connect to an orchestrator.

The stable recipe uses the core wheel's PyTorch 2.14.0 pin. Upstream's installer
can select newer development builds and override Torch with nightly packages;
those are separate from this recipe. `.[apple]` alone does not install a vLLM
runtime, and should not be combined with the Linux `.[vllm]` extra.

For native macOS CPU vLLM instead, follow the
[official source-build instructions](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/?device=apple).

## quickstart (single box)

for a full caption-to-training workflow with a web interface, use the SimpleTuner WebUI integration. the standalone flow below is best when you want to run CaptionFlow directly, contribute workers to a cluster, or export captions for your own downstream training pipeline.

1. copy + edit the sample configs

```bash
cp examples/orchestrator/local_image_files.yaml my-orchestrator.yaml
cp examples/worker.yaml my-worker.yaml
cp examples/monitor.yaml my-monitor.yaml   # optional terminal interface
```

set a unique shared token in both `my-orchestrator.yaml` and `my-worker.yaml` (see `auth.worker_tokens` in the orchestrator config and `worker.token` in the worker config).

if you use private hugging face datasets/models, export `HUGGINGFACE_HUB_TOKEN` before starting anything.

2. start the orchestrator

```bash
caption-flow orchestrator --config my-orchestrator.yaml
```

3. start one or more vLLM workers

```bash
# gpu 0 on the same host
caption-flow worker --config my-worker.yaml --gpu-id 0

# your second GPU
caption-flow worker --config my-worker.yaml --gpu-id 1

# on a remote host
caption-flow worker --config my-worker.yaml --server ws://your.hostname.address:8765
```

4. (optional) start the monitor

```bash
caption-flow monitor --config my-monitor.yaml
```

5. export the data

```bash
% caption-flow export --help                                                                                                                                      
Usage: caption-flow export [OPTIONS]

  Export caption data to various formats.

Options:
  --format [jsonl|json|csv|txt|parquet|webshart|lance|huggingface_hub|all] Export format (default: jsonl)
```

* **jsonl**: create JSON line file in the specified `--output` path
* **csv**: exports CSV-compatible data columns to the `--output` path containing incomplete metadata
* **json**: creates a `.json` file for each sample inside the `--output` subdirectory containing **complete** metadata; useful for webdatasets
* **txt**: creates `.txt` file for each sample inside the `--output` subdirectory containing ONLY captions
* **webshart**: updates an **existing per-shard metadata `.json` file** by writing captions under the plural `captions` key. Strings, native JSON objects, and mixed lists are supported. JSON objects stored as text in Lance output columns are decoded during export, preserving nested values instead of writing Python dictionary representations. Native objects supplied through `Caption.outputs` are encoded as JSON only for Lance persistence. Requires webshart 0.7.x (included in the base dependencies), including native JSON caption support. For this format, pass `--output` as the path to the existing shard metadata JSON file when exporting one shard. If you export multiple shards, pass `--output` as a directory containing one existing `{shard_name}.json` file per shard.
* **huggingface_hub**: creates a dataset on Hugging Face Hub, possibly `--private` and `--nsfw` where necessary
* **all**: creates the directory/file-generating export formats in a specified `--output` directory. prefer a directory here; `webshart` is a special case that expects existing per-shard metadata `.json` files rather than creating new metadata files.

> note: `--output` paths ending in `.json` are treated specially for `webshart`. use a directory for normal multi-format exports and an existing shard metadata JSON file only when intentionally updating a `webshart` shard.

---

## how it’s wired

### orchestrator

* **websocket server** (default `0.0.0.0:8765`) with three client roles: workers, data-feeders, and admin.
* **dataset control**: the orchestrator centrally defines the dataset (`huggingface` or `local`) and version/name. it chunk-slices shards and assigns work.
* **data serving to remote workers**: local files can be captioned by remote workers that don't have access to the same files, automatically.
* **vLLM config broadcast**: model, tp size, dtype, max seq len, memory targets, batching, sampling params, and **inference prompts** are all pushed to workers; workers can apply many changes without a model reload.
* **storage + checkpoints**: captions buffer to disk with periodic checkpoints. chunk state is tracked so restarts don’t double-work.
* **auth**: token lists for `worker`, `monitor`, and `admin` roles.

### vLLM worker

* **one process per gpu**. select the device with `--gpu-id` (or `worker.gpu_id` in YAML).
* **gets its marching orders** from the orchestrator: dataset info, model, prompts, batch size, and sampling.
* **resilient**: detects disconnects, abandons the current chunk cleanly, clears queues, reconnects, and resumes.
* **batched generate()**: images are resized down for consistent batching; each image can get multiple captions (one per prompt).

### OpenAI-compatible BYOK worker

The API backend is a separate worker process, not an orchestrator integration.
Provider keys remain in local environment variables; the CaptionFlow server sees
only the normal worker token, submitted outputs, and non-secret capacity metrics.

```bash
cp examples/worker.openai-compatible.yaml my-api-worker.yaml
export ZAI_API_KEY="..."
caption-flow worker --config my-api-worker.yaml --openai-compatible
```

The worker sends OpenAI Chat Completions-compatible multimodal requests with an
inline image data URL. Every configured endpoint can override the shared stage
model with its own `model`, or use `model_map` when several orchestrator model
names need explicit aliases.

Capacity is discovered independently for each endpoint:

* it starts at `initial_concurrency`, adds a slot after sustained successful calls, and never exceeds `max_concurrency`;
* a 429 or recognizable concurrency/capacity response halves the active limit and honors `Retry-After` or common rate-reset headers;
* `requests_per_minute` adds conservative start-time pacing when a plan publishes an RPM limit;
* retries can spill onto another endpoint, so multiple accounts or providers form one local pool without exposing their credentials.

The sample is ready for Z.AI's OpenAI-compatible Coding Plan endpoint and
`glm-5.3-flash`. Change `base_url` and `model` to use any other compatible
multimodal service; no orchestrator changes are needed.

The orchestrator may use the backend-neutral `inference:` key for shared stages,
prompts, sampling, and output fields. Existing `vllm:` configurations are also
accepted by API workers for backward compatibility; endpoint-local `model`
values take precedence over the broadcast model name.

Direct vLLM and API workers share JSON validation, output transformations,
refusal detection, and semantic retries. Configure those under
`inference.output_processing`; use `response_format` and `retry_response_format`
for constrained decoding on either backend. See the
[structured JSON recipe](recipes/STRUCTURED_JSON.md) for examples and native
vLLM tuning details.

---

## dataset formats

* huggingface hub or local based URL list datasets that are compatible with the datasets library
* huggingface hub datasets that are simple containers of raw image files
* webdatasets shards containing full image data; also can be hosted on the hub
* local folder filled with images; orchestrator will serve the data to workers

For indexed Webshart reads, set `orchestrator.webshart_prefetch_shards: 1` to
download the next shard in the background while captioning the current one.
The default is `0` (disabled). This works with both direct vLLM and API workers,
does not move the sample cursor, and is ignored for explicit remote-range reads.
Size `shard_cache_gb` to hold the current shard plus the lookahead. Prefetch
retries are spaced by `webshart_cache_retry_interval` (30 seconds by default);
a prefetch failure never marks samples processed or prevents reading the
current cached shard.

## configuration path

### config discovery order

for any component, the CLI looks for config in this order (first match wins):

1. `--config /path/to/file.yaml`
2. `./<component>.yaml` (current directory)
3. `~/.caption-flow/<component>.yaml`
4. `$XDG_CONFIG_HOME/caption-flow/<component>.yaml`
5. `/etc/caption-flow/<component>.yaml`
6. any `$XDG_CONFIG_DIRS` entries under `caption-flow/`
7. `./examples/<component>.yaml` (fallback)

---

## tls / certificates

use the built-in helpers during development:

```bash
# self-signed certs for quick local testing
caption-flow generate_cert --self-signed --domain localhost --output-dir ./certs

# inspect any certificate file
caption-flow inspect_cert ./certs/fullchain.pem
```

then point the orchestrator at the resulting cert/key (or run `--no-ssl` for dev-only ws\://).

---

## tips & notes

* **multi-gpu**: start one worker process per gpu (set `--gpu-id` or `worker.gpu_id`).
* **throughput**: tune `vllm.batch_size` in the orchestrator config (or override with `--batch-size` at worker start). higher isn’t always better; watch VRAM.
* **prompts**: add more strings under `vllm.inference_prompts` to get multiple captions per image; the worker returns only non-empty generations.
* **private HF**: if your dataset/model needs auth, export `HUGGINGFACE_HUB_TOKEN` before `caption-flow worker ...`.
* **self-signed ssl**: pass `--no-verify-ssl` to workers/monitors in dev.
* **recovery**: if you hard-crash mid-run, `caption-flow scan_chunks --fix` can reset abandoned chunks so the orchestrator can reissue them cleanly.

---

## roadmap

* hot config reload via the admin websocket path.
* dedicated data-feeder clients (separate from gpu workers) that push samples into the orchestrator.
* richer monitor TUI.

PRs welcome. keep it simple and fast.

## architecture

```
┌─────────────┐     WebSocket      ┌─────────────┐
│   Worker    │◄──────────────────►│             │
│             │                    │             │     ┌──────────────┐
│             │◄───────────────────│             │────►│Arrow/Parquet │
└─────────────┘   HTTP (img data)  │ Orchestrator│     │   Storage    │
                                   │             │     └──────────────┘
┌─────────────┐                    │             │
│   Worker    │◄──────────────────►│             │
│             │                    │             │
│             │◄───────────────────│             │
└─────────────┘   HTTP (img data)  └─────────────┘
                                           ▲
┌─────────────┐                           │
│   Monitor   │◄──────────────────────────┘
└─────────────┘
```

## Community Clusters

To contribute compute to a cluster:

1. Install caption-flow: `pip install "caption-flow[vllm]"` for a GPU worker or `pip install "caption-flow[openai]"` for a BYOK API worker
2. Get a worker token from the project maintainer
3. Run: `caption-flow worker --server wss://project.domain.com:8765 --token YOUR_TOKEN`

Your contributions will be tracked and attributed in the final dataset!

## License

AGPLv3
