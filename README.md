# CaptionFlow

<!-- [![Tests](https://github.com/bghira/CaptionFlow/workflows/tests/badge.svg)](https://github.com/bghira/CaptionFlow/actions/workflows/tests.yml) -->
[![codecov](https://codecov.io/github/bghira/CaptionFlow/graph/badge.svg?token=PRAQPNGYAS)](https://codecov.io/github/bghira/CaptionFlow)
[![PyPI version](https://badge.fury.io/py/caption-flow.svg)](https://badge.fury.io/py/caption-flow)

scalable, fault-tolerant **vLLM-powered image captioning**.

a fast websocket-based orchestrator paired with lightweight gpu workers achieves exceptional performance for batched requests through vLLM.

* **orchestrator**: hands out work in chunked shards, collects captions, checkpoints progress, and keeps simple stats.
* **workers (vLLM)**: connect to the orchestrator, stream in image samples, batch them, and generate 1..N captions per image using prompts supplied by the orchestrator.
* **config-driven**: all components read YAML config; flags can override.

> no conda. just `venv` + `pip`.

---

## install

```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .  # installs the `caption-flow` command
```

## quickstart (single box)

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
  --format [jsonl|json|csv|txt|huggingface_hub|all] Export format (default: jsonl)
```

* **jsonl**: create JSON line file in the specified `--output` path
* **csv**: exports CSV-compatible data columns to the `--output` path containing incomplete metadata
* **json**: creates a `.json` file for each sample inside the `--output` subdirectory containing **complete** metadata; useful for webdatasets
* **txt**: creates `.txt` file for each sample inside the `--output` subdirectory containing ONLY captions
* **huggingface_hub**: creates a dataset on Hugging Face Hub, possibly `--private` and `--nsfw` where necessary
* **all**: creates all export formats in a specified `--output` directory

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

---

## dataset formats

* huggingface hub or local based URL list datasets that are compatible with the datasets library
* webdatasets shards containing full image data; also can be hosted on the hub
* local folder filled with images; orchestrator will serve the data to workers

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

1. Install caption-flow: `pip install caption-flow`
2. Get a worker token from the project maintainer
3. Run: `caption-flow worker --server wss://project.domain.com:8765 --token YOUR_TOKEN`

Your contributions will be tracked and attributed in the final dataset!

## License

AGPLv3
