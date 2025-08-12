# CaptionFlow

scalable, fault-tolerant **vLLM-powered image captioning**. this "first round" focuses on a fast websocket orchestrator plus lightweight gpu workers that batch requests through vLLM.

* **orchestrator**: hands out work in chunked shards, collects captions, checkpoints progress, and keeps simple stats.
* **workers (vLLM)**: connect to the orchestrator, stream in image samples, batch them, and generate 1..N captions per image using prompts supplied by the orchestrator.
* **config-driven**: all components read YAML config; flags can override.
* **tui monitor (optional)**: a monitor client is wired into the CLI; ship a `monitor` module to enable it.

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
cp orchestrator.yaml my-orchestrator.yaml
cp worker.yaml my-worker.yaml
cp monitor.yaml my-monitor.yaml   # optional; requires a monitor module
```

set a unique shared token in both `my-orchestrator.yaml` and `my-worker.yaml` (see `auth.worker_tokens` in the orchestrator config and `worker.token` in the worker config). if you use private hugging face datasets/models, export `HUGGINGFACE_HUB_TOKEN` before starting workers.

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
```

4. (optional) start the monitor

```bash
caption-flow monitor --config my-monitor.yaml
```

5. (optional) scan/fix chunks on disk if you had crashes

```bash
caption-flow scan_chunks --data-dir ./caption_data --checkpoint-dir ./checkpoints --fix
```

---

## how it’s wired

### orchestrator

* **websocket server** (default `0.0.0.0:8765`) with three client roles: workers, data-feeders, and admin.
* **dataset control**: the orchestrator centrally defines the dataset (`huggingface` or `local`) and version/name. it chunk-slices shards and assigns work.
* **vLLM config broadcast**: model, tp size, dtype, max seq len, memory targets, batching, sampling params, and **inference prompts** are all pushed to workers; workers can apply many changes without a model reload.
* **storage + checkpoints**: captions buffer to disk with periodic checkpoints. chunk state is tracked so restarts don’t double-work.
* **auth**: token lists for `worker`, `monitor`, and `admin` roles.

start flags you’ll likely use:

```text
--config PATH                # yaml config for the orchestrator
--port INT, --host STR       # bind controls
--data-dir PATH              # overrides storage.data_dir
--cert PATH, --key PATH      # enable TLS (or use --no-ssl for ws:// in dev)
--vllm                       # use the vLLM-style orchestrator (webdataset/hf)
```

### vLLM worker

* **one process per gpu**. select the device with `--gpu-id` (or `worker.gpu_id` in YAML).
* **gets its marching orders** from the orchestrator: dataset info, model, prompts, batch size, and sampling.
* **resilient**: detects disconnects, abandons the current chunk cleanly, clears queues, reconnects, and resumes.
* **batched generate()**: images are resized down for consistent batching; each image can get multiple captions (one per prompt).

start flags you’ll likely use:

```text
--config PATH                 # yaml for the worker
--server URL                  # ws(s)://host:port
--token STR                   # must match an allowed worker token on the orchestrator
--name STR                    # display name
--batch-size INT              # override vLLM batch size
--vllm                        # use the vLLM worker implementation
--gpu-id INT                  # which gpu to use
--precision STR, --model STR  # optional overrides for dtype/model
--no-verify-ssl               # accept self-signed certs in dev
```

### (optional) monitor

* a CLI entry exists for a TUI monitor; wire in a `monitor` module to enable it. config lives in `monitor.yaml` or inside `orchestrator.yaml` under `monitor:`.

---

## configuration

### config discovery order

for any component, the CLI looks for config in this order (first match wins):

1. `--config /path/to/file.yaml`
2. `./<component>.yaml` (current directory)
3. `~/.caption-flow/<component>.yaml`
4. `$XDG_CONFIG_HOME/caption-flow/<component>.yaml`
5. `/etc/caption-flow/<component>.yaml`
6. any `$XDG_CONFIG_DIRS` entries under `caption-flow/`
7. `./examples/<component>.yaml` (fallback)

### orchestrator.yaml (highlights)

```yaml
orchestrator:
  host: 0.0.0.0
  port: 8765
  # ssl:
  #   cert: /path/fullchain.pem
  #   key:  /path/privkey.pem

  dataset:
    type: huggingface   # or "local"
    path: <hf-dataset-or-local-path>
    name: <logical-name>
    version: "1.0"

  vllm:
    model: Qwen/Qwen2.5-VL-3B-Instruct
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: float16
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt: { image: 1 }

    batch_size: 8

    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
      repetition_penalty: 1.05
      skip_special_tokens: true
      stop: ["<|end|>", "<|endoftext|>", "<|im_end|>"]

    inference_prompts:
      - "describe this image in detail"
      - "provide a comprehensive description of the visual content"
      - "what are the key elements in this image?"

  storage:
    data_dir: ./caption_data
    checkpoint_dir: ./checkpoints
    caption_buffer_size: 100
    checkpoint_interval: 1000

  # chunking/queueing
  chunk_size: 1000
  chunks_per_request: 2
  chunk_buffer_multiplier: 3
  min_chunk_buffer: 10

  auth:
    worker_tokens:
      - { token: "example-worker-token", name: "Example Worker" }
    monitor_tokens:
      - { token: "letmein", name: "Default monitor" }
    admin_tokens:
      - { token: "admin-secret-2024", name: "Admin" }
```

### worker.yaml (highlights)

```yaml
worker:
  server: ws://localhost:8765   # use wss:// in prod
  token: example-worker-token
  name: local-gpu
  gpu_id: 0
  vllm: true

  # local queues
  readahead_size: 256
  inference_queue_size: 128
```

### monitor.yaml (optional)

```yaml
monitor:
  server: ws://localhost:8765
  token: letmein
  refresh_rate: 1.0
  show_contributors: true
  show_quality_metrics: true
  max_activity_items: 20
  show_chunk_progress: true
  show_worker_queues: true
  show_throughput_graph: true
```

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
└─────────────┘                    │             │     ┌──────────────┐
                                   │ Orchestrator│────►│Arrow/Parquet │
┌─────────────┐                    │             │     │   Storage    │
│   Worker    │◄──────────────────►│             │     └──────────────┘
└─────────────┘                    └─────────────┘
                                           ▲
┌─────────────┐                           │
│   Monitor   │◄──────────────────────────┘
└─────────────┘
```

## Storage Schema

### captions.parquet
- `job_id`: Unique job identifier
- `dataset`: Dataset name
- `shard`: Shard identifier
- `item_key`: Item within shard
- `caption`: Generated caption text
- `contributor_id`: Worker who generated it
- `timestamp`: Generation time
- `quality_score`: Optional quality metric

### jobs.parquet
- `job_id`: Unique identifier
- `dataset`: Dataset name
- `shard`: Shard identifier
- `status`: pending/processing/completed/failed
- `assigned_to`: Worker ID
- `timestamp`: Status change time

### contributors.parquet
- `contributor_id`: Unique identifier
- `name`: Display name
- `total_captions`: Lifetime count
- `trust_level`: Quality tier (0-5)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff --fix src/

# Type checking
mypy src/
```

## Community Contribution

To contribute compute:

1. Install caption-flow: `pip install caption-flow`
2. Get a worker token from the project maintainer
3. Run: `caption-flow worker --server wss://project.domain.com:8765 --token YOUR_TOKEN`

Your contributions will be tracked and attributed in the final dataset!

## License

MIT
