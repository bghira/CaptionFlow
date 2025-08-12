# captionflow

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/67eae1b1-7545-4ade-a0b1-31484ba57af9" />

```bash
$ pip install caption-flow
$ caption-flow orchestrator|worker|monitor
```

scalable, fault-tolerant **vllm-powered image captioning**. this "first round" focuses on a blazing fast websocket orchestrator plus lightweight gpu workers that batch requests through vllm. 

allows communities to caption large datasets together, as members can join or leave the cluster without issue.

**performance**: consumer 4090s often outpace h100s on smaller models (3b-7b) due to higher clock speeds and lower overhead. we've seen 150+ images/sec on a single 4090 with qwen2.5-vl-3b.

* **orchestrator**: hands out work in chunked shards, collects captions, checkpoints progress, and keeps simple stats. handles 10k+ chunks/sec on commodity hardware.
* **workers (vllm)**: connect to the orchestrator, stream in image samples, batch them, and generate 1..n captions per image using prompts supplied by the orchestrator.
* **dataworkers** (coming soon): separate non-gpu clients that fetch/preprocess images and feed them to the orchestrator, freeing gpu workers to focus purely on inference.
* **config-driven**: all components read yaml config; flags can override.
* **tui monitor (optional)**: a monitor client is wired into the cli; ship a `monitor` module to enable it.

> no conda. just `venv` + `pip`.

---

## install

```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .  # installs the `caption-flow` command
# or: pip install -e git+ssh://git@github.com/bghira/captionflow
```

## quickstart (single box)

1. copy + edit the sample configs

```bash
cp examples/orchestrator.yaml config/orchestrator.yaml
cp examples/worker.yaml config/worker.yaml
cp examples/monitor.yaml config/monitor.yaml
```

set a unique shared token in both `config/orchestrator.yaml` and `config/worker.yaml` (see `auth.worker_tokens` in the orchestrator config and `worker.token` in the worker config). if you use private hugging face datasets/models, export `HUGGINGFACE_HUB_TOKEN` or use `hf auth login` (old style: `huggingface-cli login`) before starting workers.

2. start the orchestrator

```bash
caption-flow orchestrator
```

3. start one or more vllm workers

```bash
# gpu 0 on the same host
caption-flow worker --gpu-id 0

# your second gpu
caption-flow worker --gpu-id 1
```

4. (optional) start the monitor to check on status

```bash
caption-flow monitor
```

5. (optional) scan/fix chunks on disk if you had crashes or want to ensure you're actually receiving all captions correctly

```bash
caption-flow scan_chunks --data-dir ./caption_data --checkpoint-dir ./checkpoints --fix
```

---

## how it's wired

### orchestrator

```bash
$ caption-flow orchestrator --help
```

* **websocket server** (default `0.0.0.0:8765`) with four client roles: workers, dataworkers, monitors, and admin.
* **blazing fast**: handles 10,000+ chunks/sec, 100k+ concurrent connections. the bottleneck is always gpu inference, never the orchestrator.
* **dataset control**: the orchestrator centrally defines the dataset (`huggingface` or `local`) and version/name. it chunk-slices shards and assigns work.
* **worker config distribution**: most vLLM parameters, **inference prompts** and dataset details are all pushed from the orchestrator to workers when they join the cluster.
* **storage + checkpoints**: captions buffer to disk with periodic checkpoints. chunk state is tracked so restarts don't double-work.
* **auth**: token lists for `worker`, `dataworker`, `monitor`, and `admin` roles.

### GPU worker

```bash
$ caption-flow worker --help
```

* **uses vLLM for robust captioning**. hugging face transformers has major performance and reliability issues.
* **one process per gpu**. select the device with `--gpu-id` (or `worker.gpu_id` in yaml).
* **gets its marching orders** from the orchestrator: dataset info, model, prompts, batch size, and sampling.
* **resilient**: detects worker/orchestrator disconnections and handles job resumption without wasting resources.
* **batched generate()**: images can be resized down for consistent batching; each image can get multiple captions (one per prompt).
* **optimized for consumer gpus**: 4090s often beat h100s on 3b-7b models due to higher CPU boost clocks + lower kernel overhead = faster tokens/sec.

### dataworker

```bash
$ caption-flow dataworker --help
```

* **cpu-only image fetching**: separate clients that handle dataset i/o, image loading, and preprocessing
* **frees gpu workers**: gpu workers receive pre-loaded images, spending 100% of time on inference
* **scales horizontally**: spin up dozens of dataworkers on cpu nodes to saturate gpu throughput
* **smart prefetching**: predictive loading keeps gpu workers fed with zero wait time

### status monitoring

```bash
$ caption-flow monitor
```

this command reveals a terminal user interface with cluster statistics, progress, and recent log activity.

### dynamic reloading

```bash
$ caption-flow config-reload --new-config config.yaml
```

reload critical settings and adjust caption results on the fly; only reloads your caption models if you change vLLM settings.

---

## configuration

### config discovery order

for any component, the cli looks for config in this order (first match wins):

1. `--config /path/to/file.yaml`
2. `./<component>.yaml` (current directory)
3. `./config/<component>.yaml` (config subdirectory)
4. `~/.caption-flow/<component>.yaml`
5. `$XDG_CONFIG_HOME/caption-flow/<component>.yaml`
6. `/etc/caption-flow/<component>.yaml`
7. any `$XDG_CONFIG_DIRS` entries under `caption-flow/`
8. `./examples/<component>.yaml` (fallback)

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
    model: qwen/qwen2.5-vl-3b-instruct
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
      - { token: "example-worker-token", name: "example worker" }
    dataworker_tokens:
      - { token: "dataworker-token", name: "data feeder 1" }
    monitor_tokens:
      - { token: "letmein", name: "default monitor" }
    admin_tokens:
      - { token: "admin-secret-2024", name: "admin" }
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

## performance notes

**consumer gpus shine on smaller models where CPU bottlenecks arise**:
- 4090 @ 3b model: 8-15 images/sec
- 4090 @ 7b model: 8-12 images/sec
- h100 @ 3b model: 2-10 images/sec (lower CPU clocks)
- h100 @ 70b model: 2-10 images/sec (where the H100 belongs)

**orchestrator throughput**:
- 10,000+ chunks/sec on a typical Ryzen / Intel virtual machine
- 10,000+ concurrent websocket connections
- sub-millisecond chunk assignment latency
- bottleneck is always gpu inference, never the orchestrator

**scaling tips**:
- use smaller models (3b-7b) for first-pass captioning
- consumer gpus (4090/4080) offer best perf/$ on these models
- add dataworkers to prefetch and saturate gpu throughput
- run multiple workers per node (one per gpu)
- for B200, RTX 6000 Pro, and other fast GPUs, using two worker processes per GPU (two tokens required) can provide added GPU utilisation

---

## tls / certificates

use the built-in helpers during development:

```bash
# self-signed certs for quick local testing
caption-flow generate_cert --self-signed --domain localhost --output-dir ./certs

# inspect any certificate file
caption-flow inspect_cert ./certs/fullchain.pem
```

then point the orchestrator at the resulting cert/key (or run `--no-ssl` for dev-only ws://).

---

## tips & notes

* **multi-gpu**: start one worker process per gpu (set `--gpu-id` or `worker.gpu_id`).
* **throughput**: tune `vllm.batch_size` in the orchestrator config (or override with `--batch-size` at worker start). higher isn't always better; watch vram.
* **prompts**: add more strings under `vllm.inference_prompts` to get multiple captions per image; the worker returns only non-empty generations.
* **private hf**: if your dataset/model needs auth, export `HUGGINGFACE_HUB_TOKEN` before `caption-flow worker ...`.
* **self-signed ssl**: pass `--no-verify-ssl` to workers/monitors in dev.
* **recovery**: if you hard-crash mid-run and want to verify your database, `caption-flow scan_chunks --fix` can help but is basically never needed.

---

## architecture

```
                                   ┌──────────────┐
                                   │              │
┌─────────────┐     websocket      │              │      ┌──────────────┐
│ gpu worker  │◄───────────────────┤              ├─────►│arrow/parquet │
└─────────────┘                    │              │      │   storage    │
                                   │ orchestrator │      └──────────────┘
┌─────────────┐                    │              │
│ gpu worker  │◄───────────────────┤   10k+       │      ┌──────────────┐
└─────────────┘                    │ chunks/sec   ├─────►│ checkpoints  │
                                   │              │      └──────────────┘
┌─────────────┐                    │              │
│ dataworker  │◄───────────────────┤              │
└─────────────┘                    │              │
                                   │              │
┌─────────────┐                    │              │
│   monitor   │◄───────────────────┤              │
└─────────────┘                    └──────────────┘
```

## storage schema

### captions.parquet
- `job_id`: unique job identifier
- `dataset`: dataset name
- `shard`: shard identifier
- `item_key`: item within shard
- `caption`: generated caption text
- `contributor_id`: worker who generated it
- `timestamp`: generation time
- `quality_score`: optional quality metric

### jobs.parquet
- `job_id`: unique identifier
- `dataset`: dataset name
- `shard`: shard identifier
- `status`: pending/processing/completed/failed
- `assigned_to`: worker id
- `timestamp`: status change time

### contributors.parquet
- `contributor_id`: unique identifier
- `name`: display name
- `total_captions`: lifetime count
- `trust_level`: quality tier (0-5)

## development

```bash
# install with dev dependencies
pip install -e ".[dev]"

# run tests
pytest

# format code
black src/
ruff --fix src/

# type checking
mypy src/
```

## community contribution

to contribute compute:

1. install caption-flow: `pip install caption-flow`
2. get a worker token from the project maintainer
3. run: `caption-flow worker --server wss://project.domain.com:8765 --token YOUR_TOKEN`

your contributions will be tracked and attributed in the final dataset!

## roadmap

in no particular order:

* video captioning
* web interface
* automatic huggingface hub dataset continuous exports
* sequence-parallel inference for large vision models
* discord interface
* more in-depth integration for non-wds datasets (local folder captioning)
* support chaining of workflows, for 2nd/3rd pass after use of initial tag model etc
* distributed orchestrator clustering for planet-scale captioning
* validation of community caption results randomly to boost the trust levels of contributors

prs welcome. keep it simple and fast.

## license

AGPLv3
