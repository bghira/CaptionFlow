# CaptionFlow

Self-contained distributed community captioning system with WebSocket orchestration and Arrow/Parquet storage.

## Features

- **Zero external dependencies**: No Redis, no database - just WebSocket and Arrow files
- **Community-powered**: Contributors run workers on their GPUs
- **Attribution tracking**: Every caption credits its contributor
- **Quality tiers**: Trust levels based on contribution history
- **Real-time monitoring**: TUI dashboard for progress and stats
- **Automatic failover**: Jobs requeue on worker disconnect

## Installation

```bash
pip install caption-flow
```

## Quick Start

### 1. Generate SSL Certificates

For production with a domain:
```bash
# Using Let's Encrypt (requires port 80 access)
caption-flow generate-cert --domain your.domain.com --email admin@domain.com

# This creates:
# - /etc/letsencrypt/live/your.domain.com/fullchain.pem
# - /etc/letsencrypt/live/your.domain.com/privkey.pem
```

For testing (self-signed):
```bash
# Generate self-signed certificate
caption-flow generate-cert --self-signed --output-dir ./certs

# This creates:
# - ./certs/cert.pem
# - ./certs/key.pem
```

### 2. Start Orchestrator

```bash
# Production with SSL
caption-flow orchestrator \
    --port 8765 \
    --cert /etc/letsencrypt/live/your.domain.com/fullchain.pem \
    --key /etc/letsencrypt/live/your.domain.com/privkey.pem \
    --data-dir ./caption_data

# Development without SSL
caption-flow orchestrator --port 8765 --no-ssl --data-dir ./caption_data
```

### 3. Connect Workers

```bash
# Production
caption-flow worker \
    --server wss://your.domain.com:8765 \
    --token YOUR_WORKER_TOKEN \
    --name "Worker-1"

# Development (self-signed cert)
caption-flow worker \
    --server wss://localhost:8765 \
    --token YOUR_WORKER_TOKEN \
    --no-verify-ssl
```

### 4. Monitor Progress

```bash
caption-flow monitor \
    --server wss://your.domain.com:8765 \
    --token YOUR_ADMIN_TOKEN
```

## Configuration

Create a `config.yaml`:

```yaml
orchestrator:
  host: 0.0.0.0
  port: 8765
  ssl:
    cert: /path/to/cert.pem
    key: /path/to/key.pem
  storage:
    data_dir: ./caption_data
    checkpoint_interval: 1000
  auth:
    worker_tokens:
      - token: "worker-token-1"
        name: "GPU-Server-1"
      - token: "worker-token-2"
        name: "Community-Worker"
    admin_tokens:
      - "admin-token-1"

worker:
  batch_size: 32
  model: "llava-v1.6-34b"
  max_retries: 3
  heartbeat_interval: 30

monitor:
  refresh_rate: 1.0
  show_contributors: true
  show_quality_metrics: true
```

## Architecture

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
