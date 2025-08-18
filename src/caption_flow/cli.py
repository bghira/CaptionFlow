"""Command-line interface for CaptionFlow with smart configuration handling."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from datetime import datetime

from .orchestrator import Orchestrator
from .monitor import Monitor
from .utils.certificates import CertificateManager

console = Console()


class ConfigManager:
    """Smart configuration discovery and management following XDG Base Directory spec."""

    CONFIG_NAMES = {
        "orchestrator": "orchestrator.yaml",
        "worker": "worker.yaml",
        "monitor": "monitor.yaml",
    }

    @classmethod
    def get_xdg_config_home(cls) -> Path:
        """Get XDG_CONFIG_HOME or default."""
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config)
        return Path.home() / ".config"

    @classmethod
    def get_xdg_config_dirs(cls) -> List[Path]:
        """Get XDG_CONFIG_DIRS or defaults."""
        xdg_dirs = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")
        return [Path(d) for d in xdg_dirs]

    @classmethod
    def find_config(
        cls, component: str, explicit_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find and load configuration for a component.

        Search order:
        1. Explicit path if provided
        2. Current directory
        3. ~/.caption-flow/<component_config>.yaml
        4. $XDG_CONFIG_HOME/caption-flow/<component_config>.yaml
        5. /etc/caption-flow/<component_config>.yaml (system-wide)
        6. $XDG_CONFIG_DIRS/caption-flow/<component_config>.yaml
        7. ./examples/<component_config>.yaml (fallback)
        """
        config_name = cls.CONFIG_NAMES.get(component, "config.yaml")

        # If explicit path provided, use only that
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                console.print(f"[dim]Using config: {path}[/dim]")
                return cls.load_yaml(path)
            console.print(f"[yellow]Config not found: {path}[/yellow]")
            return None

        # Search paths in order
        search_paths = [
            Path.cwd() / config_name,  # Current directory
            Path.cwd() / "config" / config_name,  # Current directory / config subdir
            Path.home() / ".caption-flow" / config_name,  # Home directory
            cls.get_xdg_config_home() / "caption-flow" / config_name,  # XDG config home
            Path("/etc/caption-flow") / config_name,  # System-wide
        ]

        # Add XDG config dirs
        for xdg_dir in cls.get_xdg_config_dirs():
            search_paths.append(xdg_dir / "caption-flow" / config_name)

        # Fallback to examples
        search_paths.append(Path("examples") / config_name)

        # Try each path
        for path in search_paths:
            if path.exists():
                console.print(f"[dim]Found config: {path}[/dim]")
                return cls.load_yaml(path)

        return None

    @classmethod
    def load_yaml(cls, path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse YAML config file."""
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")
            return None

    @classmethod
    def merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge override config into base config."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls.merge_configs(result[key], value)
            else:
                result[key] = value

        return result


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler, including timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                show_time=True,  # Enables timestamp in RichHandler output
            )
        ],
    )


def apply_cli_overrides(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Apply CLI arguments as overrides to config, filtering out None values."""
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    return ConfigManager.merge_configs(config, overrides)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, verbose: bool):
    """CaptionFlow - Distributed community captioning system."""
    setup_logging(verbose)
    ctx.obj = {"verbose": verbose}


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--port", type=int, help="WebSocket server port")
@click.option("--host", help="Bind address")
@click.option("--data-dir", help="Storage directory")
@click.option("--cert", help="SSL certificate path")
@click.option("--key", help="SSL key path")
@click.option("--no-ssl", is_flag=True, help="Disable SSL (development only)")
@click.option("--vllm", is_flag=True, help="Use vLLM orchestrator for WebDataset/HF datasets")
@click.pass_context
def orchestrator(ctx, config: Optional[str], **kwargs):
    """Start the orchestrator server."""
    # Load configuration
    base_config = ConfigManager.find_config("orchestrator", config) or {}

    # Extract orchestrator section if it exists
    if "orchestrator" in base_config:
        config_data = base_config["orchestrator"]
    else:
        config_data = base_config

    # Apply CLI overrides
    if kwargs.get("port"):
        config_data["port"] = kwargs["port"]
    if kwargs.get("host"):
        config_data["host"] = kwargs["host"]
    if kwargs.get("data_dir"):
        config_data.setdefault("storage", {})["data_dir"] = kwargs["data_dir"]

    # Handle SSL configuration
    if not kwargs.get("no_ssl"):
        if kwargs.get("cert") and kwargs.get("key"):
            config_data.setdefault("ssl", {})
            config_data["ssl"]["cert"] = kwargs["cert"]
            config_data["ssl"]["key"] = kwargs["key"]
        elif not config_data.get("ssl"):
            console.print(
                "[yellow]Warning: Running without SSL. Use --cert and --key for production.[/yellow]"
            )

    if kwargs.get("vllm") and "vllm" not in config_data:
        raise ValueError("Must provide vLLM config.")

    orchestrator_instance = Orchestrator(config_data)

    try:
        asyncio.run(orchestrator_instance.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down orchestrator...[/yellow]")
        asyncio.run(orchestrator_instance.shutdown())


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--server", help="Orchestrator WebSocket URL")
@click.option("--token", help="Worker authentication token")
@click.option("--name", help="Worker display name")
@click.option("--batch-size", type=int, help="Inference batch size")
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
@click.option("--vllm", is_flag=True, help="Use vLLM worker for GPU inference")
@click.option("--gpu-id", type=int, help="GPU device ID (for vLLM)")
@click.option("--precision", help="Model precision (for vLLM)")
@click.option("--model", help="Model name (for vLLM)")
@click.pass_context
def worker(ctx, config: Optional[str], **kwargs):
    """Start a worker node."""
    # Load configuration
    base_config = ConfigManager.find_config("worker", config) or {}

    # Extract worker section if it exists
    if "worker" in base_config:
        config_data = base_config["worker"]
    else:
        config_data = base_config

    # Apply CLI overrides (only non-None values)
    for key in ["server", "token", "name", "batch_size", "gpu_id", "precision", "model"]:
        if kwargs.get(key) is not None:
            config_data[key] = kwargs[key]

    if kwargs.get("no_verify_ssl"):
        config_data["verify_ssl"] = False

    # Validate required fields
    if not config_data.get("server"):
        console.print("[red]Error: --server required (or set in config)[/red]")
        sys.exit(1)
    if not config_data.get("token"):
        console.print("[red]Error: --token required (or set in config)[/red]")
        sys.exit(1)

    # Choose worker type
    if kwargs.get("vllm") or config_data.get("vllm"):
        from .workers.caption import CaptionWorker

        worker_instance = CaptionWorker(config_data)
    else:
        raise ValueError(f"Not sure how to handle worker for {config_data.get('type')} type setup.")

    try:
        asyncio.run(worker_instance.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down worker...[/yellow]")
        asyncio.run(worker_instance.shutdown())


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--server", help="Orchestrator WebSocket URL")
@click.option("--token", help="Authentication token")
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.pass_context
def monitor(
    ctx,
    config: Optional[str],
    server: Optional[str],
    token: Optional[str],
    no_verify_ssl: bool,
    debug: bool,
):
    """Start the monitoring TUI."""

    # Enable debug logging if requested
    if debug:
        setup_logging(verbose=True)
        console.print("[yellow]Debug mode enabled[/yellow]")

    # Load configuration
    base_config = ConfigManager.find_config("monitor", config)

    if not base_config:
        # Try to find monitor config in orchestrator config as fallback
        orch_config = ConfigManager.find_config("orchestrator")
        if orch_config and "monitor" in orch_config:
            base_config = {"monitor": orch_config["monitor"]}
            console.print("[dim]Using monitor config from orchestrator.yaml[/dim]")
        else:
            base_config = {}
            if not server or not token:
                console.print("[yellow]No monitor config found, using CLI args[/yellow]")

    # Handle different config structures
    # Case 1: Config has top-level 'monitor' section
    if "monitor" in base_config:
        config_data = base_config["monitor"]
    # Case 2: Config IS the monitor config (no wrapper)
    else:
        config_data = base_config

    # Apply CLI overrides (CLI always wins)
    if server:
        config_data["server"] = server
    if token:
        config_data["token"] = token
    if no_verify_ssl:
        config_data["verify_ssl"] = False

    # Debug output
    if debug:
        console.print("\n[cyan]Final monitor configuration:[/cyan]")
        console.print(f"  Server: {config_data.get('server', 'NOT SET')}")
        console.print(
            f"  Token: {'***' + config_data.get('token', '')[-4:] if config_data.get('token') else 'NOT SET'}"
        )
        console.print(f"  Verify SSL: {config_data.get('verify_ssl', True)}")
        console.print()

    # Validate required fields
    if not config_data.get("server"):
        console.print("[red]Error: --server required (or set 'server' in monitor.yaml)[/red]")
        console.print("\n[dim]Example monitor.yaml:[/dim]")
        console.print("server: wss://localhost:8765")
        console.print("token: your-token-here")
        sys.exit(1)

    if not config_data.get("token"):
        console.print("[red]Error: --token required (or set 'token' in monitor.yaml)[/red]")
        console.print("\n[dim]Example monitor.yaml:[/dim]")
        console.print("server: wss://localhost:8765")
        console.print("token: your-token-here")
        sys.exit(1)

    # Set defaults for optional settings
    config_data.setdefault("refresh_interval", 1.0)
    config_data.setdefault("show_inactive_workers", False)
    config_data.setdefault("max_log_lines", 100)

    # Create and start monitor
    try:
        monitor_instance = Monitor(config_data)

        if debug:
            console.print("[green]Starting monitor...[/green]")
            console.print(f"[dim]Connecting to: {config_data['server']}[/dim]")
            sys.exit(1)

        asyncio.run(monitor_instance.start())

    except KeyboardInterrupt:
        console.print("\n[yellow]Closing monitor...[/yellow]")
    except ConnectionRefusedError:
        console.print(f"\n[red]Error: Cannot connect to {config_data['server']}[/red]")
        console.print("[yellow]Check that the orchestrator is running and accessible[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error starting monitor: {e}[/red]")
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--server", help="Orchestrator WebSocket URL")
@click.option("--token", help="Admin authentication token")
@click.option(
    "--new-config", type=click.Path(exists=True), required=True, help="New configuration file"
)
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
def reload_config(
    config: Optional[str],
    server: Optional[str],
    token: Optional[str],
    new_config: str,
    no_verify_ssl: bool,
):
    """Reload orchestrator configuration via admin connection."""
    import websockets
    import ssl

    # Load base config to get server/token if not provided via CLI
    if not server or not token:
        base_config = ConfigManager.find_config("orchestrator", config) or {}
        admin_config = base_config.get("admin", {})
        admin_tokens = base_config.get("orchestrator", {}).get("auth", {}).get("admin_tokens", [])
        has_admin_tokens = False
        if len(admin_tokens) > 0:
            has_admin_tokens = True
            first_admin_token = admin_tokens[0].get("token", None)
        # Do not print sensitive admin token to console.

        if not server:
            server = admin_config.get("server", "ws://localhost:8765")
        if not token:
            token = admin_config.get("token", None)
            if token is None and has_admin_tokens:
                # grab the first one, we'll just assume we're localhost.
                console.print("Using first admin token.")
                token = first_admin_token

    if not server:
        console.print("[red]Error: --server required (or set in config)[/red]")
        sys.exit(1)
    if not token:
        console.print("[red]Error: --token required (or set in config)[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading configuration from {new_config}...[/cyan]")

    # Load the new configuration
    new_cfg = ConfigManager.load_yaml(Path(new_config))
    if not new_cfg:
        console.print("[red]Failed to load configuration[/red]")
        sys.exit(1)

    # Setup SSL
    ssl_context = None
    if server.startswith("wss://"):
        if no_verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = ssl.create_default_context()

    async def send_reload():
        try:
            async with websockets.connect(server, ssl=ssl_context) as websocket:
                # Authenticate as admin
                await websocket.send(json.dumps({"token": token, "role": "admin"}))

                response = await websocket.recv()
                auth_response = json.loads(response)

                if "error" in auth_response:
                    console.print(f"[red]Authentication failed: {auth_response['error']}[/red]")
                    return False

                console.print("[green]✓ Authenticated as admin[/green]")

                # Send reload command
                await websocket.send(json.dumps({"type": "reload_config", "config": new_cfg}))

                response = await websocket.recv()
                reload_response = json.loads(response)

                if reload_response.get("type") == "reload_complete":
                    if "message" in reload_response and "No changes" in reload_response["message"]:
                        console.print(f"[yellow]{reload_response['message']}[/yellow]")
                    else:
                        console.print("[green]✓ Configuration reloaded successfully![/green]")

                        if "updated" in reload_response and reload_response["updated"]:
                            console.print("\n[cyan]Updated sections:[/cyan]")
                            for section in reload_response["updated"]:
                                console.print(f"  • {section}")

                        if "warnings" in reload_response and reload_response["warnings"]:
                            console.print("\n[yellow]Warnings:[/yellow]")
                            for warning in reload_response["warnings"]:
                                console.print(f"  ⚠ {warning}")

                    return True
                else:
                    error = reload_response.get("error", "Unknown error")
                    console.print(f"[red]Reload failed: {error} ({reload_response=})[/red]")
                    return False

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False

    success = asyncio.run(send_reload())
    if not success:
        sys.exit(1)


@main.command()
@click.option("--data-dir", default="./caption_data", help="Storage directory")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
@click.option("--fix", is_flag=True, help="Fix issues by resetting abandoned chunks")
@click.option("--verbose", is_flag=True, help="Show detailed information")
def scan_chunks(data_dir: str, checkpoint_dir: str, fix: bool, verbose: bool):
    """Scan for sparse or abandoned chunks and optionally fix them."""
    from .utils.chunk_tracker import ChunkTracker
    from .storage import StorageManager
    import pyarrow.parquet as pq

    console.print("[bold cyan]Scanning for sparse/abandoned chunks...[/bold cyan]\n")

    checkpoint_path = Path(checkpoint_dir) / "chunks.json"
    if not checkpoint_path.exists():
        console.print("[red]No chunk checkpoint found![/red]")
        return

    tracker = ChunkTracker(checkpoint_path)
    storage = StorageManager(Path(data_dir))

    # Get and display stats
    stats = tracker.get_stats()
    console.print(f"[green]Total chunks:[/green] {stats['total']}")
    console.print(f"[green]Completed:[/green] {stats['completed']}")
    console.print(f"[yellow]Pending:[/yellow] {stats['pending']}")
    console.print(f"[yellow]Assigned:[/yellow] {stats['assigned']}")
    console.print(f"[red]Failed:[/red] {stats['failed']}\n")

    # Find abandoned chunks
    abandoned_chunks = []
    stale_threshold = 3600  # 1 hour
    current_time = datetime.utcnow()

    for chunk_id, chunk_state in tracker.chunks.items():
        if chunk_state.status == "assigned" and chunk_state.assigned_at:
            age = (current_time - chunk_state.assigned_at).total_seconds()
            if age > stale_threshold:
                abandoned_chunks.append((chunk_id, chunk_state, age))

    if abandoned_chunks:
        console.print(f"[red]Found {len(abandoned_chunks)} abandoned chunks:[/red]")
        for chunk_id, chunk_state, age in abandoned_chunks[:10]:
            age_str = f"{age/3600:.1f} hours" if age > 3600 else f"{age/60:.1f} minutes"
            console.print(f"  • {chunk_id} (assigned to {chunk_state.assigned_to} {age_str} ago)")

        if len(abandoned_chunks) > 10:
            console.print(f"  ... and {len(abandoned_chunks) - 10} more")

        if fix:
            console.print("\n[yellow]Resetting abandoned chunks to pending...[/yellow]")
            for chunk_id, _, _ in abandoned_chunks:
                tracker.mark_failed(chunk_id)
            console.print(f"[green]✓ Reset {len(abandoned_chunks)} chunks[/green]")

    # Check for sparse shards
    console.print("\n[bold cyan]Checking for sparse shards...[/bold cyan]")

    shards_summary = tracker.get_shards_summary()
    sparse_shards = []

    for shard_name, shard_info in shards_summary.items():
        if not shard_info["is_complete"]:
            chunks = sorted(shard_info["chunks"], key=lambda c: c.start_index)
            expected_index = 0
            has_gaps = False

            for chunk in chunks:
                if chunk.start_index != expected_index:
                    has_gaps = True
                    break
                expected_index = chunk.start_index + chunk.chunk_size

            if has_gaps or shard_info["failed_chunks"] > 0:
                sparse_shards.append((shard_name, shard_info, has_gaps))

    if sparse_shards:
        console.print(f"\n[yellow]Found {len(sparse_shards)} sparse/incomplete shards:[/yellow]")
        for shard_name, shard_info, has_gaps in sparse_shards[:5]:
            status = []
            if shard_info["pending_chunks"] > 0:
                status.append(f"{shard_info['pending_chunks']} pending")
            if shard_info["assigned_chunks"] > 0:
                status.append(f"{shard_info['assigned_chunks']} assigned")
            if shard_info["failed_chunks"] > 0:
                status.append(f"{shard_info['failed_chunks']} failed")
            if has_gaps:
                status.append("has gaps")

            console.print(f"  • {shard_name}: {', '.join(status)}")
            console.print(
                f"    Progress: {shard_info['completed_chunks']}/{shard_info['total_chunks']} chunks"
            )

        if len(sparse_shards) > 5:
            console.print(f"  ... and {len(sparse_shards) - 5} more")

    # Cross-check with storage if verbose
    if storage.captions_path.exists() and verbose:
        console.print("\n[bold cyan]Cross-checking with stored captions...[/bold cyan]")

        try:
            table = pq.read_table(storage.captions_path, columns=["chunk_id"])
            stored_chunk_ids = set(c for c in table["chunk_id"].to_pylist() if c)

            tracker_completed = set(c for c, s in tracker.chunks.items() if s.status == "completed")

            missing_in_storage = tracker_completed - stored_chunk_ids
            missing_in_tracker = stored_chunk_ids - set(tracker.chunks.keys())

            if missing_in_storage:
                console.print(
                    f"\n[red]Chunks marked complete but missing from storage:[/red] {len(missing_in_storage)}"
                )
                for chunk_id in list(missing_in_storage)[:5]:
                    console.print(f"  • {chunk_id}")

                if fix:
                    console.print("[yellow]Resetting these chunks to pending...[/yellow]")
                    for chunk_id in missing_in_storage:
                        tracker.mark_failed(chunk_id)
                    console.print(f"[green]✓ Reset {len(missing_in_storage)} chunks[/green]")

            if missing_in_tracker:
                console.print(
                    f"\n[yellow]Chunks in storage but not tracked:[/yellow] {len(missing_in_tracker)}"
                )

        except Exception as e:
            console.print(f"[red]Error reading storage: {e}[/red]")

    # Summary
    console.print("\n[bold cyan]Summary:[/bold cyan]")

    total_issues = len(abandoned_chunks) + len(sparse_shards)
    if total_issues == 0:
        console.print("[green]✓ No issues found![/green]")
    else:
        console.print(f"[yellow]Found {total_issues} total issues[/yellow]")

        if not fix:
            console.print(
                "\n[cyan]Run with --fix flag to automatically reset abandoned chunks[/cyan]"
            )
        else:
            console.print(
                "\n[green]✓ Issues have been fixed. Restart orchestrator to reprocess.[/green]"
            )

    if fix:
        tracker.save_checkpoint()


@main.command()
@click.option("--domain", help="Domain for Let's Encrypt certificate")
@click.option("--email", help="Email for Let's Encrypt registration")
@click.option("--self-signed", is_flag=True, help="Generate self-signed certificate")
@click.option("--output-dir", default="./certs", help="Output directory for certificates")
@click.option("--staging", is_flag=True, help="Use Let's Encrypt staging server (for testing)")
def generate_cert(
    domain: Optional[str], email: Optional[str], self_signed: bool, output_dir: str, staging: bool
):
    """Generate SSL certificates."""
    cert_manager = CertificateManager()

    if self_signed:
        console.print("[yellow]Generating self-signed certificate...[/yellow]")
        cert_domain = domain or "localhost"
        cert_path, key_path = cert_manager.generate_self_signed(Path(output_dir), cert_domain)
        console.print(f"[green]✓[/green] Certificate: {cert_path}")
        console.print(f"[green]✓[/green] Key: {key_path}")
        console.print(f"\n[cyan]Use these paths in your config or CLI:[/cyan]")
        console.print(f"  --cert {cert_path}")
        console.print(f"  --key {key_path}")
    elif domain and email:
        mode = "staging" if staging else "production"
        console.print(
            f"[yellow]Requesting Let's Encrypt {mode} certificate for {domain}...[/yellow]"
        )

        le_output = Path(output_dir) if output_dir != "./certs" else None

        try:
            cert_path, key_path = cert_manager.generate_letsencrypt(
                domain, email, output_dir=le_output, staging=staging
            )
            console.print(f"[green]✓[/green] Certificate: {cert_path}")
            console.print(f"[green]✓[/green] Key: {key_path}")
            console.print(f"\n[cyan]Use these paths in your config or CLI:[/cyan]")
            console.print(f"  --cert {cert_path}")
            console.print(f"  --key {key_path}")

            if staging:
                console.print(
                    "\n[yellow]⚠ This is a staging certificate (not trusted by browsers)[/yellow]"
                )
                console.print(
                    "[yellow]  Remove --staging flag for production certificates[/yellow]"
                )
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("  • Ensure port 80 is accessible for Let's Encrypt validation")
            console.print("  • Check that the domain points to this server")
            console.print("  • Try --staging flag for testing")
            sys.exit(1)
    else:
        console.print("[red]Error: Specify either --self-signed or --domain with --email[/red]")
        sys.exit(1)


@main.command()
@click.argument("cert_path", type=click.Path(exists=True))
def inspect_cert(cert_path: str):
    """Inspect an SSL certificate."""
    cert_manager = CertificateManager()

    try:
        info = cert_manager.get_cert_info(Path(cert_path))

        console.print("\n[bold cyan]Certificate Information[/bold cyan]")
        console.print(f"[green]Subject:[/green] {info['subject']}")
        console.print(f"[green]Issuer:[/green] {info['issuer']}")
        console.print(f"[green]Valid From:[/green] {info['not_before']}")
        console.print(f"[green]Valid Until:[/green] {info['not_after']}")
        console.print(f"[green]Serial Number:[/green] {info['serial_number']}")

        if info["is_self_signed"]:
            console.print("[yellow]⚠ This is a self-signed certificate[/yellow]")

        from datetime import datetime

        if info["not_after"] < datetime.utcnow():
            console.print("[red]✗ Certificate has expired![/red]")
        elif (info["not_after"] - datetime.utcnow()).days < 30:
            days_left = (info["not_after"] - datetime.utcnow()).days
            console.print(f"[yellow]⚠ Certificate expires in {days_left} days[/yellow]")
        else:
            days_left = (info["not_after"] - datetime.utcnow()).days
            console.print(f"[green]✓ Certificate valid for {days_left} more days[/green]")

    except Exception as e:
        console.print(f"[red]Error reading certificate: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
