"""Command-line interface for CaptionFlow."""

import asyncio
import logging
import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from datetime import datetime

from .orchestrator import Orchestrator
from .worker import Worker
from .monitor import Monitor
from .utils.certificates import CertificateManager

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_path=False, show_time=False)
        ],
    )


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool):
    """CaptionFlow - Distributed community captioning system."""
    setup_logging(verbose)

    if config:
        with open(config) as f:
            ctx.obj = yaml.safe_load(f)
    else:
        ctx.obj = {}


@main.command()
@click.option("--port", default=8765, help="WebSocket server port")
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--data-dir", default="./caption_data", help="Storage directory")
@click.option("--cert", help="SSL certificate path")
@click.option("--key", help="SSL key path")
@click.option("--no-ssl", is_flag=True, help="Disable SSL (development only)")
@click.option("--vllm", is_flag=True, help="Use vLLM orchestrator for WebDataset/HF datasets")
@click.pass_context
def orchestrator(
    ctx,
    port: int,
    host: str,
    data_dir: str,
    cert: Optional[str],
    key: Optional[str],
    no_ssl: bool,
    vllm: bool,
):
    """Start the orchestrator server."""
    config = ctx.obj.get("orchestrator", {})

    # Override with CLI arguments
    if port:
        config["port"] = port
    if host:
        config["host"] = host
    if data_dir:
        config.setdefault("storage", {})["data_dir"] = data_dir

    if not no_ssl:
        if cert and key:
            config.setdefault("ssl", {})["cert"] = cert
            config["ssl"]["key"] = key
        elif not config.get("ssl"):
            console.print(
                "[yellow]Warning: Running without SSL. Use --cert and --key for production.[/yellow]"
            )

    orchestrator = Orchestrator(config)

    try:
        asyncio.run(orchestrator.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down orchestrator...[/yellow]")
        asyncio.run(orchestrator.shutdown())


@main.command()
@click.option("--server", required=True, help="Orchestrator WebSocket URL")
@click.option("--token", required=True, help="Worker authentication token")
@click.option("--name", help="Worker display name")
@click.option("--batch-size", default=32, help="Inference batch size")
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
@click.option("--vllm", is_flag=True, help="Use vLLM worker for GPU inference")
@click.option("--gpu-id", type=int, default=0, help="GPU device ID (for vLLM)")
@click.option("--precision", default="fp16", help="Model precision (for vLLM)")
@click.option("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name (for vLLM)")
@click.pass_context
def worker(
    ctx,
    server: str,
    token: str,
    name: Optional[str],
    batch_size: int,
    no_verify_ssl: bool,
    vllm: bool,
    gpu_id: int,
    precision: str,
    model: str,
):
    """Start a worker node."""
    config = ctx.obj.get("worker", {})

    # Override with CLI arguments
    config["server"] = server
    config["token"] = token
    if name:
        config["name"] = name
    config["batch_size"] = batch_size
    config["verify_ssl"] = not no_verify_ssl

    if vllm:
        # Use vLLM worker for GPU inference
        from .worker_vllm import VLLMWorker

        config["gpu_id"] = gpu_id
        config["precision"] = precision
        config["model"] = model
        worker = VLLMWorker(config)
    else:
        # Use standard worker
        worker = Worker(config)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down worker...[/yellow]")
        asyncio.run(worker.shutdown())


@main.command()
@click.option("--server", required=True, help="Orchestrator WebSocket URL")
@click.option("--token", required=True, help="Admin authentication token")
@click.option(
    "--config", type=click.Path(exists=True), required=True, help="New configuration file"
)
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
def reload_config(server: str, token: str, config: str, no_verify_ssl: bool):
    """Reload orchestrator configuration via admin connection."""
    import websockets
    import ssl

    console.print(f"[cyan]Loading configuration from {config}...[/cyan]")

    # Load the new configuration
    try:
        with open(config) as f:
            new_config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        return

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
                await websocket.send(
                    json.dumps({"token": token, "role": "admin"})  # Special admin role
                )

                # Wait for response
                response = await websocket.recv()
                auth_response = json.loads(response)

                if "error" in auth_response:
                    console.print(f"[red]Authentication failed: {auth_response['error']}[/red]")
                    return False

                console.print("[green]✓ Authenticated as admin[/green]")

                # Send reload command
                await websocket.send(json.dumps({"type": "reload_config", "config": new_config}))

                # Wait for reload response
                response = await websocket.recv()
                reload_response = json.loads(response)

                # In the reload_config command, update the response handling:
                if reload_response.get("type") == "reload_complete":
                    if "message" in reload_response and "No changes" in reload_response["message"]:
                        console.print(f"[yellow]{reload_response['message']}[/yellow]")
                    else:
                        console.print("[green]✓ Configuration reloaded successfully![/green]")

                        # Show what was updated
                        if "updated" in reload_response and reload_response["updated"]:
                            console.print("\n[cyan]Updated sections:[/cyan]")
                            for section in reload_response["updated"]:
                                console.print(f"  • {section}")

                        # Show warnings if any
                        if "warnings" in reload_response and reload_response["warnings"]:
                            console.print("\n[yellow]Warnings:[/yellow]")
                            for warning in reload_response["warnings"]:
                                console.print(f"  ⚠ {warning}")

                    return True

                else:
                    error = reload_response.get("error", "Unknown error")
                    console.print(f"[red]Reload failed: {error}[/red]")
                    return False

        except websockets.exceptions.ConnectionClosed as e:
            console.print(f"[red]Connection closed: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback

            traceback.print_exc()
            return False

    # Run the async function
    success = asyncio.run(send_reload())

    if not success:
        sys.exit(1)


@main.command()
@click.option("--server", required=True, help="Orchestrator WebSocket URL")
@click.option("--token", required=True, help="Admin authentication token")
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
@click.pass_context
def monitor(ctx, server: str, token: str, no_verify_ssl: bool):
    """Start the monitoring TUI."""
    config = ctx.obj.get("monitor", {})

    config["server"] = server
    config["token"] = token
    config["verify_ssl"] = not no_verify_ssl

    monitor = Monitor(config)

    try:
        asyncio.run(monitor.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Closing monitor...[/yellow]")


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

    # Load chunk tracker
    checkpoint_path = Path(checkpoint_dir) / "chunks.json"
    if not checkpoint_path.exists():
        console.print("[red]No chunk checkpoint found![/red]")
        return

    tracker = ChunkTracker(checkpoint_path)
    storage = StorageManager(Path(data_dir))

    # Get chunk statistics
    stats = tracker.get_stats()
    console.print(f"[green]Total chunks:[/green] {stats['total']}")
    console.print(f"[green]Completed:[/green] {stats['completed']}")
    console.print(f"[yellow]Pending:[/yellow] {stats['pending']}")
    console.print(f"[yellow]Assigned:[/yellow] {stats['assigned']}")
    console.print(f"[red]Failed:[/red] {stats['failed']}\n")

    # Find abandoned chunks (assigned for too long)
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
        for chunk_id, chunk_state, age in abandoned_chunks[:10]:  # Show first 10
            age_str = f"{age/3600:.1f} hours" if age > 3600 else f"{age/60:.1f} minutes"
            console.print(f"  • {chunk_id} (assigned to {chunk_state.assigned_to} {age_str} ago)")

        if len(abandoned_chunks) > 10:
            console.print(f"  ... and {len(abandoned_chunks) - 10} more")

        if fix:
            console.print("\n[yellow]Resetting abandoned chunks to pending...[/yellow]")
            for chunk_id, chunk_state, _ in abandoned_chunks:
                tracker.mark_failed(chunk_id)  # This resets to pending
            console.print(f"[green]✓ Reset {len(abandoned_chunks)} chunks[/green]")

    # Check for sparse shards (shards with gaps in chunk coverage)
    console.print("\n[bold cyan]Checking for sparse shards...[/bold cyan]")

    shards_summary = tracker.get_shards_summary()
    sparse_shards = []

    for shard_name, shard_info in shards_summary.items():
        if not shard_info["is_complete"]:
            # Check if chunks have gaps
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

    # Cross-check with actual stored data
    if storage.captions_path.exists() and verbose:
        console.print("\n[bold cyan]Cross-checking with stored captions...[/bold cyan]")

        try:
            # Read chunk_ids from storage
            table = pq.read_table(storage.captions_path, columns=["chunk_id"])
            stored_chunk_ids = set(c for c in table["chunk_id"].to_pylist() if c)

            # Find discrepancies
            tracker_completed = set(c for c, s in tracker.chunks.items() if s.status == "completed")

            missing_in_storage = tracker_completed - stored_chunk_ids
            missing_in_tracker = stored_chunk_ids - set(tracker.chunks.keys())

            if missing_in_storage:
                console.print(
                    f"\n[red]Chunks marked complete but missing from storage:[/red] {len(missing_in_storage)}"
                )
                for chunk_id in list(missing_in_storage)[:5]:
                    console.print(f"  • {chunk_id}")

                if fix and missing_in_storage:
                    console.print("[yellow]Resetting these chunks to pending...[/yellow]")
                    for chunk_id in missing_in_storage:
                        tracker.mark_failed(chunk_id)
                    console.print(f"[green]✓ Reset {len(missing_in_storage)} chunks[/green]")

            if missing_in_tracker:
                console.print(
                    f"\n[yellow]Chunks in storage but not tracked:[/yellow] {len(missing_in_tracker)}"
                )
                # These are likely from before chunk tracking was implemented

        except Exception as e:
            console.print(f"[red]Error reading storage: {e}[/red]")

    # Summary and recommendations
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

    # Save any changes
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

        # For Let's Encrypt, allow custom output dir but default to system location
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

        # Check expiration
        from datetime import datetime

        if info["not_after"] < datetime.utcnow():
            console.print("[red]✗ Certificate has expired![/red]")
        elif (info["not_after"] - datetime.utcnow()).days < 30:
            console.print(
                f"[yellow]⚠ Certificate expires in {(info['not_after'] - datetime.utcnow()).days} days[/yellow]"
            )
        else:
            console.print(
                f"[green]✓ Certificate valid for {(info['not_after'] - datetime.utcnow()).days} more days[/green]"
            )

    except Exception as e:
        console.print(f"[red]Error reading certificate: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
