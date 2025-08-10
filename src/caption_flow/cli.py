"""Command-line interface for CaptionFlow."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

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
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
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
