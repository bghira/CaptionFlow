"""Command-line interface for CaptionFlow with smart configuration handling."""

import asyncio
import datetime as _datetime
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

from .monitor import Monitor
from .orchestrator import Orchestrator
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
        """Find and load configuration for a component.

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
    """Configure logging with rich handler and file output to XDG state directory."""
    level = logging.DEBUG if verbose else logging.INFO

    # Determine log directory based on environment or XDG spec
    log_dir_env = os.environ.get("CAPTIONFLOW_LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env)
    else:
        # Use XDG_STATE_HOME for logs, with platform-specific fallbacks
        xdg_state_home = os.environ.get("XDG_STATE_HOME")
        if xdg_state_home:
            base_dir = Path(xdg_state_home)
        elif sys.platform == "darwin":
            base_dir = Path.home() / "Library" / "Logs"
        else:
            # Default to ~/.local/state on Linux and other systems
            base_dir = Path.home() / ".local" / "state"
        log_dir = base_dir / "caption-flow"

    try:
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "caption_flow.log"

        # Set up handlers
        handlers: List[logging.Handler] = [
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                show_time=True,
            )
        ]

        # Add file handler
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)
        log_msg = f"Logging to {log_file_path}"

    except (OSError, PermissionError) as e:
        # Fallback to only console logging if file logging fails
        handlers = [
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                show_time=True,
            )
        ]
        log_file = log_dir / "caption_flow.log"
        log_msg = f"[yellow]Warning: Could not write to log file {log_file}: {e}[/yellow]"

    logging.basicConfig(
        level=level,
        format="%(message)s",  # RichHandler overrides this format for console
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=handlers,
    )

    # Suppress noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("pyarrow").setLevel(logging.WARNING)

    # Use a dedicated logger to print the log file path to avoid format issues
    if "log_msg" in locals():
        logging.getLogger("setup").info(log_msg)


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
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
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
            warning_msg = (
                "[yellow]Warning: Running without SSL. "
                "Use --cert and --key for production.[/yellow]"
            )
            console.print(warning_msg)

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


def _load_monitor_config(config, server, token):
    """Load monitor configuration from file or fallback to orchestrator config."""
    base_config = ConfigManager.find_config("monitor", config)

    if not base_config:
        orch_config = ConfigManager.find_config("orchestrator")
        if orch_config and "monitor" in orch_config:
            base_config = {"monitor": orch_config["monitor"]}
            console.print("[dim]Using monitor config from orchestrator.yaml[/dim]")
        else:
            base_config = {}
            if not server or not token:
                console.print("[yellow]No monitor config found, using CLI args[/yellow]")

    return base_config.get("monitor", base_config)


def _apply_monitor_overrides(config_data, server, token, no_verify_ssl):
    """Apply CLI overrides to monitor configuration."""
    if server:
        config_data["server"] = server
    if token:
        config_data["token"] = token
    if no_verify_ssl:
        config_data["verify_ssl"] = False


def _debug_monitor_config(config_data):
    """Print debug information about monitor configuration."""
    console.print("\n[cyan]Final monitor configuration:[/cyan]")
    console.print(f"  Server: {config_data.get('server', 'NOT SET')}")
    console.print(
        f"  Token: {'***' + config_data.get('token', '')[-4:] if config_data.get('token') else 'NOT SET'}"
    )
    console.print(f"  Verify SSL: {config_data.get('verify_ssl', True)}")
    console.print()


def _validate_monitor_config(config_data):
    """Validate required monitor configuration fields."""
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


def _set_monitor_defaults(config_data):
    """Set default values for optional monitor settings."""
    config_data.setdefault("refresh_interval", 1.0)
    config_data.setdefault("show_inactive_workers", False)
    config_data.setdefault("max_log_lines", 100)


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
    if debug:
        setup_logging(verbose=True)
        console.print("[yellow]Debug mode enabled[/yellow]")

    config_data = _load_monitor_config(config, server, token)
    _apply_monitor_overrides(config_data, server, token, no_verify_ssl)

    if debug:
        _debug_monitor_config(config_data)

    _validate_monitor_config(config_data)
    _set_monitor_defaults(config_data)

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


# Add this command after the export command in cli.py


@main.command()
@click.option("--data-dir", default="./caption_data", help="Storage directory")
@click.option("--refresh-rate", default=10, type=int, help="Display refresh rate (Hz)")
@click.option("--no-images", is_flag=True, help="Disable image preview")
@click.pass_context
def view(ctx, data_dir: str, refresh_rate: int, no_images: bool):
    """Browse captioned dataset with interactive TUI viewer."""
    from .viewer import DatasetViewer

    data_path = Path(data_dir)

    if not data_path.exists():
        console.print(f"[red]Storage directory not found: {data_dir}[/red]")
        sys.exit(1)

    if not (data_path / "captions.parquet").exists():
        console.print(f"[red]No captions file found in {data_dir}[/red]")
        console.print("[yellow]Have you exported any captions yet?[/yellow]")
        sys.exit(1)

    # Check for term-image if images are enabled
    if not no_images:
        try:
            import term_image
        except ImportError:
            console.print("[yellow]Warning: term-image not installed[/yellow]")
            console.print("Install with: pip install term-image")
            console.print("Running without image preview...")
            no_images = True

    try:
        viewer = DatasetViewer(data_path)
        if no_images:
            viewer.disable_images = True
        viewer.refresh_rate = refresh_rate

        console.print("[cyan]Starting dataset viewer...[/cyan]")
        console.print(f"[dim]Data directory: {data_path}[/dim]")

        asyncio.run(viewer.run())

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Viewer closed[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _load_admin_credentials(config, server, token):
    """Load admin server and token from config if not provided."""
    if server and token:
        return server, token

    base_config = ConfigManager.find_config("orchestrator", config) or {}
    admin_config = base_config.get("admin", {})
    admin_tokens = base_config.get("orchestrator", {}).get("auth", {}).get("admin_tokens", [])

    final_server = server or admin_config.get("server", "ws://localhost:8765")
    final_token = token or admin_config.get("token")

    if not final_token and admin_tokens:
        console.print("Using first admin token.")
        final_token = admin_tokens[0].get("token")

    return final_server, final_token


def _setup_ssl_context(server, no_verify_ssl):
    """Setup SSL context for websocket connection."""
    import ssl

    ssl_context = None
    if server.startswith("wss://"):
        ssl_context = ssl.create_default_context()
        if no_verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    return ssl_context


async def _authenticate_admin(websocket, token):
    """Authenticate as admin with the websocket."""
    await websocket.send(json.dumps({"token": token, "role": "admin"}))

    response = await websocket.recv()
    auth_response = json.loads(response)

    if "error" in auth_response:
        console.print(f"[red]Authentication failed: {auth_response['error']}[/red]")
        return False

    console.print("[green]✓ Authenticated as admin[/green]")
    return True


async def _send_reload_command(websocket, new_cfg):
    """Send reload command and handle response."""
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


def _add_token_to_config(config_data: Dict[str, Any], role: str, name: str, token: str) -> bool:
    """Add a new token to the config data."""
    # Ensure the auth section exists
    if "orchestrator" not in config_data:
        config_data["orchestrator"] = {}
    if "auth" not in config_data["orchestrator"]:
        config_data["orchestrator"]["auth"] = {}

    auth_config = config_data["orchestrator"]["auth"]
    token_key = f"{role}_tokens"

    # Initialize token list if it doesn't exist
    if token_key not in auth_config:
        auth_config[token_key] = []

    # Check if token already exists
    for existing_token in auth_config[token_key]:
        if existing_token.get("token") == token:
            console.print(f"[yellow]Token already exists for {role}: {name}[/yellow]")
            return False
        if existing_token.get("name") == name:
            console.print(f"[yellow]Name already exists for {role}: {name}[/yellow]")
            return False

    # Add the new token
    auth_config[token_key].append({"name": name, "token": token})
    console.print(f"[green]✓ Added {role} token for {name}[/green]")
    return True


def _remove_token_from_config(config_data: Dict[str, Any], role: str, identifier: str) -> bool:
    """Remove a token from the config data by name or token."""
    auth_config = config_data.get("orchestrator", {}).get("auth", {})
    token_key = f"{role}_tokens"

    if token_key not in auth_config:
        console.print(f"[red]No {role} tokens found in config[/red]")
        return False

    tokens = auth_config[token_key]
    removed = False

    for i, token_entry in enumerate(tokens):
        if token_entry.get("name") == identifier or token_entry.get("token") == identifier:
            removed_entry = tokens.pop(i)
            console.print(f"[green]✓ Removed {role} token: {removed_entry['name']}[/green]")
            removed = True
            break

    if not removed:
        console.print(f"[red]Token not found for {role}: {identifier}[/red]")

    return removed


def _list_tokens_in_config(config_data: Dict[str, Any], role: Optional[str] = None):
    """List tokens in the config data."""
    auth_config = config_data.get("orchestrator", {}).get("auth", {})

    if not auth_config:
        console.print("[yellow]No auth configuration found[/yellow]")
        return

    roles_to_show = [role] if role else ["worker", "admin", "monitor"]

    for token_role in roles_to_show:
        token_key = f"{token_role}_tokens"
        tokens = auth_config.get(token_key, [])

        if tokens:
            console.print(f"\n[cyan]{token_role.title()} tokens:[/cyan]")
            for token_entry in tokens:
                name = token_entry.get("name", "Unknown")
                token = token_entry.get("token", "")
                masked_token = f"***{token[-4:]}" if len(token) > 4 else "***"
                console.print(f"  • {name}: {masked_token}")
        else:
            console.print(f"\n[dim]No {token_role} tokens configured[/dim]")


def _save_config_file(config_data: Dict[str, Any], config_path: Path) -> bool:
    """Save the config data to a file."""
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✓ Configuration saved to {config_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")
        return False


async def _reload_orchestrator_config(
    server: str, token: str, config_data: Dict[str, Any], no_verify_ssl: bool
) -> bool:
    """Reload the orchestrator configuration."""
    import websockets

    ssl_context = _setup_ssl_context(server, no_verify_ssl)

    try:
        async with websockets.connect(
            server, ssl=ssl_context, ping_interval=20, ping_timeout=60, close_timeout=10
        ) as websocket:
            if not await _authenticate_admin(websocket, token):
                return False

            return await _send_reload_command(websocket, config_data)
    except Exception as e:
        console.print(f"[red]Error connecting to orchestrator: {e}[/red]")
        return False


@main.group()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--server", help="Orchestrator WebSocket URL")
@click.option("--token", help="Admin authentication token")
@click.option("--no-verify-ssl", is_flag=True, help="Skip SSL verification")
@click.pass_context
def auth(
    ctx, config: Optional[str], server: Optional[str], token: Optional[str], no_verify_ssl: bool
):
    """Manage authentication tokens for the orchestrator."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        {"config": config, "server": server, "token": token, "no_verify_ssl": no_verify_ssl}
    )


@auth.command()
@click.argument("role", type=click.Choice(["worker", "admin", "monitor"]))
@click.argument("name")
@click.argument("token_value")
@click.option(
    "--no-reload", is_flag=True, help="Don't reload orchestrator config after adding token"
)
@click.pass_context
def add(ctx, role: str, name: str, token_value: str, no_reload: bool):
    """Add a new authentication token.

    ROLE: Type of token (worker, admin, monitor)
    NAME: Display name for the token
    TOKEN_VALUE: The actual token string
    """
    config_file = ctx.obj.get("config")
    server = ctx.obj.get("server")
    admin_token = ctx.obj.get("token")
    no_verify_ssl = ctx.obj.get("no_verify_ssl", False)

    # Load config
    config_data = ConfigManager.find_config("orchestrator", config_file)
    if not config_data:
        console.print("[red]No orchestrator config found[/red]")
        console.print("[dim]Use --config to specify config file path[/dim]")
        sys.exit(1)

    # Find config file path for saving
    config_path = None
    if config_file:
        config_path = Path(config_file)
    else:
        # Try to find the config file that was loaded
        for search_path in [
            Path.cwd() / "orchestrator.yaml",
            Path.cwd() / "config" / "orchestrator.yaml",
            Path.home() / ".caption-flow" / "orchestrator.yaml",
            ConfigManager.get_xdg_config_home() / "caption-flow" / "orchestrator.yaml",
        ]:
            if search_path.exists():
                config_path = search_path
                break

    if not config_path:
        console.print("[red]Could not determine config file to save to[/red]")
        console.print("[dim]Use --config to specify config file path[/dim]")
        sys.exit(1)

    # Add token to config
    if not _add_token_to_config(config_data, role, name, token_value):
        sys.exit(1)

    # Save config file
    if not _save_config_file(config_data, config_path):
        sys.exit(1)

    # Reload orchestrator if requested
    if not no_reload:
        server, admin_token = _load_admin_credentials(config_file, server, admin_token)

        if not server:
            console.print("[yellow]No server specified, skipping orchestrator reload[/yellow]")
            console.print("[dim]Use --server to reload orchestrator config[/dim]")
        elif not admin_token:
            console.print("[yellow]No admin token specified, skipping orchestrator reload[/yellow]")
            console.print("[dim]Use --token to reload orchestrator config[/dim]")
        else:
            console.print(f"[cyan]Reloading orchestrator config...[/cyan]")
            success = asyncio.run(
                _reload_orchestrator_config(server, admin_token, config_data, no_verify_ssl)
            )
            if not success:
                console.print("[yellow]Config file updated but orchestrator reload failed[/yellow]")
                console.print("[dim]You may need to restart the orchestrator manually[/dim]")


@auth.command()
@click.argument("role", type=click.Choice(["worker", "admin", "monitor"]))
@click.argument("identifier")
@click.option(
    "--no-reload", is_flag=True, help="Don't reload orchestrator config after removing token"
)
@click.pass_context
def remove(ctx, role: str, identifier: str, no_reload: bool):
    """Remove an authentication token.

    ROLE: Type of token (worker, admin, monitor)
    IDENTIFIER: Name or token value to remove
    """
    config_file = ctx.obj.get("config")
    server = ctx.obj.get("server")
    admin_token = ctx.obj.get("token")
    no_verify_ssl = ctx.obj.get("no_verify_ssl", False)

    # Load config
    config_data = ConfigManager.find_config("orchestrator", config_file)
    if not config_data:
        console.print("[red]No orchestrator config found[/red]")
        sys.exit(1)

    # Find config file path for saving
    config_path = None
    if config_file:
        config_path = Path(config_file)
    else:
        # Try to find the config file that was loaded
        for search_path in [
            Path.cwd() / "orchestrator.yaml",
            Path.cwd() / "config" / "orchestrator.yaml",
            Path.home() / ".caption-flow" / "orchestrator.yaml",
            ConfigManager.get_xdg_config_home() / "caption-flow" / "orchestrator.yaml",
        ]:
            if search_path.exists():
                config_path = search_path
                break

    if not config_path:
        console.print("[red]Could not determine config file to save to[/red]")
        sys.exit(1)

    # Remove token from config
    if not _remove_token_from_config(config_data, role, identifier):
        sys.exit(1)

    # Save config file
    if not _save_config_file(config_data, config_path):
        sys.exit(1)

    # Reload orchestrator if requested
    if not no_reload:
        server, admin_token = _load_admin_credentials(config_file, server, admin_token)

        if not server:
            console.print("[yellow]No server specified, skipping orchestrator reload[/yellow]")
        elif not admin_token:
            console.print("[yellow]No admin token specified, skipping orchestrator reload[/yellow]")
        else:
            console.print(f"[cyan]Reloading orchestrator config...[/cyan]")
            success = asyncio.run(
                _reload_orchestrator_config(server, admin_token, config_data, no_verify_ssl)
            )
            if not success:
                console.print("[yellow]Config file updated but orchestrator reload failed[/yellow]")


@auth.command()
@click.argument("role", type=click.Choice(["worker", "admin", "monitor", "all"]), required=False)
@click.pass_context
def list(ctx, role: Optional[str]):
    """List authentication tokens.

    ROLE: Type of tokens to list (worker, admin, monitor, all). Default: all
    """
    config_file = ctx.obj.get("config")

    # Load config
    config_data = ConfigManager.find_config("orchestrator", config_file)
    if not config_data:
        console.print("[red]No orchestrator config found[/red]")
        sys.exit(1)

    # Show tokens
    if role == "all" or role is None:
        _list_tokens_in_config(config_data)
    else:
        _list_tokens_in_config(config_data, role)


@auth.command()
@click.option("--length", default=32, help="Token length (default: 32)")
@click.option("--count", default=1, help="Number of tokens to generate (default: 1)")
def generate(length: int, count: int):
    """Generate random authentication tokens."""
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits + "-_"

    console.print(
        f"[cyan]Generated {count} token{'s' if count > 1 else ''} ({length} characters each):[/cyan]\n"
    )

    for i in range(count):
        token = "".join(secrets.choice(alphabet) for _ in range(length))
        console.print(f"  {i + 1}: {token}")


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

    server, token = _load_admin_credentials(config, server, token)

    if not server:
        console.print("[red]Error: --server required (or set in config)[/red]")
        sys.exit(1)
    if not token:
        console.print("[red]Error: --token required (or set in config)[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading configuration from {new_config}...[/cyan]")

    new_cfg = ConfigManager.load_yaml(Path(new_config))
    if not new_cfg:
        console.print("[red]Failed to load configuration[/red]")
        sys.exit(1)

    ssl_context = _setup_ssl_context(server, no_verify_ssl)

    async def send_reload():
        try:
            async with websockets.connect(
                server, ssl=ssl_context, ping_interval=20, ping_timeout=60, close_timeout=10
            ) as websocket:
                if not await _authenticate_admin(websocket, token):
                    return False

                return await _send_reload_command(websocket, new_cfg)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False

    success = asyncio.run(send_reload())
    if not success:
        sys.exit(1)


def _display_chunk_stats(stats):
    """Display chunk statistics."""
    console.print(f"[green]Total chunks:[/green] {stats['total']}")
    console.print(f"[green]Completed:[/green] {stats['completed']}")
    console.print(f"[yellow]Pending:[/yellow] {stats['pending']}")
    console.print(f"[yellow]Assigned:[/yellow] {stats['assigned']}")
    console.print(f"[red]Failed:[/red] {stats['failed']}\n")


def _find_abandoned_chunks(tracker):
    """Find chunks that have been assigned for too long."""
    abandoned_chunks = []
    stale_threshold = 3600  # 1 hour
    current_time = datetime.now(_datetime.UTC)

    for chunk_id, chunk_state in tracker.chunks.items():
        if chunk_state.status == "assigned" and chunk_state.assigned_at:
            age = (current_time - chunk_state.assigned_at).total_seconds()
            if age > stale_threshold:
                abandoned_chunks.append((chunk_id, chunk_state, age))

    return abandoned_chunks


def _display_abandoned_chunks(abandoned_chunks, fix, tracker):
    """Display abandoned chunks and optionally fix them."""
    if not abandoned_chunks:
        return

    console.print(f"[red]Found {len(abandoned_chunks)} abandoned chunks:[/red]")
    for chunk_id, chunk_state, age in abandoned_chunks[:10]:
        age_str = f"{age / 3600:.1f} hours" if age > 3600 else f"{age / 60:.1f} minutes"
        console.print(f"  • {chunk_id} (assigned to {chunk_state.assigned_to} {age_str} ago)")

    if len(abandoned_chunks) > 10:
        console.print(f"  ... and {len(abandoned_chunks) - 10} more")

    if fix:
        console.print("\n[yellow]Resetting abandoned chunks to pending...[/yellow]")
        for chunk_id, _, _ in abandoned_chunks:
            tracker.mark_failed(chunk_id)
        console.print(f"[green]✓ Reset {len(abandoned_chunks)} chunks[/green]")


def _find_sparse_shards(tracker):
    """Find shards with gaps or issues."""
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

    return sparse_shards


def _display_sparse_shards(sparse_shards):
    """Display sparse/incomplete shards."""
    if not sparse_shards:
        return

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


def _cross_check_storage(storage, tracker, fix):
    """Cross-check chunk tracker against storage."""
    import pyarrow.parquet as pq

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


@main.command()
@click.option("--data-dir", default="./caption_data", help="Storage directory")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
@click.option("--fix", is_flag=True, help="Fix issues by resetting abandoned chunks")
@click.option("--verbose", is_flag=True, help="Show detailed information")
def scan_chunks(data_dir: str, checkpoint_dir: str, fix: bool, verbose: bool):
    """Scan for sparse or abandoned chunks and optionally fix them."""
    from .storage import StorageManager
    from .utils.chunk_tracker import ChunkTracker

    console.print("[bold cyan]Scanning for sparse/abandoned chunks...[/bold cyan]\n")

    checkpoint_path = Path(checkpoint_dir) / "chunks.json"
    if not checkpoint_path.exists():
        console.print("[red]No chunk checkpoint found![/red]")
        return

    tracker = ChunkTracker(checkpoint_path)
    storage = StorageManager(Path(data_dir))

    # Get and display stats
    stats = tracker.get_stats()
    _display_chunk_stats(stats)

    # Find and handle abandoned chunks
    abandoned_chunks = _find_abandoned_chunks(tracker)
    _display_abandoned_chunks(abandoned_chunks, fix, tracker)

    # Check for sparse shards
    console.print("\n[bold cyan]Checking for sparse shards...[/bold cyan]")
    sparse_shards = _find_sparse_shards(tracker)
    _display_sparse_shards(sparse_shards)

    # Cross-check with storage if verbose
    if storage.captions_path.exists() and verbose:
        _cross_check_storage(storage, tracker, fix)

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


def _display_export_stats(stats):
    """Display storage statistics."""
    console.print("\n[bold cyan]Storage Statistics:[/bold cyan]")
    console.print(f"[green]Total rows:[/green] {stats['total_rows']:,}")
    console.print(f"[green]Total outputs:[/green] {stats['total_outputs']:,}")
    console.print(f"[green]Shards:[/green] {stats['shard_count']} ({', '.join(stats['shards'])})")
    console.print(f"[green]Output fields:[/green] {', '.join(stats['output_fields'])}")

    if stats.get("field_stats"):
        console.print("\n[cyan]Field breakdown:[/cyan]")
        for field, count in stats["field_stats"].items():
            console.print(f"  • {field}: {count['total_items']:,} items")


def _prepare_export_params(shard, shards, columns):
    """Prepare shard filter and column list."""
    shard_filter = None
    if shard:
        shard_filter = [shard]
    elif shards:
        shard_filter = [s.strip() for s in shards.split(",")]

    column_list = None
    if columns:
        column_list = [col.strip() for col in columns.split(",")]
        console.print(f"\n[cyan]Exporting columns:[/cyan] {', '.join(column_list)}")

    return shard_filter, column_list


async def _export_all_formats(
    exporter, output, shard_filter, column_list, limit, filename_column, export_column
):
    """Export to all formats."""
    base_name = output or "caption_export"
    base_path = Path(base_name)
    results = {}

    for export_format in ["jsonl", "csv", "parquet", "json", "txt"]:
        console.print(f"\n[cyan]Exporting to {export_format.upper()}...[/cyan]")
        try:
            format_results = await exporter.export_all_shards(
                export_format,
                base_path,
                columns=column_list,
                limit_per_shard=limit,
                shard_filter=shard_filter,
                filename_column=filename_column,
                export_column=export_column,
            )
            results[export_format] = sum(format_results.values())
        except Exception as e:
            console.print(f"[yellow]Skipping {export_format}: {e}[/yellow]")
            results[export_format] = 0

    console.print("\n[green]✓ Export complete![/green]")
    for fmt, count in results.items():
        if count > 0:
            console.print(f"  • {fmt.upper()}: {count:,} items")


async def _export_to_lance(exporter, output, column_list, shard_filter):
    """Export to Lance dataset."""
    output_path = output or "exported_captions.lance"
    console.print(f"\n[cyan]Exporting to Lance dataset:[/cyan] {output_path}")
    total_rows = await exporter.export_to_lance(
        output_path, columns=column_list, shard_filter=shard_filter
    )
    console.print(f"[green]✓ Exported {total_rows:,} rows to Lance dataset[/green]")


async def _export_to_huggingface(exporter, hf_dataset, license, private, nsfw, tags, shard_filter):
    """Export to Hugging Face Hub."""
    if not hf_dataset:
        console.print("[red]Error: --hf-dataset required for huggingface_hub format[/red]")
        console.print("[dim]Example: --hf-dataset username/my-caption-dataset[/dim]")
        sys.exit(1)

    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]

    console.print(f"\n[cyan]Uploading to Hugging Face Hub:[/cyan] {hf_dataset}")
    if private:
        console.print("[dim]Privacy: Private dataset[/dim]")
    if nsfw:
        console.print("[dim]Content: Not for all audiences[/dim]")
    if tag_list:
        console.print(f"[dim]Tags: {', '.join(tag_list)}[/dim]")
    if shard_filter:
        console.print(f"[dim]Shards: {', '.join(shard_filter)}[/dim]")

    url = await exporter.export_to_huggingface_hub(
        dataset_name=hf_dataset,
        license=license,
        private=private,
        nsfw=nsfw,
        tags=tag_list,
        shard_filter=shard_filter,
    )
    console.print(f"[green]✓ Dataset uploaded to: {url}[/green]")


async def _export_single_format(
    exporter,
    format,
    output,
    shard_filter,
    column_list,
    limit,
    filename_column,
    export_column,
    verbose,
):
    """Export to a single format."""
    output_path = output or "export"

    if shard_filter and len(shard_filter) == 1:
        console.print(f"\n[cyan]Exporting shard {shard_filter[0]} to {format.upper()}...[/cyan]")
        count = await exporter.export_shard(
            shard_filter[0],
            format,
            output_path,
            columns=column_list,
            limit=limit,
            filename_column=filename_column,
            export_column=export_column,
        )
        console.print(f"[green]✓ Exported {count:,} items[/green]")
    else:
        console.print(f"\n[cyan]Exporting to {format.upper()}...[/cyan]")
        results = await exporter.export_all_shards(
            format,
            output_path,
            columns=column_list,
            limit_per_shard=limit,
            shard_filter=shard_filter,
            filename_column=filename_column,
            export_column=export_column,
        )

        total = sum(results.values())
        console.print(f"[green]✓ Exported {total:,} items total[/green]")

        if verbose and len(results) > 1:
            console.print("\n[dim]Per-shard breakdown:[/dim]")
            for shard_name, count in sorted(results.items()):
                console.print(f"  • {shard_name}: {count:,} items")


def _validate_export_setup(data_dir):
    """Validate export setup and create storage manager."""
    from .storage import StorageManager

    storage_path = Path(data_dir)
    if not storage_path.exists():
        console.print(f"[red]Storage directory not found: {data_dir}[/red]")
        sys.exit(1)

    return StorageManager(storage_path)


async def _run_export_process(
    storage,
    format,
    output,
    shard,
    shards,
    columns,
    limit,
    filename_column,
    export_column,
    verbose,
    hf_dataset,
    license,
    private,
    nsfw,
    tags,
    stats_only,
    optimize,
    include_empty,
):
    """Execute the main export process."""
    from .storage.exporter import LanceStorageExporter

    await storage.initialize()

    stats = await storage.get_caption_stats()
    _display_export_stats(stats)

    if stats_only:
        return

    if optimize:
        console.print("\n[yellow]Optimizing storage...[/yellow]")
        await storage.optimize_storage()

    shard_filter, column_list = _prepare_export_params(shard, shards, columns)
    exporter = LanceStorageExporter(storage)

    if format == "all":
        await _export_all_formats(
            exporter, output, shard_filter, column_list, limit, filename_column, export_column
        )
    elif format == "lance":
        await _export_to_lance(exporter, output, column_list, shard_filter)
    elif format == "huggingface_hub":
        await _export_to_huggingface(
            exporter, hf_dataset, license, private, nsfw, tags, shard_filter
        )
    else:
        await _export_single_format(
            exporter,
            format,
            output,
            shard_filter,
            column_list,
            limit,
            filename_column,
            export_column,
            verbose,
        )


@main.command()
@click.option("--data-dir", default="./caption_data", help="Storage directory")
@click.option(
    "--format",
    type=click.Choice(
        ["jsonl", "json", "csv", "txt", "parquet", "lance", "huggingface_hub", "all"],
        case_sensitive=False,
    ),
    default="jsonl",
    help="Export format (default: jsonl)",
)
@click.option("--output", help="Output filename or directory")
@click.option("--limit", type=int, help="Maximum number of items to export")
@click.option("--columns", help="Comma-separated list of columns to include")
@click.option("--export-column", default="captions", help="Column to export (default: captions)")
@click.option("--filename-column", default="filename", help="Filename column (default: filename)")
@click.option("--shard", help="Export only specific shard (e.g., 'data-001')")
@click.option("--shards", help="Comma-separated list of shards to export")
@click.option("--include-empty", is_flag=True, help="Include items with empty/null export column")
@click.option("--stats-only", is_flag=True, help="Show statistics only, don't export")
@click.option("--optimize", is_flag=True, help="Optimize storage before export")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--hf-dataset", help="HuggingFace Hub dataset name (for huggingface_hub format)")
@click.option("--license", default="MIT", help="Dataset license (default: MIT)")
@click.option("--private", is_flag=True, help="Make HuggingFace dataset private")
@click.option("--nsfw", is_flag=True, help="Mark dataset as NSFW")
@click.option("--tags", help="Comma-separated tags for HuggingFace dataset")
def export(
    data_dir: str,
    format: str,
    output: Optional[str],
    limit: Optional[int],
    columns: Optional[str],
    export_column: str,
    filename_column: str,
    shard: Optional[str],
    shards: Optional[str],
    include_empty: bool,
    stats_only: bool,
    optimize: bool,
    verbose: bool,
    hf_dataset: Optional[str],
    license: str,
    private: bool,
    nsfw: bool,
    tags: Optional[str],
):
    """Export caption data to various formats with per-shard support."""
    from .storage.exporter import ExportError

    storage = _validate_export_setup(data_dir)

    try:
        asyncio.run(
            _run_export_process(
                storage,
                format,
                output,
                shard,
                shards,
                columns,
                limit,
                filename_column,
                export_column,
                verbose,
                hf_dataset,
                license,
                private,
                nsfw,
                tags,
                stats_only,
                optimize,
                include_empty,
            )
        )
    except ExportError as e:
        console.print(f"[red]Export error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Export cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


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
        console.print("\n[cyan]Use these paths in your config or CLI:[/cyan]")
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
            console.print("\n[cyan]Use these paths in your config or CLI:[/cyan]")
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

        if info["not_after"] < datetime.now(_datetime.UTC):
            console.print("[red]✗ Certificate has expired![/red]")
        elif (info["not_after"] - datetime.now(_datetime.UTC)).days < 30:
            days_left = (info["not_after"] - datetime.now(_datetime.UTC)).days
            console.print(f"[yellow]⚠ Certificate expires in {days_left} days[/yellow]")
        else:
            days_left = (info["not_after"] - datetime.now(_datetime.UTC)).days
            console.print(f"[green]✓ Certificate valid for {days_left} more days[/green]")

    except Exception as e:
        console.print(f"[red]Error reading certificate: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
