"""TUI monitor for CaptionFlow system."""

import asyncio
import json
import logging
import ssl
from datetime import datetime
from typing import Any, Dict, Optional

import websockets
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class Monitor:
    """Real-time monitoring interface for CaptionFlow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print(f"Config loaded: {self.config}")
        self.server_url = config["server"]
        self.token = config["token"]

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Display state
        self.stats = {}
        self.leaderboard = []
        self.recent_activity = []
        self.running = False

        # Rate tracking
        self.rate_info = {
            "current_rate": 0.0,
            "average_rate": 0.0,
            "expected_rate": 0.0,
        }

        # Rich console
        self.console = Console()

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context."""
        if not self.config.get("verify_ssl", True):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        return ssl.create_default_context()

    async def start(self):
        """Start the monitor interface."""
        self.running = True

        # Connect to orchestrator
        asyncio.create_task(self._connect_to_orchestrator())

        # Start display loop
        await self._display_loop()

    async def _connect_to_orchestrator(self):
        """Maintain connection to orchestrator."""
        while self.running:
            try:
                async with websockets.connect(
                    self.server_url,
                    ssl=self.ssl_context if self.server_url.startswith("wss://") else None,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=10,
                ) as websocket:
                    # Authenticate
                    await websocket.send(json.dumps({"token": self.token}))

                    # Receive updates
                    async for message in websocket:
                        data = json.loads(message)
                        await self._handle_update(data)

            except Exception as e:
                logger.error(f"Connection error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _handle_update(self, data: Dict):
        """Process update from orchestrator."""
        msg_type = data.get("type")

        if msg_type == "stats":
            self.stats = data["data"]
            # Extract rate info if present
            self.rate_info["current_rate"] = self.stats.get("current_rate", 0.0)
            self.rate_info["average_rate"] = self.stats.get("average_rate", 0.0)
            self.rate_info["expected_rate"] = self.stats.get("expected_rate", 0.0)
        elif msg_type == "leaderboard":
            self.leaderboard = data["data"]
        elif msg_type == "activity":
            self.recent_activity.append(data["data"])
            # Keep only recent activity
            self.recent_activity = self.recent_activity[-20:]

    async def _display_loop(self):
        """Main display update loop."""
        layout = self._create_layout()

        with Live(layout, console=self.console, refresh_per_second=1, screen=True):
            while self.running:
                self._update_layout(layout)
                await asyncio.sleep(0.25)

    def _create_layout(self) -> Layout:
        """Create the display layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="rates", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="leaderboard", ratio=1),
            Layout(name="activity", ratio=1),
        )

        return layout

    def _update_layout(self, layout: Layout):
        """Update layout with current data."""
        # Header
        layout["header"].update(
            Panel(
                Text("CaptionFlow Monitor", style="bold magenta", justify="center"),
                border_style="bright_blue",
            )
        )

        # Rates panel
        rates_table = Table(show_header=False, expand=True)
        rates_table.add_column("Metric", style="bold")
        rates_table.add_column("Value", style="cyan", justify="right")

        rates_table.add_row("Current Rate", f"{self.rate_info['current_rate']:.1f} captions/min")
        rates_table.add_row("Average Rate", f"{self.rate_info['average_rate']:.1f} captions/min")
        rates_table.add_row("Expected Rate", f"{self.rate_info['expected_rate']:.1f} captions/min")

        # Add efficiency percentage if we have expected rate
        if self.rate_info["expected_rate"] > 0:
            efficiency = (self.rate_info["current_rate"] / self.rate_info["expected_rate"]) * 100
            color = "green" if efficiency >= 80 else "yellow" if efficiency >= 50 else "red"
            rates_table.add_row("Efficiency", f"[{color}]{efficiency:.1f}%[/{color}]")

        layout["rates"].update(Panel(rates_table, title="Processing Rates", border_style="magenta"))

        # Statistics panel
        stats_table = Table(show_header=False, expand=True)
        stats_table.add_column("Metric")
        stats_table.add_column("Value", style="cyan")

        # Filter out rate stats (already shown in rates panel)
        for key, value in self.stats.items():
            if key not in ["current_rate", "average_rate", "expected_rate"]:
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                stats_table.add_row(key.replace("_", " ").title(), str(value))

        layout["stats"].update(Panel(stats_table, title="System Statistics", border_style="green"))

        # Leaderboard panel
        leaderboard_table = Table(expand=True)
        leaderboard_table.add_column("Rank", style="yellow")
        leaderboard_table.add_column("Contributor")
        leaderboard_table.add_column("Captions", style="cyan")
        leaderboard_table.add_column("Trust", style="green")

        for i, contributor in enumerate(self.leaderboard[:10], 1):
            # Format name with active worker count
            name = contributor.get("name", "Unknown")
            active_workers = contributor.get("active_workers", 0)

            if active_workers > 0:
                name_display = f"{name} [bright_green](x{active_workers})[/bright_green]"
            else:
                name_display = f"{name} [dim](offline)[/dim]"

            leaderboard_table.add_row(
                str(i),
                name_display,
                str(contributor.get("total_captions", 0)),
                "⭐" * contributor.get("trust_level", 0),
            )

        layout["leaderboard"].update(
            Panel(leaderboard_table, title="Top Contributors", border_style="yellow")
        )
        # Activity panel
        activity_text = Text()
        for activity in self.recent_activity[-10:]:
            activity_text.append(f"{activity}\n", style="dim")

        layout["activity"].update(
            Panel(activity_text, title="Recent Activity", border_style="blue")
        )

        # Footer
        layout["footer"].update(
            Panel(
                Text(
                    f"Updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit",
                    justify="center",
                    style="dim",
                ),
                border_style="bright_black",
            )
        )
