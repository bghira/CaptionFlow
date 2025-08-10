"""TUI monitor for CaptionFlow system."""

import asyncio
import json
import logging
import ssl
from datetime import datetime
from typing import Dict, Any, List, Optional

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
        self.server_url = config["server"]
        self.token = config["token"]
        
        # SSL configuration
        self.ssl_context = self._setup_ssl()
        
        # Display state
        self.stats = {}
        self.leaderboard = []
        self.recent_activity = []
        self.running = False
        
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
                    # ssl=self.ssl_context
                ) as websocket:
                    # Authenticate
                    await websocket.send(json.dumps({
                        "token": self.token
                    }))
                    
                    # Receive updates
                    async for message in websocket:
                        data = json.loads(message)
                        await self._handle_update(data)
                        
            except Exception as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_update(self, data: Dict):
        """Process update from orchestrator."""
        msg_type = data.get("type")
        
        if msg_type == "stats":
            self.stats = data["data"]
        elif msg_type == "leaderboard":
            self.leaderboard = data["data"]
        elif msg_type == "activity":
            self.recent_activity.append(data["data"])
            # Keep only recent activity
            self.recent_activity = self.recent_activity[-20:]
    
    async def _display_loop(self):
        """Main display update loop."""
        layout = self._create_layout()
        
        with Live(layout, console=self.console, refresh_per_second=1) as live:
            while self.running:
                self._update_layout(layout)
                await asyncio.sleep(1)
    
    def _create_layout(self) -> Layout:
        """Create the display layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="leaderboard", ratio=1),
            Layout(name="activity", ratio=1)
        )
        
        return layout
    
    def _update_layout(self, layout: Layout):
        """Update layout with current data."""
        # Header
        layout["header"].update(Panel(
            Text("CaptionFlow Monitor", style="bold magenta", justify="center"),
            border_style="bright_blue"
        ))
        
        # Statistics panel
        stats_table = Table(show_header=False, expand=True)
        stats_table.add_column("Metric")
        stats_table.add_column("Value", style="cyan")
        
        for key, value in self.stats.items():
            stats_table.add_row(
                key.replace("_", " ").title(),
                str(value)
            )
        
        layout["stats"].update(Panel(
            stats_table,
            title="System Statistics",
            border_style="green"
        ))
        
        # Leaderboard panel
        leaderboard_table = Table(expand=True)
        leaderboard_table.add_column("Rank", style="yellow")
        leaderboard_table.add_column("Contributor")
        leaderboard_table.add_column("Captions", style="cyan")
        leaderboard_table.add_column("Trust", style="green")
        
        for i, contributor in enumerate(self.leaderboard[:10], 1):
            leaderboard_table.add_row(
                str(i),
                contributor.get("name", "Unknown"),
                str(contributor.get("total_captions", 0)),
                "⭐" * contributor.get("trust_level", 0)
            )
        
        layout["leaderboard"].update(Panel(
            leaderboard_table,
            title="Top Contributors",
            border_style="yellow"
        ))
        
        # Activity panel
        activity_text = Text()
        for activity in self.recent_activity[-10:]:
            activity_text.append(f"{activity}\n", style="dim")
        
        layout["activity"].update(Panel(
            activity_text,
            title="Recent Activity",
            border_style="blue"
        ))
        
        # Footer
        layout["footer"].update(Panel(
            Text(f"Updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit", 
                 justify="center", style="dim"),
            border_style="bright_black"
        ))