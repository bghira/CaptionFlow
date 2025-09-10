"""Base worker class for WebSocket-based distributed workers."""

import asyncio
import json
import logging
import ssl
from abc import ABC, abstractmethod
from threading import Event
from typing import Any, Dict, Optional

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Base class for all WebSocket-based workers with common connection logic."""

    gpu_id: Optional[int] = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[ClientConnection] = None
        self.running = False
        self.connected = Event()
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Metrics (subclasses can extend)
        self._init_metrics()

    def _init_metrics(self):
        """Initialize basic metrics. Subclasses can override to add more."""
        pass

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context."""
        if self.server_url.startswith("ws://"):
            logger.warning("Using insecure WebSocket connection")
            return None

        if not self.config.get("verify_ssl", True):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

        return ssl.create_default_context()

    async def start(self):
        """Start the worker with automatic reconnection."""
        self.running = True

        # Allow subclasses to initialize before connection
        await self._pre_start()

        # Capture the main event loop
        self.main_loop = asyncio.get_running_loop()

        # Reconnection with exponential backoff
        reconnect_delay = 5
        max_delay = 60

        while self.running:
            try:
                await self._connect_and_run()
                reconnect_delay = 5  # Reset delay on successful connection
            except Exception as e:
                logger.error(f"Connection error: {e}", exc_info=True)
                self.connected.clear()
                self.websocket = None

                # Let subclass handle disconnection
                await self._on_disconnect()

            if self.running:
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

    async def _connect_and_run(self):
        """Connect to orchestrator and run main loop."""
        logger.info(f"Connecting to {self.server_url}")
        async with websockets.connect(
            self.server_url,
            ssl=self.ssl_context,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10,
        ) as websocket:
            self.websocket = websocket
            self.connected.set()

            # Authenticate
            auth_data = self._get_auth_data()
            await websocket.send(json.dumps(auth_data))

            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if "error" in welcome_data:
                logger.error(f"Authentication failed: {welcome_data['error']}")
                self.running = False
                return

            self.worker_id = welcome_data.get("worker_id")
            logger.info(f"Connected as {self.worker_id}")

            # Let subclass handle welcome data
            await self._handle_welcome(welcome_data)

            # Start processing
            try:
                tasks = await self._create_tasks()
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            finally:
                self.connected.clear()
                self.websocket = None

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        try:
            while self.running and self.connected.is_set():
                try:
                    if self.websocket:
                        heartbeat_data = self._get_heartbeat_data()
                        await self.websocket.send(json.dumps(heartbeat_data))
                    await asyncio.sleep(30)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"Connection lost during heartbeat: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    raise
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
            raise

    async def _base_message_handler(self):
        """Base message handler that delegates to subclass."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid message format: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed by orchestrator: {e}")
            raise
        except Exception as e:
            logger.error(f"Message handler error: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info(f"Shutting down {self.__class__.__name__}...")
        self.running = False
        self.connected.clear()

        # Let subclass do cleanup
        await self._pre_shutdown()

        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        logger.info(f"{self.__class__.__name__} shutdown complete")

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _get_auth_data(self) -> Dict[str, Any]:
        """Get authentication data to send to orchestrator."""
        pass

    @abstractmethod
    async def _handle_welcome(self, welcome_data: Dict[str, Any]):
        """Handle welcome message from orchestrator."""
        pass

    @abstractmethod
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle a message from orchestrator."""
        pass

    @abstractmethod
    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Get data to include in heartbeat."""
        pass

    @abstractmethod
    async def _create_tasks(self) -> list:
        """Create async tasks to run. Must include _heartbeat_loop and _base_message_handler."""
        pass

    # Optional hooks for subclasses

    async def _pre_start(self):
        """Called before starting connection loop. Override to initialize components."""
        pass

    async def _on_disconnect(self):
        """Called when disconnected. Override to clean up state."""
        pass

    async def _pre_shutdown(self):
        """Called before shutdown. Override for cleanup."""
        pass
