"""Worker node for distributed captioning - Fixed WebSocket handling."""

import asyncio
import json
import logging
import ssl
from typing import Dict, Any, Optional
from pathlib import Path

import websockets
import websockets.exceptions
from websockets.client import WebSocketClientProtocol

from .models import Job, JobStatus
from .utils.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class Worker:
    """Worker node that processes captioning jobs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config["server"]
        self.token = config["token"]
        self.name = config.get("name", "worker")
        self.batch_size = config.get("batch_size", 32)

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Components
        self.image_processor = ImageProcessor()

        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.current_job: Optional[Job] = None

        # Event to signal when a job is received
        self.job_received = asyncio.Event()

        # Metrics
        self.processed_count = 0
        self.error_count = 0

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context."""
        if self.server_url.startswith("ws://"):
            logger.warning(
                "Using insecure WebSocket connection (ws://). Consider using wss:// for production."
            )
            return None
        if not self.config.get("verify_ssl", True):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        return ssl.create_default_context()

    async def start(self):
        """Start the worker and connect to orchestrator."""
        self.running = True

        while self.running:
            try:
                await self._connect_and_run()
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def _connect_and_run(self):
        """Connect to orchestrator and process jobs."""
        logger.info(f"Connecting to {self.server_url}")

        try:
            async with websockets.connect(self.server_url, ssl=self.ssl_context) as websocket:
                self.websocket = websocket

                # Authenticate
                await self._authenticate()

                # Wait for welcome
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)

                if "error" in welcome_data:
                    logger.error(f"Authentication failed: {welcome_data['error']}")
                    self.running = False
                    return

                self.worker_id = welcome_data.get("worker_id")
                logger.info(f"Connected as {self.worker_id}")

                # Create tasks for concurrent operations
                tasks = [
                    asyncio.create_task(self._heartbeat_loop()),
                    asyncio.create_task(self._job_processing_loop()),
                    asyncio.create_task(self._message_handler()),
                ]

                try:
                    # Wait for any task to complete
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check for errors in completed tasks
                    for task in done:
                        try:
                            task.result()
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("WebSocket connection closed")
                        except Exception as e:
                            logger.error(f"Task error: {e}")

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by orchestrator")

        except Exception as e:
            logger.error(f"Unexpected error in connection: {e}")
            raise
        finally:
            self.websocket = None
            self.current_job = None
            self.job_received.clear()

    async def _authenticate(self):
        """Send authentication message."""
        await self.websocket.send(json.dumps({"token": self.token, "name": self.name}))

    async def _job_processing_loop(self):
        """Main loop for requesting and processing jobs."""
        while self.running and self.websocket:
            try:
                # Clear the event before requesting
                self.job_received.clear()
                self.current_job = None

                # Request a job
                await self._send_message({"type": "request_job"})

                # Wait for job to be received (with timeout)
                try:
                    await asyncio.wait_for(self.job_received.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    # No job received, continue loop
                    logger.debug("No job received within timeout")
                    await asyncio.sleep(5)
                    continue

                # Process the job if we have one
                if self.current_job:
                    await self._process_job(self.current_job)
                    self.current_job = None

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed during job processing")
                break
            except Exception as e:
                logger.error(f"Job processing error: {e}")
                self.error_count += 1
                await asyncio.sleep(1)

    async def _message_handler(self):
        """Handle ALL incoming messages from orchestrator."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid message format: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed while waiting for messages")
        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def _handle_message(self, data: Dict):
        """Process a single message from orchestrator."""
        msg_type = data.get("type")

        if msg_type == "job":
            job_data = data["job"]
            self.current_job = self._create_job_from_data(job_data)
            logger.info(f"Received job {self.current_job.job_id}")
            # Signal that a job was received
            self.job_received.set()

        elif msg_type == "no_jobs":
            logger.debug("No jobs available")
            # Signal that we got a response (even if no job)
            self.job_received.set()

        elif msg_type == "ack":
            logger.debug(f"Job {data['job_id']} acknowledged")
            self.processed_count += 1

        elif msg_type == "error":
            logger.error(f"Server error: {data.get('message', 'Unknown error')}")

        else:
            logger.debug(f"Received message type: {msg_type}")

    def _create_job_from_data(self, job_data: Dict) -> Job:
        """Create Job object from dictionary data."""
        # Handle status field properly
        if isinstance(job_data.get("status"), str):
            job_data["status"] = JobStatus(job_data["status"])
        return Job(**job_data)

    async def _process_job(self, job: Job):
        """Process a single captioning job."""
        logger.info(f"Processing job {job.job_id}")

        try:
            # Load and preprocess images
            images = await self._load_images(job)

            # Generate caption (override in subclass for actual implementation)
            caption = await self._generate_caption(job, images)

            # Submit result
            await self._send_message(
                {
                    "type": "submit_caption",
                    "job_id": job.job_id,
                    "dataset": job.dataset,
                    "shard": job.shard,
                    "item_key": job.item_key,
                    "caption": caption,
                }
            )

            logger.info(f"Completed job {job.job_id}")

        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")

            # Report failure
            await self._send_message({"type": "job_failed", "job_id": job.job_id, "error": str(e)})

    async def _generate_caption(self, job: Job, images: Any) -> str:
        """Generate caption for images. Override in subclass."""
        # Placeholder - override in VLLMWorker
        return f"[Generated caption for {job.item_key}]"

    async def _load_images(self, job: Job):
        """Load and preprocess images for a job."""
        # Placeholder - implement actual loading
        return []

    async def _send_message(self, data: Dict):
        """Send a message to the orchestrator."""
        if not self.websocket:
            logger.warning("No websocket connection")
            return

        try:
            await self.websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while sending message")
            raise

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        while self.running and self.websocket:
            try:
                await self._send_message(
                    {
                        "type": "heartbeat",
                        "processed": self.processed_count,
                        "errors": self.error_count,
                    }
                )
                await asyncio.sleep(30)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed during heartbeat")
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        self.running = False

        if self.websocket:
            await self.websocket.close()
