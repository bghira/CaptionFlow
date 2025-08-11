"""Worker node for distributed captioning."""

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

        # Dataset configuration will be received from orchestrator
        self.dataset_config = None
        self.dataset_type = None
        self.dataset_path = None

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Components
        self.image_processor = ImageProcessor()

        # State
        self.worker_id: Optional[str] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.current_job: Optional[Job] = None

        # Metrics
        self.processed_count = 0
        self.error_count = 0

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context."""
        # Check if URL is WSS (requires SSL)
        if self.server_url.startswith("ws://"):
            logger.warning(
                "Using insecure WebSocket connection (ws://). Consider using wss:// for production."
            )
            return None  # No SSL for ws://

        if not self.config.get("verify_ssl", True):
            # Disable SSL verification for development
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
                await websocket.send(json.dumps({"token": self.token, "name": self.name}))

                # Wait for welcome message with dataset configuration
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)

                if "error" in welcome_data:
                    logger.error(f"Authentication failed: {welcome_data['error']}")
                    self.running = False
                    return

                self.worker_id = welcome_data.get("worker_id")

                # Extract and store dataset configuration from orchestrator
                if "dataset_config" in welcome_data:
                    self.dataset_config = welcome_data["dataset_config"]
                    self.dataset_type = self.dataset_config.get("dataset_type")
                    self.dataset_path = self.dataset_config.get("dataset_path")
                    logger.info(
                        f"Received dataset configuration from orchestrator: "
                        f"type={self.dataset_type}, path={self.dataset_path}"
                    )
                else:
                    logger.warning("No dataset configuration received from orchestrator")

                logger.info(f"Connected as {self.worker_id}")

                # Create tasks for concurrent operations
                tasks = [
                    asyncio.create_task(self._heartbeat_loop()),
                    asyncio.create_task(self._job_processing_loop()),
                    asyncio.create_task(self._message_handler()),
                ]

                try:
                    # Wait for any task to complete (usually due to connection close)
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check if we had an error in completed tasks
                    for task in done:
                        try:
                            task.result()
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("WebSocket connection closed")
                        except Exception as e:
                            logger.error(f"Task error: {e}")

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by orchestrator")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Failed to connect: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in connection: {e}")
            raise
        finally:
            self.websocket = None
            self.current_job = None

    async def _job_processing_loop(self):
        """Main loop for requesting and processing jobs."""
        while self.running and self.websocket:
            try:
                # Request a job
                await self.websocket.send(json.dumps({"type": "request_job"}))

                # Wait a bit for response
                await asyncio.sleep(1)

                if self.current_job:
                    await self._process_job(self.current_job)
                    self.current_job = None
                else:
                    # No job available, wait before requesting again
                    await asyncio.sleep(5)

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed during job processing")
                break
            except Exception as e:
                logger.error(f"Job processing error: {e}")
                self.error_count += 1
                await asyncio.sleep(1)

    async def _message_handler(self):
        """Handle incoming messages from orchestrator."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "job":
                        job_data = data["job"]
                        self.current_job = Job(**job_data)
                        logger.info(f"Received job {self.current_job.job_id}")

                    elif msg_type == "no_jobs":
                        logger.debug("No jobs available")

                    elif msg_type == "ack":
                        logger.debug(f"Job {data['job_id']} acknowledged")
                        self.processed_count += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid message format: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed while waiting for messages")
        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def _process_job(self, job: Job):
        """Process a single captioning job."""
        if not self.websocket:
            logger.warning(f"No websocket connection, skipping job {job.job_id}")
            return

        logger.info(f"Processing job {job.job_id}")

        try:
            # Load and preprocess images
            images = await self._load_images(job)

            # TODO: Here you would integrate your captioning model
            # For now, using placeholder
            caption = f"[Generated caption for {job.item_key}]"

            # Submit result
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "submit_caption",
                        "job_id": job.job_id,
                        "dataset": job.dataset,
                        "shard": job.shard,
                        "item_key": job.item_key,
                        "caption": caption,
                    }
                )
            )

            logger.info(f"Completed job {job.job_id}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection lost while processing job {job.job_id}")
            raise  # Re-raise to trigger reconnection
        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")

            # Report failure if still connected
            if self.websocket:
                try:
                    await self.websocket.send(
                        json.dumps({"type": "job_failed", "job_id": job.job_id, "error": str(e)})
                    )
                except:
                    pass  # Connection might be closed

    async def _load_images(self, job: Job):
        """Load and preprocess images for a job."""
        # This would load actual images from the dataset
        # Now can use self.dataset_type and self.dataset_path received from orchestrator
        # For now, returning placeholder
        return []

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        while self.running and self.websocket:
            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": "heartbeat",
                            "processed": self.processed_count,
                            "errors": self.error_count,
                        }
                    )
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
