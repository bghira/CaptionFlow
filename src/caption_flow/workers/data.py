"""DataWorker for retrieving data from various sources and forwarding to orchestrator or storage."""

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any, AsyncIterator, Dict, Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config

from .base import BaseWorker

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """A single data sample to process."""

    sample_id: str
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None


class DataWorker(BaseWorker):
    """Worker that retrieves data from various sources and forwards to orchestrator/storage."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Data source configuration
        self.data_source = config.get("data_source")
        self.source_type = config.get("source_type", "auto")
        self.batch_size = config.get("batch_size", 10)

        # Storage configuration (will be updated from orchestrator)
        self.storage_config = None
        self.s3_client = None

        # State specific to data worker
        self.can_send = Event()  # For backpressure

        # Queues
        self.send_queue = Queue(maxsize=100)

    def _init_metrics(self):
        """Initialize data worker metrics."""
        self.samples_sent = 0
        self.samples_stored = 0
        self.samples_failed = 0

    def _get_auth_data(self) -> Dict[str, Any]:
        """Get authentication data."""
        return {"token": self.token, "name": self.name, "role": "data_worker"}

    async def _handle_welcome(self, welcome_data: Dict[str, Any]):
        """Handle welcome message from orchestrator."""
        self.storage_config = welcome_data.get("storage_config", {})

        # Setup S3 if configured
        if self.storage_config.get("s3", {}).get("enabled"):
            self._setup_s3_client(self.storage_config["s3"])

        logger.info(f"Storage config: {self.storage_config}")

        # Start with ability to send
        self.can_send.set()

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle message from orchestrator."""
        msg_type = data.get("type")

        if msg_type == "backpressure":
            # Orchestrator is overwhelmed
            self.can_send.clear()
            logger.info("Received backpressure signal")

        elif msg_type == "resume":
            # Orchestrator ready for more
            self.can_send.set()
            logger.info("Received resume signal")

    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Get heartbeat data."""
        return {
            "type": "heartbeat",
            "sent": self.samples_sent,
            "stored": self.samples_stored,
            "failed": self.samples_failed,
            "queue_size": self.send_queue.qsize(),
        }

    async def _create_tasks(self) -> list:
        """Create async tasks to run."""
        return [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._base_message_handler()),
            asyncio.create_task(self._data_processor()),
            asyncio.create_task(self._send_loop()),
        ]

    async def _on_disconnect(self):
        """Handle disconnection."""
        # Clear send capability
        self.can_send.clear()

        # Clear send queue
        try:
            while True:
                self.send_queue.get_nowait()
        except Empty:
            pass

    def _setup_s3_client(self, s3_config: Dict[str, Any]):
        """Setup S3 client from config."""
        if not s3_config:
            return None

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=s3_config.get("endpoint_url"),
                aws_access_key_id=s3_config.get("access_key"),
                aws_secret_access_key=s3_config.get("secret_key"),
                region_name=s3_config.get("region", "us-east-1"),
                config=Config(signature_version="s3v4"),
            )
            self.s3_bucket = s3_config.get("bucket")
            logger.info(f"S3 client configured for bucket: {self.s3_bucket}")
            return self.s3_client
        except Exception as e:
            logger.error(f"Failed to setup S3 client: {e}")
            return None

    async def _data_processor(self):
        """Process data from source."""
        try:
            batch = []

            async for sample in self._load_data_source():
                # Get image data
                if sample.image_data:
                    image_data = sample.image_data
                elif sample.image_url:
                    image_data = await self._download_image(sample.image_url)
                    if not image_data:
                        self.samples_failed += 1
                        continue
                else:
                    logger.warning(f"No image data for sample {sample.sample_id}")
                    continue

                # Store if configured
                if self.storage_config.get("forward_to_orchestrator", True):
                    # Add to send queue
                    batch.append(
                        {
                            "sample_id": sample.sample_id,
                            "image_data": image_data,
                            "metadata": sample.metadata,
                        }
                    )

                    if len(batch) >= self.batch_size:
                        # Wait for backpressure clearance
                        await asyncio.wait_for(self.can_send.wait(), timeout=300)

                        # Add batch to send queue
                        try:
                            self.send_queue.put_nowait(batch)
                            batch = []
                        except Exception:
                            # Queue full, wait
                            await asyncio.sleep(1)

                # Store locally/S3 if configured
                if self.storage_config.get("local", {}).get("enabled") or self.storage_config.get(
                    "s3", {}
                ).get("enabled"):
                    if await self._store_sample(sample, image_data):
                        self.samples_stored += 1

            # Send remaining batch
            if batch and self.storage_config.get("forward_to_orchestrator", True):
                await asyncio.wait_for(self.can_send.wait(), timeout=300)
                self.send_queue.put_nowait(batch)

        except Exception as e:
            logger.error(f"Data processing error: {e}")

    async def _send_loop(self):
        """Send data samples to orchestrator."""
        while self.running and self.connected.is_set():
            try:
                # Get batch from queue
                batch = await asyncio.get_event_loop().run_in_executor(
                    None, self.send_queue.get, True, 1
                )

                if batch and self.websocket:
                    # Send samples
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "submit_samples",
                                "samples": [
                                    {"sample_id": s["sample_id"], "metadata": s["metadata"]}
                                    for s in batch
                                ],
                                "batch_size": len(batch),
                            }
                        )
                    )

                    # Send actual image data separately
                    for sample in batch:
                        await self.websocket.send(sample["image_data"])

                    self.samples_sent += len(batch)
                    logger.info(f"Sent batch of {len(batch)} samples")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")

    async def _load_data_source(self) -> AsyncIterator[DataSample]:
        """Load data from configured source."""
        source_type = self.source_type

        if source_type == "auto":
            # Auto-detect based on file extension
            if self.data_source.endswith(".jsonl"):
                source_type = "jsonl"
            elif self.data_source.endswith(".csv"):
                source_type = "csv"
            elif self.data_source.endswith(".parquet"):
                source_type = "parquet"
            elif self.data_source.startswith("hf://") or "/" in self.data_source:
                source_type = "huggingface"

        logger.info(f"Loading data from {source_type} source: {self.data_source}")

        if source_type == "jsonl":
            async for sample in self._load_jsonl():
                yield sample
        elif source_type == "csv":
            async for sample in self._load_csv():
                yield sample
        elif source_type == "parquet":
            async for sample in self._load_parquet():
                yield sample
        elif source_type == "huggingface":
            async for sample in self._load_huggingface():
                yield sample
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    async def _load_jsonl(self) -> AsyncIterator[DataSample]:
        """Load data from JSONL file with URL list."""
        with open(self.data_source, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    sample = DataSample(
                        sample_id=data.get("id", f"sample_{line_num}"),
                        image_url=data.get("url") or data.get("image_url"),
                        metadata=data,
                    )
                    yield sample
                except Exception as e:
                    logger.error(f"Error loading line {line_num}: {e}")

    async def _load_csv(self) -> AsyncIterator[DataSample]:
        """Load data from CSV file."""
        df = pd.read_csv(self.data_source)

        # Try to find URL column
        url_cols = [col for col in df.columns if "url" in col.lower() or "link" in col.lower()]
        url_col = url_cols[0] if url_cols else None

        for idx, row in df.iterrows():
            sample = DataSample(
                sample_id=str(row.get("id", idx)),
                image_url=row.get(url_col) if url_col else None,
                metadata=row.to_dict(),
            )
            yield sample

    async def _load_parquet(self) -> AsyncIterator[DataSample]:
        """Load data from Parquet file."""
        table = pq.read_table(self.data_source)
        df = table.to_pandas()

        # Try to find URL column
        url_cols = [col for col in df.columns if "url" in col.lower() or "link" in col.lower()]
        url_col = url_cols[0] if url_cols else None

        for idx, row in df.iterrows():
            sample = DataSample(
                sample_id=str(row.get("id", idx)),
                image_url=row.get(url_col) if url_col else None,
                metadata=row.to_dict(),
            )
            yield sample

    async def _load_huggingface(self) -> AsyncIterator[DataSample]:
        """Load data from HuggingFace dataset."""
        from datasets import load_dataset

        # Parse dataset path
        if self.data_source.startswith("hf://"):
            dataset_path = self.data_source[5:]
        else:
            dataset_path = self.data_source

        # Load dataset
        ds = load_dataset(dataset_path, split="train", streaming=True)

        for idx, item in enumerate(ds):
            # Try to find image data
            image_url = None
            image_data = None

            if "image" in item and hasattr(item["image"], "save"):
                # PIL Image
                buffer = io.BytesIO()
                item["image"].save(buffer, format="PNG")
                image_data = buffer.getvalue()
            elif "url" in item:
                image_url = item["url"]
            elif "image_url" in item:
                image_url = item["image_url"]

            sample = DataSample(
                sample_id=item.get("id", f"hf_{idx}"),
                image_url=image_url,
                image_data=image_data,
                metadata=item,
            )
            yield sample

    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
        return None

    async def _store_sample(self, sample: DataSample, image_data: bytes) -> bool:
        """Store sample according to storage config."""
        stored = False

        # Store locally if configured
        if self.storage_config.get("local", {}).get("enabled"):
            local_dir = Path(self.storage_config["local"].get("path", "./data"))
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Save image
                image_path = local_dir / f"{sample.sample_id}.jpg"
                with open(image_path, "wb") as f:
                    f.write(image_data)

                # Save metadata
                meta_path = local_dir / f"{sample.sample_id}.json"
                with open(meta_path, "w") as f:
                    json.dump(sample.metadata or {}, f)

                stored = True
            except Exception as e:
                logger.error(f"Failed to store locally: {e}")

        # Store to S3 if configured
        if self.storage_config.get("s3", {}).get("enabled") and self.s3_client:
            try:
                # Upload image
                self.s3_client.put_object(
                    Bucket=self.s3_bucket, Key=f"images/{sample.sample_id}.jpg", Body=image_data
                )

                # Upload metadata
                if sample.metadata:
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=f"metadata/{sample.sample_id}.json",
                        Body=json.dumps(sample.metadata),
                    )

                stored = True
            except Exception as e:
                logger.error(f"Failed to store to S3: {e}")

        return stored
