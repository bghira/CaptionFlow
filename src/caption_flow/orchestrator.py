"""Enhanced orchestrator with WebDataset integration for vLLM workers.

This orchestrator serves as the central collection point for all captions:
1. Distributes jobs to workers via WebSocket
2. Receives captions back from workers
3. Buffers captions in memory for efficiency
4. Periodically commits batches to Arrow/Parquet storage
5. Tracks attribution (which worker generated each caption)
6. Manages checkpoints for fault tolerance
"""

import asyncio
import json
import logging
import ssl
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, Any, List
import threading
from queue import Queue, Empty

import websockets
from websockets.server import WebSocketServerProtocol

from .storage import StorageManager
from .models import Job, JobStatus, Caption, Contributor
from .utils.auth import AuthManager
from .utils.job_queue import JobQueue
from .utils.dataset_loader import DatasetLoader, ShardTracker

logger = logging.getLogger(__name__)

class Orchestrator:
    """Enhanced orchestrator for vLLM-based distributed captioning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)
        
        # Dataset configuration
        self.dataset_config = config.get("dataset", {})
        self.dataset_path = self.dataset_config.get("path")
        self.dataset_type = self.dataset_config.get("type", "huggingface")
        
        # Initialize components
        storage_config = config.get("storage", {})
        self.storage = StorageManager(
            Path(storage_config.get("data_dir", "./caption_data")),
            caption_buffer_size=storage_config.get("caption_buffer_size", 100),
            job_buffer_size=storage_config.get("job_buffer_size", 100),
            contributor_buffer_size=storage_config.get("contributor_buffer_size", 10)
        )
        self.auth = AuthManager(config.get("auth", {}))
        self.job_queue = JobQueue()
        
        # Dataset components
        self.dataset_loader = None
        self.shard_tracker = None
        if self.dataset_path:
            self.dataset_loader = DatasetLoader(self.dataset_path, self.dataset_type)
            checkpoint_dir = Path(config.get("storage", {}).get("checkpoint_dir", "./checkpoints"))
            self.shard_tracker = ShardTracker(checkpoint_dir / "shards.json")
        
        # Track connections
        self.workers: Dict[str, WebSocketServerProtocol] = {}
        self.monitors: Set[WebSocketServerProtocol] = set()
        
        # SSL configuration
        self.ssl_context = self._setup_ssl()
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_captions": 0,
            "connected_workers": 0,
            "total_shards": 0,
            "completed_shards": 0,
            "current_shard": None,
            "buffer_size": 0,
            "total_written": 0,
            "last_checkpoint": None
        }
        
        # Shard processing state
        self.current_shard = None
        self.shard_items = []
        self.shard_index = 0
        self.all_shards = []
        self.shard_lock = threading.Lock()
        
        # Background job creation
        self.job_creation_thread = None
        self.stop_job_creation = threading.Event()
    
    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL if certificates are provided."""
        ssl_config = self.config.get("ssl", {})
        if not ssl_config.get("cert") or not ssl_config.get("key"):
            return None
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(ssl_config["cert"], ssl_config["key"])
        return context
    
    def _create_jobs_from_shard(self):
        """Background thread to create jobs from dataset shards."""
        if not self.dataset_loader:
            logger.warning("No dataset configured, skipping job creation")
            return
        
        logger.info("Starting job creation thread")
        
        # Get all shards
        self.all_shards = self.dataset_loader.get_shard_list()
        remaining_shards = self.shard_tracker.get_remaining_shards(self.all_shards)
        
        self.stats["total_shards"] = len(self.all_shards)
        self.stats["completed_shards"] = len(self.all_shards) - len(remaining_shards)
        
        logger.info(
            f"Total shards: {len(self.all_shards)}, "
            f"Remaining: {len(remaining_shards)}"
        )
        
        for shard_url in remaining_shards:
            if self.stop_job_creation.is_set():
                break
            
            shard_name = Path(shard_url).stem
            self.current_shard = shard_name
            self.stats["current_shard"] = shard_name
            
            # Get already processed keys for this shard
            processed_keys = self.shard_tracker.get_processed_keys(shard_name)
            
            logger.info(f"Processing shard {shard_name} (skipping {len(processed_keys)} processed keys)")
            
            try:
                # Load all items from shard
                shard_items = []
                for key, url, image_data in self.dataset_loader.iterate_shard(shard_url, processed_keys):
                    if self.stop_job_creation.is_set():
                        break
                    
                    # Create job
                    job = Job(
                        job_id=f"{shard_name}_{key}",
                        dataset=self.dataset_path,
                        shard=shard_name,
                        item_key=key,
                        status=JobStatus.PENDING
                    )
                    
                    # Add to queue
                    asyncio.run_coroutine_threadsafe(
                        self.job_queue.add(job),
                        asyncio.get_event_loop()
                    )
                    
                    shard_items.append(key)
                    self.stats["total_jobs"] += 1
                    
                    # Small delay to avoid overwhelming the queue
                    if len(shard_items) % 100 == 0:
                        logger.debug(f"Created {len(shard_items)} jobs from shard {shard_name}")
                
                # Update shard tracker
                if shard_items:
                    with self.shard_lock:
                        self.shard_items = shard_items
                
                logger.info(f"Created {len(shard_items)} jobs from shard {shard_name}")
                
            except Exception as e:
                logger.error(f"Error processing shard {shard_name}: {e}")
        
        logger.info("Job creation thread finished")
    
    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting vLLM orchestrator on {self.host}:{self.port}")
        
        # Load existing state
        await self.storage.initialize()
        await self._restore_state()
        
        # Start job creation thread if dataset is configured
        if self.dataset_loader:
            self.job_creation_thread = threading.Thread(
                target=self._create_jobs_from_shard,
                daemon=True
            )
            self.job_creation_thread.start()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._shard_completion_check())
        
        # Start WebSocket server
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context
        ):
            logger.info("vLLM Orchestrator ready for connections")
            await asyncio.Future()  # Run forever
    
    async def _shard_completion_check(self):
        """Periodically check if current shard is complete."""
        while True:
            await asyncio.sleep(30)
            
            if not self.current_shard or not self.shard_tracker:
                continue
            
            # Check if all jobs for current shard are done
            with self.shard_lock:
                if self.shard_items:
                    # Check completion in storage
                    completed_count = 0
                    for key in self.shard_items:
                        job_id = f"{self.current_shard}_{key}"
                        job = await self.storage.get_job(job_id)
                        if job and job.status == JobStatus.COMPLETED:
                            completed_count += 1
                    
                    if completed_count == len(self.shard_items):
                        logger.info(f"Shard {self.current_shard} complete!")
                        self.shard_tracker.mark_complete(self.current_shard)
                        self.stats["completed_shards"] += 1
                        self.shard_items = []
                    else:
                        # Update partial progress
                        completed_keys = []
                        for key in self.shard_items:
                            job_id = f"{self.current_shard}_{key}"
                            job = await self.storage.get_job(job_id)
                            if job and job.status == JobStatus.COMPLETED:
                                completed_keys.append(key)
                        
                        if completed_keys:
                            self.shard_tracker.update_partial(self.current_shard, completed_keys)
    
    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection."""
        try:
            # Authenticate
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)
            
            role = self.auth.authenticate(auth_data.get("token"))
            if not role:
                await websocket.send(json.dumps({"error": "Invalid token"}))
                return
            
            # Route by role
            if role == "worker":
                await self._handle_worker(websocket, auth_data)
            elif role == "monitor":
                await self._handle_monitor(websocket)
            else:
                await websocket.send(json.dumps({"error": "Unknown role"}))
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            await websocket.close()
    
    async def _handle_worker(self, websocket: WebSocketServerProtocol, auth_data: Dict):
        """Handle worker connection lifecycle."""
        worker_id = auth_data.get("name", str(uuid.uuid4()))
        self.workers[worker_id] = websocket
        self.stats["connected_workers"] = len(self.workers)
        
        # Register contributor
        contributor = Contributor(
            contributor_id=worker_id,
            name=worker_id,
            total_captions=0,
            trust_level=1
        )
        await self.storage.save_contributor(contributor)
        
        logger.info(f"Worker {worker_id} connected")
        await self._broadcast_stats()
        
        try:
            await websocket.send(json.dumps({
                "type": "welcome",
                "worker_id": worker_id
            }))
            
            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        finally:
            del self.workers[worker_id]
            self.stats["connected_workers"] = len(self.workers)
            await self._handle_worker_disconnect(worker_id)
            await self._broadcast_stats()
    
    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")
        
        if msg_type == "request_job":
            job = await self.job_queue.get_next()
            if job:
                job.assigned_to = worker_id
                job.status = JobStatus.PROCESSING
                await self.storage.save_job(job)
                
                await self.workers[worker_id].send(json.dumps({
                    "type": "job",
                    "job": asdict(job)
                }))
                
                logger.debug(f"Assigned job {job.job_id} to worker {worker_id}")
            else:
                await self.workers[worker_id].send(json.dumps({
                    "type": "no_jobs"
                }))
        
        elif msg_type == "submit_caption":
            await self._handle_caption_submission(worker_id, data)
        
        elif msg_type == "job_failed":
            await self._handle_job_failure(worker_id, data)
        
        elif msg_type == "heartbeat":
            # Update worker stats
            pass
    
    async def _handle_caption_submission(self, worker_id: str, data: Dict):
        """Process caption submission from worker - central collection point."""
        job_id = data["job_id"]
        caption_text = data["caption"]
        
        logger.debug(f"Received caption for job {job_id} from worker {worker_id}")
        
        # Create caption record with attribution
        caption = Caption(
            job_id=job_id,
            dataset=data.get("dataset"),
            shard=data.get("shard"),
            item_key=data.get("item_key"),
            caption=caption_text,
            contributor_id=worker_id,  # Track who generated this
            timestamp=datetime.utcnow(),
            quality_score=None
        )
        
        # Add to central storage buffer (will batch write when full)
        await self.storage.save_caption(caption)
        
        # Update job status
        job = await self.storage.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            await self.storage.save_job(job)
        
        # Update statistics
        self.stats["completed_jobs"] += 1
        self.stats["total_captions"] += 1
        self.stats["buffer_size"] = len(self.storage.caption_buffer)
        
        # Update contributor stats
        contributor = await self.storage.get_contributor(worker_id)
        if contributor:
            contributor.total_captions += 1
            await self.storage.save_contributor(contributor)
        
        # Broadcast updated stats to monitors
        await self._broadcast_stats()
        
        # Acknowledge receipt to worker
        await self.workers[worker_id].send(json.dumps({
            "type": "ack",
            "job_id": job_id
        }))
        
        # Log progress periodically
        if self.stats["total_captions"] % 100 == 0:
            logger.info(f"Collected {self.stats['total_captions']} captions centrally")
    
    async def _handle_job_failure(self, worker_id: str, data: Dict):
        """Handle job failure from worker."""
        job_id = data["job_id"]
        error = data.get("error", "Unknown error")
        
        logger.warning(f"Job {job_id} failed on worker {worker_id}: {error}")
        
        # Update job status
        job = await self.storage.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            await self.storage.save_job(job)
        
        self.stats["failed_jobs"] += 1
        await self._broadcast_stats()
    
    async def _handle_worker_disconnect(self, worker_id: str):
        """Handle worker disconnection - requeue jobs."""
        # Find and requeue any assigned jobs
        jobs = await self.storage.get_jobs_by_worker(worker_id)
        for job in jobs:
            if job.status == JobStatus.PROCESSING:
                job.status = JobStatus.PENDING
                job.assigned_to = None
                await self.storage.save_job(job)
                await self.job_queue.add(job)
                logger.info(f"Requeued job {job.job_id}")
    
    async def _handle_monitor(self, websocket: WebSocketServerProtocol):
        """Handle monitor connection."""
        self.monitors.add(websocket)
        logger.info("Monitor connected")
        
        try:
            # Send initial stats
            await websocket.send(json.dumps({
                "type": "stats",
                "data": self.stats
            }))
            
            # Send contributor leaderboard
            contributors = await self.storage.get_top_contributors(10)
            await websocket.send(json.dumps({
                "type": "leaderboard",
                "data": [asdict(c) for c in contributors]
            }))
            
            # Keep connection alive
            async for _ in websocket:
                pass
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        finally:
            self.monitors.discard(websocket)
    
    async def _broadcast_stats(self):
        """Broadcast statistics to all monitors."""
        if not self.monitors:
            return
        
        message = json.dumps({
            "type": "stats",
            "data": self.stats
        })
        
        # Send to all monitors
        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(monitor)
        
        # Clean up disconnected monitors
        self.monitors -= disconnected
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections."""
        while True:
            await asyncio.sleep(30)
            
            # Ping workers
            disconnected = []
            for worker_id, ws in self.workers.items():
                try:
                    await ws.ping()
                except:
                    disconnected.append(worker_id)
            
            # Clean up disconnected workers
            for worker_id in disconnected:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    await self._handle_worker_disconnect(worker_id)
    
    async def _checkpoint_loop(self):
        """Periodically checkpoint storage - forces central buffer flush."""
        interval = self.config.get("storage", {}).get("checkpoint_interval", 1000)
        
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Force checkpoint at regular intervals
            if self.stats["total_captions"] > 0 and self.stats["total_captions"] % interval == 0:
                logger.info(f"Triggering checkpoint at {self.stats['total_captions']} captions")
                await self.storage.checkpoint()
                
                # Update stats
                self.stats["last_checkpoint"] = datetime.utcnow().isoformat()
                self.stats["total_written"] = self.storage.total_captions_written
                self.stats["buffer_size"] = len(self.storage.caption_buffer)
                
                await self._broadcast_stats()
                logger.info(f"Checkpoint complete. Total written to disk: {self.stats['total_written']}")
    
    async def _restore_state(self):
        """Restore state from storage on startup."""
        # Load pending jobs back into queue
        pending_jobs = await self.storage.get_pending_jobs()
        for job in pending_jobs:
            await self.job_queue.add(job)
        
        # Update statistics
        self.stats["total_jobs"] = await self.storage.count_jobs()
        self.stats["completed_jobs"] = await self.storage.count_completed_jobs()
        self.stats["total_captions"] = await self.storage.count_captions()
        
        logger.info(f"Restored state: {self.stats['total_jobs']} jobs, {self.stats['total_captions']} captions")
    
    async def shutdown(self):
        """Graceful shutdown with final buffer flush."""
        logger.info("Shutting down orchestrator...")
        
        # Stop job creation
        self.stop_job_creation.set()
        if self.job_creation_thread:
            self.job_creation_thread.join(timeout=5)
        
        # Close all connections
        for ws in list(self.workers.values()):
            await ws.close()
        for ws in list(self.monitors):
            await ws.close()
        
        # Final checkpoint - flush all remaining captions to disk
        logger.info(f"Final flush: {len(self.storage.caption_buffer)} captions in buffer")
        await self.storage.checkpoint()
        
        # Log final statistics
        logger.info(f"Shutdown complete. Total captions collected and written: {self.storage.total_captions_written}")
        
        await self.storage.close()