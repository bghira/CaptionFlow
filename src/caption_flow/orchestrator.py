import asyncio
import datetime as _datetime
import json
import logging
import os
import ssl
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection

from .models import Caption, Contributor, JobId
from .processors import (
    HuggingFaceDatasetOrchestratorProcessor,
    LocalFilesystemOrchestratorProcessor,
    ProcessorConfig,
    WebDatasetOrchestratorProcessor,
    WorkAssignment,
    WorkResult,
)
from .storage import StorageManager
from .utils.auth import AuthManager
from .utils.json_utils import safe_json_dumps

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class Orchestrator:
    """Generic orchestrator for distributed work processing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)

        # Processor configuration
        processor_type = config.get("dataset", {}).get("processor_type", None)
        assert (
            processor_type is not None
        ), "You must supply processor_type in your orchestrator dataset configuration."
        processor_config = ProcessorConfig(processor_type=processor_type, config=config)

        # Initialize processor
        if processor_type == "webdataset":
            self.processor = WebDatasetOrchestratorProcessor()
        elif processor_type == "huggingface_datasets":
            self.processor = HuggingFaceDatasetOrchestratorProcessor()
        elif processor_type == "local_filesystem":
            self.processor = LocalFilesystemOrchestratorProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

        # Initialize components
        storage_config = config.get("storage", {})
        self.storage = StorageManager(
            Path(storage_config.get("data_dir", "./caption_data")),
            caption_buffer_size=storage_config.get("caption_buffer_size", 1000),
        )
        self.auth = AuthManager(config.get("auth", {}))
        self.processor.initialize(processor_config, self.storage)

        # Processing configuration
        self.chunks_per_request = config.get("chunks_per_request", 2)

        # Track connections
        self.workers: Dict[str, ServerConnection] = {}
        self.monitors: Set[ServerConnection] = set()
        self.workers_by_user = defaultdict(set)

        # SSL configuration
        self.ssl_context = self._setup_ssl()

        # Statistics
        self.stats = {
            "connected_workers": 0,
            "total_outputs": 0,
            "last_checkpoint": None,
            "processor_stats": {},
        }

        # Cache for leaderboard
        self._cached_leaderboard = None

        # Data worker stuff
        self.data_workers = {}
        self.data_sample_queue = asyncio.Queue()
        self.backpressure_threshold = config.get("backpressure_threshold", 1000)

        # Rate tracking
        self.rate_tracker = {
            "start_time": time.time(),
            "last_update_time": time.time(),
            "last_output_count": 0,
            "current_rate": 0.0,
            "average_rate": 0.0,
        }

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL if certificates are provided."""
        ssl_config = self.config.get("ssl", {})
        if not ssl_config.get("cert") or not ssl_config.get("key"):
            return None

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(ssl_config["cert"], ssl_config["key"])
        return context

    async def start(self):
        """Start the orchestrator server."""
        logger.info(f"Starting orchestrator on {self.host}:{self.port}")
        processor_type = self.config.get("dataset", {}).get("processor_type", None)
        if not processor_type:
            logger.info(f"Config: {self.config}")
            raise ValueError(
                "You must supply processor_type in your orchestrator dataset configuration."
            )
        logger.info(f"Processor type: {processor_type}")

        # Initialize storage
        await self.storage.initialize()

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._stats_update_loop())

        await self.update_unprocessed_ranges()

        # Start WebSocket server
        websocket_logger = logging.getLogger("websockets")
        websocket_logger.setLevel(logging.WARNING)
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            logger=websocket_logger,
        ):
            logger.info("Orchestrator ready for connections")
            await asyncio.Future()  # Run forever

    def get_workers_by_user_stats(self) -> Dict[str, Dict]:
        """Get worker statistics grouped by user."""
        stats = {}
        for user, worker_ids in self.workers_by_user.items():
            stats[user] = {"worker_ids": list(worker_ids), "count": len(worker_ids)}
        return stats

    async def update_unprocessed_ranges(self):
        """Update unprocessed ranges based on what's actually in storage."""
        if not self.processor or not self.storage:
            return

        processed_job_ids = self.storage.get_all_processed_job_ids()
        self.processor.update_from_storage(processed_job_ids)

    async def _send_leaderboard_to_monitor(self, websocket: ServerConnection):
        """Alias for _send_monitor_leaderboard for backward compatibility."""
        await self._send_monitor_leaderboard(websocket)

    async def handle_connection(self, websocket: ServerConnection):
        """Handle new WebSocket connection."""
        try:
            # Authenticate
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)

            auth_ticket = self.auth.authenticate(auth_data.get("token"))
            if not auth_ticket.role:
                await websocket.send(safe_json_dumps({"error": "Invalid token"}))
                return

            if auth_ticket.role == "worker":
                await self._handle_worker(websocket, auth_ticket)
            elif auth_ticket.role == "monitor":
                await self._handle_monitor(websocket)
            elif auth_ticket.role == "admin":
                await self._handle_admin(websocket, auth_ticket)
            elif auth_ticket.role == "data_worker":
                await self._handle_data_worker(websocket, auth_ticket)
            else:
                await websocket.send(
                    safe_json_dumps({"error": f"Unknown role: {auth_ticket.role}"})
                )

        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            await websocket.close()

    async def _handle_worker(self, websocket: ServerConnection, auth_ticket):
        """Handle worker connection lifecycle."""
        # Generate unique worker ID
        base_name = getattr(auth_ticket, "name", "worker")
        worker_id = f"{base_name}_{str(uuid.uuid4())[:8]}"
        worker_user = base_name

        self.workers[worker_id] = websocket
        self.workers_by_user[worker_user].add(worker_id)
        self.stats["connected_workers"] = len(self.workers)

        # Register contributor
        contributor = await self.storage.get_contributor(worker_user)
        if not contributor:
            contributor = Contributor(
                contributor_id=worker_user,
                name=worker_user,
                total_captions=0,
                trust_level=1,
            )
            await self.storage.save_contributor(contributor)

        logger.info(f"Worker {worker_id} (user: {worker_user}) is retrieving configuration")
        try:
            # Send welcome message with processor config
            filtered_config = self.config.copy()
            for unwanted_key in ["auth", "orchestrator", "storage"]:
                filtered_config.pop(unwanted_key, None)
            welcome_message = {
                "type": "welcome",
                "worker_id": worker_id,
                "user_id": worker_user,
                "processor_type": self.config.get("dataset", {}).get("processor_type", None),
                "processor_config": filtered_config,
            }
            await websocket.send(safe_json_dumps(welcome_message))

            async for message in websocket:
                data = json.loads(message)
                await self._process_worker_message(worker_id, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} has disconnected due to websocket connection closure")
        finally:
            if worker_id in self.workers:
                del self.workers[worker_id]

            self.workers_by_user[worker_user].discard(worker_id)
            if not self.workers_by_user[worker_user]:
                del self.workers_by_user[worker_user]

            self.stats["connected_workers"] = len(self.workers)

            # Release assignments
            self.processor.release_assignments(worker_id)
            logger.info(f"Worker {worker_id} has safely disconnected")

    def _auth_configs_equal(
        self, current_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> bool:
        """Compare two auth configurations for equality."""

        # Helper function to normalize token lists for comparison
        def normalize_tokens(tokens):
            if not tokens:
                return []
            # Sort by token for consistent comparison
            return sorted(
                [{"name": t.get("name"), "token": t.get("token")} for t in tokens],
                key=lambda x: x.get("token", ""),
            )

        # Compare each token type
        current_workers = normalize_tokens(current_config.get("worker_tokens", []))
        new_workers = normalize_tokens(new_config.get("worker_tokens", []))

        current_admins = normalize_tokens(current_config.get("admin_tokens", []))
        new_admins = normalize_tokens(new_config.get("admin_tokens", []))

        current_monitors = normalize_tokens(current_config.get("monitor_tokens", []))
        new_monitors = normalize_tokens(new_config.get("monitor_tokens", []))

        return (
            current_workers == new_workers
            and current_admins == new_admins
            and current_monitors == new_monitors
        )

    async def _handle_config_reload(self, websocket: ServerConnection, new_config: Dict[str, Any]):
        """Handle configuration reload request."""
        logger.info("Processing configuration reload request")

        updated_sections = []
        warnings = []

        try:
            # Extract orchestrator section if present
            if "orchestrator" in new_config:
                orchestrator_config = new_config["orchestrator"]
            else:
                orchestrator_config = new_config

            # Update processor configuration if present
            if "processor_type" in orchestrator_config:
                old_type = self.config.get("processor_type")
                new_type = orchestrator_config["processor_type"]

                if old_type != new_type:
                    warnings.append("Processor type changes require orchestrator restart")
                    updated_sections.append("processor_type")
                else:
                    # Update processor config
                    self.config.update(orchestrator_config)

                    # Reinitialize processor with new config
                    processor_config = ProcessorConfig(
                        processor_type=new_type, config=orchestrator_config
                    )
                    self.processor.initialize(processor_config)
                    updated_sections.append("processor_config")

            # Update chunks per request
            if "chunks_per_request" in orchestrator_config:
                self.chunks_per_request = orchestrator_config["chunks_per_request"]
                updated_sections.append("chunks_per_request")

            # Update auth configuration
            if "auth" in orchestrator_config:
                try:
                    current_auth_config = self.config.get("auth", {})
                    new_auth_config = orchestrator_config["auth"]

                    # Only recreate AuthManager if auth config has actually changed
                    if not self._auth_configs_equal(current_auth_config, new_auth_config):
                        self.auth = AuthManager(new_auth_config)
                        updated_sections.append("auth")
                        logger.info("Auth configuration updated due to changes")
                    else:
                        logger.info("Auth configuration unchanged, preserving existing AuthManager")
                except Exception as e:
                    logger.error(f"Failed to update AuthManager: {e}")
                    warnings.append(f"Auth update failed: {e}")

            # Update storage settings
            if "storage" in orchestrator_config:
                storage_config = orchestrator_config["storage"]

                if "caption_buffer_size" in storage_config:
                    self.storage.caption_buffer_size = storage_config["caption_buffer_size"]
                    updated_sections.append("storage.caption_buffer_size")

            # Update main config
            if "orchestrator" in new_config:
                self.config = new_config["orchestrator"]
            else:
                self.config.update(orchestrator_config)

            # Send success response
            await websocket.send(
                safe_json_dumps(
                    {"type": "reload_complete", "updated": updated_sections, "warnings": warnings}
                )
            )

            logger.info(f"Configuration reloaded. Updated sections: {', '.join(updated_sections)}")
            await self._send_activity(
                f"Configuration reloaded by admin: {', '.join(updated_sections)}"
            )

        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            await websocket.send(safe_json_dumps({"type": "reload_failed", "error": str(e)}))

    async def _process_worker_message(self, worker_id: str, data: Dict):
        """Process message from worker."""
        msg_type = data.get("type")

        if msg_type == "get_work_units":
            count = data.get("count", self.chunks_per_request)
            units = self.processor.get_work_units(count, worker_id)
            logger.debug(f"Assigning units: {[unit.chunk_id for unit in units]}")

            if units:
                # Create assignment
                assignment = WorkAssignment(
                    assignment_id=str(uuid.uuid4()),
                    worker_id=worker_id,
                    units=units,
                    assigned_at=datetime.now(_datetime.UTC),
                )

                await self.workers[worker_id].send(
                    safe_json_dumps({"type": "work_assignment", "assignment": assignment.to_dict()})
                )

                logger.debug(f"Assigned {len(units)} work units to worker {worker_id}")
            else:
                if worker_id in self.workers:
                    await self.workers[worker_id].send(safe_json_dumps({"type": "no_work"}))

        elif msg_type == "work_complete":
            unit_id = data["unit_id"]
            self.processor.mark_completed(unit_id, worker_id)
            logger.debug(f"Work unit {unit_id} completed by worker {worker_id}")

        elif msg_type == "work_failed":
            unit_id = data["unit_id"]
            error = data.get("error", "Unknown error")
            self.processor.mark_failed(unit_id, worker_id, error)
            logger.warning(f"Work unit {unit_id} failed on worker {worker_id}: {error}")

        elif msg_type == "submit_results":
            await self._handle_results_submission(worker_id, data)

        elif msg_type == "heartbeat":
            logger.debug(f"Heartbeat from {worker_id}: {data}")

    async def _handle_results_submission(self, worker_id: str, data: Dict):
        """Process results submission from worker - fires off async task and returns immediately."""
        # Fire and forget - process in background
        asyncio.create_task(self._process_result_async(worker_id, data))

    async def _process_result_async(self, worker_id: str, data: Dict):
        """Actually process the result in background."""
        try:
            # Extract user from worker_id
            worker_user = worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id

            # Create work result
            _job_id = data.get("job_id")
            job_id = JobId.from_str(_job_id)
            shard_name = job_id.shard_id
            chunk_name = job_id.chunk_id

            result = WorkResult(
                unit_id=data["unit_id"],
                source_id=shard_name,
                chunk_id=job_id.get_chunk_str(),
                sample_id=data["sample_id"],
                dataset=data["dataset"],
                outputs=data["outputs"],
                metadata=data.get("metadata", {}),
                processing_time_ms=data.get("processing_time_ms", 0),
            )

            # Let processor handle any custom processing - this updates chunk tracker
            # IMPORTANT: Call this BEFORE saving to storage so chunk tracker is updated
            # regardless of whether the item is a duplicate
            processed = self.processor.handle_result(result)

            # Create caption record for storage
            total_outputs = sum(len(v) for v in result.outputs.values())

            filename = result.metadata.pop("_filename", None)
            url = result.metadata.pop("_url", None)
            image_height = result.metadata.pop("image_height", None)
            image_width = result.metadata.pop("image_width", None)
            file_size = result.metadata.pop("file_size", None)
            image_format = result.metadata.pop("image_format", None)
            result.metadata.pop("item_index", None)
            item_key = result.metadata.pop("item_key", None)

            to_delete_metadata_keys = ["_image_format", "_job_id"]
            for key in to_delete_metadata_keys:
                if key in result.metadata:
                    del result.metadata[key]

            caption = Caption(
                job_id=job_id,
                dataset=result.dataset,
                shard=processed["source_id"],
                chunk_id=chunk_name,
                item_key=item_key,
                captions=result.outputs.get("captions", []),
                outputs=result.outputs,
                contributor_id=worker_user,
                timestamp=datetime.now(_datetime.UTC),
                caption_count=total_outputs,
                processing_time_ms=result.processing_time_ms,
                metadata=result.metadata,
                image_height=image_height,
                image_width=image_width,
                filename=filename,
                url=url,
                file_size=file_size,
                image_format=image_format,
            )

            # Save to storage (might skip if duplicate)
            saved = await self.storage.save_caption(caption)

            # Update contributor stats only if actually saved
            if saved:
                contributor = await self.storage.get_contributor(worker_user)
                if contributor:
                    contributor.total_captions += total_outputs
                    await self.storage.save_contributor(contributor)

        except Exception as e:
            logger.error(
                f"Error processing result from {worker_id} for unit {data.get('unit_id', 'unknown')}: {e}",
                exc_info=True,
            )

    async def _handle_monitor(self, websocket: ServerConnection):
        """Handle monitor connection."""
        self.monitors.add(websocket)
        logger.info(f"Monitor connected (total: {len(self.monitors)})")

        try:
            # Send welcome
            await websocket.send(safe_json_dumps({"type": "welcome", "role": "monitor"}))

            # Send initial stats
            await self._send_monitor_stats(websocket)

            # Keep connection alive
            async for _message in websocket:
                pass

        except websockets.exceptions.ConnectionClosed:
            logger.info("Monitor disconnected")
        finally:
            self.monitors.discard(websocket)

    async def _handle_admin(self, websocket: ServerConnection, auth_ticket):
        """Handle admin connection."""
        admin_id = getattr(auth_ticket, "name", "admin")
        logger.info(f"Admin {admin_id} connected")

        try:
            await websocket.send(safe_json_dumps({"type": "welcome", "role": "admin"}))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "reload_config":
                        await self._handle_config_reload(websocket, data.get("config", {}))
                    elif data.get("type") == "get_stats":
                        await self._send_monitor_stats(websocket)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid admin message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Admin {admin_id} disconnected")

    async def _handle_data_worker(self, websocket: ServerConnection, auth_ticket):
        """Handle data worker connection."""
        worker_id = getattr(auth_ticket, "name", str(uuid.uuid4()))
        self.data_workers[worker_id] = websocket

        logger.info(f"Data worker {worker_id} connected")

        try:
            # Send welcome with storage config
            storage_config = self.config.get(
                "data_worker_storage",
                {
                    "forward_to_orchestrator": True,
                    "local": {"enabled": False},
                    "s3": {"enabled": False},
                },
            )

            await websocket.send(
                json.dumps(
                    {"type": "welcome", "worker_id": worker_id, "storage_config": storage_config}
                )
            )

            # Track if we've sent backpressure
            backpressure_sent = False

            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "submit_samples":
                    # Check queue size for backpressure
                    if self.data_sample_queue.qsize() > self.backpressure_threshold:
                        if not backpressure_sent:
                            await websocket.send(json.dumps({"type": "backpressure"}))
                            backpressure_sent = True
                            logger.warning(f"Backpressure applied to data worker {worker_id}")
                    else:
                        if backpressure_sent:
                            await websocket.send(json.dumps({"type": "resume"}))
                            backpressure_sent = False

                    # Receive image data for each sample
                    samples = data["samples"]
                    for sample in samples:
                        # Receive binary image data
                        image_data = await websocket.recv()

                        # Create job and add to queue
                        job = {
                            "job_id": f"data_{worker_id}_{sample['sample_id']}",
                            "sample_id": sample["sample_id"],
                            "image_data": image_data,
                            "metadata": sample.get("metadata", {}),
                            "source": "data_worker",
                            "worker_id": worker_id,
                        }

                        await self.data_sample_queue.put(job)

                elif msg_type == "heartbeat":
                    logger.debug(f"Data worker {worker_id} heartbeat: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Data worker {worker_id} disconnected")
        finally:
            del self.data_workers[worker_id]

    async def _send_monitor_initial_data(self, websocket: ServerConnection):
        """Send initial data to monitor in a separate task to avoid blocking."""
        total_start = time.time()
        try:
            # Check if websocket is still in monitors set
            if websocket not in self.monitors:
                logger.debug("Monitor disconnected before initial data send")
                return

            # Send current stats (already in memory)
            stats_start = time.time()
            await websocket.send(safe_json_dumps({"type": "stats", "data": self.stats}))
            logger.debug(f"Monitor stats sent in {(time.time() - stats_start)*1000:.1f}ms")

            # Get processor stats instead of chunk stats
            processor_stats_start = time.time()
            processor_stats = self.processor.get_stats()
            logger.debug(
                f"Processor stats retrieved in {(time.time() - processor_stats_start)*1000:.1f}ms"
            )

            stats_send_start = time.time()
            await websocket.send(
                safe_json_dumps({"type": "processor_stats", "data": processor_stats})
            )
            logger.debug(f"Processor stats sent in {(time.time() - stats_send_start)*1000:.1f}ms")

            if websocket not in self.monitors:
                return

            # For leaderboard, check if we have a cached version first
            if hasattr(self, "_cached_leaderboard") and self._cached_leaderboard:
                # Use cached leaderboard if available
                cache_send_start = time.time()
                await websocket.send(
                    safe_json_dumps({"type": "leaderboard", "data": self._cached_leaderboard})
                )
                logger.debug(
                    f"Cached leaderboard sent in {(time.time() - cache_send_start)*1000:.1f}ms"
                )
            else:
                # Schedule leaderboard update separately
                leaderboard_task_start = time.time()
                asyncio.create_task(self._send_leaderboard_to_monitor(websocket))
                logger.debug(
                    f"Leaderboard task created in {(time.time() - leaderboard_task_start)*1000:.1f}ms"
                )

            logger.debug(
                f"Monitor initial data send completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Monitor disconnected during initial data send")
        except Exception as e:
            logger.error(f"Error sending initial monitor data: {e}")

    async def _send_monitor_leaderboard(self, websocket: ServerConnection):
        """Send leaderboard data to a specific monitor."""
        total_start = time.time()
        try:
            if websocket not in self.monitors:
                return

            # Get contributors asynchronously
            contributors_start = time.time()
            contributors = await self.storage.get_top_contributors(10)
            logger.debug(
                f"Contributors retrieved in {(time.time() - contributors_start)*1000:.1f}ms"
            )

            # Get worker counts in thread pool
            worker_counts_start = time.time()
            loop = asyncio.get_event_loop()
            worker_counts = await loop.run_in_executor(
                None,
                lambda: (
                    self.get_workers_by_user_stats() if hasattr(self, "workers_by_user") else {}
                ),
            )
            logger.debug(
                f"Worker counts retrieved in {(time.time() - worker_counts_start)*1000:.1f}ms"
            )

            # Build enhanced contributors list
            build_start = time.time()
            enhanced_contributors = []
            for contributor in contributors:
                contrib_dict = {
                    "contributor_id": contributor.contributor_id,
                    "name": contributor.name,
                    "total_captions": contributor.total_captions,
                    "trust_level": contributor.trust_level,
                    "active_workers": len(
                        worker_counts.get(contributor.contributor_id, {}).get("worker_ids", [])
                    ),
                }
                enhanced_contributors.append(contrib_dict)
            logger.debug(f"Enhanced contributors built in {(time.time() - build_start)*1000:.1f}ms")

            # Cache for future monitors
            self._cached_leaderboard = enhanced_contributors

            # Send if still connected
            if websocket in self.monitors:
                send_start = time.time()
                await websocket.send(
                    safe_json_dumps({"type": "leaderboard", "data": enhanced_contributors})
                )
                logger.debug(
                    f"Leaderboard sent to monitor in {(time.time() - send_start)*1000:.1f}ms"
                )

            logger.debug(
                f"Leaderboard send to monitor completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Monitor disconnected during leaderboard send")
        except Exception as e:
            logger.error(f"Error sending leaderboard to monitor: {e}")

    async def _send_monitor_stats(self, websocket: ServerConnection):
        """Send current stats to a monitor."""
        # Get processor stats
        processor_stats = self.processor.get_stats()

        # Get storage stats
        storage_stats = await self.storage.get_storage_stats()

        # Combine all stats
        all_stats = {
            **self.stats,
            **storage_stats,
            "processor_stats": processor_stats,
            "current_rate": self.rate_tracker["current_rate"],
            "average_rate": self.rate_tracker["average_rate"],
        }

        await websocket.send(safe_json_dumps({"type": "stats", "data": all_stats}))

    async def _send_activity(self, activity: str):
        """Send activity update to monitors."""
        if not self.monitors:
            return

        message = safe_json_dumps(
            {"type": "activity", "data": f"[{datetime.now().strftime('%H:%M:%S')}] {activity}"}
        )

        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(monitor)

        self.monitors -= disconnected

    async def _broadcast_leaderboard(self):
        """Send leaderboard updates to monitors - separate from stats to avoid blocking."""
        if not self.monitors:
            return

        total_start = time.time()
        try:
            # Get contributors
            contributors_start = time.time()
            contributors = await self.storage.get_top_contributors(10)
            logger.debug(
                f"Contributors retrieved for broadcast in {(time.time() - contributors_start)*1000:.1f}ms"
            )

            # Get worker counts
            worker_counts_start = time.time()
            loop = asyncio.get_event_loop()
            worker_counts = await loop.run_in_executor(
                None,
                lambda: (
                    self.get_workers_by_user_stats() if hasattr(self, "workers_by_user") else {}
                ),
            )
            logger.debug(
                f"Worker counts retrieved for broadcast in {(time.time() - worker_counts_start)*1000:.1f}ms"
            )

            # Build enhanced contributors list
            build_start = time.time()
            enhanced_contributors = []
            for contributor in contributors:
                contrib_dict = {
                    "contributor_id": contributor.contributor_id,
                    "name": contributor.name,
                    "total_captions": contributor.total_captions,
                    "trust_level": contributor.trust_level,
                    "active_workers": len(
                        worker_counts.get(contributor.contributor_id, {}).get("worker_ids", [])
                    ),
                }
                enhanced_contributors.append(contrib_dict)
            logger.debug(
                f"Enhanced contributors built for broadcast in {(time.time() - build_start)*1000:.1f}ms"
            )

            # Cache it
            self._cached_leaderboard = enhanced_contributors

            # Create message once
            message_create_start = time.time()
            leaderboard_message = safe_json_dumps(
                {"type": "leaderboard", "data": enhanced_contributors}
            )
            logger.debug(
                f"Leaderboard message created in {(time.time() - message_create_start)*1000:.1f}ms"
            )

            # Send to all monitors in parallel
            send_start = time.time()

            async def send_leaderboard(monitor):
                try:
                    await monitor.send(leaderboard_message)
                except:
                    return monitor  # Mark for removal
                return None

            monitors_copy = self.monitors.copy()
            results = await asyncio.gather(
                *[send_leaderboard(m) for m in monitors_copy], return_exceptions=True
            )

            # Remove disconnected
            disconnected = {
                m
                for m, r in zip(monitors_copy, results, strict=False)
                if r is not None and not isinstance(r, Exception)
            }
            self.monitors -= disconnected

            logger.debug(
                f"Leaderboard sent to {len(monitors_copy)} monitors in {(time.time() - send_start)*1000:.1f}ms"
            )
            logger.debug(
                f"Leaderboard broadcast completed in {(time.time() - total_start)*1000:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error broadcasting leaderboard: {e}")

    async def _broadcast_stats(self):
        """Broadcast statistics to all monitors."""
        if not self.monitors:
            return

        # Get current stats
        processor_stats = self.processor.get_stats()
        storage_stats = await self.storage.get_storage_stats()

        # Update main stats
        self.stats["processor_stats"] = processor_stats
        self.stats["total_outputs"] = storage_stats["total_captions"]

        # Create message
        stats_message = safe_json_dumps(
            {
                "type": "stats",
                "data": {
                    **self.stats,
                    **storage_stats,
                    "current_rate": self.rate_tracker["current_rate"],
                    "average_rate": self.rate_tracker["average_rate"],
                },
            }
        )

        # Send to all monitors
        disconnected = set()
        for monitor in self.monitors:
            try:
                await monitor.send(stats_message)
            except:
                disconnected.add(monitor)

        self.monitors -= disconnected

    async def _heartbeat_loop(self):
        """Collect and log worker status periodically."""
        while True:
            await asyncio.sleep(30)

            # Just collect status - no ping/pong
            active_workers = []
            for worker_id, ws in list(self.workers.items()):
                # Check if WebSocket is still open (don't ping)
                if ws.state == websockets.protocol.State.OPEN:
                    active_workers.append(worker_id)
                else:
                    # Clean up closed connections
                    logger.info(f"Worker {worker_id} connection closed")
                    del self.workers[worker_id]
                    self.processor.release_assignments(worker_id)

            # Log status
            if active_workers:
                logger.debug(
                    f"Inactive workers: {len(self.workers) - len(active_workers)}/{len(active_workers)} - {', '.join(active_workers[:5])}"
                )
            # add to self.stats
            self.stats["active_workers"] = len(active_workers)
            self.stats["inactive_workers"] = len(self.workers) - len(active_workers)

    async def _checkpoint_loop(self):
        """Periodically checkpoint storage and chunk tracker."""
        interval = self.config.get("storage", {}).get("checkpoint_interval", 60)

        while True:
            await asyncio.sleep(interval)

            try:
                # Checkpoint storage
                await self.storage.checkpoint()

                # Also checkpoint the chunk tracker if using webdataset processor
                if hasattr(self.processor, "chunk_tracker") and self.processor.chunk_tracker:
                    # Save checkpoint in thread pool to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.processor.chunk_tracker.save
                    )
                    logger.debug("Saved chunk tracker checkpoint")

                self.stats["last_checkpoint"] = datetime.now(_datetime.UTC).isoformat()
                logger.info("Storage and chunk tracker checkpoint complete")
            except Exception as e:
                logger.error(f"Error during checkpoint: {e}", exc_info=True)

    async def _stats_update_loop(self):
        """Periodically update and broadcast stats."""
        while True:
            await asyncio.sleep(10)

            # Update rate tracking
            storage_stats = await self.storage.get_storage_stats()
            current_total = storage_stats["total_captions"]
            current_time = time.time()

            elapsed = current_time - self.rate_tracker["last_update_time"]
            if elapsed > 0:
                output_diff = current_total - self.rate_tracker["last_output_count"]
                self.rate_tracker["current_rate"] = (output_diff / elapsed) * 60
                self.rate_tracker["last_output_count"] = current_total
                self.rate_tracker["last_update_time"] = current_time

                # Average rate since start
                total_elapsed = current_time - self.rate_tracker["start_time"]
                if total_elapsed > 0:
                    self.rate_tracker["average_rate"] = (current_total / total_elapsed) * 60

            await self._broadcast_stats()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down orchestrator...")

        # Close all connections
        for ws in list(self.workers.values()):
            await ws.close()
        for ws in list(self.monitors):
            await ws.close()

        # Final checkpoint
        await self.storage.checkpoint()
        await self.storage.close()

        logger.info("Shutdown complete")
