"""Storage management with Lance backend using a single dataset."""

import gc
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
import lance
import pandas as pd
import pyarrow as pa

from ..models import Caption, Contributor, JobId, StorageContents

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class StorageManager:
    """Manages Lance storage with a single dataset and dynamic columns."""

    def __init__(
        self,
        data_dir: Path,
        caption_buffer_size: int = 100,
        contributor_buffer_size: int = 50,
    ):
        self.duckdb_shard_connections = {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.captions_path = self.data_dir / "captions.lance"
        self.jobs_path = self.data_dir / "jobs.lance"
        self.contributors_path = self.data_dir / "contributors.lance"
        self.stats_path = self.data_dir / "storage_stats.json"

        # In-memory buffers
        self.caption_buffer = []
        self.job_buffer = []
        self.contributor_buffer = []

        # Buffer size configuration
        self.caption_buffer_size = caption_buffer_size
        self.contributor_buffer_size = contributor_buffer_size

        # Track existing IDs
        self.existing_caption_job_ids: Set[str] = set()
        self.existing_contributor_ids: Set[str] = set()
        self.existing_job_ids: Set[str] = set()

        # Track known output fields
        self.known_output_fields: Set[str] = set()

        # Lance datasets
        self.captions_dataset: Optional[lance.Dataset] = None
        self.jobs_dataset: Optional[lance.Dataset] = None
        self.contributors_dataset: Optional[lance.Dataset] = None

        # Statistics
        self.stats = {
            "disk_rows": 0,
            "disk_outputs": 0,
            "field_counts": {},
            "total_captions_written": 0,
            "total_caption_entries_written": 0,
            "total_flushes": 0,
            "duplicates_skipped": 0,
            "session_field_counts": {},
        }

        # Rate tracking
        self.row_additions = deque(maxlen=100)
        self.start_time = time.time()
        self.last_rate_log_time = time.time()

        # Base caption schema
        self.base_caption_fields = [
            ("job_id", pa.string()),
            ("dataset", pa.string()),
            ("shard", pa.string()),
            ("chunk_id", pa.string()),
            ("item_key", pa.string()),
            ("item_index", pa.int32()),
            ("filename", pa.string()),
            ("url", pa.string()),
            ("caption_count", pa.int32()),
            ("contributor_id", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("quality_scores", pa.list_(pa.float32())),
            ("image_width", pa.int32()),
            ("image_height", pa.int32()),
            ("image_format", pa.string()),
            ("file_size", pa.int64()),
            ("processing_time_ms", pa.float32()),
            ("metadata", pa.string()),
        ]

        # Current caption schema (will be updated dynamically)
        self.caption_schema = None

        self.job_schema = pa.schema(
            [
                ("job_id", pa.string()),
                ("dataset", pa.string()),
                ("shard", pa.string()),
                ("item_key", pa.string()),
                ("status", pa.string()),
                ("assigned_to", pa.string()),
                ("created_at", pa.timestamp("us")),
                ("updated_at", pa.timestamp("us")),
            ]
        )

        self.contributor_schema = pa.schema(
            [
                ("contributor_id", pa.string()),
                ("name", pa.string()),
                ("total_captions", pa.int64()),
                ("trust_level", pa.int32()),
            ]
        )

    def init_duckdb_connection(
        self, output_shard: Optional[str] = None
    ) -> duckdb.DuckDBPyConnection:
        """Initialize or retrieve a DuckDB connection for a given output shard.
        Currently, we just use a single output shard, but this allows for future implementation of multiple.

        Args:
        ----
            output_shard (Optional[str]): The output shard identifier. If None, uses default shard.

        Returns:
        -------
            duckdb.DuckDBPyConnection: The DuckDB connection for the specified shard.

        """
        shard_key = output_shard or "default"
        if shard_key in self.duckdb_shard_connections:
            return self.duckdb_shard_connections[shard_key]

        conn = duckdb.connect(database=":memory:")

        # For the default shard, register the captions Lance dataset if it exists
        if shard_key == "default":
            # Force refresh the dataset to handle cases where it was recreated due to schema evolution
            if self.captions_path.exists():
                try:
                    # Always reload from disk to ensure we have the latest version
                    logger.debug(f"Reloading Lance dataset from {self.captions_path}")
                    self.captions_dataset = lance.dataset(str(self.captions_path))
                    logger.debug("Successfully loaded Lance dataset, converting to Arrow table")
                    # Convert Lance dataset to Arrow table for DuckDB compatibility
                    arrow_table = self.captions_dataset.to_table()
                    logger.debug(
                        f"Successfully converted Lance dataset to Arrow table with {arrow_table.num_rows} rows"
                    )
                    # Register the Arrow table in DuckDB so it can be queried
                    conn.register("captions", arrow_table)
                    logger.debug(
                        f"Registered Lance dataset {self.captions_path} as 'captions' table in DuckDB"
                    )

                    # Verify the table was registered
                    tables = conn.execute("SHOW TABLES").fetchall()
                    logger.debug(f"Available tables in DuckDB: {tables}")
                except Exception as e:
                    logger.warning(f"Failed to register Lance dataset in DuckDB: {e}")
                    # Fall back to direct file path queries

        self.duckdb_shard_connections[shard_key] = conn

        return conn

    def _init_lance_dataset(self) -> Optional[lance.LanceDataset]:
        """Initialize or retrieve the captions Lance dataset."""
        if self.captions_dataset:
            logger.debug("Captions dataset already initialized")
            return self.captions_dataset

        if not self.captions_path.exists():
            logger.debug("Captions dataset does not exist, creating new one")
            # Create initial schema with just base fields
            self.caption_schema = self._build_caption_schema(set())

            # Create empty dataset on disk with proper schema
            empty_dict = {}
            for field_name, field_type in self.base_caption_fields:
                if field_type == pa.string():
                    empty_dict[field_name] = []
                elif field_type == pa.int32():
                    empty_dict[field_name] = []
                elif field_type == pa.int64():
                    empty_dict[field_name] = []
                elif field_type == pa.float32():
                    empty_dict[field_name] = []
                elif field_type == pa.timestamp("us"):
                    empty_dict[field_name] = []
                elif field_type == pa.list_(pa.float32()):
                    empty_dict[field_name] = []
                else:
                    empty_dict[field_name] = []

            empty_table = pa.Table.from_pydict(empty_dict, schema=self.caption_schema)
            self.captions_dataset = lance.write_dataset(
                empty_table, str(self.captions_path), mode="create"
            )
            logger.info(f"Created empty captions storage at {self.captions_path}")

            return self.captions_dataset

        try:
            logger.debug(f"Loading Lance dataset from {self.captions_path}")
            self.captions_dataset = lance.dataset(str(self.captions_path))
            return self.captions_dataset
        except Exception as e:
            logger.error(f"Failed to load Lance dataset from {self.captions_path}: {e}")
            return None

    def _update_duckdb_connections_after_schema_change(self):
        """Update DuckDB connections after dataset schema has changed."""
        logger.debug(
            f"Updating {len(self.duckdb_shard_connections)} DuckDB connections after schema change"
        )
        for shard_key, conn in self.duckdb_shard_connections.items():
            if shard_key == "default" and self.captions_dataset:
                try:
                    # Re-register the updated dataset
                    arrow_table = self.captions_dataset.to_table()
                    conn.register("captions", arrow_table)
                    logger.debug(
                        f"Updated DuckDB registration for {self.captions_path} after schema change"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update DuckDB connection after schema change: {e}")

    def _build_caption_schema(self, output_fields: Set[str]) -> pa.Schema:
        """Build caption schema with dynamic output fields."""
        fields = self.base_caption_fields.copy()

        # Add dynamic output fields
        for field_name in sorted(output_fields):
            fields.append((field_name, pa.list_(pa.string())))

        return pa.schema(fields)

    def _save_stats(self):
        """Persist current stats to disk."""
        try:
            with open(self.stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def _load_stats(self):
        """Load stats from disk if available."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path, "r") as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                    logger.info(f"Loaded stats from {self.stats_path}")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    async def initialize(self):
        """Initialize storage and load existing data."""
        self._load_stats()

        # Initialize caption storage
        if self._init_lance_dataset():
            logger.debug(f"Initialized captions dataset from {self.captions_path}")
            # Load existing job IDs
            job_ids = (
                self.captions_dataset.to_table(columns=["job_id"]).column("job_id").to_pylist()
            )
            self.existing_caption_job_ids = set(job_ids)

            # Detect output fields
            schema = self.captions_dataset.schema
            base_field_names = {field[0] for field in self.base_caption_fields}
            self.known_output_fields = set(schema.names) - base_field_names

            # Update caption schema
            self.caption_schema = self._build_caption_schema(self.known_output_fields)

            logger.info(
                f"Loaded Lance dataset: {len(job_ids)} rows, output fields: {sorted(self.known_output_fields)}"
            )

            # Calculate stats if not loaded
            if self.stats["disk_rows"] == 0:
                await self._calculate_initial_stats()
        else:
            logger.warning("No existing captions dataset found, starting fresh.")

        # Initialize contributors storage
        if not self.contributors_path.exists():
            # Create empty contributors dataset
            empty_dict = {"contributor_id": [], "name": [], "total_captions": [], "trust_level": []}
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.contributor_schema)
            self.contributors_dataset = lance.write_dataset(
                empty_table, str(self.contributors_path), mode="create"
            )
            logger.info(f"Created empty contributor storage at {self.contributors_path}")
        else:
            self.contributors_dataset = lance.dataset(str(self.contributors_path))
            contributor_ids = (
                self.contributors_dataset.to_table(columns=["contributor_id"])
                .column("contributor_id")
                .to_pylist()
            )
            self.existing_contributor_ids = set(contributor_ids)
            logger.info(f"Loaded contributors dataset: {len(contributor_ids)} contributors")

        # Initialize jobs storage
        if not self.jobs_path.exists():
            # Create empty jobs dataset
            empty_dict = {
                "job_id": [],
                "dataset": [],
                "shard": [],
                "item_key": [],
                "status": [],
                "assigned_to": [],
                "created_at": [],
                "updated_at": [],
            }
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.job_schema)
            self.jobs_dataset = lance.write_dataset(empty_table, str(self.jobs_path), mode="create")
            logger.info(f"Created empty jobs storage at {self.jobs_path}")
        else:
            self.jobs_dataset = lance.dataset(str(self.jobs_path))
            logger.info(f"Loaded jobs dataset: {self.jobs_dataset.count_rows()} rows")

    async def _calculate_initial_stats(self):
        """Calculate initial statistics from Lance dataset."""
        if not self.captions_dataset:
            return

        logger.info("Calculating initial statistics...")

        try:
            self.stats["disk_rows"] = self.captions_dataset.count_rows()

            if self.known_output_fields and self.stats["disk_rows"] > 0:
                # Sample data to calculate stats efficiently
                table = self.captions_dataset.to_table()
                df = table.to_pandas()

                total_outputs = 0
                field_counts = {}

                for field_name in self.known_output_fields:
                    if field_name in df.columns:
                        field_count = 0
                        column_data = df[field_name]

                        for value in column_data:
                            if value is not None and isinstance(value, list) and len(value) > 0:
                                field_count += len(value)

                        if field_count > 0:
                            field_counts[field_name] = field_count
                            total_outputs += field_count

                self.stats["disk_outputs"] = total_outputs
                self.stats["field_counts"] = field_counts

                del df, table
                gc.collect()
            else:
                self.stats["disk_outputs"] = 0
                self.stats["field_counts"] = {}

            logger.info(
                f"Initial stats: {self.stats['disk_rows']} rows, "
                f"{self.stats['disk_outputs']} outputs, "
                f"fields: {list(self.stats['field_counts'].keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to calculate initial stats: {e}", exc_info=True)

        self._save_stats()

    async def save_caption(self, caption: Caption):
        """Save a caption entry."""
        caption_dict = asdict(caption)

        # Extract item_index from metadata
        if "metadata" in caption_dict and isinstance(caption_dict["metadata"], dict):
            item_index = caption_dict["metadata"].get("_item_index")
            if item_index is not None:
                caption_dict["item_index"] = item_index

        # Extract outputs
        outputs = caption_dict.pop("outputs", {})
        caption_dict.pop("captions", None)

        # Get job_id for deduplication - convert to string early
        _job_id = caption_dict.get("job_id")
        job_id = JobId.from_dict(_job_id).get_sample_str() if isinstance(_job_id, dict) else _job_id
        caption_dict["job_id"] = job_id  # Update dict with string version

        # Check for duplicate
        if job_id in self.existing_caption_job_ids:
            self.stats["duplicates_skipped"] += 1
            logger.debug(f"Skipping duplicate job_id: {job_id}")
            return

        # Try to find existing buffered row
        for _idx, row in enumerate(self.caption_buffer):
            if row.get("job_id") == job_id:
                # Merge outputs
                for field_name, field_values in outputs.items():
                    if field_name not in self.known_output_fields:
                        self.known_output_fields.add(field_name)
                        logger.info(f"New output field detected: {field_name}")
                    if field_name in row and isinstance(row[field_name], list):
                        row[field_name].extend(field_values)
                    else:
                        row[field_name] = list(field_values)
                if "caption_count" in caption_dict:
                    old_count = row.get("caption_count", 0)
                    row["caption_count"] = old_count + caption_dict["caption_count"]
                return

        # Create new row
        for field_name, field_values in outputs.items():
            if field_name not in self.known_output_fields:
                self.known_output_fields.add(field_name)
                logger.info(f"New output field detected: {field_name}")
            caption_dict[field_name] = list(field_values)

        # Serialize metadata
        if "metadata" in caption_dict:
            caption_dict["metadata"] = json.dumps(caption_dict.get("metadata", {}))
        else:
            caption_dict["metadata"] = "{}"

        self.caption_buffer.append(caption_dict)

        if len(self.caption_buffer) >= self.caption_buffer_size:
            logger.debug("Caption buffer full, flushing.")
            await self._flush_captions()

    async def _flush_captions(self):
        """Flush caption buffer to Lance dataset."""
        if not self.caption_buffer:
            return

        try:
            num_rows = len(self.caption_buffer)
            captions_to_write = list(self.caption_buffer)

            # Prepare data
            prepared_buffer = []
            new_job_ids = []

            for row in self.caption_buffer:
                prepared_row = row.copy()
                job_id = prepared_row.get("job_id")
                if job_id:
                    new_job_ids.append(job_id)

                # Ensure all base fields are present
                for field_name, _field_type in self.base_caption_fields:
                    if field_name not in prepared_row:
                        prepared_row[field_name] = None

                # Ensure all output fields are present
                for field_name in self.known_output_fields:
                    if field_name not in prepared_row:
                        prepared_row[field_name] = None

                prepared_buffer.append(prepared_row)

            # Build schema and create table
            schema = self._build_caption_schema(self.known_output_fields)
            table = pa.Table.from_pylist(prepared_buffer, schema=schema)

            # Write to Lance
            if self.captions_dataset is None and self.captions_path.exists():
                try:
                    self.captions_dataset = lance.dataset(str(self.captions_path))
                except Exception:
                    # Dataset might be corrupted or incomplete
                    self.captions_dataset = None

            if self.captions_dataset is not None:
                # Check if schema has changed (new output fields added)
                existing_schema_fields = set(self.captions_dataset.schema.names)
                new_schema_fields = set(schema.names)

                if new_schema_fields != existing_schema_fields:
                    # Schema has changed, need to merge existing data with new schema
                    logger.info(
                        f"Schema evolution detected. New fields: {new_schema_fields - existing_schema_fields}"
                    )

                    # Read existing data
                    existing_table = self.captions_dataset.to_table()
                    existing_df = existing_table.to_pandas()

                    # Add missing columns to existing data
                    for field_name in new_schema_fields - existing_schema_fields:
                        existing_df[field_name] = None

                    # Convert back to table with new schema
                    existing_table_updated = pa.Table.from_pandas(existing_df, schema=schema)

                    # Concatenate existing and new data
                    combined_table = pa.concat_tables([existing_table_updated, table])

                    # Recreate dataset with combined data
                    self.captions_dataset = lance.write_dataset(
                        combined_table, str(self.captions_path), mode="overwrite"
                    )

                    # Update DuckDB connections after schema evolution
                    logger.debug("Updating DuckDB connections after schema evolution")
                    self._update_duckdb_connections_after_schema_change()
                else:
                    # Schema hasn't changed, normal append
                    self.captions_dataset = lance.write_dataset(
                        table, str(self.captions_path), mode="append"
                    )
            else:
                # Create new dataset
                self.captions_dataset = lance.write_dataset(
                    table, str(self.captions_path), mode="create"
                )

            # Update tracking
            self.existing_caption_job_ids.update(new_job_ids)

            # Update stats
            self._update_stats_for_new_captions(captions_to_write, num_rows)
            self.stats["total_flushes"] += 1

            # Track row additions
            current_time = time.time()
            self.row_additions.append((current_time, num_rows))
            self._log_rates(num_rows)

            # Clear buffer only on success
            self.caption_buffer.clear()

            logger.info(f"Flushed {num_rows} rows to Lance dataset")

            # Save stats periodically
            if self.stats["total_flushes"] % 10 == 0:
                self._save_stats()

        except Exception as e:
            logger.error(f"Failed to flush captions: {e}")
            # Don't clear buffer on failure - preserve data
            raise
        finally:
            gc.collect()

    def _update_stats_for_new_captions(self, captions_added: List[dict], rows_added: int):
        """Update stats incrementally."""
        self.stats["disk_rows"] += rows_added
        self.stats["total_captions_written"] += rows_added

        outputs_added = 0
        for caption in captions_added:
            for field_name in self.known_output_fields:
                if field_name in caption and isinstance(caption[field_name], list):
                    count = len(caption[field_name])
                    outputs_added += count

                    if field_name not in self.stats["field_counts"]:
                        self.stats["field_counts"][field_name] = 0
                    self.stats["field_counts"][field_name] += count

                    if field_name not in self.stats["session_field_counts"]:
                        self.stats["session_field_counts"][field_name] = 0
                    self.stats["session_field_counts"][field_name] += count

        self.stats["disk_outputs"] += outputs_added
        self.stats["total_caption_entries_written"] += outputs_added

    def _calculate_rates(self) -> Dict[str, float]:
        """Calculate row addition rates."""
        current_time = time.time()
        rates = {}

        windows = {"1min": 1, "5min": 5, "15min": 15, "60min": 60}
        cutoff_time = current_time - (60 * 60)

        while self.row_additions and self.row_additions[0][0] < cutoff_time:
            self.row_additions.popleft()

        for window_name, window_minutes in windows.items():
            window_seconds = window_minutes * 60
            window_start = current_time - window_seconds
            rows_in_window = sum(
                count for timestamp, count in self.row_additions if timestamp >= window_start
            )
            elapsed = current_time - self.start_time
            actual_window = min(window_seconds, elapsed)
            if actual_window > 0:
                rates[window_name] = rows_in_window / actual_window
            else:
                rates[window_name] = 0.0

        # Add instant and overall rates
        rates["instant"] = rates.get("1min", 0.0)
        total_elapsed = current_time - self.start_time
        if total_elapsed > 0:
            rates["overall"] = self.stats["total_captions_written"] / total_elapsed
        else:
            rates["overall"] = 0.0

        return rates

    def _log_rates(self, rows_added: int):
        """Log rate information."""
        current_time = time.time()
        time_since_last_log = current_time - self.last_rate_log_time

        if time_since_last_log < 10 and rows_added < 50:
            return

        rates = self._calculate_rates()
        rate_str = (
            f"Rate stats - Instant: {rates.get('1min', 0):.1f} rows/s | "
            f"Avg (5m): {rates.get('5min', 0):.1f} | "
            f"Avg (15m): {rates.get('15min', 0):.1f} | "
            f"Overall: {rates['overall']:.1f} rows/s"
        )
        logger.info(rate_str)
        self.last_rate_log_time = current_time

    async def save_contributor(self, contributor: Contributor):
        """Save or update contributor stats."""
        self.contributor_buffer.append(asdict(contributor))

        if len(self.contributor_buffer) >= self.contributor_buffer_size:
            await self._flush_contributors()

    async def _flush_contributors(self):
        """Flush contributor buffer to Lance."""
        if not self.contributor_buffer:
            return

        table = pa.Table.from_pylist(self.contributor_buffer, schema=self.contributor_schema)

        mode = "append" if self.contributors_path.exists() else "create"
        self.contributors_dataset = lance.write_dataset(
            table, str(self.contributors_path), mode=mode
        )
        if mode == "create":
            logger.info(f"Created contributor storage at {self.contributors_path}")

        self.contributor_buffer.clear()

    async def _flush_jobs(self):
        """Flush job buffer to Lance."""
        if not self.job_buffer:
            return

        table = pa.Table.from_pylist(self.job_buffer, schema=self.job_schema)

        table = pa.Table.from_pylist(self.job_buffer, schema=self.job_schema)

        mode = "append" if self.jobs_path.exists() else "create"
        self.jobs_dataset = lance.write_dataset(table, str(self.jobs_path), mode=mode)
        if mode == "create":
            logger.info(f"Created jobs storage at {self.jobs_path}")

        self.job_buffer.clear()

    async def checkpoint(self):
        """Flush all buffers to disk."""
        logger.info("Checkpoint: Flushing buffers")

        await self._flush_captions()
        await self._flush_contributors()
        await self._flush_jobs()

        self._save_stats()

        logger.info(
            f"Checkpoint complete. Total rows: {self.stats['disk_rows']}, "
            f"Total outputs: {self.stats['disk_outputs']}"
        )

    def get_all_processed_job_ids(self) -> Set[str]:
        """Get all processed job_ids."""
        all_job_ids = self.existing_caption_job_ids.copy()

        for row in self.caption_buffer:
            if "job_id" in row:
                all_job_ids.add(row["job_id"])

        return all_job_ids

    async def get_storage_contents(
        self,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        include_metadata: bool = True,
    ) -> StorageContents:
        """Get storage contents for export using DuckDB."""
        # Flush buffers first
        await self.checkpoint()

        if not self.captions_path.exists() or not self.captions_dataset:
            return StorageContents(
                rows=[],
                columns=[],
                output_fields=list(self.known_output_fields),
                total_rows=0,
                metadata={"message": "No data available"},
            )

        try:
            logger.debug("Getting DuckDB connection")
            con = self.init_duckdb_connection()
            logger.debug("Got DuckDB connection, building query")

            # Build query
            column_str = "*"
            if columns:
                # Quote column names to handle special characters
                column_str = ", ".join([f'"{c}"' for c in columns])

            query = f"SELECT {column_str} FROM captions"
            if limit:
                query += f" LIMIT {limit}"

            logger.debug(f"Executing DuckDB query: {query}")
            # Execute query and fetch data
            table = con.execute(query).fetch_arrow_table()
            logger.debug(f"Query executed successfully, got {table.num_rows} rows")
            rows = table.to_pylist()
            actual_columns = table.schema.names

            # Parse metadata
            if "metadata" in actual_columns:
                for row in rows:
                    if row.get("metadata"):
                        try:
                            row["metadata"] = json.loads(row["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass  # Keep as string if not valid JSON

            metadata = {}
            if include_metadata:
                metadata = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_available_rows": self.stats["disk_rows"],
                    "rows_exported": len(rows),
                    "storage_path": str(self.captions_path),
                    "field_stats": self.stats["field_counts"],
                }

            return StorageContents(
                rows=rows,
                columns=actual_columns,
                output_fields=list(self.known_output_fields),
                total_rows=len(rows),
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to get storage contents with DuckDB: {e}", exc_info=True)
            return StorageContents(
                rows=[],
                columns=[],
                output_fields=list(self.known_output_fields),
                total_rows=0,
                metadata={"error": str(e)},
            )

    async def get_processed_jobs_for_chunk(self, chunk_id: str) -> Set[str]:
        """Get all processed job_ids for a given chunk."""
        if not self.captions_dataset:
            return set()

        table = self.captions_dataset.to_table(
            columns=["job_id", "chunk_id"], filter=f"chunk_id = '{chunk_id}'"
        )
        return set(table.column("job_id").to_pylist())

    async def get_caption_stats(self) -> Dict[str, Any]:
        """Get statistics about stored captions."""
        total_rows = self.stats["disk_rows"] + len(self.caption_buffer)

        # Count outputs in buffer
        buffer_outputs = 0
        buffer_field_counts = defaultdict(int)
        for row in self.caption_buffer:
            for field_name in self.known_output_fields:
                if field_name in row and isinstance(row[field_name], list):
                    count = len(row[field_name])
                    buffer_outputs += count
                    buffer_field_counts[field_name] += count

        # Merge buffer counts with disk counts
        field_stats = {}
        for field_name in self.known_output_fields:
            disk_count = self.stats["field_counts"].get(field_name, 0)
            buffer_count = buffer_field_counts.get(field_name, 0)
            total_count = disk_count + buffer_count

            if total_count > 0:
                field_stats[field_name] = {
                    "total_items": total_count,
                    "disk_items": disk_count,
                    "buffer_items": buffer_count,
                }

        total_outputs = self.stats["disk_outputs"] + buffer_outputs

        return {
            "total_rows": total_rows,
            "total_outputs": total_outputs,
            "output_fields": sorted(list(self.known_output_fields)),
            "field_stats": field_stats,
            # Compatibility fields for CLI
            "shard_count": 1,
            "shards": ["default"],
        }

    async def count_captions(self) -> int:
        """Count total outputs across all fields."""
        stats = await self.get_caption_stats()
        return stats["total_outputs"]

    async def count_caption_rows(self) -> int:
        """Count total rows."""
        stats = await self.get_caption_stats()
        return stats["total_rows"]

    async def get_contributor(self, contributor_id: str) -> Optional[Contributor]:
        """Retrieve a contributor by ID."""
        # Check buffer first
        for buffered in self.contributor_buffer:
            if buffered["contributor_id"] == contributor_id:
                return Contributor(**buffered)

        if not self.contributors_dataset:
            return None

        try:
            table = self.contributors_dataset.to_table(
                filter=f"contributor_id = '{contributor_id}'"
            )
            if table.num_rows == 0:
                return None

            df = table.to_pandas()
            row = df.iloc[0]
            return Contributor(
                contributor_id=row["contributor_id"],
                name=row["name"],
                total_captions=int(row["total_captions"]),
                trust_level=int(row["trust_level"]),
            )
        except Exception as e:
            logger.error(f"Failed to get contributor {contributor_id}: {e}")
            return None

    async def get_top_contributors(self, limit: int = 10) -> List[Contributor]:
        """Get top contributors by caption count."""
        contributors = []

        if self.contributors_dataset:
            table = self.contributors_dataset.to_table()
            df = table.to_pandas()
            df = df.sort_values("total_captions", ascending=False).head(limit)

            for _, row in df.iterrows():
                contributors.append(
                    Contributor(
                        contributor_id=row["contributor_id"],
                        name=row["name"],
                        total_captions=int(row["total_captions"]),
                        trust_level=int(row["trust_level"]),
                    )
                )

        return contributors

    async def get_output_field_stats(self) -> Dict[str, Any]:
        """Get statistics about output fields."""
        stats = await self.get_caption_stats()
        field_counts = {field: info["total_items"] for field, info in stats["field_stats"].items()}
        total_outputs = sum(field_counts.values())

        return {
            "total_fields": len(field_counts),
            "field_counts": field_counts,
            "total_outputs": total_outputs,
            "fields": sorted(list(field_counts.keys())),
        }

    async def close(self):
        """Close storage and flush buffers."""
        await self.checkpoint()

        rates = self._calculate_rates()
        logger.info(
            f"Storage closed. Total rows written: {self.stats['total_captions_written']}, "
            f"Total outputs: {self.stats['total_caption_entries_written']}, "
            f"Overall rate: {rates['overall']:.1f} rows/s"
        )

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get all storage-related statistics."""
        caption_stats = await self.get_caption_stats()
        rates = self._calculate_rates()

        # Format field_breakdown to match expected format (dict of dicts with total_items)
        field_breakdown = {}
        for field, stats in caption_stats.get("field_stats", {}).items():
            if isinstance(stats, dict):
                # Already in correct format
                field_breakdown[field] = stats
            else:
                # Convert simple int to expected format
                field_breakdown[field] = {"total_items": stats}

        return {
            "total_captions": caption_stats["total_outputs"],
            "total_rows": caption_stats["total_rows"],
            "buffer_size": len(self.caption_buffer),
            "total_written": self.stats["total_captions_written"],
            "total_entries_written": self.stats["total_caption_entries_written"],
            "duplicates_skipped": self.stats["duplicates_skipped"],
            "total_flushes": self.stats["total_flushes"],
            "output_fields": sorted(list(self.known_output_fields)),
            "field_breakdown": field_breakdown,
            "contributor_buffer_size": len(self.contributor_buffer),
            "rates": {
                "instant": f"{rates.get('instant', 0.0):.1f} rows/s",
                "5min": f"{rates.get('5min', 0.0):.1f} rows/s",
                "15min": f"{rates.get('15min', 0.0):.1f} rows/s",
                "60min": f"{rates.get('60min', 0.0):.1f} rows/s",
                "overall": f"{rates.get('overall', 0.0):.1f} rows/s",
            },
        }

    async def optimize_storage(self):
        """Optimize storage by compacting Lance dataset."""
        if self.captions_dataset:
            logger.info("Optimizing Lance dataset...")
            self.captions_dataset.optimize.compact_files()
            self.captions_dataset.cleanup_old_versions()
            logger.info("Storage optimization complete")

    def _is_column_empty(self, df: pd.DataFrame, column_name: str) -> bool:
        """Check if a column is entirely empty."""
        if column_name not in df.columns:
            return True

        col = df[column_name]
        if col.isna().all():
            return True

        if pd.api.types.is_numeric_dtype(col):
            non_null_values = col.dropna()
            if len(non_null_values) > 0 and (non_null_values == 0).all():
                return True

        if col.dtype == "object":
            non_null_values = col.dropna()
            if len(non_null_values) == 0:
                return True
            all_empty_lists = True
            for val in non_null_values:
                if isinstance(val, list) and len(val) > 0:
                    all_empty_lists = False
                    break
                elif not isinstance(val, list):
                    all_empty_lists = False
                    break
            if all_empty_lists:
                return True

        return False

    def _get_non_empty_columns(
        self, df: pd.DataFrame, preserve_base_fields: bool = True
    ) -> List[str]:
        """Get list of columns that contain actual data."""
        base_field_names = {field[0] for field in self.base_caption_fields}
        non_empty_columns = []

        for col in df.columns:
            if preserve_base_fields and col in base_field_names:
                non_empty_columns.append(col)
            elif not self._is_column_empty(df, col):
                non_empty_columns.append(col)

        return non_empty_columns

    def _get_existing_output_columns(self) -> Set[str]:
        """Get output field columns that exist - for API compatibility."""
        return self.known_output_fields.copy()

    # Compatibility methods for LanceStorageExporter
    @property
    def shard_datasets(self) -> Dict[str, Any]:
        """Compatibility property for exporter - returns single default shard."""
        if self.captions_dataset:
            return {"default": self.captions_dataset}
        return {}

    @property
    def shard_output_fields(self) -> Dict[str, Set[str]]:
        """Compatibility property for exporter - returns output fields for default shard."""
        return {"default": self.known_output_fields.copy()}

    async def get_shard_contents(
        self,
        shard_name: str,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        include_metadata: bool = True,
    ) -> StorageContents:
        """Compatibility method for exporter - delegates to get_storage_contents for default shard."""
        if shard_name != "default":
            return StorageContents(
                rows=[],
                columns=[],
                output_fields=list(self.known_output_fields),
                total_rows=0,
                metadata={
                    "error": f"Shard '{shard_name}' not found. Only 'default' shard is supported."
                },
            )

        return await self.get_storage_contents(
            limit=limit, columns=columns, include_metadata=include_metadata
        )
