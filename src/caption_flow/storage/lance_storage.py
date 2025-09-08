"""Lance-based storage management with per-shard datasets and dynamic column support."""

import asyncio
import gc
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple
from collections import defaultdict, deque
import time
import numpy as np

import lance
import pyarrow as pa
import pandas as pd

from ..models import Job, Caption, Contributor, StorageContents, JobId

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LanceStorageManager:
    """Manages Lance storage with per-shard datasets and dynamic columns."""

    def __init__(
        self,
        data_dir: Path,
        caption_buffer_size: int = 100,
        contributor_buffer_size: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Directory structure
        self.shards_dir = self.data_dir / "shards"
        self.shards_dir.mkdir(exist_ok=True)

        self.jobs_path = self.data_dir / "jobs.lance"
        self.contributors_path = self.data_dir / "contributors.lance"
        self.stats_path = self.data_dir / "storage_stats.json"

        # Per-shard buffers
        self.shard_buffers: Dict[str, List[dict]] = defaultdict(list)
        self.caption_buffer_size = caption_buffer_size

        # Other buffers
        self.job_buffer = []
        self.contributor_buffer = []
        self.contributor_buffer_size = contributor_buffer_size

        # Track existing IDs per shard
        self.shard_existing_job_ids: Dict[str, Set[str]] = defaultdict(set)
        self.existing_contributor_ids: Set[str] = set()

        # Track known output fields per shard
        self.shard_output_fields: Dict[str, Set[str]] = defaultdict(set)

        # Lance datasets cache
        self.shard_datasets: Dict[str, lance.Dataset] = {}
        self.jobs_dataset: Optional[lance.Dataset] = None
        self.contributors_dataset: Optional[lance.Dataset] = None

        # Statistics per shard
        self.shard_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "disk_rows": 0,
                "disk_outputs": 0,
                "field_counts": {},
                "duplicates_skipped": 0,
            }
        )

        # Global statistics
        self.global_stats = {
            "total_captions_written": 0,
            "total_caption_entries_written": 0,
            "total_flushes": 0,
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

    def _get_shard_path(self, shard_name: str) -> Path:
        """Get the Lance dataset path for a shard."""
        return self.shards_dir / f"{shard_name}.lance"

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
            stats_data = {
                "shard_stats": self.shard_stats,
                "global_stats": self.global_stats,
            }
            with open(self.stats_path, "w") as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def _load_stats(self):
        """Load stats from disk if available."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path, "r") as f:
                    loaded_stats = json.load(f)
                    self.shard_stats.update(loaded_stats.get("shard_stats", {}))
                    self.global_stats.update(loaded_stats.get("global_stats", {}))
                    logger.info(f"Loaded stats from {self.stats_path}")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    async def initialize(self):
        """Initialize storage and load existing data."""
        self._load_stats()

        # Check for old Parquet files and migrate if needed
        old_captions_path = self.data_dir / "captions.parquet"
        if old_captions_path.exists() and not any(self.shards_dir.glob("*.lance")):
            logger.info("Found old Parquet storage, migrating to Lance format...")
            await self._migrate_from_parquet(old_captions_path)

        # Initialize shard datasets
        for shard_path in self.shards_dir.glob("*.lance"):
            shard_name = shard_path.stem
            try:
                dataset = lance.dataset(str(shard_path))
                self.shard_datasets[shard_name] = dataset

                # Load existing job IDs for this shard
                job_ids = dataset.to_table(columns=["job_id"]).column("job_id").to_pylist()
                self.shard_existing_job_ids[shard_name] = set(job_ids)

                # Detect output fields
                schema = dataset.schema
                base_field_names = {field[0] for field in self.base_caption_fields}
                output_fields = set(schema.names) - base_field_names
                self.shard_output_fields[shard_name] = output_fields

                # Update stats if not loaded from file
                if shard_name not in self.shard_stats:
                    self.shard_stats[shard_name]["disk_rows"] = dataset.count_rows()
                    await self._calculate_shard_stats(shard_name)

                logger.info(
                    f"Loaded shard {shard_name}: {len(job_ids)} rows, "
                    f"output fields: {sorted(output_fields)}"
                )
            except Exception as e:
                logger.error(f"Failed to load shard {shard_name}: {e}")

        # Initialize jobs dataset
        if self.jobs_path.exists():
            self.jobs_dataset = lance.dataset(str(self.jobs_path))
            logger.info(f"Loaded jobs dataset: {self.jobs_dataset.count_rows()} rows")

        # Initialize contributors dataset
        if self.contributors_path.exists():
            self.contributors_dataset = lance.dataset(str(self.contributors_path))
            contributor_ids = (
                self.contributors_dataset.to_table(columns=["contributor_id"])
                .column("contributor_id")
                .to_pylist()
            )
            self.existing_contributor_ids = set(contributor_ids)
            logger.info(f"Loaded contributors dataset: {len(contributor_ids)} contributors")

    async def _calculate_shard_stats(self, shard_name: str):
        """Calculate stats for a specific shard."""
        dataset = self.shard_datasets.get(shard_name)
        if not dataset:
            return

        stats = self.shard_stats[shard_name]
        output_fields = self.shard_output_fields[shard_name]

        if not output_fields:
            stats["disk_outputs"] = 0
            stats["field_counts"] = {}
            return

        try:
            # Get all data for counting
            table = dataset.to_table()
            df = table.to_pandas()

            total_outputs = 0
            field_counts = {}

            for field_name in output_fields:
                if field_name in df.columns:
                    field_count = 0
                    column_data = df[field_name]

                    for value in column_data:
                        if value is not None and isinstance(value, list) and len(value) > 0:
                            field_count += len(value)

                    if field_count > 0:
                        field_counts[field_name] = field_count
                        total_outputs += field_count

            stats["disk_outputs"] = total_outputs
            stats["field_counts"] = field_counts

            logger.info(
                f"Shard {shard_name} stats: {stats['disk_rows']} rows, "
                f"{total_outputs} outputs, fields: {list(field_counts.keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to calculate stats for shard {shard_name}: {e}")

    async def _migrate_from_parquet(self, parquet_path: Path):
        """Migrate old Parquet file to Lance shard format."""
        try:
            import pyarrow.parquet as pq
            import numpy as np
            import json

            logger.info(f"Reading {parquet_path}...")
            table = pq.read_table(str(parquet_path))
            df = table.to_pandas()

            # Debug: Check what columns we actually have
            logger.info(f"Parquet columns found: {list(df.columns)}")
            logger.info(f"Parquet schema: {table.schema}")

            total_rows = len(df)
            logger.info(f"Found {total_rows:,} rows to migrate")

            # Detect if we need to migrate from JSON outputs
            if "outputs" in df.columns:
                logger.info("Migrating from JSON outputs to dynamic columns...")
                df = await self._migrate_outputs_to_columns(df)

            # Group by shard
            if "shard" not in df.columns:
                logger.warning("No shard column found, assigning all rows to 'data-0001'")
                df["shard"] = "data-0001"

            shard_groups = df.groupby("shard")
            logger.info(f"Found {len(shard_groups)} shards to migrate")

            # Migrate each shard
            for shard_name, shard_df in shard_groups:
                logger.info(f"Migrating shard {shard_name}: {len(shard_df)} rows...")

                # Reset index to avoid issues
                shard_df = shard_df.reset_index(drop=True)

                # Detect output fields for this shard
                base_field_names = {field[0] for field in self.base_caption_fields}
                output_fields = set(shard_df.columns) - base_field_names - {"outputs"}

                # Add 'captions' to output fields if it exists in the dataframe
                if "captions" in shard_df.columns:
                    output_fields.add("captions")

                self.shard_output_fields[shard_name] = output_fields

                # Build schema and create table
                schema = self._build_caption_schema(output_fields)

                # Ensure all fields have correct types
                shard_records = shard_df.to_dict("records")
                prepared_records = []

                # Helper function to convert numpy arrays to lists
                def convert_numpy_to_list(value):
                    """Recursively convert numpy arrays to lists."""
                    if isinstance(value, np.ndarray):
                        return value.tolist()
                    elif isinstance(value, list):
                        return [convert_numpy_to_list(item) for item in value]
                    else:
                        return value

                for idx, record in enumerate(shard_records):
                    # First pass: convert ALL numpy arrays in the record
                    for key in list(record.keys()):
                        record[key] = convert_numpy_to_list(record[key])

                    # Ensure base fields exist and have correct types
                    for field_name, field_type in self.base_caption_fields:
                        if field_name not in record or pd.isna(record[field_name]):
                            # Set appropriate defaults based on PyArrow type
                            if pa.types.is_list(field_type):
                                record[field_name] = []
                            elif pa.types.is_integer(field_type):
                                record[field_name] = 0
                            elif pa.types.is_floating(field_type):
                                record[field_name] = 0.0
                            elif pa.types.is_timestamp(field_type):
                                record[field_name] = pd.Timestamp.now()
                            elif pa.types.is_string(field_type):
                                record[field_name] = None if field_name != "metadata" else "{}"
                            else:
                                record[field_name] = None
                        else:
                            # Validate existing values match expected type
                            value = record[field_name]
                            if pa.types.is_list(field_type):
                                # Ensure list fields contain lists
                                if not isinstance(value, list):
                                    if pd.isna(value).all() or value is None:
                                        record[field_name] = []
                                    elif isinstance(value, (int, float)):
                                        # This is the problematic case - convert scalar to empty list
                                        logger.warning(
                                            f"Field {field_name} has scalar value {value}, converting to empty list"
                                        )
                                        record[field_name] = []
                                    else:
                                        record[field_name] = [value]

                    # Ensure output fields are lists
                    for field_name in output_fields:
                        if field_name in record:
                            value = record[field_name]
                            if pd.isna(value).all() or value is None:
                                record[field_name] = []
                            elif not isinstance(value, list):
                                # For captions field, check if it's a string that looks like a JSON list
                                if field_name == "captions" and isinstance(value, str):
                                    try:
                                        # Try to parse as JSON
                                        parsed = json.loads(value)
                                        if isinstance(parsed, list):
                                            record[field_name] = parsed
                                        else:
                                            record[field_name] = [value]
                                    except:
                                        record[field_name] = [value]
                                else:
                                    record[field_name] = [value] if value else []
                            # If it's a list, flatten any nested lists (from numpy conversion)
                            elif (
                                isinstance(value, list)
                                and len(value) == 1
                                and isinstance(value[0], list)
                            ):
                                # Flatten single-element lists containing lists
                                record[field_name] = value[0]

                    # Remove any leftover "outputs" field
                    record.pop("outputs", None)

                    # Special handling for quality_scores field
                    if "quality_scores" in record:
                        value = record["quality_scores"]
                        if isinstance(value, (int, float)):
                            record["quality_scores"] = []
                        elif pd.isna(value).all() or value is None:
                            record["quality_scores"] = []
                        elif not isinstance(value, list):
                            record["quality_scores"] = []

                    # Final check: ensure no numpy arrays remain
                    for key, value in record.items():
                        if isinstance(value, np.ndarray):
                            logger.error(
                                f"Found unconverted numpy array in field {key} for record {idx}"
                            )
                            record[key] = value.tolist()

                    prepared_records.append(record)

                # Create Lance dataset
                try:
                    # Before creating table, log sample for debugging
                    if prepared_records:
                        sample = prepared_records[0].copy()
                        # Check types in sample
                        for k, v in sample.items():
                            if isinstance(v, (list, np.ndarray)):
                                logger.debug(f"Field {k}: type={type(v)}, len={len(v) if v else 0}")
                                if isinstance(v, list) and v:
                                    logger.debug(f"  First element type: {type(v[0])}")

                    shard_table = pa.Table.from_pylist(prepared_records, schema=schema)
                    shard_path = self._get_shard_path(shard_name)

                    self.shard_datasets[shard_name] = lance.write_dataset(
                        shard_table, str(shard_path), mode="create"
                    )

                    # Track job IDs
                    job_ids = [r.get("job_id") for r in prepared_records if r.get("job_id")]
                    self.shard_existing_job_ids[shard_name] = set(job_ids)

                    # Update stats
                    self.shard_stats[shard_name]["disk_rows"] = len(shard_df)
                    await self._calculate_shard_stats(shard_name)

                    logger.info(f"✓ Migrated shard {shard_name}: {len(prepared_records)} rows")

                except Exception as e:
                    logger.error(f"Failed to create Lance dataset for shard {shard_name}: {e}")
                    # Log a sample record for debugging
                    if prepared_records:
                        sample_record = prepared_records[0]
                        logger.error(f"Sample record: {sample_record}")
                        logger.error(f"Schema fields: {schema.names}")
                        # Check for numpy arrays
                        for k, v in sample_record.items():
                            if isinstance(v, np.ndarray):
                                logger.error(f"Field {k} is still a numpy array!")
                    raise

            # Rename old file
            backup_path = parquet_path.with_suffix(".parquet.migrated")
            parquet_path.rename(backup_path)
            logger.info(f"Renamed old file to: {backup_path}")

            # Save stats
            self._save_stats()

            logger.info(
                f"✓ Migration complete! Migrated {total_rows:,} rows across {len(shard_groups)} shards"
            )

        except Exception as e:
            logger.error(f"Failed to migrate from Parquet: {e}", exc_info=True)
            raise

    async def _migrate_outputs_to_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Migrate from JSON outputs column to dynamic columns."""
        # Collect all unique output field names
        output_fields = set()
        for outputs_json in df.get("outputs", []):
            if outputs_json and pd.notna(outputs_json):
                try:
                    outputs = json.loads(outputs_json)
                    if isinstance(outputs, dict):
                        output_fields.update(outputs.keys())
                except:
                    continue

        # Add legacy "captions" field if it exists and isn't already a base field
        base_field_names = {f[0] for f in self.base_caption_fields}
        if "captions" in df.columns and "captions" not in base_field_names:
            output_fields.add("captions")

        logger.info(f"Found output fields to migrate: {sorted(output_fields)}")

        # Create new columns for each output field
        for field_name in output_fields:
            if field_name not in df.columns:
                df[field_name] = None

        # Migrate data from outputs JSON to columns
        for idx, row in df.iterrows():
            if pd.notna(row.get("outputs")):
                try:
                    outputs = json.loads(row["outputs"])
                    if isinstance(outputs, dict):
                        for field_name, field_values in outputs.items():
                            df.at[idx, field_name] = field_values
                except:
                    continue

            # Handle legacy captions column if it's becoming a dynamic field
            if "captions" in output_fields and pd.notna(row.get("captions")):
                if pd.isna(df.at[idx, "captions"]):
                    df.at[idx, "captions"] = row["captions"]

        # Drop the old outputs column
        if "outputs" in df.columns:
            df = df.drop(columns=["outputs"])

        return df

    async def _migrate_jobs_from_parquet(self, parquet_path: Path):
        """Migrate old jobs Parquet file to Lance."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(parquet_path))
            self.jobs_dataset = lance.write_dataset(table, str(self.jobs_path), mode="create")

            # Rename old file
            backup_path = parquet_path.with_suffix(".parquet.migrated")
            parquet_path.rename(backup_path)

            logger.info(f"✓ Migrated jobs to Lance format ({table.num_rows} rows)")

        except Exception as e:
            logger.error(f"Failed to migrate jobs: {e}")

    async def _migrate_contributors_from_parquet(self, parquet_path: Path):
        """Migrate old contributors Parquet file to Lance."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(parquet_path))
            self.contributors_dataset = lance.write_dataset(
                table, str(self.contributors_path), mode="create"
            )

            # Load contributor IDs
            contributor_ids = table.column("contributor_id").to_pylist()
            self.existing_contributor_ids = set(contributor_ids)

            # Rename old file
            backup_path = parquet_path.with_suffix(".parquet.migrated")
            parquet_path.rename(backup_path)

            logger.info(
                f"✓ Migrated contributors to Lance format ({len(contributor_ids)} contributors)"
            )

        except Exception as e:
            logger.error(f"Failed to migrate contributors: {e}")

    async def save_caption(self, caption: Caption):
        """Save a caption to the appropriate shard buffer."""
        caption_dict = asdict(caption)

        # Extract shard name
        shard_name = caption_dict.get("shard", "unknown")

        # Extract item_index from metadata
        if "metadata" in caption_dict and isinstance(caption_dict["metadata"], dict):
            item_index = caption_dict["metadata"].get("_item_index")
            if item_index is not None:
                caption_dict["item_index"] = item_index

        # Extract outputs
        outputs = caption_dict.pop("outputs", {})
        caption_dict.pop("captions", None)

        # Get job_id for deduplication
        _job_id = caption_dict.get("job_id")
        job_id = JobId.from_dict(_job_id).get_sample_str() if isinstance(_job_id, dict) else _job_id

        # Check for duplicate in this shard
        if job_id in self.shard_existing_job_ids[shard_name]:
            self.shard_stats[shard_name]["duplicates_skipped"] += 1
            logger.debug(f"Skipping duplicate job_id in shard {shard_name}: {job_id}")
            return

        # Try to find existing buffered row for this job in the shard
        shard_buffer = self.shard_buffers[shard_name]
        found_row = False

        for idx, row in enumerate(shard_buffer):
            if row.get("job_id") == job_id:
                found_row = True
                # Merge outputs
                for field_name, field_values in outputs.items():
                    self.shard_output_fields[shard_name].add(field_name)
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
            self.shard_output_fields[shard_name].add(field_name)
            caption_dict[field_name] = list(field_values)

        # Serialize metadata
        if "metadata" in caption_dict:
            caption_dict["metadata"] = json.dumps(caption_dict.get("metadata", {}))
        else:
            caption_dict["metadata"] = "{}"

        if isinstance(caption_dict.get("job_id"), dict):
            caption_dict["job_id"] = job_id

        shard_buffer.append(caption_dict)

        # Check if we should flush this shard
        if len(shard_buffer) >= self.caption_buffer_size:
            logger.debug(f"Shard {shard_name} buffer full, flushing.")
            await self._flush_shard(shard_name)

    async def _flush_shard(self, shard_name: str):
        """Flush a specific shard buffer to Lance."""
        buffer = self.shard_buffers[shard_name]
        if not buffer:
            return

        try:
            num_rows = len(buffer)

            # Prepare data with all fields
            prepared_buffer = []
            new_job_ids = []

            for row in buffer:
                prepared_row = row.copy()

                # Track job_ids
                job_id = prepared_row.get("job_id")
                if job_id:
                    new_job_ids.append(job_id)

                # Ensure all base fields are present
                for field_name, field_type in self.base_caption_fields:
                    if field_name not in prepared_row:
                        prepared_row[field_name] = None

                # Ensure all output fields are present
                for field_name in self.shard_output_fields[shard_name]:
                    if field_name not in prepared_row:
                        prepared_row[field_name] = None

                prepared_buffer.append(prepared_row)

            # Build schema and create table
            schema = self._build_caption_schema(self.shard_output_fields[shard_name])
            table = pa.Table.from_pylist(prepared_buffer, schema=schema)

            # Get or create dataset for this shard
            shard_path = self._get_shard_path(shard_name)

            if shard_name in self.shard_datasets:
                # Append to existing dataset
                self.shard_datasets[shard_name] = lance.write_dataset(
                    table,
                    str(shard_path),
                    mode="append",
                )
            else:
                # Create new dataset
                self.shard_datasets[shard_name] = lance.write_dataset(
                    table,
                    str(shard_path),
                    mode="create",
                )

            # Update tracking
            self.shard_existing_job_ids[shard_name].update(new_job_ids)

            # Update stats
            self._update_shard_stats(shard_name, buffer, num_rows)
            self.global_stats["total_flushes"] += 1

            # Track row additions
            current_time = time.time()
            self.row_additions.append((current_time, num_rows))
            self._log_rates(num_rows)

            # Clear buffer
            buffer.clear()

            logger.info(f"Flushed {num_rows} rows to shard {shard_name}")

            # Save stats periodically
            if self.global_stats["total_flushes"] % 10 == 0:
                self._save_stats()

        except Exception as e:
            logger.error(f"Failed to flush shard {shard_name}: {e}")
        finally:
            self.shard_buffers[shard_name].clear()
            gc.collect()

    def _update_shard_stats(self, shard_name: str, captions: List[dict], rows_added: int):
        """Update statistics for a shard."""
        stats = self.shard_stats[shard_name]
        stats["disk_rows"] += rows_added

        # Update global stats
        self.global_stats["total_captions_written"] += rows_added

        # Count outputs
        outputs_added = 0
        for caption in captions:
            for field_name in self.shard_output_fields[shard_name]:
                if field_name in caption and isinstance(caption[field_name], list):
                    count = len(caption[field_name])
                    outputs_added += count

                    # Update field counts
                    if field_name not in stats["field_counts"]:
                        stats["field_counts"][field_name] = 0
                    stats["field_counts"][field_name] += count

                    if field_name not in self.global_stats["session_field_counts"]:
                        self.global_stats["session_field_counts"][field_name] = 0
                    self.global_stats["session_field_counts"][field_name] += count

        stats["disk_outputs"] += outputs_added
        self.global_stats["total_caption_entries_written"] += outputs_added

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

        total_elapsed = current_time - self.start_time
        if total_elapsed > 0:
            rates["overall"] = self.global_stats["total_captions_written"] / total_elapsed
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

        if self.contributors_dataset:
            # For updates, we need to handle this differently in Lance
            # For now, just append (in production, you'd implement proper upsert)
            self.contributors_dataset = lance.write_dataset(
                table, str(self.contributors_path), mode="append"
            )
        else:
            self.contributors_dataset = lance.write_dataset(
                table, str(self.contributors_path), mode="create"
            )

        self.contributor_buffer.clear()

    async def checkpoint(self):
        """Flush all buffers to disk."""
        logger.info(f"Checkpoint: Flushing {len(self.shard_buffers)} shards")

        # Flush all shard buffers
        for shard_name in list(self.shard_buffers.keys()):
            if self.shard_buffers[shard_name]:
                await self._flush_shard(shard_name)

        # Flush other buffers
        await self._flush_contributors()

        # Save stats
        self._save_stats()

        # Log summary
        total_rows = sum(s["disk_rows"] for s in self.shard_stats.values())
        total_outputs = sum(s["disk_outputs"] for s in self.shard_stats.values())
        logger.info(
            f"Checkpoint complete. Total rows: {total_rows}, " f"Total outputs: {total_outputs}"
        )

    def get_all_processed_job_ids(self) -> Set[str]:
        """Get all processed job_ids across all shards."""
        all_job_ids = set()

        # Add from existing sets
        for job_ids in self.shard_existing_job_ids.values():
            all_job_ids.update(job_ids)

        # Add from buffers
        for buffer in self.shard_buffers.values():
            for row in buffer:
                if "job_id" in row:
                    all_job_ids.add(row["job_id"])

        return all_job_ids

    async def get_shard_contents(
        self,
        shard_name: str,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> StorageContents:
        """Get contents for a specific shard."""
        # Flush the shard first
        if shard_name in self.shard_buffers:
            await self._flush_shard(shard_name)

        if shard_name not in self.shard_datasets:
            return StorageContents(
                rows=[],
                columns=[],
                output_fields=[],
                total_rows=0,
                metadata={"message": f"No data for shard {shard_name}"},
            )

        dataset = self.shard_datasets[shard_name]

        # Build scanner
        scanner = dataset.scanner(columns=columns)
        if limit:
            scanner = scanner.limit(limit)

        # Get data
        table = scanner.to_table()
        df = table.to_pandas()

        # Convert to records
        rows = df.to_dict("records")

        # Parse metadata
        if "metadata" in df.columns:
            for row in rows:
                if row.get("metadata"):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except:
                        pass

        stats = self.shard_stats[shard_name]

        return StorageContents(
            rows=rows,
            columns=list(df.columns),
            output_fields=list(self.shard_output_fields[shard_name]),
            total_rows=len(df),
            metadata={
                "shard": shard_name,
                "total_available_rows": stats["disk_rows"],
                "total_outputs": stats["disk_outputs"],
                "field_stats": stats["field_counts"],
            },
        )

    async def get_storage_contents(
        self,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        include_metadata: bool = True,
        shard_filter: Optional[List[str]] = None,
    ) -> StorageContents:
        """Get combined contents from all shards or specific shards."""
        await self.checkpoint()

        all_rows = []
        all_columns = set()
        all_output_fields = set()
        total_rows = 0

        # Determine which shards to include
        shards_to_include = shard_filter if shard_filter else list(self.shard_datasets.keys())

        for shard_name in shards_to_include:
            if shard_name not in self.shard_datasets:
                continue

            shard_contents = await self.get_shard_contents(shard_name, columns=columns)
            all_rows.extend(shard_contents.rows)
            all_columns.update(shard_contents.columns)
            all_output_fields.update(shard_contents.output_fields)
            total_rows += shard_contents.total_rows

            if limit and len(all_rows) >= limit:
                all_rows = all_rows[:limit]
                break

        metadata = {}
        if include_metadata:
            metadata = {
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "total_rows": total_rows,
                "rows_exported": len(all_rows),
                "shards_included": shards_to_include,
                "shard_stats": {
                    shard: self.shard_stats[shard]
                    for shard in shards_to_include
                    if shard in self.shard_stats
                },
            }

        return StorageContents(
            rows=all_rows,
            columns=sorted(all_columns),
            output_fields=sorted(all_output_fields),
            total_rows=total_rows,
            metadata=metadata,
        )

    async def get_caption_stats(self) -> Dict[str, Any]:
        """Get overall statistics across all shards."""
        total_rows = sum(s["disk_rows"] for s in self.shard_stats.values())
        total_outputs = sum(s["disk_outputs"] for s in self.shard_stats.values())

        # Add buffer counts
        for shard_name, buffer in self.shard_buffers.items():
            total_rows += len(buffer)
            for row in buffer:
                for field_name in self.shard_output_fields[shard_name]:
                    if field_name in row and isinstance(row[field_name], list):
                        total_outputs += len(row[field_name])

        # Aggregate field stats
        all_field_stats = defaultdict(int)
        for stats in self.shard_stats.values():
            for field, count in stats.get("field_counts", {}).items():
                all_field_stats[field] += count

        # Get all output fields
        all_output_fields = set()
        for fields in self.shard_output_fields.values():
            all_output_fields.update(fields)

        return {
            "total_rows": total_rows,
            "total_outputs": total_outputs,
            "output_fields": sorted(all_output_fields),
            "field_stats": dict(all_field_stats),
            "shard_count": len(self.shard_datasets),
            "shards": sorted(self.shard_datasets.keys()),
        }

    async def close(self):
        """Close storage and flush buffers."""
        await self.checkpoint()

        rates = self._calculate_rates()
        logger.info(
            f"Storage closed. Total rows written: {self.global_stats['total_captions_written']}, "
            f"Total outputs: {self.global_stats['total_caption_entries_written']}, "
            f"Overall rate: {rates['overall']:.1f} rows/s"
        )

    async def optimize_shard(self, shard_name: str):
        """Optimize a specific shard by compacting and cleaning up."""
        if shard_name not in self.shard_datasets:
            logger.warning(f"Shard {shard_name} not found")
            return

        dataset = self.shard_datasets[shard_name]

        # Compact the dataset (Lance handles this efficiently)
        logger.info(f"Compacting shard {shard_name}...")
        dataset.optimize.compact_files()

        # Clean up old versions
        dataset.cleanup_old_versions()

        logger.info(f"Optimization complete for shard {shard_name}")

    async def optimize_storage(self):
        """Optimize all shards."""
        logger.info(f"Optimizing {len(self.shard_datasets)} shards...")

        for shard_name in self.shard_datasets:
            await self.optimize_shard(shard_name)

        logger.info("Storage optimization complete")
