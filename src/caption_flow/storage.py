"""Arrow/Parquet storage management with dynamic column support for outputs."""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs
import pandas as pd
from collections import defaultdict

from .models import Job, Caption, Contributor, JobStatus

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages Arrow/Parquet storage with dynamic columns for output fields."""

    def __init__(
        self,
        data_dir: Path,
        caption_buffer_size: int = 100,
        job_buffer_size: int = 100,
        contributor_buffer_size: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.captions_path = self.data_dir / "captions.parquet"
        self.jobs_path = self.data_dir / "jobs.parquet"
        self.contributors_path = self.data_dir / "contributors.parquet"

        # In-memory buffers for batching writes
        self.caption_buffer = []
        self.job_buffer = []
        self.contributor_buffer = []

        # Buffer size configuration
        self.caption_buffer_size = caption_buffer_size
        self.job_buffer_size = job_buffer_size
        self.contributor_buffer_size = contributor_buffer_size

        # Track existing job_ids to prevent duplicates
        self.existing_contributor_ids: Set[str] = set()
        self.existing_caption_job_ids: Set[str] = set()
        self.existing_job_ids: Set[str] = set()

        # Track known output fields for schema evolution
        self.known_output_fields: Set[str] = set()

        # Statistics
        self.total_captions_written = 0
        self.total_caption_entries_written = 0  # Total individual captions
        self.total_flushes = 0
        self.duplicates_skipped = 0

        # Base caption schema without dynamic output fields
        self.base_caption_fields = [
            ("job_id", pa.string()),
            ("dataset", pa.string()),
            ("shard", pa.string()),
            ("chunk_id", pa.string()),
            ("item_key", pa.string()),
            ("item_index", pa.int32()),
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

    def _get_existing_output_columns(self) -> Set[str]:
        """Get output field columns that actually exist in the parquet file."""
        if not self.captions_path.exists():
            return set()

        table_metadata = pq.read_metadata(self.captions_path)
        existing_columns = set(table_metadata.schema.names)
        base_field_names = {field[0] for field in self.base_caption_fields}

        return existing_columns - base_field_names

    def _build_caption_schema(self, output_fields: Set[str]) -> pa.Schema:
        """Build caption schema with dynamic output fields."""
        fields = self.base_caption_fields.copy()

        # Add dynamic output fields (all as list of strings for now)
        for field_name in sorted(output_fields):  # Sort for consistent ordering
            fields.append((field_name, pa.list_(pa.string())))

        return pa.schema(fields)

    async def initialize(self):
        """Initialize storage files if they don't exist."""
        if not self.captions_path.exists():
            # Create initial schema with just base fields
            self.caption_schema = self._build_caption_schema(set())

            # Create empty table
            empty_dict = {field[0]: [] for field in self.base_caption_fields}
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.caption_schema)
            pq.write_table(empty_table, self.captions_path)
            logger.info(f"Created empty caption storage at {self.captions_path}")
        else:
            # Load existing schema and detect output fields
            existing_table = pq.read_table(self.captions_path)
            existing_columns = set(existing_table.column_names)

            # Identify output fields (columns not in base schema)
            base_field_names = {field[0] for field in self.base_caption_fields}
            self.known_output_fields = existing_columns - base_field_names

            # Check if we need to migrate from old "outputs" JSON column
            if "outputs" in existing_columns:
                logger.info("Migrating from JSON outputs to dynamic columns...")
                await self._migrate_outputs_to_columns(existing_table)
            else:
                # Build current schema from existing columns
                self.caption_schema = self._build_caption_schema(self.known_output_fields)

            # Load existing caption job_ids
            self.existing_caption_job_ids = set(existing_table["job_id"].to_pylist())
            logger.info(f"Loaded {len(self.existing_caption_job_ids)} existing caption job_ids")
            logger.info(f"Known output fields: {sorted(self.known_output_fields)}")

        # Initialize other storage files...
        if not self.contributors_path.exists():
            empty_dict = {"contributor_id": [], "name": [], "total_captions": [], "trust_level": []}
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.contributor_schema)
            pq.write_table(empty_table, self.contributors_path)
            logger.info(f"Created empty contributor storage at {self.contributors_path}")
        else:
            existing_contributors = pq.read_table(
                self.contributors_path, columns=["contributor_id"]
            )
            self.existing_contributor_ids = set(existing_contributors["contributor_id"].to_pylist())
            logger.info(f"Loaded {len(self.existing_contributor_ids)} existing contributor IDs")

    async def _migrate_outputs_to_columns(self, existing_table: pa.Table):
        """Migrate from JSON outputs column to dynamic columns."""
        df = existing_table.to_pandas()

        # Collect all unique output field names
        output_fields = set()
        for outputs_json in df.get("outputs", []):
            if outputs_json:
                try:
                    outputs = json.loads(outputs_json)
                    output_fields.update(outputs.keys())
                except:
                    continue

        # Add legacy "captions" field if it exists and isn't already a base field
        if "captions" in df.columns and "captions" not in {f[0] for f in self.base_caption_fields}:
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

        # Update known fields and schema
        self.known_output_fields = output_fields
        self.caption_schema = self._build_caption_schema(output_fields)

        # Write migrated table
        migrated_table = pa.Table.from_pandas(df, schema=self.caption_schema)
        pq.write_table(migrated_table, self.captions_path)
        logger.info("Migration complete - outputs now stored in dynamic columns")

    async def save_caption(self, caption: Caption):
        """Save a caption entry with dynamic output columns."""
        # Convert to dict
        caption_dict = asdict(caption)

        # Extract item_index from metadata if present
        if "metadata" in caption_dict and isinstance(caption_dict["metadata"], dict):
            item_index = caption_dict["metadata"].get("_item_index")
            if item_index is not None:
                caption_dict["item_index"] = item_index

        # Extract outputs and handle them separately
        outputs = caption_dict.pop("outputs", {})

        # Remove old "captions" field if it exists (will be in outputs)
        caption_dict.pop("captions", None)

        new_fields = set()
        for field_name, field_values in outputs.items():
            caption_dict[field_name] = field_values
            if field_name not in self.known_output_fields:
                new_fields.add(field_name)
                self.known_output_fields.add(field_name)  # Add immediately

        if new_fields:
            logger.info(f"New output fields detected: {sorted(new_fields)}")
            logger.info(f"Total known output fields: {sorted(self.known_output_fields)}")

        # Serialize metadata to JSON if present
        if "metadata" in caption_dict:
            caption_dict["metadata"] = json.dumps(caption_dict.get("metadata", {}))
        else:
            caption_dict["metadata"] = "{}"

        # Add to buffer
        self.caption_buffer.append(caption_dict)

        # Log buffer status
        logger.debug(f"Caption buffer size: {len(self.caption_buffer)}/{self.caption_buffer_size}")

        # Flush if buffer is large enough
        if len(self.caption_buffer) >= self.caption_buffer_size:
            await self._flush_captions()

    async def save_captions(self, caption_data: Dict[str, Any]):
        """Save captions for an image - compatible with dict input."""
        job_id = caption_data["job_id"]

        # Check if we already have captions for this job_id
        if job_id in self.existing_caption_job_ids:
            self.duplicates_skipped += 1
            logger.debug(f"Skipping duplicate captions for job_id: {job_id}")
            return

        # Check if it's already in the buffer
        for buffered in self.caption_buffer:
            if buffered["job_id"] == job_id:
                logger.debug(f"Captions for job_id {job_id} already in buffer")
                return

        # Handle outputs if present
        if "outputs" in caption_data:
            outputs = caption_data.pop("outputs")
            # Add each output field directly to caption_data
            for field_name, field_values in outputs.items():
                caption_data[field_name] = field_values
                if field_name not in self.known_output_fields:
                    self.known_output_fields.add(field_name)
                    logger.info(f"New output field detected: {field_name}")

        # Handle legacy captions field
        if "captions" in caption_data and "captions" not in self.known_output_fields:
            self.known_output_fields.add("captions")

        # Count all outputs
        caption_count = 0
        for field_name in self.known_output_fields:
            if field_name in caption_data and isinstance(caption_data[field_name], list):
                caption_count += len(caption_data[field_name])

        caption_data["caption_count"] = caption_count

        # Add default values for optional fields
        if "quality_scores" not in caption_data:
            caption_data["quality_scores"] = None

        if "metadata" in caption_data and isinstance(caption_data["metadata"], dict):
            caption_data["metadata"] = json.dumps(caption_data["metadata"])
        elif "metadata" not in caption_data:
            caption_data["metadata"] = "{}"

        self.caption_buffer.append(caption_data)
        self.existing_caption_job_ids.add(job_id)

        # Flush if buffer is large enough
        if len(self.caption_buffer) >= self.caption_buffer_size:
            await self._flush_captions()

    async def _flush_captions(self):
        """Write caption buffer to parquet with dynamic schema."""
        if not self.caption_buffer:
            return

        num_rows = len(self.caption_buffer)

        # Count total outputs across all fields
        total_outputs = 0
        for row in self.caption_buffer:
            for field_name in self.known_output_fields:
                if field_name in row and isinstance(row[field_name], list):
                    total_outputs += len(row[field_name])

        logger.info(f"Flushing {num_rows} rows with {total_outputs} total outputs to disk")

        # Check if we need to evolve the schema
        current_schema_fields = set(self.caption_schema.names) if self.caption_schema else set()
        all_fields_needed = set(
            self.base_caption_fields[i][0] for i in range(len(self.base_caption_fields))
        )
        all_fields_needed.update(self.known_output_fields)

        if all_fields_needed != current_schema_fields:
            # Schema evolution needed
            logger.info(
                f"Evolving schema to include new fields: {all_fields_needed - current_schema_fields}"
            )
            self.caption_schema = self._build_caption_schema(self.known_output_fields)

            # If file exists, we need to migrate it
            if self.captions_path.exists():
                await self._evolve_schema_on_disk()

        # Prepare data with all required columns
        prepared_buffer = []
        for row in self.caption_buffer:
            prepared_row = row.copy()

            # Ensure all base fields are present
            for field_name, field_type in self.base_caption_fields:
                if field_name not in prepared_row:
                    prepared_row[field_name] = None

            # Ensure all output fields are present (even if None)
            for field_name in self.known_output_fields:
                if field_name not in prepared_row:
                    prepared_row[field_name] = None

            prepared_buffer.append(prepared_row)

        # Create table from buffer
        table = pa.Table.from_pylist(prepared_buffer, schema=self.caption_schema)

        if self.captions_path.exists():
            # Read existing table
            existing = pq.read_table(self.captions_path)

            # Get existing job_ids for deduplication
            existing_job_ids = set(existing.column("job_id").to_pylist())

            # Filter new data to exclude duplicates
            new_rows = []
            for row in prepared_buffer:
                if row["job_id"] not in existing_job_ids:
                    new_rows.append(row)

            if new_rows:
                # Create table from new rows only
                new_table = pa.Table.from_pylist(new_rows, schema=self.caption_schema)

                # Combine tables
                combined = pa.concat_tables([existing, new_table])

                # Write with proper preservation
                pq.write_table(combined, self.captions_path, compression="snappy")

                logger.info(
                    f"Added {len(new_rows)} new rows (skipped {num_rows - len(new_rows)} duplicates)"
                )
                actual_new = len(new_rows)
            else:
                logger.info(f"All {num_rows} rows were duplicates, skipping write")
                actual_new = 0
        else:
            # Write new file
            pq.write_table(table, self.captions_path, compression="snappy")
            actual_new = num_rows

        self.total_captions_written += actual_new
        self.total_caption_entries_written += total_outputs
        self.total_flushes += 1
        self.caption_buffer.clear()

        logger.info(
            f"Successfully wrote captions (rows: {self.total_captions_written}, "
            f"total outputs: {self.total_caption_entries_written}, "
            f"duplicates skipped: {self.duplicates_skipped})"
        )

    async def _evolve_schema_on_disk(self):
        """Evolve the schema of the existing parquet file to include new columns."""
        logger.info("Evolving schema on disk to add new columns...")

        # Read existing data
        existing_table = pq.read_table(self.captions_path)
        df = existing_table.to_pandas()

        # Add missing columns with None values
        for field_name in self.known_output_fields:
            if field_name not in df.columns:
                df[field_name] = None
                logger.info(f"Added new column: {field_name}")

        # Recreate table with new schema
        evolved_table = pa.Table.from_pandas(df, schema=self.caption_schema)
        pq.write_table(evolved_table, self.captions_path, compression="snappy")
        logger.info("Schema evolution complete")

    async def get_captions(self, job_id: str) -> Optional[Dict[str, List[str]]]:
        """Retrieve all output fields for a specific job_id."""
        # Check buffer first
        for buffered in self.caption_buffer:
            if buffered["job_id"] == job_id:
                outputs = {}
                for field_name in self.known_output_fields:
                    if field_name in buffered and buffered[field_name]:
                        outputs[field_name] = buffered[field_name]
                return outputs

        if not self.captions_path.exists():
            return None

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        row = df[df["job_id"] == job_id]
        if row.empty:
            return None

        # Collect all output fields
        outputs = {}
        for field_name in self.known_output_fields:
            if field_name in row.columns:
                value = row.iloc[0][field_name]
                if pd.notna(value) and value is not None:
                    outputs[field_name] = value

        return outputs if outputs else None

    async def save_job(self, job: Job):
        """Save or update a job - buffers until batch size reached."""
        # For updates, we still add to buffer (will be handled in flush)
        self.job_buffer.append(
            {
                "job_id": job.job_id,
                "dataset": job.dataset,
                "shard": job.shard,
                "item_key": job.item_key,
                "status": job.status.value,
                "assigned_to": job.assigned_to,
                "created_at": job.created_at,
                "updated_at": datetime.utcnow(),
            }
        )

        self.existing_job_ids.add(job.job_id)

        if len(self.job_buffer) >= self.job_buffer_size:
            await self._flush_jobs()

    async def save_contributor(self, contributor: Contributor):
        """Save or update contributor stats - buffers until batch size reached."""
        self.contributor_buffer.append(asdict(contributor))

        if len(self.contributor_buffer) >= self.contributor_buffer_size:
            await self._flush_contributors()

    async def _flush_jobs(self):
        """Write job buffer to parquet."""
        if not self.job_buffer:
            return

        table = pa.Table.from_pylist(self.job_buffer, schema=self.job_schema)

        # For jobs, we need to handle updates (upsert logic)
        if self.jobs_path.exists():
            existing = pq.read_table(self.jobs_path).to_pandas()
            new_df = table.to_pandas()

            # Update existing records or add new ones
            for _, row in new_df.iterrows():
                mask = existing["job_id"] == row["job_id"]
                if mask.any():
                    # Update existing
                    for col in row.index:
                        existing.loc[existing[mask].index, col] = row[col]
                else:
                    # Add new
                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

            updated_table = pa.Table.from_pandas(existing, schema=self.job_schema)
            pq.write_table(updated_table, self.jobs_path)
        else:
            pq.write_table(table, self.jobs_path)

        self.job_buffer.clear()
        logger.debug(f"Flushed {len(self.job_buffer)} jobs")

    async def _flush_contributors(self):
        """Write contributor buffer to parquet."""
        if not self.contributor_buffer:
            return

        table = pa.Table.from_pylist(self.contributor_buffer, schema=self.contributor_schema)

        # Handle updates for contributors
        if self.contributors_path.exists():
            existing = pq.read_table(self.contributors_path).to_pandas()
            new_df = table.to_pandas()

            for _, row in new_df.iterrows():
                mask = existing["contributor_id"] == row["contributor_id"]
                if mask.any():
                    for col in row.index:
                        existing.loc[mask, col] = row[col]
                else:
                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

            updated_table = pa.Table.from_pandas(existing, schema=self.contributor_schema)
            pq.write_table(updated_table, self.contributors_path)
        else:
            pq.write_table(table, self.contributors_path)

        self.contributor_buffer.clear()

    async def checkpoint(self):
        """Force flush all buffers to disk - called periodically by orchestrator."""
        logger.info(
            f"Checkpoint: Flushing buffers (captions: {len(self.caption_buffer)}, "
            f"jobs: {len(self.job_buffer)}, contributors: {len(self.contributor_buffer)})"
        )

        await self._flush_captions()
        await self._flush_jobs()
        await self._flush_contributors()

        logger.info(
            f"Checkpoint complete. Total rows: {self.total_captions_written}, "
            f"Total caption entries: {self.total_caption_entries_written}, "
            f"Duplicates skipped: {self.duplicates_skipped}"
        )

    async def job_exists(self, job_id: str) -> bool:
        """Check if a job already exists in storage or buffer."""
        if job_id in self.existing_job_ids:
            return True

        # Check buffer
        for buffered in self.job_buffer:
            if buffered["job_id"] == job_id:
                return True

        return False

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        # Check buffer first
        for buffered in self.job_buffer:
            if buffered["job_id"] == job_id:
                return Job(
                    job_id=buffered["job_id"],
                    dataset=buffered["dataset"],
                    shard=buffered["shard"],
                    item_key=buffered["item_key"],
                    status=JobStatus(buffered["status"]),
                    assigned_to=buffered["assigned_to"],
                    created_at=buffered["created_at"],
                )

        if not self.jobs_path.exists():
            return None

        table = pq.read_table(self.jobs_path)
        df = table.to_pandas()

        row = df[df["job_id"] == job_id]
        if row.empty:
            return None

        return Job(
            job_id=row.iloc[0]["job_id"],
            dataset=row.iloc[0]["dataset"],
            shard=row.iloc[0]["shard"],
            item_key=row.iloc[0]["item_key"],
            status=JobStatus(row.iloc[0]["status"]),
            assigned_to=row.iloc[0]["assigned_to"],
            created_at=row.iloc[0]["created_at"],
        )

    async def get_jobs_by_worker(self, worker_id: str) -> List[Job]:
        """Get all jobs assigned to a worker."""
        if not self.jobs_path.exists():
            return []

        table = pq.read_table(self.jobs_path)
        df = table.to_pandas()

        rows = df[df["assigned_to"] == worker_id]

        jobs = []
        for _, row in rows.iterrows():
            jobs.append(
                Job(
                    job_id=row["job_id"],
                    dataset=row["dataset"],
                    shard=row["shard"],
                    item_key=row["item_key"],
                    status=JobStatus(row["status"]),
                    assigned_to=row["assigned_to"],
                    created_at=row["created_at"],
                )
            )

        return jobs

    async def get_caption_stats(self) -> Dict[str, Any]:
        """Get statistics about stored captions including field-specific stats."""
        if not self.captions_path.exists():
            return {"total_rows": 0, "total_outputs": 0, "output_fields": [], "field_stats": {}}

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        if len(df) == 0:
            return {"total_rows": 0, "total_outputs": 0, "output_fields": [], "field_stats": {}}

        # Get actual columns in the dataframe
        existing_columns = set(df.columns)

        # Calculate stats per field (only for fields that exist in the file)
        field_stats = {}
        total_outputs = 0

        for field_name in self.known_output_fields:
            if field_name in existing_columns:
                # Count non-null entries
                non_null_mask = df[field_name].notna()
                non_null_count = non_null_mask.sum()

                # Count total items in lists
                field_total = 0
                field_lengths = []

                for value in df.loc[non_null_mask, field_name]:
                    if isinstance(value, list):
                        length = len(value)
                        field_total += length
                        field_lengths.append(length)
                    elif pd.notna(value):
                        length = 1
                        field_total += length
                        field_lengths.append(length)

                if field_lengths:
                    field_stats[field_name] = {
                        "rows_with_data": non_null_count,
                        "total_items": field_total,
                        "avg_items_per_row": sum(field_lengths) / len(field_lengths),
                    }
                    if min(field_lengths) != max(field_lengths):
                        field_stats[field_name].update(
                            {
                                "min_items": min(field_lengths),
                                "max_items": max(field_lengths),
                            }
                        )
                    total_outputs += field_total

        return {
            "total_rows": len(df),
            "total_outputs": total_outputs,
            "output_fields": sorted(list(self.known_output_fields)),
            "field_stats": field_stats,
            "caption_count_stats": {
                "mean": df["caption_count"].mean() if "caption_count" in df.columns else 0,
                "min": df["caption_count"].min() if "caption_count" in df.columns else 0,
                "max": df["caption_count"].max() if "caption_count" in df.columns else 0,
            },
        }

    async def get_sample_captions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of caption entries showing all output fields."""
        if not self.captions_path.exists():
            return []

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        if len(df) == 0:
            return []

        sample_df = df.sample(min(n, len(df)))
        samples = []

        for _, row in sample_df.iterrows():
            # Collect outputs from dynamic columns
            outputs = {}
            total_outputs = 0

            for field_name in self.known_output_fields:
                if field_name in row and pd.notna(row[field_name]):
                    value = row[field_name]
                    outputs[field_name] = value
                    if isinstance(value, list):
                        total_outputs += len(value)

            samples.append(
                {
                    "job_id": row["job_id"],
                    "item_key": row["item_key"],
                    "outputs": outputs,
                    "field_count": len(outputs),
                    "total_outputs": total_outputs,
                    "image_dims": f"{row.get('image_width', 'N/A')}x{row.get('image_height', 'N/A')}",
                    "has_metadata": bool(row.get("metadata") and row["metadata"] != "{}"),
                }
            )

        return samples

    async def count_captions(self) -> int:
        """Count total outputs across all dynamic fields."""
        total = 0

        if self.captions_path.exists():
            # Get actual columns in the file
            table_metadata = pq.read_metadata(self.captions_path)
            existing_columns = set(table_metadata.schema.names)

            # Only read output fields that actually exist in the file
            columns_to_read = [f for f in self.known_output_fields if f in existing_columns]

            if columns_to_read:
                table = pq.read_table(self.captions_path, columns=columns_to_read)
                df = table.to_pandas()

                for field_name in columns_to_read:
                    for value in df[field_name]:
                        if pd.notna(value) and isinstance(value, list):
                            total += len(value)

        # Add buffer counts
        for row in self.caption_buffer:
            for field_name in self.known_output_fields:
                if field_name in row and isinstance(row[field_name], list):
                    total += len(row[field_name])

        return total

    async def count_caption_rows(self) -> int:
        """Count total rows (unique images with captions)."""
        if not self.captions_path.exists():
            return 0

        table = pq.read_table(self.captions_path)
        return len(table)

    async def get_contributor(self, contributor_id: str) -> Optional[Contributor]:
        """Retrieve a contributor by ID."""
        # Check buffer first
        for buffered in self.contributor_buffer:
            if buffered["contributor_id"] == contributor_id:
                return Contributor(**buffered)

        if not self.contributors_path.exists():
            return None

        table = pq.read_table(self.contributors_path)
        df = table.to_pandas()

        row = df[df["contributor_id"] == contributor_id]
        if row.empty:
            return None

        return Contributor(
            contributor_id=row.iloc[0]["contributor_id"],
            name=row.iloc[0]["name"],
            total_captions=int(row.iloc[0]["total_captions"]),
            trust_level=int(row.iloc[0]["trust_level"]),
        )

    async def get_top_contributors(self, limit: int = 10) -> List[Contributor]:
        """Get top contributors by caption count."""
        contributors = []

        if self.contributors_path.exists():
            table = pq.read_table(self.contributors_path)
            df = table.to_pandas()

            # Sort by total_captions descending
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
        """Get statistics about output fields in stored captions."""
        if not self.captions_path.exists():
            return {"total_fields": 0, "field_counts": {}}

        if not self.known_output_fields:
            return {"total_fields": 0, "field_counts": {}}

        # Get actual columns in the file
        table_metadata = pq.read_metadata(self.captions_path)
        existing_columns = set(table_metadata.schema.names)

        # Only read output fields that actually exist in the file
        columns_to_read = [f for f in self.known_output_fields if f in existing_columns]

        if not columns_to_read:
            return {"total_fields": 0, "field_counts": {}}

        table = pq.read_table(self.captions_path, columns=columns_to_read)
        df = table.to_pandas()

        if len(df) == 0:
            return {"total_fields": 0, "field_counts": {}}

        # Count outputs by field
        field_counts = {}
        total_outputs = 0

        for field_name in columns_to_read:
            field_count = 0
            for value in df[field_name]:
                if pd.notna(value) and isinstance(value, list):
                    field_count += len(value)

            if field_count > 0:
                field_counts[field_name] = field_count
                total_outputs += field_count

        return {
            "total_fields": len(field_counts),
            "field_counts": field_counts,
            "total_outputs": total_outputs,
            "fields": sorted(list(field_counts.keys())),
        }

    async def get_captions_with_field(
        self, field_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get captions that have a specific output field."""
        if not self.captions_path.exists():
            return []

        if field_name not in self.known_output_fields:
            logger.warning(f"Field '{field_name}' not found in known output fields")
            return []

        # Check if the field actually exists in the file
        existing_output_columns = self._get_existing_output_columns()
        if field_name not in existing_output_columns:
            logger.warning(
                f"Field '{field_name}' exists in known fields but not in parquet file yet"
            )
            return []

        # Only read necessary columns
        columns_to_read = ["job_id", "item_key", field_name]

        try:
            table = pq.read_table(self.captions_path, columns=columns_to_read)
        except Exception as e:
            logger.error(f"Error reading field '{field_name}': {e}")
            return []

        df = table.to_pandas()

        # Filter rows where field has data
        mask = df[field_name].notna()
        filtered_df = df[mask].head(limit)

        results = []
        for _, row in filtered_df.iterrows():
            results.append(
                {
                    "job_id": row["job_id"],
                    "item_key": row["item_key"],
                    field_name: row[field_name],
                    "value_count": len(row[field_name]) if isinstance(row[field_name], list) else 1,
                }
            )

        return results

    async def export_by_field(self, field_name: str, output_path: Path, format: str = "jsonl"):
        """Export all captions for a specific field."""
        if not self.captions_path.exists():
            logger.warning("No captions to export")
            return 0

        if field_name not in self.known_output_fields:
            logger.warning(f"Field '{field_name}' not found in known output fields")
            return 0

        # Check if the field actually exists in the file
        existing_output_columns = self._get_existing_output_columns()
        if field_name not in existing_output_columns:
            logger.warning(f"Field '{field_name}' not found in parquet file")
            return 0

        # Read only necessary columns
        columns_to_read = ["item_key", "dataset", field_name]
        table = pq.read_table(self.captions_path, columns=columns_to_read)
        df = table.to_pandas()

        exported = 0
        with open(output_path, "w") as f:
            for _, row in df.iterrows():
                if pd.notna(row[field_name]) and row[field_name]:
                    if format == "jsonl":
                        record = {
                            "item_key": row["item_key"],
                            "dataset": row["dataset"],
                            field_name: row[field_name],
                        }
                        f.write(json.dumps(record) + "\n")
                        exported += 1

        logger.info(f"Exported {exported} items with field '{field_name}' to {output_path}")
        return exported

    async def get_pending_jobs(self) -> List[Job]:
        """Get all pending jobs for restoration on startup."""
        if not self.jobs_path.exists():
            return []

        table = pq.read_table(self.jobs_path)
        df = table.to_pandas()

        # Get jobs with PENDING or PROCESSING status
        pending_df = df[df["status"].isin([JobStatus.PENDING.value, JobStatus.PROCESSING.value])]

        jobs = []
        for _, row in pending_df.iterrows():
            jobs.append(
                Job(
                    job_id=row["job_id"],
                    dataset=row["dataset"],
                    shard=row["shard"],
                    item_key=row["item_key"],
                    status=JobStatus(row["status"]),
                    assigned_to=row.get("assigned_to"),
                    created_at=row["created_at"],
                )
            )

        return jobs

    async def count_jobs(self) -> int:
        """Count total jobs."""
        if not self.jobs_path.exists():
            return 0

        table = pq.read_table(self.jobs_path)
        return len(table)

    async def count_completed_jobs(self) -> int:
        """Count completed jobs."""
        if not self.jobs_path.exists():
            return 0

        table = pq.read_table(self.jobs_path)
        df = table.to_pandas()
        return len(df[df["status"] == JobStatus.COMPLETED.value])

    async def close(self):
        """Close storage and flush buffers."""
        await self.checkpoint()
        logger.info(
            f"Storage closed. Total rows: {self.total_captions_written}, "
            f"Total caption entries: {self.total_caption_entries_written}, "
            f"Duplicates skipped: {self.duplicates_skipped}"
        )

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get all storage-related statistics."""
        # Count outputs on disk
        disk_outputs = await self.count_captions()

        # Count outputs in buffer
        buffer_outputs = 0
        for row in self.caption_buffer:
            for field_name in self.known_output_fields:
                if field_name in row and isinstance(row[field_name], list):
                    buffer_outputs += len(row[field_name])

        # Get field-specific stats
        field_stats = await self.get_caption_stats()
        total_rows_including_buffer = await self.count_caption_rows() + len(self.caption_buffer)

        return {
            "total_captions": disk_outputs + buffer_outputs,
            "total_rows": total_rows_including_buffer,
            "buffer_size": len(self.caption_buffer),
            "total_written": self.total_captions_written,
            "total_entries_written": self.total_caption_entries_written,
            "duplicates_skipped": self.duplicates_skipped,
            "total_flushes": self.total_flushes,
            "output_fields": sorted(list(self.known_output_fields)),
            "field_breakdown": field_stats.get("field_stats", None),
            "job_buffer_size": len(self.job_buffer),
            "contributor_buffer_size": len(self.contributor_buffer),
        }
