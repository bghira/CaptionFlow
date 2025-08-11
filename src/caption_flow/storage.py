"""Arrow/Parquet storage management with list column support for captions."""

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

from .models import Job, Caption, Contributor, JobStatus

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages Arrow/Parquet storage for captions and jobs with list column support."""

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

        # Statistics
        self.total_captions_written = 0
        self.total_caption_entries_written = 0  # Total individual captions
        self.total_flushes = 0
        self.duplicates_skipped = 0

        # Schemas - Updated caption schema to support list of captions
        self.caption_schema = pa.schema(
            [
                ("job_id", pa.string()),
                ("dataset", pa.string()),
                ("shard", pa.string()),
                ("item_key", pa.string()),
                ("captions", pa.list_(pa.string())),  # Changed from single caption to list
                ("caption_count", pa.int32()),  # Number of captions for this item
                ("contributor_id", pa.string()),
                ("timestamp", pa.timestamp("us")),
                ("quality_scores", pa.list_(pa.float32())),  # Optional quality scores per caption
                ("image_width", pa.int32()),
                ("image_height", pa.int32()),
                ("image_format", pa.string()),
                ("file_size", pa.int64()),
                ("processing_time_ms", pa.float32()),
            ]
        )

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

    async def initialize(self):
        """Initialize storage files if they don't exist."""
        # Create empty parquet files if needed
        if not self.captions_path.exists():
            # Create empty table with schema using from_pydict
            empty_dict = {
                "job_id": [],
                "dataset": [],
                "shard": [],
                "item_key": [],
                "captions": [],
                "caption_count": [],
                "contributor_id": [],
                "timestamp": [],
                "quality_scores": [],
                "image_width": [],
                "image_height": [],
                "image_format": [],
                "file_size": [],
                "processing_time_ms": [],
            }
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.caption_schema)
            pq.write_table(empty_table, self.captions_path)
            logger.info(f"Created empty caption storage at {self.captions_path}")
        else:
            # Load existing caption job_ids to prevent duplicates
            existing_captions = pq.read_table(self.captions_path, columns=["job_id"])
            self.existing_caption_job_ids = set(existing_captions["job_id"].to_pylist())
            logger.info(f"Loaded {len(self.existing_caption_job_ids)} existing caption job_ids")

        if not self.jobs_path.exists():
            # Create empty table with schema using from_pydict
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
            pq.write_table(empty_table, self.jobs_path)
            logger.info(f"Created empty job storage at {self.jobs_path}")
        else:
            # Load existing job_ids
            existing_jobs = pq.read_table(self.jobs_path, columns=["job_id"])
            self.existing_job_ids = set(existing_jobs["job_id"].to_pylist())
            logger.info(f"Loaded {len(self.existing_job_ids)} existing job_ids")

        if not self.contributors_path.exists():
            # Create empty table with schema using from_pydict
            empty_dict = {"contributor_id": [], "name": [], "total_captions": [], "trust_level": []}
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.contributor_schema)
            pq.write_table(empty_table, self.contributors_path)
            logger.info(f"Created empty contributor storage at {self.contributors_path}")
        else:
            # Load existing contributors
            existing_contributors = pq.read_table(
                self.contributors_path, columns=["contributor_id"]
            )
            self.existing_contributor_ids = set(existing_contributors["contributor_id"].to_pylist())
            logger.info(f"Loaded {len(self.existing_contributor_ids)} existing contributor IDs")

    async def save_captions(self, caption_data: Dict[str, Any]):
        """Save captions for an image - single row with list of captions."""
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

        # Ensure captions is a list (not a JSON string)
        captions = caption_data.get("captions")
        if isinstance(captions, str):
            # If it's a JSON string, decode it
            import json

            try:
                captions = json.loads(captions)
                caption_data["captions"] = captions
                logger.warning(f"Decoded JSON string to list for job_id {job_id}")
            except json.JSONDecodeError:
                logger.error(f"Invalid captions format for job_id {job_id}")
                return

        if not isinstance(captions, list):
            logger.error(f"Captions must be a list for job_id {job_id}, got {type(captions)}")
            return

        # Add caption count
        caption_data["caption_count"] = len(captions)

        # Add default values for optional fields if not present
        if "quality_scores" not in caption_data:
            caption_data["quality_scores"] = None

        self.caption_buffer.append(caption_data)
        self.existing_caption_job_ids.add(job_id)

        # Log buffer status
        logger.debug(f"Caption buffer size: {len(self.caption_buffer)}/{self.caption_buffer_size}")
        logger.debug(f"  Added captions for {job_id}: {len(captions)} captions")

        # Flush if buffer is large enough
        if len(self.caption_buffer) >= self.caption_buffer_size:
            await self._flush_captions()

    async def save_caption(self, caption: Caption):
        """Save a single caption entry."""
        # Convert to dict and ensure it's a list of captions
        caption_dict = asdict(caption)
        if "captions" in caption_dict and not isinstance(caption_dict["captions"], list):
            caption_dict["captions"] = [caption_dict["captions"]]
        elif "caption" in caption_dict and isinstance(caption_dict["caption"], str):
            # If it's a single caption string, wrap it in a list
            caption_dict["captions"] = [caption_dict["caption"]]
            del caption_dict["caption"]

        # Add to buffer
        self.caption_buffer.append(caption_dict)

        # Log buffer status
        logger.debug(f"Caption buffer size: {len(self.caption_buffer)}/{self.caption_buffer_size}")

        # Flush if buffer is large enough
        if len(self.caption_buffer) >= self.caption_buffer_size:
            await self._flush_captions()

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

    async def _flush_captions(self):
        """Write caption buffer to parquet with deduplication."""
        if not self.caption_buffer:
            return

        num_rows = len(self.caption_buffer)
        num_captions = sum(len(row["captions"]) for row in self.caption_buffer)
        logger.info(f"Flushing {num_rows} rows with {num_captions} total captions to disk")

        # Ensure all captions are proper lists before creating table
        for row in self.caption_buffer:
            if isinstance(row["captions"], str):
                import json

                try:
                    row["captions"] = json.loads(row["captions"])
                except:
                    logger.error(f"Failed to decode captions for {row['job_id']}")
                    row["captions"] = [row["captions"]]  # Wrap string in list as fallback

        # Create table from buffer with explicit schema
        table = pa.Table.from_pylist(self.caption_buffer, schema=self.caption_schema)

        if self.captions_path.exists():
            # Read existing table
            existing = pq.read_table(self.captions_path)

            # Get existing job_ids for deduplication
            existing_job_ids = set(existing.column("job_id").to_pylist())

            # Filter new data to exclude duplicates
            new_rows = []
            for row in self.caption_buffer:
                if row["job_id"] not in existing_job_ids:
                    new_rows.append(row)

            if new_rows:
                # Create table from new rows only
                new_table = pa.Table.from_pylist(new_rows, schema=self.caption_schema)

                # Combine tables using PyArrow concat (preserves list types better)
                combined = pa.concat_tables([existing, new_table])

                # Write with proper list column preservation
                pq.write_table(combined, self.captions_path, compression="snappy")

                logger.info(
                    f"Added {len(new_rows)} new rows (skipped {num_rows - len(new_rows)} duplicates)"
                )
                actual_new = len(new_rows)
            else:
                logger.info(f"All {num_rows} rows were duplicates, skipping write")
                actual_new = 0
        else:
            # Write new file with proper list columns
            pq.write_table(table, self.captions_path, compression="snappy")
            actual_new = num_rows

        self.total_captions_written += actual_new
        self.total_caption_entries_written += sum(
            len(row["captions"]) for row in self.caption_buffer[:actual_new]
        )
        self.total_flushes += 1
        self.caption_buffer.clear()

        logger.info(
            f"Successfully wrote captions (rows: {self.total_captions_written}, "
            f"total captions: {self.total_caption_entries_written}, "
            f"duplicates skipped: {self.duplicates_skipped})"
        )

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

    async def get_captions(self, job_id: str) -> Optional[List[str]]:
        """Retrieve captions for a specific job_id."""
        # Check buffer first
        for buffered in self.caption_buffer:
            if buffered["job_id"] == job_id:
                return buffered["captions"]

        if not self.captions_path.exists():
            return None

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        row = df[df["job_id"] == job_id]
        if row.empty:
            return None

        captions = row.iloc[0]["captions"]

        # Handle both correct list storage and incorrect JSON string storage
        if isinstance(captions, str):
            # This shouldn't happen with correct storage, but handle legacy data
            try:
                captions = json.loads(captions)
                logger.warning(f"Had to decode JSON string for job_id {job_id} - file needs fixing")
            except json.JSONDecodeError:
                captions = [captions]  # Wrap single string as list

        return captions

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
        """Get statistics about stored captions."""
        if not self.captions_path.exists():
            return {
                "total_rows": 0,
                "total_captions": 0,
                "avg_captions_per_image": 0,
                "min_captions": 0,
                "max_captions": 0,
            }

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        if len(df) == 0:
            return {
                "total_rows": 0,
                "total_captions": 0,
                "avg_captions_per_image": 0,
                "min_captions": 0,
                "max_captions": 0,
            }

        caption_counts = df["caption_count"].values

        return {
            "total_rows": len(df),
            "total_captions": caption_counts.sum(),
            "avg_captions_per_image": caption_counts.mean(),
            "min_captions": caption_counts.min(),
            "max_captions": caption_counts.max(),
            "std_captions": caption_counts.std(),
        }

    async def get_sample_captions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of caption entries for inspection."""
        if not self.captions_path.exists():
            return []

        table = pq.read_table(self.captions_path)
        df = table.to_pandas()

        if len(df) == 0:
            return []

        sample_df = df.sample(min(n, len(df)))
        samples = []

        for _, row in sample_df.iterrows():
            samples.append(
                {
                    "job_id": row["job_id"],
                    "item_key": row["item_key"],
                    "captions": row["captions"],
                    "caption_count": row["caption_count"],
                    "image_dims": f"{row.get('image_width', 'N/A')}x{row.get('image_height', 'N/A')}",
                }
            )

        return samples

    async def count_captions(self) -> int:
        """Count total caption entries (not rows)."""
        if not self.captions_path.exists():
            return 0

        table = pq.read_table(self.captions_path, columns=["caption_count"])
        df = table.to_pandas()
        return df["caption_count"].sum()

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
