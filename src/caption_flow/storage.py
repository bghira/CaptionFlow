"""Arrow/Parquet storage management."""

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

from .models import Job, Caption, Contributor, JobStatus

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages Arrow/Parquet storage for captions and jobs."""
    
    def __init__(self, data_dir: Path, 
                 caption_buffer_size: int = 100,
                 job_buffer_size: int = 100,
                 contributor_buffer_size: int = 10):
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
        
        # Statistics
        self.total_captions_written = 0
        self.total_flushes = 0
        
        # Schemas
        self.caption_schema = pa.schema([
            ("job_id", pa.string()),
            ("dataset", pa.string()),
            ("shard", pa.string()),
            ("item_key", pa.string()),
            ("caption", pa.string()),
            ("contributor_id", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("quality_score", pa.float32())
        ])
        
        self.job_schema = pa.schema([
            ("job_id", pa.string()),
            ("dataset", pa.string()),
            ("shard", pa.string()),
            ("item_key", pa.string()),
            ("status", pa.string()),
            ("assigned_to", pa.string()),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us"))
        ])
        
        self.contributor_schema = pa.schema([
            ("contributor_id", pa.string()),
            ("name", pa.string()),
            ("total_captions", pa.int64()),
            ("trust_level", pa.int32())
        ])
    
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
                "caption": [],
                "contributor_id": [],
                "timestamp": [],
                "quality_score": []
            }
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.caption_schema)
            pq.write_table(empty_table, self.captions_path)
            logger.info(f"Created empty caption storage at {self.captions_path}")
        
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
                "updated_at": []
            }
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.job_schema)
            pq.write_table(empty_table, self.jobs_path)
            logger.info(f"Created empty job storage at {self.jobs_path}")
        
        if not self.contributors_path.exists():
            # Create empty table with schema using from_pydict
            empty_dict = {
                "contributor_id": [],
                "name": [],
                "total_captions": [],
                "trust_level": []
            }
            empty_table = pa.Table.from_pydict(empty_dict, schema=self.contributor_schema)
            pq.write_table(empty_table, self.contributors_path)
            logger.info(f"Created empty contributor storage at {self.contributors_path}")
    
    async def save_caption(self, caption: Caption):
        """Save a caption to storage - buffers until batch size reached."""
        self.caption_buffer.append(asdict(caption))
        
        # Flush if buffer is large enough
        if len(self.caption_buffer) >= self.caption_buffer_size:
            await self._flush_captions()
    
    async def save_job(self, job: Job):
        """Save or update a job - buffers until batch size reached."""
        self.job_buffer.append({
            "job_id": job.job_id,
            "dataset": job.dataset,
            "shard": job.shard,
            "item_key": job.item_key,
            "status": job.status.value,
            "assigned_to": job.assigned_to,
            "created_at": job.created_at,
            "updated_at": datetime.utcnow()
        })
        
        if len(self.job_buffer) >= self.job_buffer_size:
            await self._flush_jobs()
    
    async def save_contributor(self, contributor: Contributor):
        """Save or update contributor stats - buffers until batch size reached."""
        self.contributor_buffer.append(asdict(contributor))
        
        if len(self.contributor_buffer) >= self.contributor_buffer_size:
            await self._flush_contributors()
    
    async def _flush_captions(self):
        """Write caption buffer to parquet - called by orchestrator."""
        if not self.caption_buffer:
            return
        
        num_captions = len(self.caption_buffer)
        logger.info(f"Flushing {num_captions} captions to disk")
        
        table = pa.Table.from_pylist(self.caption_buffer, schema=self.caption_schema)
        
        if self.captions_path.exists():
            existing = pq.read_table(self.captions_path)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, self.captions_path)
        else:
            pq.write_table(table, self.captions_path)
        
        self.total_captions_written += num_captions
        self.total_flushes += 1
        self.caption_buffer.clear()
        
        logger.info(f"Successfully wrote {num_captions} captions (total: {self.total_captions_written})")
    
    async def _flush_jobs(self):
        """Write job buffer to parquet."""
        if not self.job_buffer:
            return
        
        table = pa.Table.from_pylist(self.job_buffer, schema=self.job_schema)
        
        # For jobs, we need to handle updates (upsert logic)
        if self.jobs_path.exists():
            existing = pq.read_table(self.jobs_path).to_pandas()
            new_df = table.to_pandas()
            
            # Update existing records
            for _, row in new_df.iterrows():
                mask = existing["job_id"] == row["job_id"]
                if mask.any():
                    existing.loc[mask] = row
                else:
                    existing = existing.append(row, ignore_index=True)
            
            updated_table = pa.Table.from_pandas(existing, schema=self.job_schema)
            pq.write_table(updated_table, self.jobs_path)
        else:
            pq.write_table(table, self.jobs_path)
        
        self.job_buffer.clear()
    
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
                    existing.loc[mask] = row
                else:
                    existing = existing.append(row, ignore_index=True)
            
            updated_table = pa.Table.from_pandas(existing, schema=self.contributor_schema)
            pq.write_table(updated_table, self.contributors_path)
        else:
            pq.write_table(table, self.contributors_path)
        
        self.contributor_buffer.clear()
    
    async def checkpoint(self):
        """Force flush all buffers to disk - called periodically by orchestrator."""
        logger.info(f"Checkpoint: Flushing buffers (captions: {len(self.caption_buffer)}, "
                   f"jobs: {len(self.job_buffer)}, contributors: {len(self.contributor_buffer)})")
        
        await self._flush_captions()
        await self._flush_jobs()
        await self._flush_contributors()
        
        logger.info(f"Checkpoint complete. Total captions written: {self.total_captions_written}")
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
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
            created_at=row.iloc[0]["created_at"]
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
            jobs.append(Job(
                job_id=row["job_id"],
                dataset=row["dataset"],
                shard=row["shard"],
                item_key=row["item_key"],
                status=JobStatus(row["status"]),
                assigned_to=row["assigned_to"],
                created_at=row["created_at"]
            ))
        
        return jobs
    
    async def get_pending_jobs(self) -> List[Job]:
        """Get all pending jobs."""
        if not self.jobs_path.exists():
            return []
        
        table = pq.read_table(self.jobs_path)
        df = table.to_pandas()
        
        rows = df[df["status"] == JobStatus.PENDING.value]
        
        jobs = []
        for _, row in rows.iterrows():
            jobs.append(Job(
                job_id=row["job_id"],
                dataset=row["dataset"],
                shard=row["shard"],
                item_key=row["item_key"],
                status=JobStatus.PENDING,
                assigned_to=None,
                created_at=row["created_at"]
            ))
        
        return jobs
    
    async def get_contributor(self, contributor_id: str) -> Optional[Contributor]:
        """Get contributor by ID."""
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
            total_captions=row.iloc[0]["total_captions"],
            trust_level=row.iloc[0]["trust_level"]
        )
    
    async def get_top_contributors(self, limit: int = 10) -> List[Contributor]:
        """Get top contributors by caption count."""
        if not self.contributors_path.exists():
            return []
        
        table = pq.read_table(self.contributors_path)
        df = table.to_pandas()
        
        df = df.nlargest(limit, "total_captions")
        
        contributors = []
        for _, row in df.iterrows():
            contributors.append(Contributor(
                contributor_id=row["contributor_id"],
                name=row["name"],
                total_captions=row["total_captions"],
                trust_level=row["trust_level"]
            ))
        
        return contributors
    
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
    
    async def count_captions(self) -> int:
        """Count total captions."""
        if not self.captions_path.exists():
            return 0
        
        table = pq.read_table(self.captions_path)
        return len(table)
    
    async def close(self):
        """Close storage and flush buffers."""
        await self.checkpoint()