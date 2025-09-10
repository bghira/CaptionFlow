"""Data models for CaptionFlow."""

import datetime as _datetime
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class JobStatus(Enum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self):
        return self.value

    def to_json(self):
        return self.value


@dataclass
class Job:
    """Captioning job."""

    job_id: str
    dataset: str
    shard: str
    item_key: str
    status: JobStatus = JobStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(_datetime.UTC)


@dataclass
class JobId:
    shard_id: str
    chunk_id: str
    sample_id: str

    def get_shard_str(self):
        return f"{self.shard_id}"

    def get_chunk_str(self):
        return f"{self.shard_id}:chunk:{self.chunk_id}"

    def get_sample_str(self):
        return f"{self.shard_id}:chunk:{self.chunk_id}:idx:{self.sample_id}"

    @staticmethod
    def from_dict(job: dict) -> "JobId":
        return JobId(shard_id=job["shard_id"], chunk_id=job["chunk_id"], sample_id=job["sample_id"])

    @staticmethod
    def from_values(shard_id: str, chunk_id: str, sample_id: str) -> "JobId":
        return JobId(shard_id=shard_id, chunk_id=chunk_id, sample_id=sample_id)

    @staticmethod
    def from_str(job_id: str):
        # from data-0000:chunk:0:idx:0
        parts = job_id.split(":")
        if len(parts) != 5:
            raise ValueError(f"Invalid job_id format: {job_id}")

        shard_id = parts[0]
        chunk_keyword = parts[1]
        chunk_id = parts[2]
        idx_keyword = parts[3]
        sample_id = parts[4]

        # Validate format
        if not shard_id:
            raise ValueError(f"Invalid job_id format: empty shard_id in {job_id}")
        if chunk_keyword != "chunk":
            raise ValueError(
                f"Invalid job_id format: expected 'chunk' keyword, got '{chunk_keyword}' in {job_id}"
            )
        if idx_keyword != "idx":
            raise ValueError(
                f"Invalid job_id format: expected 'idx' keyword, got '{idx_keyword}' in {job_id}"
            )

        # Validate numeric fields
        try:
            int(chunk_id)
        except ValueError:
            raise ValueError(
                f"Invalid job_id format: chunk_id must be numeric, got '{chunk_id}' in {job_id}"
            )

        # sample_id can be empty/None for some use cases, but if provided must be numeric
        if sample_id:
            try:
                int(sample_id)
            except ValueError:
                raise ValueError(
                    f"Invalid job_id format: sample_id must be numeric if provided, got '{sample_id}' in {job_id}"
                )

        return JobId(shard_id=shard_id, chunk_id=chunk_id, sample_id=sample_id)


@dataclass
class Caption:
    """Generated caption with attribution and image metadata."""

    # Core fields
    job_id: str
    dataset: str
    shard: str
    item_key: str
    contributor_id: str
    timestamp: datetime
    caption_count: int = 1  # Number of captions generated for this item
    caption: Optional[str] = None
    captions: Optional[List[str]] = None
    outputs: Dict[str, List[str]] = field(default_factory=dict)
    quality_score: Optional[float] = None
    quality_scores: Optional[List[float]] = None

    # Image metadata
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None
    file_size: Optional[int] = None
    filename: Optional[str] = None
    url: Optional[str] = None

    # Processing metadata
    caption_index: Optional[int] = None  # Which caption this is (0, 1, 2...)
    total_captions: Optional[int] = None  # Total captions for this image
    processing_time_ms: Optional[float] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.caption is None and self.captions is None:
            raise ValueError("At least one of 'caption' or 'captions' must be provided")


@dataclass
class Contributor:
    """Contributor information."""

    contributor_id: str
    name: str
    total_captions: int = 0
    trust_level: int = 1


@dataclass
class ProcessingStage:
    """Configuration for a single processing stage."""

    name: str
    model: str
    prompts: List[str]
    output_field: str
    requires: List[str] = field(default_factory=list)
    sampling: Optional[Dict[str, Any]] = None

    # Model-specific overrides
    tensor_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None


@dataclass
class StageResult:
    """Results from a single stage."""

    stage_name: str
    output_field: str
    outputs: List[str]  # Multiple outputs from multiple prompts
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.error is None and bool(self.outputs)


@dataclass
class ShardChunk:
    """Shard chunk assignment with unprocessed ranges."""

    chunk_id: str
    shard_url: str
    shard_name: str
    start_index: int
    chunk_size: int
    unprocessed_ranges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ProcessingItem:
    """Item being processed."""

    chunk_id: str
    item_key: str
    image: Image.Image
    image_data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)  # Accumulated results


@dataclass
class ProcessedResult:
    """Result with multi-stage outputs."""

    chunk_id: str
    shard_name: str
    item_key: str
    outputs: Dict[str, List[str]]  # field_name -> list of outputs
    image_width: int
    image_height: int
    image_format: str
    file_size: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageContents:
    """Container for storage data to be exported."""

    rows: List[Dict[str, Any]]
    columns: List[str]
    output_fields: List[str]
    total_rows: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency."""
        if self.rows and self.columns:
            # Ensure all rows have the expected columns
            for row in self.rows:
                missing_cols = set(self.columns) - set(row.keys())
                if missing_cols:
                    logger.warning(f"Row missing columns: {missing_cols}")


class ExportError(Exception):
    """Base exception for export-related errors."""

    pass
