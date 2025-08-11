"""Data models for CaptionFlow."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


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
            self.created_at = datetime.utcnow()


@dataclass
class Caption:
    """Generated caption with attribution and image metadata."""

    # Core fields
    job_id: str
    dataset: str
    shard: str
    item_key: str
    caption: str
    contributor_id: str
    timestamp: datetime
    quality_score: Optional[float] = None

    # Image metadata
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None
    file_size: Optional[int] = None

    # Processing metadata
    caption_index: Optional[int] = None  # Which caption this is (0, 1, 2...)
    total_captions: Optional[int] = None  # Total captions for this image
    processing_time_ms: Optional[float] = None
    chunk_id: Optional[str] = None


@dataclass
class Contributor:
    """Contributor information."""

    contributor_id: str
    name: str
    total_captions: int = 0
    trust_level: int = 1
