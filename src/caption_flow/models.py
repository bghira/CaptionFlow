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
    """Generated caption with attribution."""
    job_id: str
    dataset: str
    shard: str
    item_key: str
    caption: str
    contributor_id: str
    timestamp: datetime
    quality_score: Optional[float] = None

@dataclass
class Contributor:
    """Contributor information."""
    contributor_id: str
    name: str
    total_captions: int = 0
    trust_level: int = 1