"""Job queue management."""

import asyncio
from typing import Optional
from collections import deque

from ..models import Job


class JobQueue:
    """Priority job queue with backpressure."""

    def __init__(self):
        self.queue = deque()
        self.processing = set()
        self.lock = asyncio.Lock()

    async def add(self, job: Job):
        """Add job to queue."""
        async with self.lock:
            self.queue.append(job)

    async def get_next(self) -> Optional[Job]:
        """Get next available job."""
        async with self.lock:
            if self.queue:
                job = self.queue.popleft()
                self.processing.add(job.job_id)
                return job
        return None

    async def complete(self, job_id: str):
        """Mark job as complete."""
        async with self.lock:
            self.processing.discard(job_id)

    async def requeue(self, job: Job):
        """Requeue a job (for failures)."""
        async with self.lock:
            self.processing.discard(job.job_id)
            self.queue.appendleft(job)  # Priority requeue
