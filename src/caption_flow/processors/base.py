"""Base processor abstractions for data source handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Iterator, Tuple
from datetime import datetime
from pathlib import Path


@dataclass
class WorkUnit:
    """Generic unit of work that can be processed."""

    unit_id: str
    source_id: str  # e.g., shard name, batch ID
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def get_size_hint(self) -> int:
        """Get estimated size/complexity of this work unit."""
        return self.metadata.get("size_hint", 1)


@dataclass
class WorkAssignment:
    """Assignment of work units to a worker."""

    assignment_id: str
    worker_id: str
    units: List[WorkUnit]
    assigned_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for network transmission."""
        return {
            "assignment_id": self.assignment_id,
            "worker_id": self.worker_id,
            "units": [
                {
                    "unit_id": u.unit_id,
                    "source_id": u.source_id,
                    "data": u.data,
                    "metadata": u.metadata,
                    "priority": u.priority,
                }
                for u in self.units
            ],
            "assigned_at": self.assigned_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkAssignment":
        """Create from dict received over network."""
        units = [
            WorkUnit(
                unit_id=u["unit_id"],
                source_id=u["source_id"],
                data=u["data"],
                metadata=u.get("metadata", {}),
                priority=u.get("priority", 0),
            )
            for u in data["units"]
        ]
        return cls(
            assignment_id=data["assignment_id"],
            worker_id=data["worker_id"],
            units=units,
            assigned_at=datetime.fromisoformat(data["assigned_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkResult:
    """Result from processing a work unit."""

    unit_id: str
    source_id: str
    outputs: Dict[str, List[Any]]  # field_name -> list of outputs
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.error is None and bool(self.outputs)


@dataclass
class ProcessorConfig:
    """Configuration for a processor."""

    processor_type: str
    config: Dict[str, Any]


class OrchestratorProcessor(ABC):
    """Base processor for orchestrator side - manages work distribution."""

    @abstractmethod
    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize the processor with configuration."""
        pass

    @abstractmethod
    def get_work_units(self, count: int, worker_id: str) -> List[WorkUnit]:
        """Get available work units for a worker."""
        pass

    @abstractmethod
    def mark_completed(self, unit_id: str, worker_id: str) -> None:
        """Mark a work unit as completed."""
        pass

    @abstractmethod
    def mark_failed(self, unit_id: str, worker_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        pass

    @abstractmethod
    def release_assignments(self, worker_id: str) -> None:
        """Release all assignments for a disconnected worker."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        pass

    def handle_result(self, result: WorkResult) -> Dict[str, Any]:
        """Handle a work result - can be overridden for custom processing."""
        return {
            "unit_id": result.unit_id,
            "source_id": result.source_id,
            "outputs": result.outputs,
            "metadata": result.metadata,
        }


class WorkerProcessor(ABC):
    """Base processor for worker side - processes work units."""

    @abstractmethod
    def initialize(self, config: ProcessorConfig) -> None:
        """Initialize the processor with configuration."""
        pass

    @abstractmethod
    def process_unit(self, unit: WorkUnit, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Process a single work unit, yielding items to be captioned.

        Args:
            unit: The work unit to process
            context: Runtime context (e.g., models, sampling params)

        Yields:
            Dict containing:
                - image: PIL Image
                - metadata: Dict of metadata
                - item_key: Unique identifier for this item
        """
        pass

    def prepare_result(
        self, unit: WorkUnit, outputs: List[Dict[str, Any]], processing_time_ms: float
    ) -> WorkResult:
        """Prepare a work result from processed outputs."""
        # Aggregate outputs by field
        aggregated = {}
        for output in outputs:
            for field, values in output.items():
                if field not in aggregated:
                    aggregated[field] = []
                aggregated[field].extend(values if isinstance(values, list) else [values])

        return WorkResult(
            unit_id=unit.unit_id,
            source_id=unit.source_id,
            outputs=aggregated,
            metadata={"item_count": len(outputs), **unit.metadata},
            processing_time_ms=processing_time_ms,
        )

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset/source being processed."""
        pass
