"""Abstract base class for all data collectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CollectionResult:
    """Result of a data collection run."""

    source: str
    task_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    status: str = "running"
    records_collected: int = 0
    error_message: str | None = None


class BaseCollector(ABC):
    """Abstract collector with logging and error handling."""

    source: str = "base"

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory

    @abstractmethod
    async def collect(self) -> CollectionResult:
        """Execute the collection task. Must be implemented by subclasses."""
        ...

    async def _log_collection(self, result: CollectionResult) -> None:
        """Persist collection result to data_collection_log table."""
        from app.models.data_collection_log import DataCollectionLog

        async with self.db_session_factory() as session:
            log_entry = DataCollectionLog(
                source=result.source,
                task_name=result.task_name,
                started_at=result.started_at,
                completed_at=result.completed_at,
                status=result.status,
                records_collected=result.records_collected,
                error_message=result.error_message,
            )
            session.add(log_entry)
            await session.commit()
