"""Admin schemas — collection triggers, status, system stats."""

from datetime import datetime

from pydantic import BaseModel


class CollectionTriggerResponse(BaseModel):
    """Response when manually triggering data collection."""

    status: str  # "triggered" | "already_running"
    source: str
    task_id: str | None = None
    message: str | None = None


class CollectionLogItem(BaseModel):
    """Single collection log entry."""

    id: int
    source: str
    task_name: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    records_collected: int = 0
    error_message: str | None = None

    model_config = {"from_attributes": True}


class CollectionStatusResponse(BaseModel):
    """Status of recent collection runs."""

    last_runs: list[CollectionLogItem]
    scheduler_enabled: bool
    collection_interval_minutes: int


class SystemStatsResponse(BaseModel):
    """Database and system statistics."""

    teams_count: int = 0
    players_count: int = 0
    games_count: int = 0
    completed_games_count: int = 0
    active_injuries_count: int = 0
    pending_injuries_count: int = 0
    predictions_count: int = 0
    evaluated_predictions_count: int = 0
    last_collection: datetime | None = None
    active_model_version: str | None = None
    model_last_trained: datetime | None = None


class ModelTrainResponse(BaseModel):
    """Response when triggering model training."""

    status: str  # "triggered" | "failed"
    version: str | None = None
    message: str | None = None


class AccuracyTrendPoint(BaseModel):
    """Single data point in the accuracy trend."""

    date: str
    win_accuracy: float | None = None
    spread_mae: float | None = None
    total_mae: float | None = None
    sample_count: int = 0
    window_days: int = 7


class AccuracyTrendSummary(BaseModel):
    """Overall accuracy summary across all time."""

    win_accuracy: float
    spread_mae: float | None = None
    total_mae: float | None = None
    total_evaluated: int = 0


class AccuracyTrendResponse(BaseModel):
    """Accuracy trend data for the dashboard."""

    trends: list[AccuracyTrendPoint]
    overall: AccuracyTrendSummary
    model_version: str | None = None
