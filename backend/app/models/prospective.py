"""Immutable, paired real-pregame observations under a locked protocol."""

from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class ProspectiveRun(Base):
    __tablename__ = 'prospective_runs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    protocol: Mapped[dict] = mapped_column(JSON, nullable=False)
    protocol_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    target_games: Mapped[int] = mapped_column(Integer, nullable=False, default=200)


class ProspectiveForecast(Base):
    __tablename__ = 'prospective_forecasts'
    __table_args__ = (UniqueConstraint('run_id', 'game_id', name='uq_prospective_run_game'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('prospective_runs.id'), nullable=False)
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id'), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scheduled_start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False)
    feature_vector: Mapped[list] = mapped_column(JSON, nullable=False)
    feature_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    baseline_output: Mapped[dict] = mapped_column(JSON, nullable=False)
    candidate_output: Mapped[dict] = mapped_column(JSON, nullable=False)
    data_quality: Mapped[dict] = mapped_column(JSON, nullable=False)
    market_snapshot: Mapped[dict | None] = mapped_column(JSON)
