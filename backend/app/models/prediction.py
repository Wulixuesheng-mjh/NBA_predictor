"""Prediction ORM model."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("games.id"), nullable=True)
    home_team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    away_team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    model_name: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Win/loss (from home team perspective)
    win_prob_home: Mapped[float] = mapped_column(Float, nullable=False)
    win_prob_away: Mapped[float] = mapped_column(Float, nullable=False)
    # Spread (home_score - away_score)
    predicted_spread: Mapped[float | None] = mapped_column(Float)
    spread_confidence: Mapped[float | None] = mapped_column(Float)
    # Total points
    predicted_total: Mapped[float | None] = mapped_column(Float)
    total_confidence: Mapped[float | None] = mapped_column(Float)
    # Calibration
    calibrated: Mapped[bool] = mapped_column(default=False)
    calibration_factors: Mapped[dict | None] = mapped_column(JSON)
    # Debug / reproducibility
    features_json: Mapped[dict | None] = mapped_column(JSON)
    lookback_games: Mapped[int] = mapped_column(Integer, default=10)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="predictions")
    result: Mapped["PredictionResult | None"] = relationship("PredictionResult", back_populates="prediction", uselist=False)
