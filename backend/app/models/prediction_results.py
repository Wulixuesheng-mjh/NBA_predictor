"""Prediction results ORM model for backtesting / evaluation."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    actual_spread: Mapped[float | None] = mapped_column(Float)
    actual_total: Mapped[int | None] = mapped_column(Integer)
    actual_winner: Mapped[str | None] = mapped_column(String(4))
    spread_correct: Mapped[bool | None] = mapped_column(Boolean)
    total_correct: Mapped[bool | None] = mapped_column(Boolean)
    win_correct: Mapped[bool | None] = mapped_column(Boolean)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="result")
