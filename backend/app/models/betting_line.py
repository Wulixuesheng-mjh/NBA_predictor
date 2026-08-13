"""Betting line (盘口) ORM model — point spread & over/under per game.

One row per game. ``spread_line`` is the expected *home margin*
(home_score - away_score); positive = home favored. ``total_line`` is the
expected combined total (home_score + away_score).

Sign convention matches the prediction pipeline: the raw regression target
``predicted_spread`` is also ``home - away``, so the cover label is
``1 if (home_score - away_score) > spread_line else 0`` and the over label is
``1 if (home_score + away_score) > total_line else 0``.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class BettingLine(Base):
    __tablename__ = "betting_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id"), unique=True, nullable=False
    )
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    # Expected home margin (home_score - away_score); positive = home favored.
    spread_line: Mapped[float] = mapped_column(Float, nullable=False)
    # Expected combined total (home_score + away_score).
    total_line: Mapped[float] = mapped_column(Float, nullable=False)
    # Raw per-side spread lines (vegas Average_Line_Spread) for reference.
    home_spread: Mapped[float | None] = mapped_column(Float)
    away_spread: Mapped[float | None] = mapped_column(Float)
    # Where the line came from: "erichqiu" (historical) or "manual" (frontend entry).
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="manual")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
