"""Game ORM model — the central fact table."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nba_game_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    game_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="Scheduled")
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    quarter: Mapped[int | None] = mapped_column(Integer)
    time_remaining: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    team_stats: Mapped[list["TeamGameStats"]] = relationship("TeamGameStats", back_populates="game")
    player_stats: Mapped[list["GamePlayerStats"]] = relationship("GamePlayerStats", back_populates="game")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="game")
