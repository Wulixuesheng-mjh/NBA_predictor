"""Team game stats ORM model."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TeamGameStats(Base):
    __tablename__ = "team_game_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pts: Mapped[int] = mapped_column(Integer, default=0)
    reb: Mapped[int] = mapped_column(Integer, default=0)
    ast: Mapped[int] = mapped_column(Integer, default=0)
    stl: Mapped[int] = mapped_column(Integer, default=0)
    blk: Mapped[int] = mapped_column(Integer, default=0)
    tov: Mapped[int] = mapped_column(Integer, default=0)
    pf: Mapped[int] = mapped_column(Integer, default=0)
    fgm: Mapped[int] = mapped_column(Integer, default=0)
    fga: Mapped[int] = mapped_column(Integer, default=0)
    fg3m: Mapped[int] = mapped_column(Integer, default=0)
    fg3a: Mapped[int] = mapped_column(Integer, default=0)
    ftm: Mapped[int] = mapped_column(Integer, default=0)
    fta: Mapped[int] = mapped_column(Integer, default=0)
    off_rtg: Mapped[float | None] = mapped_column(Float)
    def_rtg: Mapped[float | None] = mapped_column(Float)
    pace: Mapped[float | None] = mapped_column(Float)
    ts_pct: Mapped[float | None] = mapped_column(Float)
    efg_pct: Mapped[float | None] = mapped_column(Float)
    reb_pct: Mapped[float | None] = mapped_column(Float)
    ast_pct: Mapped[float | None] = mapped_column(Float)
    tov_pct: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="team_stats")
    team: Mapped["Team"] = relationship("Team")
