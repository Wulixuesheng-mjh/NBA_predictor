"""Historical player injury ORM model (daily snapshots from NBA official reports).

Unlike ``PlayerInjury`` (which tracks the live, user-reviewed injury state for
prediction), this table stores immutable per-day snapshots imported from the
NBA official injury report PDFs (via the ``nbainjuries`` package). It is used
only for *training* — to give the model historical injury variance so the
injury features learn a real signal.

Key difference from ``PlayerInjury``:
  - ``team_id`` is denormalized from the report itself, NOT ``Player.team_id``,
    because players change teams across seasons.
  - One row per (game_date, player, team) snapshot.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class HistoricalPlayerInjury(Base):
    __tablename__ = "historical_player_injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    # Team as recorded on the injury report (players move teams across seasons).
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    injury_status: Mapped[str] = mapped_column(String(20), nullable=False)  # Out/Doubtful/Questionable/Probable
    injury_type: Mapped[str | None] = mapped_column(String(100))  # from Reason column
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="nbainjuries")
    game_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("games.id"))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    player: Mapped["Player"] = relationship("Player", back_populates="historical_injuries")
    team: Mapped["Team"] = relationship("Team")

    __table_args__ = (
        UniqueConstraint("player_id", "team_id", "game_date", name="uq_hist_inj_player_team_date"),
        Index("idx_hist_inj_game_date", "game_date"),
        Index("idx_hist_inj_team_game", "team_id", "game_date"),
        Index("idx_hist_inj_player_game", "player_id", "game_date"),
    )
