"""Player injury ORM model with confirmation workflow."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PlayerInjury(Base):
    """Tracks player injuries from nba_api (pending confirmation) or manual entry.

    Confirmation workflow:
      - nba_api auto-fetched records start as `pending`
      - User reviews and either confirms (`confirmed`), edits then confirms (`modified`),
        or deletes the record
      - Manual entries are always `confirmed` by default
      - Prediction uses: confirmed = modified > pending
    """

    __tablename__ = "player_injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(10), nullable=False, default="nba_api")
    confirmation: Mapped[str] = mapped_column(String(15), nullable=False, default="pending")
    injury_status: Mapped[str] = mapped_column(String(20), nullable=False)
    injury_type: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    reported_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_return: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_json: Mapped[str | None] = mapped_column(Text)
    game_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("games.id"))
    team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="injuries")

    __table_args__ = (
        Index("idx_injuries_player", "player_id", "is_active"),
        Index("idx_injuries_active", "is_active"),
    )
