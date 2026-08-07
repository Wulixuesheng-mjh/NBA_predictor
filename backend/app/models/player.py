"""Player ORM model."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nba_player_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    first_name: Mapped[str | None] = mapped_column(String(50))
    last_name: Mapped[str | None] = mapped_column(String(50))
    full_name: Mapped[str | None] = mapped_column(String(100))
    team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"))
    jersey_number: Mapped[str | None] = mapped_column(String(5))
    position: Mapped[str | None] = mapped_column(String(5))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    team: Mapped["Team | None"] = relationship("Team", back_populates="players")
    game_stats: Mapped[list["GamePlayerStats"]] = relationship("GamePlayerStats", back_populates="player")
    injuries: Mapped[list["PlayerInjury"]] = relationship("PlayerInjury", back_populates="player")
