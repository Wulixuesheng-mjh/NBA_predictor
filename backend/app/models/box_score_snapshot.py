"""Immutable evidence for validated official box-score repairs."""
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class BoxScoreSnapshot(Base):
    __tablename__='box_score_snapshots'
    __table_args__=(UniqueConstraint('game_id','sha256',name='uq_box_score_source_payload'),)
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    game_id: Mapped[int]=mapped_column(ForeignKey('games.id'),nullable=False,index=True)
    source: Mapped[str]=mapped_column(String(40),nullable=False)
    source_url: Mapped[str]=mapped_column(Text,nullable=False)
    source_file: Mapped[str]=mapped_column(Text,nullable=False)
    sha256: Mapped[str]=mapped_column(String(64),nullable=False)
    observed_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    published_at: Mapped[datetime|None]=mapped_column(DateTime(timezone=True))
    payload: Mapped[dict]=mapped_column(JSON,nullable=False)
    validation: Mapped[dict]=mapped_column(JSON,nullable=False)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
