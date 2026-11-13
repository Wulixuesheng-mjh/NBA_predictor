"""Diagnostic forward observations, deliberately separate from release acceptance."""
from datetime import datetime
from sqlalchemy import Boolean,DateTime,ForeignKey,Integer,JSON,String,UniqueConstraint,func
from sqlalchemy.orm import Mapped,mapped_column
from app.database import Base


class ShadowObservationRun(Base):
    __tablename__='shadow_observation_runs'
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    name: Mapped[str]=mapped_column(String(100),unique=True,nullable=False)
    protocol: Mapped[dict]=mapped_column(JSON,nullable=False)
    protocol_hash: Mapped[str]=mapped_column(String(64),nullable=False)
    enabled: Mapped[bool]=mapped_column(Boolean,nullable=False,default=True)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())


class ShadowObservation(Base):
    __tablename__='shadow_observations'
    __table_args__=(UniqueConstraint('run_id','game_id',name='uq_shadow_observation_run_game'),)
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    run_id: Mapped[int]=mapped_column(ForeignKey('shadow_observation_runs.id'),nullable=False)
    game_id: Mapped[int]=mapped_column(ForeignKey('games.id'),nullable=False)
    as_of: Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    scheduled_start_utc: Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    payload: Mapped[dict]=mapped_column(JSON,nullable=False)
    payload_hash: Mapped[str]=mapped_column(String(64),nullable=False)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
