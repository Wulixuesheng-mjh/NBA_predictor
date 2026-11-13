"""Injury schemas with confirmation workflow support."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InjuryBase(BaseModel):
    """Base injury fields."""

    player_id: int
    injury_status: Literal['Out','Doubtful','Questionable','Probable','Available']
    injury_type: str | None = None
    description: str | None = None
    expected_return: datetime | None = None


class InjuryCreate(InjuryBase):
    """Manual injury creation request — auto-confirmed with source='manual'."""

    game_id: int | None = None
    team_id: int | None = None
    published_at: datetime | None = None


class InjuryUpdate(BaseModel):
    """Update an existing injury record.

    All fields optional — only provided fields are updated.
    Updating sets confirmation='modified' (unless it was 'manual').
    """

    injury_status: Literal['Out','Doubtful','Questionable','Probable','Available'] | None = None
    injury_type: str | None = None
    description: str | None = None
    expected_return: datetime | None = None
    is_active: bool | None = None


class InjuryResponse(InjuryBase):
    """Injury as returned by the API."""

    id: int
    game_id: int | None = None
    team_id: int | None = None
    source: str
    confirmation: str
    reported_date: datetime
    is_active: bool
    player_name: str | None = None
    team_abbreviation: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class InjuryListResponse(BaseModel):
    """List of injuries with total count."""

    injuries: list[InjuryResponse]
    total: int
    pending_count: int = 0


class ConfirmAllResponse(BaseModel):
    """Response for one-click confirm-all."""

    status: str = "ok"
    confirmed_count: int


class InjuryReportCreate(BaseModel):
    game_id: int
    team_id: int
    source: str = "manual"
    published_at: datetime | None = None
    observed_at: datetime | None = None
    is_complete: bool = False


class InjuryReportResponse(InjuryReportCreate):
    id: int
    observed_at: datetime

    model_config = {"from_attributes": True}
