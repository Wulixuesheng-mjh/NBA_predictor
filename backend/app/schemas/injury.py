"""Injury schemas with confirmation workflow support."""

from datetime import datetime

from pydantic import BaseModel, Field


class InjuryBase(BaseModel):
    """Base injury fields."""

    player_id: int
    injury_status: str = Field(..., description="Out / Doubtful / Questionable / Probable / Available")
    injury_type: str | None = None
    description: str | None = None
    expected_return: datetime | None = None


class InjuryCreate(InjuryBase):
    """Manual injury creation request — auto-confirmed with source='manual'."""

    pass


class InjuryUpdate(BaseModel):
    """Update an existing injury record.

    All fields optional — only provided fields are updated.
    Updating sets confirmation='modified' (unless it was 'manual').
    """

    injury_status: str | None = None
    injury_type: str | None = None
    description: str | None = None
    expected_return: datetime | None = None
    is_active: bool | None = None


class InjuryResponse(InjuryBase):
    """Injury as returned by the API."""

    id: int
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
