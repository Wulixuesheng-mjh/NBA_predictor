"""Team schemas."""

from datetime import datetime

from pydantic import BaseModel


class TeamBase(BaseModel):
    """Base team fields."""

    abbreviation: str
    city: str
    name: str
    full_name: str
    conference: str | None = None
    division: str | None = None


class TeamResponse(TeamBase):
    """Team as returned by the API."""

    id: int
    nba_team_id: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class TeamListResponse(BaseModel):
    """List of teams with total count."""

    teams: list[TeamResponse]
    total: int


class TeamDetailResponse(TeamResponse):
    """Team detail with recent performance summary."""

    recent_record: str | None = None  # e.g. "7-3"
    avg_pts: float | None = None
    avg_pts_allowed: float | None = None
    net_rating: float | None = None


class PlayerBrief(BaseModel):
    """Minimal player info for roster listing."""

    id: int
    nba_player_id: int
    full_name: str | None = None
    position: str | None = None
    jersey_number: str | None = None

    model_config = {"from_attributes": True}


class TeamRosterResponse(BaseModel):
    """Team roster."""

    team_id: int
    team_name: str
    players: list[PlayerBrief]
    total: int
