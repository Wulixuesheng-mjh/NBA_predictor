"""Player schemas."""

from datetime import datetime

from pydantic import BaseModel


class PlayerBase(BaseModel):
    """Base player fields."""

    first_name: str | None = None
    last_name: str | None = None
    full_name: str | None = None
    position: str | None = None
    jersey_number: str | None = None


class PlayerResponse(PlayerBase):
    """Player as returned by the API."""

    id: int
    nba_player_id: int
    team_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PlayerListResponse(BaseModel):
    """List of players with total count."""

    players: list[PlayerResponse]
    total: int


class PlayerSeasonStats(BaseModel):
    """Player's season average statistics."""

    games_played: int = 0
    avg_pts: float = 0.0
    avg_reb: float = 0.0
    avg_ast: float = 0.0
    avg_stl: float = 0.0
    avg_blk: float = 0.0
    avg_tov: float = 0.0
    avg_minutes: float = 0.0
    fg_pct: float = 0.0
    fg3_pct: float = 0.0
    ft_pct: float = 0.0


class PlayerInjuryBrief(BaseModel):
    """Brief injury info for player detail."""

    id: int
    injury_status: str
    injury_type: str | None = None
    description: str | None = None
    confirmation: str
    source: str
    is_active: bool

    model_config = {"from_attributes": True}


class PlayerDetailResponse(PlayerResponse):
    """Player detail with season stats and injury status."""

    season_stats: PlayerSeasonStats | None = None
    active_injuries: list[PlayerInjuryBrief] = []
    team_abbreviation: str | None = None
