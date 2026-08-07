"""Game schemas."""

from datetime import datetime

from pydantic import BaseModel


class GameBase(BaseModel):
    """Base game fields."""

    nba_game_id: str
    season: str
    game_date: datetime
    status: str = "Scheduled"


class GameResponse(GameBase):
    """Game as returned by the API."""

    id: int
    home_team_id: int
    away_team_id: int
    home_team_name: str | None = None
    away_team_name: str | None = None
    home_team_abbr: str | None = None
    away_team_abbr: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    quarter: int | None = None
    time_remaining: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class GameListResponse(BaseModel):
    """List of games with total count."""

    games: list[GameResponse]
    total: int
    offset: int = 0
    limit: int = 50


class TeamGameStatsBrief(BaseModel):
    """Summary team stats for a game."""

    team_id: int
    team_name: str | None = None
    is_home: bool
    pts: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    fg_pct: float | None = None
    fg3_pct: float | None = None
    ft_pct: float | None = None

    model_config = {"from_attributes": True}


class PlayerGameStatsBrief(BaseModel):
    """Player box score stats for a game."""

    player_id: int
    player_name: str | None = None
    team_id: int
    minutes: str | None = None
    pts: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    plus_minus: int | None = None

    model_config = {"from_attributes": True}


class GameDetailResponse(GameResponse):
    """Game detail with team and player box scores."""

    home_team_name: str | None = None
    away_team_name: str | None = None
    home_team_abbr: str | None = None
    away_team_abbr: str | None = None
    team_stats: list[TeamGameStatsBrief] = []
    player_stats: list[PlayerGameStatsBrief] = []


class UpcomingGameResponse(BaseModel):
    """Lightweight upcoming game info with team names."""

    id: int
    nba_game_id: str
    game_date: datetime
    home_team_id: int
    away_team_id: int
    home_team_name: str | None = None
    away_team_name: str | None = None
    home_team_abbr: str | None = None
    away_team_abbr: str | None = None
    status: str = "Scheduled"

    model_config = {"from_attributes": True}
