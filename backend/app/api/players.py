"""Players API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.player_service import PlayerService
from app.schemas.player import (
    PlayerResponse,
    PlayerListResponse,
    PlayerDetailResponse,
    PlayerSeasonStats,
    PlayerInjuryBrief,
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.get(
    "/",
    response_model=PlayerListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_players(
    team_id: int | None = Query(None, description="Filter by team ID"),
    search: str | None = Query(None, description="Search by player name"),
    position: str | None = Query(None, description="Filter by position (G, F, C, G-F, F-C)"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List players with optional filters: team, name search, position."""
    players, total = await PlayerService.list_players(
        db, team_id=team_id, search=search, position=position,
        offset=offset, limit=limit,
    )
    return PlayerListResponse(
        players=[PlayerResponse.model_validate(p) for p in players],
        total=total,
    )


@router.get(
    "/{player_id}",
    response_model=PlayerDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_player(player_id: int, db: AsyncSession = Depends(get_db)):
    """Get player detail with season stats and active injuries."""
    detail = await PlayerService.get_player_detail(db, player_id)
    if detail is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    player = detail["player"]
    stats = detail["season_stats"]
    injuries = detail["active_injuries"]

    return PlayerDetailResponse(
        id=player.id,
        nba_player_id=player.nba_player_id,
        first_name=player.first_name,
        last_name=player.last_name,
        full_name=player.full_name,
        position=player.position,
        jersey_number=player.jersey_number,
        team_id=player.team_id,
        created_at=player.created_at,
        updated_at=player.updated_at,
        season_stats=PlayerSeasonStats(**stats) if stats else None,
        active_injuries=[PlayerInjuryBrief.model_validate(i) for i in injuries],
        team_abbreviation=detail["team_abbreviation"],
    )
