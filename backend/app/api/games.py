"""Games API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.game_service import GameService
from app.schemas.game import (
    GameResponse,
    GameListResponse,
    GameDetailResponse,
    TeamGameStatsBrief,
    PlayerGameStatsBrief,
    UpcomingGameResponse,
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.get(
    "/",
    response_model=GameListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_games(
    date: str | None = Query(None, description="Filter by date (YYYY-MM-DD)"),
    status: str | None = Query(None, description="Filter by status (Scheduled/Final/InProgress)"),
    team_id: int | None = Query(None, description="Filter by team ID"),
    season: str | None = Query(None, description="Filter by season (e.g. 2025-26)"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List games with optional filters: date, status, team, season."""
    games, total = await GameService.list_games(
        db, date=date, status=status, team_id=team_id, season=season,
        offset=offset, limit=limit,
    )
    return GameListResponse(
        games=[GameResponse(**g) for g in games],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/upcoming",
    response_model=list[UpcomingGameResponse],
    responses={500: {"model": ErrorResponse}},
)
async def upcoming_games(
    days: int = Query(7, ge=1, le=30, description="Days to look ahead"),
    db: AsyncSession = Depends(get_db),
):
    """Get upcoming scheduled games for the next N days (default 7)."""
    games = await GameService.get_upcoming_games(db, days=days)
    return [UpcomingGameResponse(**g) for g in games]


@router.get(
    "/{game_id}",
    response_model=GameDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_game(game_id: int, db: AsyncSession = Depends(get_db)):
    """Get game detail with team and player box scores."""
    detail = await GameService.get_game_detail(db, game_id)
    if detail is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    game = detail["game"]
    return GameDetailResponse(
        id=game.id,
        nba_game_id=game.nba_game_id,
        season=game.season,
        game_date=game.game_date,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
        status=game.status,
        home_score=game.home_score,
        away_score=game.away_score,
        quarter=game.quarter,
        time_remaining=game.time_remaining,
        created_at=game.created_at,
        updated_at=game.updated_at,
        home_team_name=detail["home_team_name"],
        away_team_name=detail["away_team_name"],
        home_team_abbr=detail["home_team_abbr"],
        away_team_abbr=detail["away_team_abbr"],
        team_stats=[TeamGameStatsBrief(**ts) for ts in detail["team_stats"]],
        player_stats=[PlayerGameStatsBrief(**ps) for ps in detail["player_stats"]],
    )
