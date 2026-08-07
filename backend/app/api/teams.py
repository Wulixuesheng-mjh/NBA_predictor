"""Teams API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.team_service import TeamService
from app.schemas.team import (
    TeamResponse,
    TeamListResponse,
    TeamDetailResponse,
    TeamRosterResponse,
    PlayerBrief,
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.get(
    "/",
    response_model=TeamListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_teams(
    conference: str | None = Query(None, description="Filter by conference (East/West)"),
    division: str | None = Query(None, description="Filter by division"),
    db: AsyncSession = Depends(get_db),
):
    """List all NBA teams with optional conference/division filters."""
    teams, total = await TeamService.list_teams(
        db, conference=conference, division=division
    )
    return TeamListResponse(
        teams=[TeamResponse.model_validate(t) for t in teams],
        total=total,
    )


@router.get(
    "/{team_id}",
    response_model=TeamDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_team(team_id: int, db: AsyncSession = Depends(get_db)):
    """Get team detail with recent performance summary."""
    detail = await TeamService.get_team_detail(db, team_id)
    if detail is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    team = detail["team"]
    return TeamDetailResponse(
        id=team.id,
        nba_team_id=team.nba_team_id,
        abbreviation=team.abbreviation,
        city=team.city,
        name=team.name,
        full_name=team.full_name,
        conference=team.conference,
        division=team.division,
        created_at=team.created_at,
        updated_at=team.updated_at,
        recent_record=detail["recent_record"],
        avg_pts=detail["avg_pts"],
        avg_pts_allowed=detail["avg_pts_allowed"],
        net_rating=detail["net_rating"],
    )


@router.get(
    "/{team_id}/players",
    response_model=TeamRosterResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_team_players(team_id: int, db: AsyncSession = Depends(get_db)):
    """Get current roster for a team."""
    roster = await TeamService.get_team_roster(db, team_id)
    if roster is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    return TeamRosterResponse(
        team_id=roster["team_id"],
        team_name=roster["team_name"],
        players=[PlayerBrief.model_validate(p) for p in roster["players"]],
        total=roster["total"],
    )
