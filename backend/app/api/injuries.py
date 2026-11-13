"""Injury API endpoints with confirmation workflow.

Supports:
  - List injuries with filters (team_id, confirmation, is_active)
  - Create manual injury (auto-confirmed)
  - Update/modify injury (marks as 'modified' if source != 'manual')
  - Delete (soft: set is_active=False)
  - One-click confirm-all pending injuries
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.injury_service import InjuryService
from app.schemas.injury import (
    InjuryCreate,
    InjuryUpdate,
    InjuryResponse,
    InjuryListResponse,
    ConfirmAllResponse,
    InjuryReportCreate,
    InjuryReportResponse,
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.get(
    "/",
    response_model=InjuryListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_injuries(
    team_id: int | None = Query(None, description="Filter by team ID"),
    game_id: int | None = Query(None, description="Filter by target game"),
    confirmation: str | None = Query(
        None, description="Filter by confirmation status: pending/confirmed/modified"
    ),
    is_active: bool = Query(True, description="Only show active injuries"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List injuries with optional filters.

    Useful filters:
      - `confirmation=pending` — get the review queue for AdminPage
      - `team_id=N` — get injuries for a specific team
    """
    injuries, total, pending_count = await InjuryService.list_injuries(
        db,
        team_id=team_id,
        confirmation=confirmation,
        is_active=is_active,
        offset=offset,
        limit=limit,
        game_id=game_id,
    )
    return InjuryListResponse(
        injuries=[InjuryResponse(**inj) for inj in injuries],
        total=total,
        pending_count=pending_count,
    )


@router.post(
    "/",
    response_model=InjuryResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}},
)
async def create_injury(
    body: InjuryCreate,
    db: AsyncSession = Depends(get_db),
):
    """Manually create an injury record.

    Manual entries are automatically set to:
      - source='manual'
      - confirmation='confirmed'
    """
    try:
        injury = await InjuryService.create_injury(
            db,
            player_id=body.player_id,
            injury_status=body.injury_status,
            injury_type=body.injury_type,
            description=body.description,
            expected_return=body.expected_return,
            game_id=body.game_id,
            team_id=body.team_id,
            published_at=body.published_at,
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.commit()

    # Fetch with player/team info for response
    return await _enrich_injury_response(db, injury)


@router.post("/reports", response_model=InjuryReportResponse, status_code=201)
async def create_injury_report(body: InjuryReportCreate, db: AsyncSession = Depends(get_db)):
    """Archive report coverage, including an explicit no-injury complete report."""
    from datetime import datetime, timezone
    from app.models.game import Game
    from app.models.injury_report import InjuryReport
    from app.services.data_quality_service import bump_data_revision

    game = await db.get(Game, body.game_id)
    if game is None or body.team_id not in {game.home_team_id, game.away_team_id}:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="team_id is not part of game_id")
    # The API records actual receipt time. Client supplied clocks cannot turn
    # a late manual entry into a historical pregame snapshot.
    report = await InjuryService.archive_bound_snapshot(
        db, body.game_id, body.team_id, complete=body.is_complete,
    )
    await bump_data_revision(db, f"injury-report:{report.id}")
    await db.commit()
    await db.refresh(report)
    return report


@router.put(
    "/{injury_id}",
    response_model=InjuryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_injury(
    injury_id: int,
    body: InjuryUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an injury record.

    - Sets confirmation='modified' if source is 'nba_api'
    - Manual entries (source='manual') keep their confirmation status
    - Only provided fields are updated
    """
    injury = await InjuryService.update_injury(
        db,
        injury_id=injury_id,
        injury_status=body.injury_status,
        injury_type=body.injury_type,
        description=body.description,
        expected_return=body.expected_return,
        is_active=body.is_active,
    )
    if injury is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Injury {injury_id} not found")

    await db.commit()
    return await _enrich_injury_response(db, injury)


@router.delete(
    "/{injury_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_injury(injury_id: int, db: AsyncSession = Depends(get_db)):
    """Soft-delete an injury record (sets is_active=False)."""
    deleted = await InjuryService.delete_injury(db, injury_id)
    if not deleted:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Injury {injury_id} not found")

    await db.commit()
    return {"status": "deleted", "id": injury_id}


@router.post(
    "/confirm-all",
    response_model=ConfirmAllResponse,
)
async def confirm_all_injuries(db: AsyncSession = Depends(get_db)):
    """One-click confirm all pending injuries.

    Sets all `confirmation='pending'` records to `'confirmed'`.
    Returns the count of confirmed records.
    """
    count = await InjuryService.confirm_all(db)
    await db.commit()
    return ConfirmAllResponse(status="ok", confirmed_count=count)


# ── Helper ─────────────────────────────────────────────────────

async def _enrich_injury_response(
    db: AsyncSession, injury
) -> InjuryResponse:
    """Build an InjuryResponse with player name and team abbreviation."""
    from app.models.player import Player
    from app.models.team import Team
    from sqlalchemy import select

    player_name = None
    team_abbr = None

    player_query = select(Player).where(Player.id == injury.player_id)
    result = await db.execute(player_query)
    player = result.scalar_one_or_none()
    if player:
        player_name = player.full_name
        if player.team_id:
            team_query = select(Team.abbreviation).where(Team.id == player.team_id)
            result = await db.execute(team_query)
            team_abbr = result.scalar_one_or_none()

    return InjuryResponse(
        id=injury.id,
        player_id=injury.player_id,
        game_id=injury.game_id,
        team_id=injury.team_id,
        player_name=player_name,
        team_abbreviation=team_abbr,
        source=injury.source,
        confirmation=injury.confirmation,
        injury_status=injury.injury_status,
        injury_type=injury.injury_type,
        description=injury.description,
        reported_date=injury.reported_date,
        expected_return=injury.expected_return,
        is_active=injury.is_active,
        created_at=injury.created_at,
        updated_at=injury.updated_at,
    )
