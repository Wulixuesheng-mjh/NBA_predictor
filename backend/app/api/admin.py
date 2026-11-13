"""Admin API endpoints — collection triggers, status, system stats, manual entry."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.models.team import Team
from app.models.player import Player
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats
from app.models.player_injuries import PlayerInjury
from app.models.prediction import Prediction
from app.models.prediction_results import PredictionResult
from app.models.data_collection_log import DataCollectionLog
from app.schemas.admin import (
    CollectionTriggerResponse,
    CollectionLogItem,
    CollectionStatusResponse,
    SystemStatsResponse,
    ModelTrainResponse,
)
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

# Track running collection task
_running_collection: bool = False


# ── Admin Authentication ──────────────────────────────────────

async def require_admin(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> None:
    """Verify admin API key if configured. Skips auth when key is empty (dev mode)."""
    if not settings.admin_api_key:
        return  # Auth disabled — development mode
    if not x_api_key or x_api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing admin API key. Set X-API-Key header.",
        )


# Apply auth to all admin routes
router = APIRouter(dependencies=[Depends(require_admin)])


@router.post(
    "/collect",
    response_model=CollectionTriggerResponse,
    responses={409: {"model": ErrorResponse}},
)
async def trigger_collection(
    source: str = "kaggle",
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger data collection.

    Source options:
      - 'kaggle' (default): Use nightly-updated Kaggle dataset
      - 'nba_api': Use nba_api (may be blocked from some regions)
      - 'all': Try both
    """
    global _running_collection

    if _running_collection:
        return CollectionTriggerResponse(
            status="already_running",
            source=source,
            message="A collection task is already in progress. Please wait.",
        )

    _running_collection = True
    try:
        result = None

        if source in ("kaggle", "all"):
            from app.collector.kaggle_collector import get_kaggle_collector
            collector = get_kaggle_collector()
            result = await collector.collect()

        if source in ("nba_api", "all") and (result is None or result.status != "success"):
            try:
                from app.collector.nba_collector import get_collector
                collector = get_collector()
                nba_result = await collector.collect()
                if result is None or nba_result.records_collected > 0:
                    result = nba_result
            except Exception:
                pass  # nba_api unavailable — expected

        _running_collection = False

        if result is None:
            return CollectionTriggerResponse(
                status="failed",
                source=source,
                message="No collector available for the requested source.",
            )

        return CollectionTriggerResponse(
            status="triggered" if result.status == "success" else "failed",
            source=source,
            task_id=str(result.completed_at.timestamp()) if result.completed_at else None,
            message=f"Collected {result.records_collected} records" if result.status == "success" else result.error_message,
        )
    except Exception as e:
        _running_collection = False
        logger.error("Manual collection failed: %s", e, exc_info=True)
        return CollectionTriggerResponse(
            status="failed",
            source=source,
            message=str(e),
        )


class RosterSyncResponse(BaseModel):
    status: str
    players_synced: int


@router.post(
    "/sync-rosters",
    response_model=RosterSyncResponse,
)
async def sync_rosters() -> RosterSyncResponse:
    """Re-sync every team's current-season roster from nba_api.

    Updates team assignments (trades / signings), jersey numbers and
    positions, and detaches players that left. Idempotent — safe to rerun
    whenever rosters change.
    """
    try:
        from app.collector.nba_collector import get_collector

        collector = get_collector()
        count = await collector.sync_players()
    except Exception as e:
        logger.error("Roster sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Roster sync failed: {e}")

    return RosterSyncResponse(status="ok", players_synced=count)


@router.get(
    "/collect/status",
    response_model=CollectionStatusResponse,
)
async def collection_status(db: AsyncSession = Depends(get_db)):
    """Get the status of recent collection runs."""
    query = (
        select(DataCollectionLog)
        .order_by(DataCollectionLog.started_at.desc())
        .limit(10)
    )
    result = await db.execute(query)
    logs = result.scalars().all()

    return CollectionStatusResponse(
        last_runs=[CollectionLogItem.model_validate(log) for log in logs],
        scheduler_enabled=settings.scheduler_enabled,
        collection_interval_minutes=settings.collection_interval_minutes,
    )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
)
async def system_stats(db: AsyncSession = Depends(get_db)):
    """Get system statistics: record counts, last collection, model info."""
    # Count all major tables
    counts = {}
    for model, key in [
        (Team, "teams"),
        (Player, "players"),
        (Game, "games"),
        (PlayerInjury, "active_injuries"),
        (Prediction, "predictions"),
    ]:
        query = select(func.count(model.id))
        if key == "active_injuries":
            query = query.where(PlayerInjury.is_active == True)
        result = await db.execute(query)
        counts[key] = result.scalar() or 0

    # Completed games
    completed_query = select(func.count(Game.id)).where(Game.status == "Final")
    result = await db.execute(completed_query)
    counts["completed_games"] = result.scalar() or 0

    # Pending injuries
    pending_query = select(func.count(PlayerInjury.id)).where(
        PlayerInjury.confirmation == "pending", PlayerInjury.is_active == True
    )
    result = await db.execute(pending_query)
    counts["pending_injuries"] = result.scalar() or 0

    # Evaluated predictions
    eval_query = select(func.count(PredictionResult.id))
    result = await db.execute(eval_query)
    counts["evaluated_predictions"] = result.scalar() or 0

    # Last collection time
    last_query = (
        select(DataCollectionLog.completed_at)
        .where(DataCollectionLog.status == "success")
        .order_by(DataCollectionLog.completed_at.desc())
        .limit(1)
    )
    result = await db.execute(last_query)
    last_collection = result.scalar_one_or_none()

    # Active model version (check disk)
    from app.services.model_release_service import read_active_release
    release = read_active_release()
    model_version = release['version'] if release else None
    model_trained = release.get('created_at') if release else None

    return SystemStatsResponse(
        teams_count=counts["teams"],
        players_count=counts["players"],
        games_count=counts["games"],
        completed_games_count=counts["completed_games"],
        active_injuries_count=counts["active_injuries"],
        pending_injuries_count=counts["pending_injuries"],
        predictions_count=counts["predictions"],
        evaluated_predictions_count=counts["evaluated_predictions"],
        last_collection=last_collection,
        active_model_version=model_version,
        model_last_trained=model_trained,
    )


@router.post(
    "/train-model",
    response_model=ModelTrainResponse,
)
async def trigger_model_training(
    seasons: str | None = None,
    force: bool = False,
):
    """Manually trigger model training.

    Optionally specify comma-separated seasons (e.g. '2024-25,2025-26').
    If not specified, uses all available data.
    """
    try:
        from ml.src.feedback import run_feedback_cycle

        season_list = seasons.split(",") if seasons else None
        result = await run_feedback_cycle(force_retrain=force,seasons=season_list)

        if result.errors:
            return ModelTrainResponse(status="failed",message="; ".join(result.errors))

        return ModelTrainResponse(
            status="success" if result.retrained else "triggered",
            version=result.retrain_version,
            message=(
                f"Retrained v{result.retrain_version}, {result.predictions_evaluated} predictions evaluated"
                if result.retrained
                else f"Only {result.new_games_found} new games — below retrain threshold. "
                       f"Calibration updated: {result.calibration_updated}"
            ),
        )
    except ImportError:
        return ModelTrainResponse(
            status="failed",
            message="ML module not available. Ensure ml/ package is installed.",
        )
    except Exception as e:
        logger.error("Model training trigger failed: %s", e, exc_info=True)
        return ModelTrainResponse(
            status="failed",
            message=str(e),
        )


# ── Manual Game Result Entry ────────────────────────────────────


class ManualGameResult(BaseModel):
    """Schema for manually entering a game result.

    Used when automated collection (nba_api/Kaggle) is unavailable,
    e.g., for same-day game results that haven't been published yet.
    """

    nba_game_id: str = Field(..., description="NBA game ID (e.g., '0022500001')")
    home_score: int = Field(..., ge=0, description="Home team final score")
    away_score: int = Field(..., ge=0, description="Away team final score")
    season: Optional[str] = Field(None, description="Season (e.g., '2025-26'). Auto-detected if omitted.")
    game_date: Optional[str] = Field(None, description="Game date ISO format. Auto-set to today if omitted.")
    home_team_id: Optional[int] = Field(None, description="NBA team ID. Looked up from DB if omitted.")
    away_team_id: Optional[int] = Field(None, description="NBA team ID. Looked up from DB if omitted.")


class ManualGameResultResponse(BaseModel):
    status: str
    game_id: int | None = None
    message: str


@router.post(
    "/manual-game-result",
    response_model=ManualGameResultResponse,
    responses={404: {"model": ErrorResponse}},
)
async def manual_game_result(
    payload: ManualGameResult,
    db: AsyncSession = Depends(get_db),
):
    """Manually enter or update a game result.

    Use this when:
      - A game just finished and isn't in Kaggle yet (same-day results)
      - NBA API is blocked from your region
      - You need to correct a game score

    The endpoint creates the game if it doesn't exist, or updates
    the score if it does. If home_team_id/away_team_id are omitted,
    it attempts to look them up from an existing schedule entry.
    """
    # 1. Check if game already exists
    result = await db.execute(
        select(Game).where(Game.nba_game_id == payload.nba_game_id)
    )
    game = result.scalar_one_or_none()

    if game:
        # Update existing game
        game.home_score = payload.home_score
        game.away_score = payload.away_score
        game.status = "Final"
        game.updated_at = datetime.now(timezone.utc)
        from app.services.data_quality_service import bump_data_revision, refresh_game_quality
        await refresh_game_quality(db, game.id)
        await bump_data_revision(db, f"manual-game-result:{game.id}")
        await db.commit()
        return ManualGameResultResponse(
            status="updated",
            game_id=game.id,
            message=f"Game {payload.nba_game_id} updated: {payload.home_score}-{payload.away_score}",
        )

    # 2. Game doesn't exist — need team IDs and date to create
    home_nba_id = payload.home_team_id
    away_nba_id = payload.away_team_id

    if not home_nba_id or not away_nba_id:
        raise HTTPException(
            status_code=400,
            detail="home_team_id and away_team_id are required for new games",
        )

    # Look up internal team IDs
    team_result = await db.execute(select(Team.nba_team_id, Team.id))
    team_map = {str(row[0]): row[1] for row in team_result.all()}

    home_db_id = team_map.get(str(home_nba_id))
    away_db_id = team_map.get(str(away_nba_id))

    if not home_db_id or not away_db_id:
        raise HTTPException(
            status_code=404,
            detail=f"Team not found: home={home_nba_id}, away={away_nba_id}",
        )

    # Parse game date
    if payload.game_date:
        game_date = datetime.fromisoformat(payload.game_date[:19])
    else:
        game_date = datetime.now(timezone.utc)

    # Determine season
    if payload.season:
        season = payload.season
    else:
        if game_date.month >= 10:
            season = f"{game_date.year}-{str(game_date.year + 1)[-2:]}"
        else:
            season = f"{game_date.year - 1}-{str(game_date.year)[-2:]}"

    # Create the game
    new_game = Game(
        nba_game_id=payload.nba_game_id,
        season=season,
        game_date=game_date,
        source_game_date=game_date.date(),
        scheduled_start_utc=None,
        time_precision="date",
        game_type="regular",
        home_team_id=home_db_id,
        away_team_id=away_db_id,
        status="Final",
        home_score=payload.home_score,
        away_score=payload.away_score,
    )
    db.add(new_game)
    await db.flush()
    from app.services.data_quality_service import bump_data_revision, refresh_game_quality
    await refresh_game_quality(db, new_game.id)
    await bump_data_revision(db, f"manual-game-result-create:{new_game.id}")
    await db.commit()
    await db.refresh(new_game)

    return ManualGameResultResponse(
        status="created",
        game_id=new_game.id,
        message=f"Game {payload.nba_game_id} created: {payload.home_score}-{payload.away_score} ({season})",
    )


# ── Bulk Injury Import ────────────────────────────────────────


class BulkInjuryItem(BaseModel):
    """Single injury item for bulk import."""

    player_id: int = Field(..., description="Internal player ID")
    injury_status: str = Field(default="Out", description="Out / Doubtful / Questionable / Probable")
    injury_type: str = Field(..., description="e.g. 'Ankle', 'Knee', 'Groin'")
    description: str | None = Field(None, description="Optional detail")


class BulkInjuryRequest(BaseModel):
    """Bulk injury import request."""

    injuries: list[BulkInjuryItem] = Field(..., min_length=1, max_length=100)
    replace_existing: bool = Field(
        default=False,
        description="If True, deactivate all current active injuries before import"
    )


class BulkInjuryResponse(BaseModel):
    status: str
    created: int
    skipped: int
    deactivated: int = 0
    message: str


@router.post(
    "/bulk-injuries",
    response_model=BulkInjuryResponse,
    responses={400: {"model": ErrorResponse}},
)
async def bulk_import_injuries(
    payload: BulkInjuryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Bulk import injury records for multiple players at once.

    Useful when:
      - Kaggle has no injury data and BallDontLie All-Star tier isn't enabled
      - You want to manually sync injuries from an external source (ESPN, etc.)
      - You need to batch-update the entire league's injury report

    Set `replace_existing=True` to deactivate all current active injuries
    before importing the new batch (full refresh mode).
    """
    from app.models.player_injuries import PlayerInjury

    deactivated = 0

    # Optional: deactivate all current injuries before import
    if payload.replace_existing:
        result = await db.execute(
            select(PlayerInjury).where(PlayerInjury.is_active == True)
        )
        active = result.scalars().all()
        for injury in active:
            injury.is_active = False
            injury.updated_at = datetime.now(timezone.utc)
        deactivated = len(active)

    created = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    for item in payload.injuries:
        # Check if player exists
        player_result = await db.execute(
            select(Player.id).where(Player.id == item.player_id)
        )
        if not player_result.scalar_one_or_none():
            skipped += 1
            continue

        # Check for duplicate active injury of same type
        dup = await db.execute(
            select(PlayerInjury).where(
                PlayerInjury.player_id == item.player_id,
                PlayerInjury.is_active == True,
                PlayerInjury.injury_type == item.injury_type,
            )
        )
        if dup.scalar_one_or_none():
            skipped += 1
            continue

        new_injury = PlayerInjury(
            player_id=item.player_id,
            source="manual",
            confirmation="confirmed",
            injury_status=item.injury_status,
            injury_type=item.injury_type,
            description=item.description or "",
            reported_date=now,
            is_active=True,
        )
        db.add(new_injury)
        created += 1

    await db.commit()

    return BulkInjuryResponse(
        status="success",
        created=created,
        skipped=skipped,
        deactivated=deactivated,
        message=f"Created {created} injuries, skipped {skipped}"
        + (f", deactivated {deactivated} existing" if deactivated else ""),
    )
