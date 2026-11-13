"""Scheduled job definitions.

Four-tier collection:
  1. run_kaggle_collection()      — historical + full box scores (every 6h)
  2. run_balldontlie_collection() — same-day game scores (every 30min)
  3. run_roster_sync()            — current-season rosters (daily 4:17 AM)
  4. run_nightly_retraining()     — settle results only (daily 2:37 AM)
"""

import logging

from app.collector.kaggle_collector import get_kaggle_collector
from app.collector.balldontlie_collector import get_balldontlie_collector
from app.collector.nba_collector import get_collector

logger = logging.getLogger(__name__)
_prospective_milestones = {}


async def run_shadow_observation_cycle() -> None:
    """Capture locked diagnostic forecasts; settlement reads Final scores only."""
    from app.database import async_session_factory
    from app.services.shadow_observation_service import capture_due
    try:
        async with async_session_factory() as session:
            captured=await capture_due(session)
            await session.commit()
            if captured:
                logger.info('Diagnostic shadow observations captured: %s (not acceptance)',captured)
    except Exception:
        logger.exception('Diagnostic shadow capture failed; no retrospective records created')


async def run_prospective_capture() -> None:
    """Inactive until a final protocol is explicitly locked; never promotes."""
    from app.database import async_session_factory
    from app.models.prospective import ProspectiveRun
    from app.services.prospective_service import capture_due, progress_report
    from sqlalchemy import select
    try:
        async with async_session_factory() as session:
            count = await capture_due(session)
            await session.commit()
            if count:
                logger.info('Prospective capture: %s paired real-pregame forecasts', count)
            runs = (await session.execute(select(ProspectiveRun))).scalars().all()
            for run in runs:
                report = await progress_report(session, run)
                milestone = report['completed']//50
                if milestone > _prospective_milestones.get(run.id, 0):
                    logger.info('Prospective milestone: %s', report)
                    _prospective_milestones[run.id] = milestone
    except Exception:
        logger.exception('Prospective capture failed; no retrospective pregame rows created')


# ── Kaggle Collection ──────────────────────────────────────────

async def run_kaggle_collection() -> None:
    """Scheduled job: Kaggle-based historical data sync.

    The Kaggle dataset is updated nightly (US Eastern time).
    Since kagglehub caches the dataset locally, subsequent calls
    return instantly unless a new version exists.
    """
    logger.info("Starting Kaggle data collection...")
    try:
        collector = get_kaggle_collector()
        result = await collector.collect()

        if result.status == "success":
            if result.records_collected > 0:
                logger.info("Kaggle: imported %d records", result.records_collected)
            else:
                logger.debug("Kaggle: no new records (dataset unchanged)")
        else:
            logger.error("Kaggle collection failed: %s", result.error_message)
    except Exception as e:
        logger.error("Kaggle collection failed: %s", e, exc_info=True)


# ── BallDontLie Collection ─────────────────────────────────────

async def run_balldontlie_collection() -> None:
    """Scheduled job: BallDontLie same-day game score sync.

    Fetches today's games and updates newly Final games with scores.
    Free tier: 5 requests/minute — each run uses 1-2 requests
    (1 for games list, 0-1 for stats). At 30min intervals this
    is well within limits.

    Disabled silently if BALLDONTLIE_API_KEY is not configured.
    """
    logger.debug("Starting BallDontLie real-time collection...")
    try:
        collector = get_balldontlie_collector()
        result = await collector.collect()

        if result.status == "success":
            if result.records_collected > 0:
                logger.info(
                    "BallDontLie: %d games updated to Final",
                    result.records_collected,
                )
        elif result.error_message and "API_KEY not set" in result.error_message:
            # Not configured — log once at info, then skip
            logger.debug("BallDontLie skipped: API key not configured")
        else:
            logger.warning("BallDontLie failed: %s", result.error_message)
    except Exception as e:
        logger.error("BallDontLie collection failed: %s", e, exc_info=True)


# ── Roster Sync ────────────────────────────────────────────────

async def run_roster_sync() -> None:
    """Scheduled job: refresh every team's current-season roster.

    Keeps the UI roster accurate through trades, signings and the
    February deadline — the Kaggle path only ships a players snapshot.
    """
    logger.info("Starting roster sync...")
    try:
        collector = get_collector()
        count = await collector.sync_players()
        logger.info("Roster sync: %d player records updated", count)
    except Exception as e:
        logger.error("Roster sync failed: %s", e, exc_info=True)


# ── Nightly Retraining ─────────────────────────────────────────

async def run_nightly_retraining() -> None:
    """Settle completed games without fitting models in the API event loop.

    The historical job name is retained. Explicit candidate training belongs
    in the trainer container, with a locked policy and writable output paths.
    """
    logger.info("Starting nightly prediction evaluation...")
    try:
        import sys
        from pathlib import Path

        ml_path = str(Path(__file__).parent.parent.parent / "ml")
        if ml_path not in sys.path:
            sys.path.insert(0, ml_path)

        from ml.src.feedback import FeedbackLoop

        loop = FeedbackLoop()
        result = await loop.run_cycle(force_retrain=False,evaluate_only=True)
        logger.info(
            "Nightly evaluation: %d evaluated, retrained=%s v%s, calib=%s, errors=%d",
            result.predictions_evaluated,
            result.retrained,
            result.retrain_version or "N/A",
            result.calibration_updated,
            len(result.errors),
        )
        if result.errors:
            for err in result.errors:
                logger.warning("Nightly retraining error: %s", err)
    except ImportError as e:
        logger.warning("ML module not available for nightly retraining: %s", e)
    except Exception as e:
        logger.error("Nightly retraining failed: %s", e, exc_info=True)
