"""APScheduler setup and lifecycle management.

Three-tier collection strategy:
  1. Kaggle (every 6h)    — historical data + full box scores (nightly)
  2. BallDontLie (30min)  — same-day game scores (real-time)
  3. Nightly retraining   — evaluate + retrain at 2:37 AM

BallDontLie requires a free API key from https://app.balldontlie.io.
Set BALLDONTLIE_API_KEY in .env to enable real-time collection.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.scheduler.jobs import (
    run_kaggle_collection,
    run_balldontlie_collection,
    run_roster_sync,
    run_nightly_retraining,
    run_prospective_capture,
    run_shadow_observation_cycle,
)

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


def start_scheduler() -> None:
    """Start the APScheduler with all configured jobs."""
    scheduler = get_scheduler()

    import os
    if os.getenv('SHADOW_CAPTURE_ENABLED','true').lower()=='true':
        scheduler.add_job(run_shadow_observation_cycle,trigger=IntervalTrigger(minutes=5),
            id='shadow_observation_capture',name='Diagnostic shadow capture and settlement',
            replace_existing=True,max_instances=1,coalesce=True,misfire_grace_time=60)

    scheduler.add_job(
        run_prospective_capture, trigger=IntervalTrigger(minutes=5),
        id='prospective_capture', name='Locked paired pregame capture',
        replace_existing=True, max_instances=1,
    )

    # Kaggle: historical data + full box scores (nightly update)
    scheduler.add_job(
        run_kaggle_collection,
        trigger=IntervalTrigger(minutes=settings.collection_interval_minutes),
        id="kaggle_collection",
        name="Kaggle data collection",
        replace_existing=True,
    )

    # BallDontLie: same-day game scores (real-time)
    if settings.balldontlie_api_key:
        scheduler.add_job(
            run_balldontlie_collection,
            trigger=IntervalTrigger(
                minutes=settings.balldontlie_interval_minutes
            ),
            id="balldontlie_collection",
            name="BallDontLie real-time collection",
            replace_existing=True,
        )

    # Current-season rosters: trades/signings change team assignments
    scheduler.add_job(
        run_roster_sync,
        trigger=CronTrigger(hour=4, minute=17),
        id="roster_sync",
        name="Daily roster sync",
        replace_existing=True,
    )

    # Nightly model retraining: once per day at 2:37am local time
    scheduler.add_job(
        run_nightly_retraining,
        trigger=CronTrigger(hour=2, minute=37),
        id="nightly_retraining",
        name="Nightly prediction evaluation",
        replace_existing=True,
    )

    scheduler.start()


def stop_scheduler() -> None:
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
