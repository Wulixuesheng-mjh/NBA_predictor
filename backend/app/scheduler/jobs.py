"""Scheduled job definitions.

Three-tier collection:
  1. run_kaggle_collection()      — historical + full box scores (every 6h)
  2. run_balldontlie_collection() — same-day game scores (every 30min)
  3. run_nightly_retraining()     — evaluate + retrain (daily 2:37 AM)
"""

import logging

from app.collector.kaggle_collector import get_kaggle_collector
from app.collector.balldontlie_collector import get_balldontlie_collector

logger = logging.getLogger(__name__)


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


# ── Nightly Retraining ─────────────────────────────────────────

async def run_nightly_retraining() -> None:
    """Scheduled job: nightly model evaluation + optional retraining.

    Runs once per day (2:37 AM) after all games have finished.
    Evaluates pending predictions against actual results and triggers
    XGBoost full retraining + calibration update if enough new games
    are detected.
    """
    logger.info("Starting nightly retraining cycle...")
    try:
        import sys
        from pathlib import Path

        ml_path = str(Path(__file__).parent.parent.parent / "ml")
        if ml_path not in sys.path:
            sys.path.insert(0, ml_path)

        from ml.src.feedback import FeedbackLoop

        loop = FeedbackLoop()
        result = await loop.run_cycle(force_retrain=True)
        logger.info(
            "Nightly retraining: %d evaluated, retrained=%s v%s, calib=%s, errors=%d",
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
