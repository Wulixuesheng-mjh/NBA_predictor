"""Daily feedback loop — close the prediction-verification-retraining cycle.

Triggered when the collector detects ≥3 new Final games (see
_collector.py:_trigger_model_retraining()).

Workflow:
  1. Load latest predictions from database
  2. Load new game results
  3. Compare predictions vs actuals → store in prediction_results table
  4. Trigger XGBoost full retraining (incorporating new data)
  5. Optionally fine-tune MLP
  6. Update calibration coefficients
  7. Log the feedback cycle

This ensures the model continuously improves as the season progresses.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ml.src.config import ml_settings
from ml.src.database import async_session_factory
from ml.src.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class FeedbackResult:
    """Result of a feedback cycle."""

    new_games_found: int
    predictions_evaluated: int
    retrained: bool
    retrain_version: str | None
    calibration_updated: bool
    errors: list[str]
    completed_at: str


class FeedbackLoop:
    """Manages the daily feedback loop: evaluate → retrain → calibrate."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def run_cycle(
        self,
        force_retrain: bool = False,
        seasons: list[str] | None = None,
    ) -> FeedbackResult:
        """Execute one feedback cycle.

        Args:
            force_retrain: Retrain even if fewer than min_new_games_for_retrain.
            seasons: Seasons to include in retraining.

        Returns:
            FeedbackResult summarizing the cycle.
        """
        errors: list[str] = []
        new_games = 0
        evaluated = 0
        retrained = False
        retrain_version = None
        calibration_updated = False

        logger.info("=== Starting feedback cycle ===")

        # 1. Evaluate pending predictions against new results
        try:
            new_games = await self._evaluate_predictions()
            evaluated = new_games
        except Exception as e:
            msg = f"Failed to evaluate predictions: {e}"
            logger.error(msg)
            errors.append(msg)

        # 2. Determine if retraining is needed
        should_retrain = force_retrain or new_games >= ml_settings.min_new_games_for_retrain

        if should_retrain and ml_settings.auto_retrain:
            logger.info(
                "Triggering model retraining (%d new games)", new_games
            )
            try:
                trainer = Trainer(self.session_factory)
                result = await trainer.train_full_pipeline(
                    seasons=seasons,
                    lookback_games=ml_settings.default_lookback_games,
                    calibrate=True,
                )
                retrained = True
                retrain_version = result.version
                calibration_updated = True
                logger.info("Retraining complete: v%s", result.version)
            except Exception as e:
                msg = f"Retraining failed: {e}"
                logger.error(msg, exc_info=True)
                errors.append(msg)
        elif new_games > 0:
            # Update calibration only (no full retrain needed yet)
            logger.info(
                "Updating calibration with %d new games (below retrain threshold of %d)",
                new_games, ml_settings.min_new_games_for_retrain,
            )
            try:
                await self._update_calibration_only()
                calibration_updated = True
            except Exception as e:
                msg = f"Calibration update failed: {e}"
                logger.error(msg)
                errors.append(msg)

        result = FeedbackResult(
            new_games_found=new_games,
            predictions_evaluated=evaluated,
            retrained=retrained,
            retrain_version=retrain_version,
            calibration_updated=calibration_updated,
            errors=errors,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info("=== Feedback cycle complete: %s ===", result)
        return result

    async def _evaluate_predictions(self) -> int:
        """Find predictions with now-completed games and store results.

        Returns the number of newly evaluated predictions.
        """
        from app.models.prediction import Prediction
        from app.models.prediction_results import PredictionResult
        from app.models.game import Game
        from sqlalchemy import select, and_

        count = 0
        async with self.session_factory() as sess:
            # Find predictions for games that are now Final but don't have results yet
            query = (
                select(Prediction, Game)
                .join(Game, Prediction.game_id == Game.id)
                .outerjoin(
                    PredictionResult,
                    PredictionResult.prediction_id == Prediction.id,
                )
                .where(
                    and_(
                        Game.status == "Final",
                        Game.home_score.isnot(None),
                        Game.away_score.isnot(None),
                        PredictionResult.id.is_(None),  # No result recorded yet
                    )
                )
            )
            result = await sess.execute(query)
            pairs = result.all()

            for pred, game in pairs:
                try:
                    actual_spread = game.home_score - game.away_score
                    actual_total = game.home_score + game.away_score
                    actual_winner = "home" if actual_spread > 0 else "away"

                    # Determine correctness
                    pred_winner = "home" if pred.win_prob_home > 0.5 else "away"
                    win_correct = pred_winner == actual_winner

                    # For spread: correct if predicted direction is right
                    # (not checking exact value since spread betting is directional)
                    spread_correct = None
                    if pred.predicted_spread is not None:
                        predicted_dir = "home" if pred.predicted_spread > 0 else "away"
                        actual_dir = "home" if actual_spread > 0 else "away"
                        spread_correct = predicted_dir == actual_dir

                    # For total: correct if both above or below league average
                    total_correct = None
                    if pred.predicted_total is not None:
                        league_avg = 225.0
                        pred_over = pred.predicted_total > league_avg
                        actual_over = actual_total > league_avg
                        total_correct = pred_over == actual_over

                    # Store result
                    pr = PredictionResult(
                        prediction_id=pred.id,
                        actual_spread=float(actual_spread),
                        actual_total=int(actual_total),
                        actual_winner=actual_winner,
                        spread_correct=spread_correct,
                        total_correct=total_correct,
                        win_correct=win_correct,
                        evaluated_at=datetime.now(timezone.utc),
                    )
                    sess.add(pr)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to evaluate prediction %d: %s", pred.id, e)
                    continue

            if count > 0:
                await sess.commit()
                logger.info("Evaluated %d predictions against actual results", count)

        return count

    async def _update_calibration_only(self) -> None:
        """Update calibration coefficients without full retraining.

        Uses recent prediction results (last 30 days) to recompute
        calibration factors.
        """
        from app.models.prediction import Prediction
        from app.models.prediction_results import PredictionResult
        from app.models.game import Game
        from sqlalchemy import select, and_, desc

        from ml.src.models.calibration import Calibrator

        async with self.session_factory() as sess:
            # Get recent evaluated predictions
            cutoff = datetime.now(timezone.utc).replace(
                day=datetime.now(timezone.utc).day - ml_settings.calibration_window_days
            )

            query = (
                select(Prediction, PredictionResult)
                .join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
                .join(Game, Prediction.game_id == Game.id)
                .where(
                    and_(
                        Game.status == "Final",
                        PredictionResult.evaluated_at >= cutoff,
                    )
                )
                .order_by(desc(Game.game_date))
            )
            result = await sess.execute(query)
            pairs = result.all()

            if len(pairs) < ml_settings.min_games_for_calibration:
                logger.info(
                    "Not enough recent results for calibration (%d < %d)",
                    len(pairs), ml_settings.min_games_for_calibration,
                )
                return

            # Build arrays
            import numpy as np

            y_win = np.array([
                1.0 if pr.actual_winner == "home" else 0.0
                for _, pr in pairs
            ])
            y_spread = np.array([
                pr.actual_spread or 0.0 for _, pr in pairs
            ])
            y_total = np.array([
                float(pr.actual_total or 0) for _, pr in pairs
            ])

            # Create lightweight prediction objects for the calibrator
            class _Pred:
                pass

            preds = []
            for pred, _ in pairs:
                p = _Pred()
                p.win_prob_home = pred.win_prob_home
                p.predicted_spread = pred.predicted_spread or 0.0
                p.predicted_total = pred.predicted_total or 0.0
                preds.append(p)

            calibrator = Calibrator()
            calibrator.compute_coefficients(preds, y_win, y_spread, y_total)
            calibrator.save()

            logger.info("Calibration updated from %d recent games", len(pairs))


async def run_feedback_cycle(
    force_retrain: bool = False,
) -> FeedbackResult:
    """Convenience function to run one feedback cycle."""
    loop = FeedbackLoop()
    return await loop.run_cycle(force_retrain=force_retrain)
