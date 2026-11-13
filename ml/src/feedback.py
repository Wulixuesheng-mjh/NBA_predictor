"""Settle predictions automatically; stage training only on explicit requests.

Collectors and the online scheduler use evaluation-only cycles. Candidate
training runs separately and never promotes a release through this module.
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
        *,
        evaluate_only: bool = True,
    ) -> FeedbackResult:
        """Execute one feedback cycle.

        Args:
            force_retrain: Retrain even if fewer than min_new_games_for_retrain.
            seasons: Seasons to include in retraining.
            evaluate_only: Settle results without training or changing calibration.
                False is reserved for explicit training requests.

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
            evaluated = await self._evaluate_predictions()
            new_games = await self._count_ready_games_since_release()
        except Exception as e:
            msg = f"Failed to evaluate predictions: {e}"
            logger.error(msg)
            errors.append(msg)

        # 2. Determine if retraining is needed
        should_retrain = force_retrain or new_games >= ml_settings.min_new_games_for_retrain

        if evaluate_only:
            logger.info("Evaluation-only cycle: %d complete new games; training deferred to trainer", new_games)
        elif should_retrain and ml_settings.auto_retrain:
            logger.info(
                "Triggering model retraining (%d new games)", new_games
            )
            try:
                trainer = Trainer(self.session_factory)
                result = await trainer.train_full_pipeline(
                    seasons=seasons,
                    lookback_games=ml_settings.default_lookback_games,
                    calibrate=ml_settings.calibration_enabled,
                )
                retrained = True
                retrain_version = result.version
                calibration_updated = bool(result.calibration)
                logger.info("Retraining complete: v%s", result.version)
            except Exception as e:
                msg = f"Retraining failed: {e}"
                logger.error(msg, exc_info=True)
                errors.append(msg)
        elif new_games > 0 and getattr(ml_settings, "calibration_enabled", False):
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
                        Prediction.prediction_mode == "pregame",
                        Prediction.as_of.isnot(None),
                        Game.scheduled_start_utc.isnot(None),
                        Prediction.as_of < Game.scheduled_start_utc,
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

                    # Market correctness uses the exact archived line.
                    spread_correct = None
                    if pred.market_home_handicap is not None:
                        market_result = actual_spread + pred.market_home_handicap
                        if market_result != 0:
                            model_home_cover = (pred.predicted_spread or 0) + pred.market_home_handicap > 0
                            spread_correct = model_home_cover == (market_result > 0)

                    # For total: correct if both above or below league average
                    total_correct = None
                    if pred.predicted_total is not None and pred.market_total is not None:
                        if actual_total != pred.market_total:
                            pred_over = pred.predicted_total > pred.market_total
                            actual_over = actual_total > pred.market_total
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
            await self._refresh_residuals(sess)

        return count

    @staticmethod
    async def _refresh_residuals(session) -> None:
        """Rebuild a versioned store; retries cannot multiply one game's weight."""
        import json
        import os
        import tempfile
        from pathlib import Path
        from sqlalchemy import select
        from app.models.game import Game
        from app.models.prediction import Prediction
        from app.models.prediction_results import PredictionResult
        from ml.src.uncertainty import build_residual_payload

        path = Path(ml_settings.models_dir) / "residuals.json"
        pairs = (await session.execute(select(Prediction,Game,PredictionResult)
            .join(Game,Prediction.game_id==Game.id)
            .join(PredictionResult,PredictionResult.prediction_id==Prediction.id)
            .where(Prediction.prediction_mode=='pregame',Prediction.as_of.isnot(None),
                   Game.status=='Final',Game.score_complete.is_(True)))).all()
        payload = build_residual_payload(pairs)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                    prefix='residuals.', suffix='.tmp', delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(payload,stream,indent=2,allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,path)
        except OSError as exc:
            logger.warning("Could not update residual store: %s", exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _count_ready_games_since_release(self) -> int:
        """Count new games whose score and both box-score layers are usable."""
        import json
        import os
        from datetime import date

        from app.models.game import Game
        from sqlalchemy import func, select

        cutoff = None
        manifest_path = os.path.join(ml_settings.models_dir, "active.json")
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                raw = json.load(handle).get("training_data_until")
            if raw:
                cutoff = date.fromisoformat(str(raw)[:10])
        except (OSError, ValueError, TypeError):
            pass

        clauses = [
            Game.status == "Final",
            Game.score_complete.is_(True),
            Game.team_stats_complete.is_(True),
            Game.player_stats_complete.is_(True),
        ]
        if cutoff is not None:
            # The manifest cutoff is an NBA game day, not UTC midnight.
            # Unknown source days cannot certify a game as new training data.
            clauses.append(Game.source_game_date > cutoff)
        async with self.session_factory() as sess:
            value = await sess.scalar(select(func.count(Game.id)).where(*clauses))
        return int(value or 0)

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
    seasons: list[str] | None = None,
) -> FeedbackResult:
    """Explicit manual training entry; automatic jobs call evaluation-only cycles."""
    loop = FeedbackLoop()
    return await loop.run_cycle(force_retrain=force_retrain,seasons=seasons,evaluate_only=False)
