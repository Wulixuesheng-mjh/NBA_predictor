"""Historical backtesting — evaluate model performance on past data.

Walks through completed games chronologically, generating predictions
using only data available before each game (no look-ahead bias).

Reports:
  - Overall accuracy (win/spread/total)
  - Accuracy by season
  - Accuracy by confidence level
  - Accuracy trend over time
  - Spread coverage (±N points)
  - Total coverage (±N points)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ml.src.config import ml_settings
from ml.src.database import async_session_factory
from ml.src.features.feature_builder import FeatureBuilder, FeatureVector
from ml.src.models.xgboost_model import XGBoostTrainer, XGBoostPrediction
from ml.src.models.mlp_model import MLPTrainer, MLPPrediction

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Aggregated backtest evaluation metrics."""

    n_games: int = 0

    # Win/Loss
    win_accuracy: float = 0.0
    win_accuracy_by_season: dict[str, float] = field(default_factory=dict)

    # Spread — target: MAE < 5.0, gambling lines typically within 5 pts
    spread_mae: float = 0.0
    spread_rmse: float = 0.0
    spread_within_3: float = 0.0  # % within ±3 points (stretch)
    spread_within_5: float = 0.0  # % within ±5 points (target)

    # Total — target: MAE < 5.0
    total_mae: float = 0.0
    total_rmse: float = 0.0
    total_within_5: float = 0.0   # % within ±5 points (target)
    total_within_8: float = 0.0   # % within ±8 points

    # Per-game records
    per_game: list[dict] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Single-game backtest comparison."""

    nba_game_id: str
    game_date: str
    home_team_id: int
    away_team_id: int
    actual_spread: float
    actual_total: float
    actual_winner: str  # "home" or "away"
    xgboost: dict | None = None
    mlp: dict | None = None


class Backtester:
    """Runs historical backtests for model evaluation."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory
        self.feature_builder = FeatureBuilder(session_factory)

    async def run(
        self,
        xgb_trainer: XGBoostTrainer,
        mlp_trainer: MLPTrainer | None = None,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 10,
        max_games: int | None = None,
    ) -> BacktestMetrics:
        """Run backtest over completed games.

        Uses a walk-forward approach: for each completed game, builds features
        using only data available before that game, then compares prediction
        to actual result.

        IMPORTANT: This evaluates the feature+prediction pipeline but does NOT
        retrain models during the walk-forward. For a true walk-forward backtest
        with periodic retraining, use the feedback loop module.

        Args:
            xgb_trainer: Trained XGBoost model.
            mlp_trainer: Trained MLP model (optional).
            seasons: Seasons to evaluate.
            lookback_games: Lookback window for features.
            min_games_before: Minimum prior games required per team.
            max_games: Limit number of games (for quick testing).

        Returns:
            BacktestMetrics with all evaluation results.
        """
        from app.models.game import Game
        from sqlalchemy import select, and_

        async with self.session_factory() as sess:
            # Query completed games
            query = select(Game).where(Game.status == "Final")
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())

            if max_games:
                query = query.limit(max_games)

            result = await sess.execute(query)
            games = result.scalars().all()

            logger.info("Backtesting %d games...", len(games))

            metrics = BacktestMetrics()
            results: list[BacktestResult] = []

            for i, game in enumerate(games):
                if game.home_score is None or game.away_score is None:
                    continue

                # Build features as of game date (no leakage)
                try:
                    fv = await self.feature_builder.build_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                    )
                except Exception as e:
                    logger.debug("Skipping %s: %s", game.nba_game_id, e)
                    continue

                if fv.home_games_available < min_games_before or fv.away_games_available < min_games_before:
                    continue

                # Actual values
                actual_spread = game.home_score - game.away_score
                actual_total = game.home_score + game.away_score
                actual_winner = "home" if actual_spread > 0 else "away"

                X = fv.features.reshape(1, -1)

                # XGBoost prediction
                xgb_result = None
                if xgb_trainer and xgb_trainer.is_trained:
                    try:
                        pred = xgb_trainer.predict(X)
                        xgb_result = {
                            "win_prob_home": pred.win_prob_home,
                            "predicted_spread": pred.predicted_spread,
                            "predicted_total": pred.predicted_total,
                            "win_correct": (pred.win_prob_home > 0.5) == (actual_winner == "home"),
                            "spread_error": pred.predicted_spread - actual_spread,
                            "total_error": pred.predicted_total - actual_total,
                        }
                    except Exception as e:
                        logger.debug("XGB pred failed for %s: %s", game.nba_game_id, e)

                # MLP prediction
                mlp_result = None
                if mlp_trainer and mlp_trainer.model:
                    try:
                        pred = mlp_trainer.predict(X)
                        mlp_result = {
                            "win_prob_home": pred.win_prob_home,
                            "predicted_spread": pred.predicted_spread,
                            "predicted_total": pred.predicted_total,
                            "win_correct": (pred.win_prob_home > 0.5) == (actual_winner == "home"),
                            "spread_error": pred.predicted_spread - actual_spread,
                            "total_error": pred.predicted_total - actual_total,
                        }
                    except Exception as e:
                        logger.debug("MLP pred failed for %s: %s", game.nba_game_id, e)

                results.append(
                    BacktestResult(
                        nba_game_id=game.nba_game_id,
                        game_date=game.game_date.isoformat(),
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        actual_spread=float(actual_spread),
                        actual_total=float(actual_total),
                        actual_winner=actual_winner,
                        xgboost=xgb_result,
                        mlp=mlp_result,
                    )
                )

                if (i + 1) % 100 == 0:
                    logger.info("  Backtest progress: %d/%d games", i + 1, len(games))

            # Compute aggregate metrics from XGBoost results
            xgb_results = [r for r in results if r.xgboost is not None]
            if xgb_results:
                n = len(xgb_results)
                metrics.n_games = n

                # Win accuracy
                win_correct = sum(1 for r in xgb_results if r.xgboost["win_correct"])
                metrics.win_accuracy = win_correct / n

                # Spread metrics
                spread_errors = [abs(r.xgboost["spread_error"]) for r in xgb_results]
                metrics.spread_mae = float(np.mean(spread_errors))
                metrics.spread_rmse = float(np.sqrt(np.mean(np.square(spread_errors))))
                metrics.spread_within_3 = sum(1 for e in spread_errors if e <= 3) / n
                metrics.spread_within_5 = sum(1 for e in spread_errors if e <= 5) / n

                # Total metrics
                total_errors = [abs(r.xgboost["total_error"]) for r in xgb_results]
                metrics.total_mae = float(np.mean(total_errors))
                metrics.total_rmse = float(np.sqrt(np.mean(np.square(total_errors))))
                metrics.total_within_5 = sum(1 for e in total_errors if e <= 5) / n
                metrics.total_within_8 = sum(1 for e in total_errors if e <= 8) / n

                # Per-season accuracy
                season_results: dict[str, list] = {}
                for r in xgb_results:
                    season = r.game_date[:4]  # rough season extraction
                    season_results.setdefault(season, []).append(r.xgboost["win_correct"])
                metrics.win_accuracy_by_season = {
                    s: float(np.mean(v)) for s, v in season_results.items()
                }

                # Per-game records (last 50)
                metrics.per_game = [
                    {
                        "game_id": r.nba_game_id,
                        "date": r.game_date,
                        "actual_spread": r.actual_spread,
                        "predicted_spread": r.xgboost["predicted_spread"],
                        "actual_total": r.actual_total,
                        "predicted_total": r.xgboost["predicted_total"],
                        "win_correct": r.xgboost["win_correct"],
                        "spread_error": r.xgboost["spread_error"],
                        "total_error": r.xgboost["total_error"],
                    }
                    for r in xgb_results[-50:]
                ]

            logger.info(
                "Backtest complete: %d games. "
                "Win=%.3f, Spread MAE=%.2f (±5:%.1f%%), Total MAE=%.2f (±5:%.1f%%)",
                metrics.n_games,
                metrics.win_accuracy,
                metrics.spread_mae,
                metrics.spread_within_5 * 100,
                metrics.total_mae,
                metrics.total_within_5 * 100,
            )

            return metrics
