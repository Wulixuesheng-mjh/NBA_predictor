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
from ml.src.features.opponent_adjusted_features import OpponentAdjustedFeatures
from ml.src.models.xgboost_model import XGBoostTrainer, XGBoostPrediction
from ml.src.models.mlp_model import MLPTrainer, MLPPrediction
from ml.src.trainer import Trainer

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


@dataclass
class CoverOverBacktestMetrics:
    """Aggregated cover/over (盘口) walk-forward backtest metrics."""

    n_games: int = 0
    cover_accuracy: float = 0.0
    over_accuracy: float = 0.0
    cover_logloss: float = 0.0
    over_logloss: float = 0.0
    cover_accuracy_by_season: dict[str, float] = field(default_factory=dict)
    over_accuracy_by_season: dict[str, float] = field(default_factory=dict)
    # Betting value: accuracy among confident predictions (|p-0.5| >= 0.10),
    # compared against the -110 break-even of ~52.4%.
    cover_accuracy_confident: float = 0.0
    over_accuracy_confident: float = 0.0
    n_confident: int = 0
    per_game: list[dict] = field(default_factory=list)


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
                        use_historical=True,
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
                mlp_available = (
                    mlp_trainer
                    and (mlp_trainer.model is not None or mlp_trainer._onnx_session is not None)
                )
                if mlp_available:
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

                if (i + 1) % max(1, min(100, len(games) // 10)) == 0:
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

    async def run_walk_forward(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 5,
        initial_train_games: int = 200,
        retrain_every: int = 50,
        max_games: int | None = None,
    ) -> BacktestMetrics:
        """Run a true walk-forward backtest with periodic model retraining.

        Unlike ``run()`` which evaluates with a single frozen model, this method
        simulates real-world deployment: the model is retrained after every
        *retrain_every* games using all available data up to that point, then
        evaluated on the next batch of games.

        Flow::

                    Train on [0:N]  →  Predict [N:N+K]
            Train on [0:N+K]  →  Predict [N+K:N+2K]
            Train on [0:N+2K] →  Predict [N+2K:N+3K]
            ...

        where N = *initial_train_games*, K = *retrain_every*.

        This eliminates the backtest-vs-production gap caused by frozen models
        and gives a realistic estimate of predictive power in a live setting.

        Args:
            seasons: Seasons to evaluate (None = all available).
            lookback_games: Lookback window for team features.
            min_games_before: Minimum prior games required per team.
            initial_train_games: Number of games for the initial training window.
            retrain_every: Retrain the model after every N predicted games.
            max_games: Limit total number of games (for quick testing).

        Returns:
            BacktestMetrics aggregated across all prediction windows.
        """
        from app.models.game import Game
        from sqlalchemy import select

        logger.info(
            "Walk-forward backtest: initial_window=%d, retrain_every=%d",
            initial_train_games, retrain_every,
        )

        async with self.session_factory() as sess:
            # ── 1. Load all completed games ──
            query = select(Game).where(Game.status == "Final")
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())
            if max_games:
                query = query.limit(max_games)

            result = await sess.execute(query)
            games = result.scalars().all()

            logger.info("Walk-forward: loaded %d completed games", len(games))

            # ── 2. Pre-build features for ALL games (date-based, no leakage) ──
            #    We cache features + labels so retraining windows don't re-compute.
            all_samples: list[dict] = []
            for i, game in enumerate(games):
                if game.home_score is None or game.away_score is None:
                    continue

                try:
                    fv = await self.feature_builder.build_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                        use_historical=True,
                    )
                except Exception:
                    continue

                if fv.home_games_available < min_games_before or fv.away_games_available < min_games_before:
                    continue

                actual_spread = float(game.home_score - game.away_score)
                all_samples.append({
                    "features": fv.features,
                    "y_win": 1.0 if actual_spread > 0 else 0.0,
                    "y_spread": actual_spread,
                    "y_total": float(game.home_score + game.away_score),
                    "nba_game_id": game.nba_game_id,
                    "game_date": game.game_date,
                    "season": game.season,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                })

                # Progress every 500 games or ~10%
                total = len(games)
                if (i + 1) % max(1, min(500, total // 10)) == 0:
                    logger.info(
                        "  Pre-building features: %d/%d (%.0f%%), %d valid",
                        i + 1, total, (i + 1) / total * 100,
                        len(all_samples),
                    )

            logger.info(
                "Walk-forward: %d valid samples after filtering (min_games=%d)",
                len(all_samples), min_games_before,
            )

            if len(all_samples) < initial_train_games + retrain_every:
                logger.warning(
                    "Not enough samples (%d) for walk-forward backtest "
                    "(need at least %d)", len(all_samples),
                    initial_train_games + retrain_every,
                )
                return BacktestMetrics(n_games=0)

            # ── 3. Walk-forward loop ──
            all_predictions: list[dict] = []
            train_end = initial_train_games
            cycle = 0

            while train_end < len(all_samples):
                predict_end = min(train_end + retrain_every, len(all_samples))

                train_samples = all_samples[:train_end]
                predict_samples = all_samples[train_end:predict_end]

                cycle += 1
                logger.info(
                    "  Cycle %d: training on %d games, predicting %d games "
                    "(idx %d → %d)",
                    cycle, len(train_samples), len(predict_samples),
                    train_end, predict_end - 1,
                )

                # ── 3a. Train XGBoost on this window ──
                X_train = np.vstack([s["features"] for s in train_samples])
                y_win_train = np.array([s["y_win"] for s in train_samples])
                y_spread_train = np.array([s["y_spread"] for s in train_samples])
                y_total_train = np.array([s["y_total"] for s in train_samples])

                # Chronological split within training window (80/20)
                split_idx = int(len(X_train) * 0.8)
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                yw_tr, yw_val = y_win_train[:split_idx], y_win_train[split_idx:]
                ys_tr, ys_val = y_spread_train[:split_idx], y_spread_train[split_idx:]
                yt_tr, yt_val = y_total_train[:split_idx], y_total_train[split_idx:]

                xgb = XGBoostTrainer()
                xgb.train(
                    X_train=X_tr, y_win_train=yw_tr,
                    y_spread_train=ys_tr, y_total_train=yt_tr,
                    X_val=X_val, y_win_val=yw_val,
                    y_spread_val=ys_val, y_total_val=yt_val,
                )

                # ── 3b. Predict upcoming games ──
                for sample in predict_samples:
                    X_pred = sample["features"].reshape(1, -1)
                    actual_spread = sample["y_spread"]
                    actual_total = sample["y_total"]
                    actual_winner = "home" if actual_spread > 0 else "away"

                    try:
                        pred = xgb.predict(X_pred)
                        all_predictions.append({
                            "nba_game_id": sample["nba_game_id"],
                            "game_date": sample["game_date"].isoformat(),
                            "season": sample["season"],
                            "actual_spread": actual_spread,
                            "actual_total": actual_total,
                            "actual_winner": actual_winner,
                            "win_prob_home": pred.win_prob_home,
                            "predicted_spread": pred.predicted_spread,
                            "predicted_total": pred.predicted_total,
                            "win_correct": (pred.win_prob_home > 0.5) == (actual_winner == "home"),
                            "spread_error": pred.predicted_spread - actual_spread,
                            "total_error": pred.predicted_total - actual_total,
                        })
                    except Exception:
                        continue

                train_end = predict_end

            # ── 4. Aggregate metrics ──
            return self._aggregate_metrics(all_predictions)

    @staticmethod
    def _aggregate_metrics(predictions: list[dict]) -> BacktestMetrics:
        """Compute BacktestMetrics from a list of per-game prediction dicts."""
        n = len(predictions)
        if n == 0:
            return BacktestMetrics(n_games=0)

        metrics = BacktestMetrics()
        metrics.n_games = n

        # Win accuracy
        win_correct = sum(1 for p in predictions if p["win_correct"])
        metrics.win_accuracy = win_correct / n

        # Spread metrics
        spread_errors = [abs(p["spread_error"]) for p in predictions]
        metrics.spread_mae = float(np.mean(spread_errors))
        metrics.spread_rmse = float(np.sqrt(np.mean(np.square(spread_errors))))
        metrics.spread_within_3 = sum(1 for e in spread_errors if e <= 3) / n
        metrics.spread_within_5 = sum(1 for e in spread_errors if e <= 5) / n

        # Total metrics
        total_errors = [abs(p["total_error"]) for p in predictions]
        metrics.total_mae = float(np.mean(total_errors))
        metrics.total_rmse = float(np.sqrt(np.mean(np.square(total_errors))))
        metrics.total_within_5 = sum(1 for e in total_errors if e <= 5) / n
        metrics.total_within_8 = sum(1 for e in total_errors if e <= 8) / n

        # Per-season accuracy
        season_results: dict[str, list] = {}
        for p in predictions:
            season = p["game_date"][:4]
            season_results.setdefault(season, []).append(p["win_correct"])
        metrics.win_accuracy_by_season = {
            s: float(np.mean(v)) for s, v in season_results.items()
        }

        # Per-game records (last 50)
        metrics.per_game = [
            {
                "game_id": p["nba_game_id"],
                "date": p["game_date"],
                "actual_spread": p["actual_spread"],
                "predicted_spread": p["predicted_spread"],
                "actual_total": p["actual_total"],
                "predicted_total": p["predicted_total"],
                "win_correct": p["win_correct"],
                "spread_error": p["spread_error"],
                "total_error": p["total_error"],
            }
            for p in predictions[-50:]
        ]

        logger.info(
            "Walk-forward complete: %d predictions across %d retrain cycles. "
            "Win=%.3f, Spread MAE=%.2f (±5:%.1f%%), Total MAE=%.2f (±5:%.1f%%)",
            metrics.n_games,
            len(set(p["game_date"] for p in predictions)),  # approximate cycle count
            metrics.win_accuracy,
            metrics.spread_mae,
            metrics.spread_within_5 * 100,
            metrics.total_mae,
            metrics.total_within_5 * 100,
        )

        return metrics

    async def run_cover_over_walk_forward(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 5,
        initial_train_games: int = 500,
        retrain_every: int = 100,
        max_games: int | None = None,
    ) -> CoverOverBacktestMetrics:
        """Walk-forward backtest for the cover/over (盘口) binary classifiers.

        Mirrors ``run_walk_forward`` but for the two betting-line classifiers:
        cover (让分胜负) and over (大小分).  Only games with a betting line
        (2012-2019) are used; features are the reduced cross-season-safe subset.
        """
        from app.models.game import Game
        from app.models.betting_line import BettingLine
        from sqlalchemy import select

        logger.info(
            "Cover/over walk-forward: initial=%d, retrain_every=%d",
            initial_train_games, retrain_every,
        )

        async with self.session_factory() as sess:
            # 1. Load all lined games (chronological)
            query = (
                select(Game, BettingLine)
                .join(BettingLine, BettingLine.game_id == Game.id)
                .where(Game.status == "Final")
            )
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())
            if max_games:
                query = query.limit(max_games)

            result = await sess.execute(query)
            pairs = result.all()
            logger.info("Cover/over walk-forward: %d lined games", len(pairs))

            opponent_cache = await OpponentAdjustedFeatures.precompute_opponent_cache(sess)

            # 2. Pre-build reduced features + labels (date-based, no leakage)
            all_samples: list[dict] = []
            for i, (game, line) in enumerate(pairs):
                if game.home_score is None or game.away_score is None:
                    continue
                if line.spread_line is None or line.total_line is None:
                    continue

                try:
                    fv = await self.feature_builder.build_cover_over_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                        opponent_cache=opponent_cache,
                    )
                except Exception:
                    continue

                if fv.home_games_available < min_games_before or fv.away_games_available < min_games_before:
                    continue

                margin = float(game.home_score - game.away_score)
                total = float(game.home_score + game.away_score)
                all_samples.append({
                    "features": fv.features,
                    "y_cover": 1.0 if margin > line.spread_line else 0.0,
                    "y_over": 1.0 if total > line.total_line else 0.0,
                    "nba_game_id": game.nba_game_id,
                    "game_date": game.game_date,
                    "season": game.season,
                })

                if (i + 1) % 500 == 0:
                    logger.info(
                        "  Pre-building cover/over features: %d/%d (%d valid)",
                        i + 1, len(pairs), len(all_samples),
                    )

            logger.info("Cover/over walk-forward: %d valid samples", len(all_samples))

            if len(all_samples) < initial_train_games + retrain_every:
                logger.warning(
                    "Not enough cover/over samples (%d) for walk-forward (need %d)",
                    len(all_samples), initial_train_games + retrain_every,
                )
                return CoverOverBacktestMetrics(n_games=0)

            # 3. Walk-forward loop
            all_predictions: list[dict] = []
            train_end = initial_train_games
            cycle = 0

            while train_end < len(all_samples):
                predict_end = min(train_end + retrain_every, len(all_samples))
                train_samples = all_samples[:train_end]
                predict_samples = all_samples[train_end:predict_end]

                cycle += 1
                logger.info(
                    "  Cover/over cycle %d: train %d, predict %d",
                    cycle, len(train_samples), len(predict_samples),
                )

                X_train = np.vstack([s["features"] for s in train_samples])
                y_cover_train = np.array([s["y_cover"] for s in train_samples])
                y_over_train = np.array([s["y_over"] for s in train_samples])

                split_idx = int(len(X_train) * 0.8)
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                yc_tr, yc_val = y_cover_train[:split_idx], y_cover_train[split_idx:]
                yo_tr, yo_val = y_over_train[:split_idx], y_over_train[split_idx:]

                xgb = XGBoostTrainer()
                xgb.train_cover_over(
                    X_train=X_tr, y_cover_train=yc_tr, y_over_train=yo_tr,
                    X_val=X_val, y_cover_val=yc_val, y_over_val=yo_val,
                )

                for sample in predict_samples:
                    try:
                        pred = xgb.predict_cover_over(sample["features"].reshape(1, -1))
                        all_predictions.append({
                            "nba_game_id": sample["nba_game_id"],
                            "game_date": sample["game_date"].isoformat(),
                            "season": sample["season"],
                            "y_cover": sample["y_cover"],
                            "y_over": sample["y_over"],
                            "cover_prob": pred.cover_prob_home,
                            "over_prob": pred.over_prob,
                            "cover_correct": (pred.cover_prob_home > 0.5) == (sample["y_cover"] == 1.0),
                            "over_correct": (pred.over_prob > 0.5) == (sample["y_over"] == 1.0),
                        })
                    except Exception:
                        continue

                train_end = predict_end

            return self._aggregate_cover_over_metrics(all_predictions)

    @staticmethod
    def _aggregate_cover_over_metrics(
        predictions: list[dict],
    ) -> CoverOverBacktestMetrics:
        """Compute CoverOverBacktestMetrics from per-game prediction dicts."""
        n = len(predictions)
        if n == 0:
            return CoverOverBacktestMetrics(n_games=0)

        metrics = CoverOverBacktestMetrics()
        metrics.n_games = n

        metrics.cover_accuracy = sum(1 for p in predictions if p["cover_correct"]) / n
        metrics.over_accuracy = sum(1 for p in predictions if p["over_correct"]) / n

        def _logloss(probs: list[float], labels: list[float]) -> float:
            p = np.clip(np.asarray(probs, dtype=float), 1e-15, 1 - 1e-15)
            y = np.asarray(labels, dtype=float)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        metrics.cover_logloss = _logloss(
            [p["cover_prob"] for p in predictions], [p["y_cover"] for p in predictions]
        )
        metrics.over_logloss = _logloss(
            [p["over_prob"] for p in predictions], [p["y_over"] for p in predictions]
        )

        cover_season: dict[str, list] = {}
        over_season: dict[str, list] = {}
        for p in predictions:
            s = p["game_date"][:4]
            cover_season.setdefault(s, []).append(p["cover_correct"])
            over_season.setdefault(s, []).append(p["over_correct"])
        metrics.cover_accuracy_by_season = {
            s: float(np.mean(v)) for s, v in cover_season.items()
        }
        metrics.over_accuracy_by_season = {
            s: float(np.mean(v)) for s, v in over_season.items()
        }

        # Confident subset (|p-0.5| >= 0.10) — betting value vs 52.4% break-even.
        cover_conf = [p for p in predictions if abs(p["cover_prob"] - 0.5) >= 0.10]
        over_conf = [p for p in predictions if abs(p["over_prob"] - 0.5) >= 0.10]
        metrics.n_confident = max(len(cover_conf), len(over_conf))
        if cover_conf:
            metrics.cover_accuracy_confident = (
                sum(1 for p in cover_conf if (p["cover_prob"] > 0.5) == (p["y_cover"] == 1.0))
                / len(cover_conf)
            )
        if over_conf:
            metrics.over_accuracy_confident = (
                sum(1 for p in over_conf if (p["over_prob"] > 0.5) == (p["y_over"] == 1.0))
                / len(over_conf)
            )

        metrics.per_game = [
            {
                "game_id": p["nba_game_id"],
                "date": p["game_date"],
                "cover_prob": p["cover_prob"],
                "over_prob": p["over_prob"],
                "cover_correct": p["cover_correct"],
                "over_correct": p["over_correct"],
            }
            for p in predictions[-50:]
        ]

        logger.info(
            "Cover/over walk-forward complete: %d predictions. "
            "Cover=%.3f, Over=%.3f (break-even 52.4%%); confident cover=%.3f, over=%.3f (%d games)",
            n,
            metrics.cover_accuracy,
            metrics.over_accuracy,
            metrics.cover_accuracy_confident,
            metrics.over_accuracy_confident,
            metrics.n_confident,
        )
        return metrics
