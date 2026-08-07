"""Optuna-based hyperparameter tuning for XGBoost models.

Systematically searches the hyperparameter space to minimize combined
spread + total MAE. Uses chronological validation split (no data leakage).

Usage:
    python -m ml.src.models.tuner --trials 100 --lookback 10
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import optuna
import xgboost as xgb

from ml.src.config import ml_settings
from ml.src.database import async_session_factory
from ml.src.features.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Best hyperparameters and metrics from tuning."""

    study_name: str
    best_params: dict
    best_value: float  # combined MAE
    best_spread_mae: float
    best_total_mae: float
    best_win_accuracy: float
    n_trials: int
    duration_seconds: float


class XGBoostTuner:
    """Optuna tuning for XGBoost regression and classification models.

    Optimizes for minimizing (spread_mae + total_mae) / 2, which is the
    primary target metric for the gambling prediction use case.
    """

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory
        self.feature_builder = FeatureBuilder(session_factory)

        # Data cache — loaded once per tuning run
        self.X_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_win_train: np.ndarray | None = None
        self.y_win_val: np.ndarray | None = None
        self.y_spread_train: np.ndarray | None = None
        self.y_spread_val: np.ndarray | None = None
        self.y_total_train: np.ndarray | None = None
        self.y_total_val: np.ndarray | None = None

    async def load_data(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 5,
        val_ratio: float = 0.15,
    ) -> tuple[int, int]:
        """Load and split training data. Returns (n_train, n_val)."""
        logger.info("Loading data for tuning...")
        dataset = await self.feature_builder.build_training_dataset(
            seasons=seasons,
            lookback_games=lookback_games,
            min_games_before=min_games_before,
        )

        n = len(dataset.X)
        split_idx = max(1, min(int(n * (1 - val_ratio)), n - 1))

        self.X_train = dataset.X[:split_idx]
        self.X_val = dataset.X[split_idx:]
        self.y_win_train = dataset.y_win[:split_idx]
        self.y_win_val = dataset.y_win[split_idx:]
        self.y_spread_train = dataset.y_spread[:split_idx]
        self.y_spread_val = dataset.y_spread[split_idx:]
        self.y_total_train = dataset.y_total[:split_idx]
        self.y_total_val = dataset.y_total[split_idx:]

        logger.info("Tuning data: %d train, %d val samples", split_idx, n - split_idx)
        return split_idx, n - split_idx

    async def tune(
        self,
        n_trials: int = 100,
        timeout_seconds: int | None = 600,
        study_name: str = "xgboost_nba",
    ) -> TuningResult:
        """Run Optuna hyperparameter search.

        Args:
            n_trials: Number of Optuna trials.
            timeout_seconds: Max tuning time in seconds.
            study_name: Name for the Optuna study.

        Returns:
            TuningResult with best params and metrics.
        """
        if self.X_train is None:
            raise RuntimeError("Call load_data() before tune()")

        import time
        start_time = time.time()

        # ── Create study ──
        storage_path = os.path.join(ml_settings.models_dir, "xgboost", "optuna.db")
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage_url = f"sqlite:///{storage_path}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            load_if_exists=True,
        )

        logger.info(
            "Starting Optuna tuning: %d trials (timeout=%ds)",
            n_trials, timeout_seconds,
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
        )

        duration = time.time() - start_time

        # ── Evaluate best params ──
        best_params = study.best_params
        best_value = study.best_value

        # Full evaluation with best params
        spread_mae, total_mae, win_acc = self._evaluate_params(best_params)

        logger.info("Tuning complete in %.1fs", duration)
        logger.info("Best value (combined MAE): %.4f", best_value)
        logger.info("Best params: %s", best_params)
        logger.info("Final eval — Spread MAE=%.2f, Total MAE=%.2f, Win Acc=%.3f",
                     spread_mae, total_mae, win_acc)

        return TuningResult(
            study_name=study_name,
            best_params=best_params,
            best_value=best_value,
            best_spread_mae=spread_mae,
            best_total_mae=total_mae,
            best_win_accuracy=win_acc,
            n_trials=len(study.trials),
            duration_seconds=duration,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: minimize (spread_mae + total_mae) / 2.

        This is the primary target metric. Win accuracy is tracked but
        not part of the objective since it's already good enough (~72%).
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": ml_settings.random_seed,
            "n_jobs": -1,
        }

        # ── Early stopping rounds ──
        early_stopping = trial.suggest_int("early_stopping_rounds", 30, 100, step=10)

        try:
            # Spread regressor
            model_spread = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=early_stopping,
                **params,
            )
            model_spread.fit(
                self.X_train, self.y_spread_train,
                eval_set=[(self.X_val, self.y_spread_val)],
                verbose=False,
            )

            # Total regressor
            model_total = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=early_stopping,
                **params,
            )
            model_total.fit(
                self.X_train, self.y_total_train,
                eval_set=[(self.X_val, self.y_total_val)],
                verbose=False,
            )

            # Evaluate
            spread_preds = model_spread.predict(self.X_val)
            total_preds = model_total.predict(self.X_val)

            spread_mae = float(np.mean(np.abs(spread_preds - self.y_spread_val)))
            total_mae = float(np.mean(np.abs(total_preds - self.y_total_val)))

            # Combined objective: equally weighted
            combined_mae = (spread_mae + total_mae) / 2.0

            # Log individual metrics as trial user attrs for analysis
            trial.set_user_attr("spread_mae", spread_mae)
            trial.set_user_attr("total_mae", total_mae)

            return combined_mae

        except Exception as e:
            logger.debug("Trial %d failed: %s", trial.number, e)
            return 1e9  # penalty for failed trials

    def _evaluate_params(self, params: dict) -> tuple[float, float, float]:
        """Full evaluation of a parameter set: spread, total, win.

        Returns (spread_mae, total_mae, win_accuracy).
        """
        from copy import deepcopy

        p = deepcopy(params)

        # Extract non-model params
        early_stopping = p.pop("early_stopping_rounds", 80)

        # Ensure required params
        p.setdefault("random_state", ml_settings.random_seed)
        p.setdefault("n_jobs", -1)

        # Spread
        model_spread = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping,
            **p,
        )
        model_spread.fit(
            self.X_train, self.y_spread_train,
            eval_set=[(self.X_val, self.y_spread_val)],
            verbose=False,
        )
        spread_preds = model_spread.predict(self.X_val)
        spread_mae = float(np.mean(np.abs(spread_preds - self.y_spread_val)))

        # Total
        model_total = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping,
            **p,
        )
        model_total.fit(
            self.X_train, self.y_total_train,
            eval_set=[(self.X_val, self.y_total_val)],
            verbose=False,
        )
        total_preds = model_total.predict(self.X_val)
        total_mae = float(np.mean(np.abs(total_preds - self.y_total_val)))

        # Win classifier
        model_win = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=early_stopping,
            **p,
        )
        model_win.fit(
            self.X_train, self.y_win_train,
            eval_set=[(self.X_val, self.y_win_val)],
            verbose=False,
        )
        win_preds = model_win.predict(self.X_val)
        win_acc = float((win_preds == self.y_win_val).mean())

        return spread_mae, total_mae, win_acc


async def run_tuning(
    seasons: list[str] | None = None,
    lookback: int = 10,
    n_trials: int = 100,
    timeout_minutes: int = 10,
) -> TuningResult:
    """Convenience function: run a full tuning session."""
    tuner = XGBoostTuner()
    await tuner.load_data(seasons=seasons, lookback_games=lookback)
    return await tuner.tune(
        n_trials=n_trials,
        timeout_seconds=timeout_minutes * 60,
    )
