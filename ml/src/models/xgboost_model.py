"""XGBoost model training and inference.

Trains three independent XGBoost models:
  - XGBoost-Win:   Binary classifier → home team win probability
  - XGBoost-Spread: Regressor → predicted point spread (home - away)
  - XGBoost-Total:  Regressor → predicted total points (home + away)

Features:
  - Time-series cross-validation (chronological split)
  - Early stopping on validation set
  - Model persistence (save/load JSON or binary format)
  - Feature importance reporting
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import xgboost as xgb

from ml.src.config import ml_settings

logger = logging.getLogger(__name__)


@dataclass
class XGBoostPrediction:
    """Result from XGBoost inference."""

    win_prob_home: float
    win_prob_away: float
    predicted_spread: float
    spread_confidence: float
    predicted_total: float
    total_confidence: float
    model_version: str


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    win_accuracy: float
    win_logloss: float
    spread_mae: float
    spread_rmse: float
    total_mae: float
    total_rmse: float
    train_samples: int
    val_samples: int
    training_time_seconds: float


@dataclass
class CoverOverPrediction:
    """Result from cover/over (盘口) XGBoost inference."""

    cover_prob_home: float  # P(home covers the spread)
    over_prob: float        # P(game goes over the total)
    model_version: str


@dataclass
class CoverOverTrainingMetrics:
    """Metrics from a cover/over training run."""

    cover_accuracy: float
    cover_logloss: float
    over_accuracy: float
    over_logloss: float
    train_samples: int
    val_samples: int
    training_time_seconds: float


class XGBoostTrainer:
    """Trains and manages XGBoost models for NBA prediction."""

    def __init__(
        self,
        models_dir: str | None = None,
        version: str | None = None,
    ):
        self.models_dir = models_dir or ml_settings.xgboost_dir
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model_win: xgb.XGBClassifier | None = None
        self._model_spread: xgb.XGBRegressor | None = None
        self._model_total: xgb.XGBRegressor | None = None
        self._model_cover: xgb.XGBClassifier | None = None
        self._model_over: xgb.XGBClassifier | None = None

    @property
    def model_win(self) -> xgb.XGBClassifier | None:
        return self._model_win

    @property
    def model_spread(self) -> xgb.XGBRegressor | None:
        return self._model_spread

    @property
    def model_total(self) -> xgb.XGBRegressor | None:
        return self._model_total

    @property
    def model_cover(self) -> xgb.XGBClassifier | None:
        return self._model_cover

    @property
    def model_over(self) -> xgb.XGBClassifier | None:
        return self._model_over

    @property
    def is_trained(self) -> bool:
        return self._model_win is not None

    @property
    def is_cover_over_trained(self) -> bool:
        return self._model_cover is not None and self._model_over is not None

    def _common_params(self) -> dict:
        """Shared XGBoost hyperparameters (regression and classification)."""
        return {
            "n_estimators": ml_settings.xgboost_n_estimators,
            "max_depth": ml_settings.xgboost_max_depth,
            "learning_rate": ml_settings.xgboost_learning_rate,
            "subsample": ml_settings.xgboost_subsample,
            "colsample_bytree": ml_settings.xgboost_colsample_bytree,
            "reg_alpha": ml_settings.xgboost_reg_alpha,
            "reg_lambda": ml_settings.xgboost_reg_lambda,
            "min_child_weight": ml_settings.xgboost_min_child_weight,
            "gamma": ml_settings.xgboost_gamma,
            "random_state": ml_settings.random_seed,
            "n_jobs": -1,
        }

    def train(
        self,
        X_train: np.ndarray,
        y_win_train: np.ndarray,
        y_spread_train: np.ndarray,
        y_total_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_win_val: np.ndarray | None = None,
        y_spread_val: np.ndarray | None = None,
        y_total_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> TrainingMetrics:
        """Train all three XGBoost models.

        Args:
            X_train: Training features.
            y_win_train: Win labels (1=home win, 0=away win).
            y_spread_train: Spread values (home_score - away_score).
            y_total_train: Total points (home_score + away_score).
            X_val: Validation features (optional, for early stopping).
            y_win_val, y_spread_val, y_total_val: Validation targets.
            feature_names: Feature name list for importance reporting.

        Returns:
            TrainingMetrics with validation performance.
        """
        import time
        start_time = time.time()

        n_train = len(X_train)
        n_val = len(X_val) if X_val is not None else 0
        logger.info(
            "Training XGBoost models v%s: %d train, %d val samples",
            self.version, n_train, n_val,
        )

        # Common XGBoost parameters — tuned for tight spread/total accuracy
        common_params = self._common_params()

        # ── Win/Loss Classifier ──
        logger.info("Training XGBoost-Win classifier...")
        self._model_win = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            **common_params,
        )

        eval_set_win = [(X_train, y_win_train)]
        if X_val is not None and y_win_val is not None:
            eval_set_win.append((X_val, y_win_val))

        self._model_win.fit(
            X_train, y_win_train,
            eval_set=eval_set_win,
            verbose=False,
        )

        # ── Spread Regressor ──
        logger.info("Training XGBoost-Spread regressor...")
        self._model_spread = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            **common_params,
        )

        eval_set_spread = [(X_train, y_spread_train)]
        if X_val is not None and y_spread_val is not None:
            eval_set_spread.append((X_val, y_spread_val))

        self._model_spread.fit(
            X_train, y_spread_train,
            eval_set=eval_set_spread,
            verbose=False,
        )

        # ── Total Points Regressor ──
        logger.info("Training XGBoost-Total regressor...")
        self._model_total = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            **common_params,
        )

        eval_set_total = [(X_train, y_total_train)]
        if X_val is not None and y_total_val is not None:
            eval_set_total.append((X_val, y_total_val))

        self._model_total.fit(
            X_train, y_total_train,
            eval_set=eval_set_total,
            verbose=False,
        )

        training_time = time.time() - start_time

        # ── Evaluate on validation set ──
        if X_val is not None:
            metrics = self._evaluate(
                X_val, y_win_val, y_spread_val, y_total_val,
                n_train, n_val, training_time,
            )
        else:
            # Use training metrics as fallback
            metrics = TrainingMetrics(
                win_accuracy=0.0,
                win_logloss=0.0,
                spread_mae=0.0,
                spread_rmse=0.0,
                total_mae=0.0,
                total_rmse=0.0,
                train_samples=n_train,
                val_samples=0,
                training_time_seconds=training_time,
            )

        # Log feature importance (top 10)
        if feature_names and self._model_total is not None:
            importances = self._model_total.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            top_features = [(feature_names[i], importances[i]) for i in indices]
            logger.info("Top 10 features by importance: %s", top_features)

        logger.info(
            "Training complete in %.1fs. "
            "Win acc=%.3f, Spread MAE=%.2f, Total MAE=%.2f",
            training_time,
            metrics.win_accuracy,
            metrics.spread_mae,
            metrics.total_mae,
        )

        return metrics

    def train_cover_over(
        self,
        X_train: np.ndarray,
        y_cover_train: np.ndarray,
        y_over_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_cover_val: np.ndarray | None = None,
        y_over_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> CoverOverTrainingMetrics:
        """Train the two cover/over binary classifiers (让分胜负 / 大小分).

        Uses the reduced cross-season-safe feature subset (COVER_OVER_FEATURE_DIM)
        and the betting-line labels:
          y_cover = 1 if (home - away) > spread_line else 0
          y_over  = 1 if (home + away) > total_line else 0
        """
        import time
        start_time = time.time()

        n_train = len(X_train)
        n_val = len(X_val) if X_val is not None else 0
        logger.info(
            "Training cover/over classifiers v%s: %d train, %d val samples",
            self.version, n_train, n_val,
        )

        common_params = self._common_params()

        # ── Cover (spread) classifier ──
        self._model_cover = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            **common_params,
        )
        eval_set_cover = [(X_train, y_cover_train)]
        if X_val is not None and y_cover_val is not None:
            eval_set_cover.append((X_val, y_cover_val))
        self._model_cover.fit(X_train, y_cover_train, eval_set=eval_set_cover, verbose=False)

        # ── Over (total) classifier ──
        self._model_over = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            **common_params,
        )
        eval_set_over = [(X_train, y_over_train)]
        if X_val is not None and y_over_val is not None:
            eval_set_over.append((X_val, y_over_val))
        self._model_over.fit(X_train, y_over_train, eval_set=eval_set_over, verbose=False)

        training_time = time.time() - start_time

        if X_val is not None and y_cover_val is not None and y_over_val is not None:
            cover_preds = self._model_cover.predict(X_val)
            over_preds = self._model_over.predict(X_val)
            cover_acc = float((cover_preds == y_cover_val).mean())
            over_acc = float((over_preds == y_over_val).mean())

            cover_probs = np.clip(self._model_cover.predict_proba(X_val)[:, 1], 1e-15, 1 - 1e-15)
            over_probs = np.clip(self._model_over.predict_proba(X_val)[:, 1], 1e-15, 1 - 1e-15)
            cover_logloss = float(-np.mean(
                y_cover_val * np.log(cover_probs) + (1 - y_cover_val) * np.log(1 - cover_probs)
            ))
            over_logloss = float(-np.mean(
                y_over_val * np.log(over_probs) + (1 - y_over_val) * np.log(1 - over_probs)
            ))
        else:
            cover_acc = over_acc = cover_logloss = over_logloss = 0.0

        logger.info(
            "Cover/over training complete in %.1fs. Cover acc=%.3f, Over acc=%.3f",
            training_time, cover_acc, over_acc,
        )

        return CoverOverTrainingMetrics(
            cover_accuracy=cover_acc,
            cover_logloss=cover_logloss,
            over_accuracy=over_acc,
            over_logloss=over_logloss,
            train_samples=n_train,
            val_samples=n_val,
            training_time_seconds=training_time,
        )

    def predict(self, X: np.ndarray) -> XGBoostPrediction:
        """Run inference with all three models.

        Args:
            X: Feature vector(s), shape (n_samples, FEATURE_DIM) or (FEATURE_DIM,).

        Returns:
            XGBoostPrediction (if single sample) or list of them.
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train() first or load().")

        # Ensure 2D — derive single-sample flag AFTER reshape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        single = X.shape[0] == 1

        win_probs = self._model_win.predict_proba(X)  # shape: (n, 2) -> [away, home]
        spreads = self._model_spread.predict(X)
        totals = self._model_total.predict(X)

        # Confidence estimates from prediction margins
        # For classifier: how far from 0.5
        # For regressor: use the standard deviation of individual tree predictions
        win_margin = np.abs(win_probs[:, 1] - 0.5) * 2  # 0-1 scale

        results = []
        for i in range(len(X)):
            results.append(
                XGBoostPrediction(
                    win_prob_home=float(win_probs[i, 1]),
                    win_prob_away=float(win_probs[i, 0]),
                    predicted_spread=float(spreads[i]),
                    spread_confidence=float(win_margin[i]),  # proxy
                    predicted_total=float(totals[i]),
                    total_confidence=float(win_margin[i]),  # proxy
                    model_version=self.version,
                )
            )

        return results[0] if single else results

    def predict_cover_over(self, X: np.ndarray) -> CoverOverPrediction:
        """Run cover/over inference (requires train_cover_over() or load())."""
        if not self.is_cover_over_trained:
            raise RuntimeError("Cover/over models not trained. Call train_cover_over() or load().")

        if X.ndim == 1:
            X = X.reshape(1, -1)
        single = X.shape[0] == 1

        cover_probs = self._model_cover.predict_proba(X)[:, 1]
        over_probs = self._model_over.predict_proba(X)[:, 1]

        results = [
            CoverOverPrediction(
                cover_prob_home=float(cover_probs[i]),
                over_prob=float(over_probs[i]),
                model_version=self.version,
            )
            for i in range(len(X))
        ]
        return results[0] if single else results

    def _evaluate(
        self,
        X_val: np.ndarray,
        y_win_val: np.ndarray,
        y_spread_val: np.ndarray,
        y_total_val: np.ndarray,
        n_train: int,
        n_val: int,
        training_time: float,
    ) -> TrainingMetrics:
        """Compute validation metrics with sanity checks."""
        # Win accuracy
        win_preds = self._model_win.predict(X_val)
        win_acc = float((win_preds == y_win_val).mean())

        # Win log loss
        win_probs = self._model_win.predict_proba(X_val)
        # Clip to avoid log(0)
        win_probs = np.clip(win_probs, 1e-15, 1 - 1e-15)
        win_logloss = float(-np.mean(
            y_win_val * np.log(win_probs[:, 1]) + (1 - y_win_val) * np.log(win_probs[:, 0])
        ))

        # Spread metrics
        spread_preds = self._model_spread.predict(X_val)
        spread_mae = float(np.mean(np.abs(spread_preds - y_spread_val)))
        spread_rmse = float(np.sqrt(np.mean((spread_preds - y_spread_val) ** 2)))

        # Total metrics
        total_preds = self._model_total.predict(X_val)
        total_mae = float(np.mean(np.abs(total_preds - y_total_val)))
        total_rmse = float(np.sqrt(np.mean((total_preds - y_total_val) ** 2)))

        # ── Post-training sanity check ──
        self._validate_predictions(spread_preds, total_preds, win_probs[:, 1])

        return TrainingMetrics(
            win_accuracy=win_acc,
            win_logloss=win_logloss,
            spread_mae=spread_mae,
            spread_rmse=spread_rmse,
            total_mae=total_mae,
            total_rmse=total_rmse,
            train_samples=n_train,
            val_samples=n_val,
            training_time_seconds=training_time,
        )

    @staticmethod
    def _validate_predictions(
        spread_preds: np.ndarray,
        total_preds: np.ndarray,
        win_probs: np.ndarray,
    ) -> None:
        """Validate model predictions are within reasonable ranges.

        Catches severe model breakage before persisting to disk.
        Thresholds are generous — a healthy model never comes close.
        """
        issues: list[str] = []

        # 1. Spread within ±50
        max_abs_spread = float(max(abs(spread_preds)))
        if max_abs_spread > 50:
            issues.append(
                f"max |spread| = {max_abs_spread:.1f} exceeds 50 "
                f"(range [{float(spread_preds.min()):.1f}, {float(spread_preds.max()):.1f}])"
            )

        # 2. Total within [180, 350]
        t_min, t_max = float(total_preds.min()), float(total_preds.max())
        if t_min < 180 or t_max > 350:
            issues.append(
                f"total range = [{t_min:.0f}, {t_max:.0f}] outside [180, 350]"
            )

        # 3. All spreads same sign → model biased
        if bool(np.all(spread_preds > 0)) or bool(np.all(spread_preds < 0)):
            sign_str = "positive" if bool(np.all(spread_preds > 0)) else "negative"
            issues.append(f"all {len(spread_preds)} spreads are {sign_str} (model biased)")

        # 4. Win probs should vary
        unique_probs = len(set(round(float(w), 3) for w in win_probs))
        if unique_probs < 3:
            issues.append(
                f"only {unique_probs} unique win probabilities (model may be collapsed)"
            )

        if issues:
            raise ValueError(
                "XGBoost sanity check FAILED:\n  - " + "\n  - ".join(issues)
            )

        logger.info(
            "XGBoost sanity check PASSED: spread [%.1f, %.1f], total [%.0f, %.0f], "
            "%d unique win probs",
            float(spread_preds.min()), float(spread_preds.max()),
            t_min, t_max, unique_probs,
        )

    def save(self, version: str | None = None) -> str:
        """Save all three models to disk. Returns the version directory path."""
        ver = version or self.version
        model_dir = os.path.join(self.models_dir, ver)
        os.makedirs(model_dir, exist_ok=True)

        if self._model_win:
            self._model_win.save_model(os.path.join(model_dir, "win_model.json"))
        if self._model_spread:
            self._model_spread.save_model(os.path.join(model_dir, "spread_model.json"))
        if self._model_total:
            self._model_total.save_model(os.path.join(model_dir, "total_model.json"))
        if self._model_cover:
            self._model_cover.save_model(os.path.join(model_dir, "cover_model.json"))
        if self._model_over:
            self._model_over.save_model(os.path.join(model_dir, "over_model.json"))

        # Save metadata. When only cover/over was trained (_model_win is None —
        # e.g. train_cover_over_pipeline saving into an existing version dir),
        # preserve the existing feature_dim instead of clobbering it to 0.
        feature_dim = self._model_win.n_features_in_ if self._model_win else 0
        if self._model_win is None:
            meta_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        feature_dim = json.load(f).get("feature_dim", 0)
                except Exception:
                    feature_dim = 0

        metadata = {
            "version": ver,
            "created_at": datetime.now().isoformat(),
            "feature_dim": feature_dim,
            "cover_over_feature_dim": self._model_cover.n_features_in_ if self._model_cover else None,
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved XGBoost models to %s", model_dir)
        return model_dir

    def load(self, version: str | None = None) -> str:
        """Load all three models from disk. Returns the version string."""
        ver = version or self._find_latest_version()
        model_dir = os.path.join(self.models_dir, ver)

        win_path = os.path.join(model_dir, "win_model.json")
        spread_path = os.path.join(model_dir, "spread_model.json")
        total_path = os.path.join(model_dir, "total_model.json")
        cover_path = os.path.join(model_dir, "cover_model.json")
        over_path = os.path.join(model_dir, "over_model.json")

        self._model_win = xgb.XGBClassifier()
        self._model_win.load_model(win_path)

        self._model_spread = xgb.XGBRegressor()
        self._model_spread.load_model(spread_path)

        self._model_total = xgb.XGBRegressor()
        self._model_total.load_model(total_path)

        # Cover/over classifiers are optional (older versions lack them).
        if os.path.exists(cover_path) and os.path.exists(over_path):
            self._model_cover = xgb.XGBClassifier()
            self._model_cover.load_model(cover_path)
            self._model_over = xgb.XGBClassifier()
            self._model_over.load_model(over_path)

        self.version = ver
        logger.info("Loaded XGBoost models v%s from %s", ver, model_dir)
        return ver

    def _find_latest_version(self) -> str | None:
        """Find the most recent model version on disk."""
        if not os.path.isdir(self.models_dir):
            return None
        versions = [
            d for d in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, d))
            and os.path.exists(os.path.join(self.models_dir, d, "win_model.json"))
        ]
        if not versions:
            return None
        return sorted(versions)[-1]


# Convenience function
def create_trainer(version: str | None = None) -> XGBoostTrainer:
    """Create an XGBoostTrainer instance."""
    return XGBoostTrainer(version=version)
