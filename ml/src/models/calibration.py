"""Calibration layer for model predictions.

Applies a statistical correction based on recent prediction errors.
Uses a sliding window (default 30 days) to detect and correct for
systematic bias in the model predictions.

Principle:
  - If model predicts home win 65% but actual is 58% → calibrate to 58%
  - If model systematically overestimates spreads by 1.2 pts → subtract 1.2
  - If model systematically underestimates totals by 3.5 pts → add 3.5

Calibration coefficients are stored on disk and updated daily.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

from ml.src.config import ml_settings

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCoefficients:
    """Multiplicative and additive calibration factors."""

    # Win probability calibration (Platt scaling - style)
    win_bias: float = 0.0  # Additive bias correction for win prob
    win_scale: float = 1.0  # Scale factor for win prob

    # Spread calibration
    spread_bias: float = 0.0  # Additive correction (model output - bias)
    spread_scale: float = 1.0  # Multiplicative correction

    # Total calibration
    total_bias: float = 0.0
    total_scale: float = 1.0

    # Metadata
    n_games_used: int = 0
    window_days: int = 30
    computed_at: str = ""


class Calibrator:
    """Manages prediction calibration coefficients."""

    def __init__(self, calibration_path: str | None = None):
        self.calibration_path = calibration_path or os.path.join(
            ml_settings.models_dir, "xgboost", "calibration.json"
        )
        self.coefficients = CalibrationCoefficients()

    @property
    def is_fitted(self) -> bool:
        return self.coefficients.n_games_used > 0

    def compute_coefficients(
        self,
        val_predictions: list,  # List of XGBoostPrediction or similar
        y_win: np.ndarray,
        y_spread: np.ndarray,
        y_total: np.ndarray,
    ) -> CalibrationCoefficients:
        """Compute calibration coefficients from validation predictions.

        Args:
            val_predictions: List of predictions (must have win_prob_home,
                             predicted_spread, predicted_total attributes).
            y_win: Actual win (1=home, 0=away).
            y_spread: Actual spread (home - away).
            y_total: Actual total points.

        Returns:
            CalibrationCoefficients.
        """
        n = len(val_predictions)
        if n < ml_settings.min_games_for_calibration:
            logger.warning(
                "Need at least %d games for calibration, got %d",
                ml_settings.min_games_for_calibration, n,
            )
            return self.coefficients

        pred_win = np.array([p.win_prob_home for p in val_predictions])
        pred_spread = np.array([p.predicted_spread for p in val_predictions])
        pred_total = np.array([p.predicted_total for p in val_predictions])

        # Win calibration: compute bias (actual win rate - predicted win prob)
        actual_win_rate = y_win.mean()
        predicted_win_rate = pred_win.mean()
        win_bias = actual_win_rate - predicted_win_rate

        # Spread calibration: linear regression actual ~ predicted
        spread_bias = float(np.mean(y_spread - pred_spread))

        # Avoid division by zero in scale
        pred_spread_std = np.std(pred_spread)
        if pred_spread_std > 1e-6:
            spread_scale = float(np.std(y_spread) / pred_spread_std)
        else:
            spread_scale = 1.0

        # Total calibration
        total_bias = float(np.mean(y_total - pred_total))

        pred_total_std = np.std(pred_total)
        if pred_total_std > 1e-6:
            total_scale = float(np.std(y_total) / pred_total_std)
        else:
            total_scale = 1.0

        from datetime import datetime

        self.coefficients = CalibrationCoefficients(
            win_bias=win_bias,
            win_scale=1.0,  # Keep simple: only additive for win prob
            spread_bias=spread_bias,
            spread_scale=spread_scale,
            total_bias=total_bias,
            total_scale=total_scale,
            n_games_used=n,
            window_days=ml_settings.calibration_window_days,
            computed_at=datetime.now().isoformat(),
        )

        logger.info(
            "Calibration computed from %d games: "
            "win_bias=%+.3f, spread_bias=%+.2f, total_bias=%+.2f",
            n, win_bias, spread_bias, total_bias,
        )

        return self.coefficients

    def apply(self, prediction):
        """Apply calibration to a single prediction.

        Two-stage correction:
          1. Additive bias: shifts the mean to match actual outcomes.
          2. Multiplicative scale: stretches/compresses the variance to match
             actual outcome variance (applied only when |scale - 1.0| > 0.03).

        Spreads are mean-centred (~0) so direct multiplication works.
        Totals are de-meaned before scaling (league average ~227.5).

        Args:
            prediction: XGBoostPrediction or MLPPrediction object.

        Returns:
            The same prediction object with calibrated values.
        """
        if not self.is_fitted:
            return prediction

        c = self.coefficients

        # ── Win probability (additive bias, clamp to [0.01, 0.99]) ──
        raw_win = prediction.win_prob_home
        calibrated_win = np.clip(raw_win + c.win_bias, 0.01, 0.99)
        prediction.win_prob_home = float(calibrated_win)
        prediction.win_prob_away = float(1.0 - calibrated_win)

        # ── Spread: additive bias + multiplicative scale ──
        raw_spread = prediction.predicted_spread
        calibrated_spread = raw_spread + c.spread_bias
        # Apply scale if model systematically compresses/expands predictions.
        # Spread mean ≈ 0 (home/away roughly balanced), so direct scale is safe.
        # Cap effective scale at 2.0 to avoid overshooting on individual games.
        if abs(c.spread_scale - 1.0) > 0.03:
            effective_scale = min(c.spread_scale, 2.0)
            calibrated_spread = calibrated_spread * effective_scale
        prediction.predicted_spread = float(np.clip(calibrated_spread, -30.0, 30.0))

        # ── Total: additive bias + multiplicative scale (de-meaned) ──
        raw_total = prediction.predicted_total
        calibrated_total = raw_total + c.total_bias
        # De-mean before scaling because mean_total ≈ 227.5 (not 0).
        # League-average total fluctuates by season; 227.5 is a stable reference.
        if abs(c.total_scale - 1.0) > 0.03:
            LEAGUE_AVG_TOTAL = 227.5
            effective_scale = min(c.total_scale, 2.0)
            calibrated_total = ((calibrated_total - LEAGUE_AVG_TOTAL) * effective_scale) + LEAGUE_AVG_TOTAL
        prediction.predicted_total = float(np.clip(calibrated_total, 180.0, 280.0))

        return prediction

    def save(self, path: str | None = None) -> None:
        """Save calibration coefficients to disk."""
        save_path = path or self.calibration_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            "win_bias": self.coefficients.win_bias,
            "win_scale": self.coefficients.win_scale,
            "spread_bias": self.coefficients.spread_bias,
            "spread_scale": self.coefficients.spread_scale,
            "total_bias": self.coefficients.total_bias,
            "total_scale": self.coefficients.total_scale,
            "n_games_used": self.coefficients.n_games_used,
            "window_days": self.coefficients.window_days,
            "computed_at": self.coefficients.computed_at,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved calibration to %s", save_path)

    def load(self, path: str | None = None) -> CalibrationCoefficients:
        """Load calibration coefficients from disk."""
        load_path = path or self.calibration_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No calibration file at {load_path}")

        with open(load_path) as f:
            data = json.load(f)

        self.coefficients = CalibrationCoefficients(
            win_bias=data.get("win_bias", 0.0),
            win_scale=data.get("win_scale", 1.0),
            spread_bias=data.get("spread_bias", 0.0),
            spread_scale=data.get("spread_scale", 1.0),
            total_bias=data.get("total_bias", 0.0),
            total_scale=data.get("total_scale", 1.0),
            n_games_used=data.get("n_games_used", 0),
            window_days=data.get("window_days", 30),
            computed_at=data.get("computed_at", ""),
        )

        logger.info(
            "Loaded calibration from %d games (computed %s)",
            self.coefficients.n_games_used,
            self.coefficients.computed_at,
        )

        return self.coefficients
