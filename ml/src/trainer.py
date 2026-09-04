"""Training orchestration for the NBA prediction pipeline.

Coordinates:
  1. Feature building (dataset construction)
  2. Train/validation chronological split
  3. XGBoost training
  4. MLP training
  5. Model persistence
  6. Calibration computation
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from ml.src.config import ml_settings
from ml.src.database import async_session_factory
from ml.src.features.feature_builder import FeatureBuilder, TrainingDataset, CoverOverDataset
from ml.src.models.xgboost_model import XGBoostTrainer, TrainingMetrics, CoverOverTrainingMetrics
from ml.src.models.calibration import Calibrator

# MLPTrainer imported lazily — torch ~800MB, not available in backend container.
# Only the trainer container (--profile trainer) has torch installed.

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Complete result from a training run."""

    version: str
    trained_at: str
    xgboost_metrics: TrainingMetrics
    mlp_metrics: dict
    calibration: dict | None
    dataset_info: dict
    model_paths: dict[str, str]


class Trainer:
    """Orchestrates the full training pipeline."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory
        self.feature_builder = FeatureBuilder(session_factory)

    async def train_full_pipeline(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        version: str | None = None,
        calibrate: bool = True,
        use_historical: bool = True,
    ) -> TrainingResult:
        """Run the complete training pipeline.

        Args:
            seasons: Seasons to include (e.g. ["2023-24", "2024-25"]).
            lookback_games: Number of recent games for team features.
            version: Model version string; auto-generated if None.
            calibrate: Whether to compute calibration coefficients.
            use_historical: Read injury features from the historical snapshot
                table (True) or leave them zero (False — A/B baseline).

        Returns:
            TrainingResult with full metrics and paths.
        """
        ver = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=== Starting training pipeline v%s ===", ver)

        # 1. Build training dataset
        logger.info("Step 1/5: Building training dataset...")
        dataset = await self.feature_builder.build_training_dataset(
            seasons=seasons,
            lookback_games=lookback_games,
            min_games_before=5,  # minimum prior games per team
            use_historical=use_historical,
        )

        if len(dataset.X) == 0:
            raise ValueError("No training samples found. Check database has completed games.")

        logger.info(
            "Dataset: %d samples, win_rate=%.2f%%, spread_mean=%.1f, total_mean=%.0f",
            len(dataset.X),
            dataset.y_win.mean() * 100,
            dataset.y_spread.mean(),
            dataset.y_total.mean(),
        )

        # 2. Chronological train/val split
        logger.info("Step 2/5: Chronological split...")
        split_data = self._chronological_split(dataset)

        # 3. Train XGBoost
        logger.info("Step 3/5: Training XGBoost models...")
        xgb_trainer = XGBoostTrainer(version=ver)
        xgb_metrics = xgb_trainer.train(
            X_train=split_data["X_train"],
            y_win_train=split_data["y_win_train"],
            y_spread_train=split_data["y_spread_train"],
            y_total_train=split_data["y_total_train"],
            X_val=split_data["X_val"],
            y_win_val=split_data["y_win_val"],
            y_spread_val=split_data["y_spread_val"],
            y_total_val=split_data["y_total_val"],
            feature_names=dataset.sample_info[0].get("feature_names") if dataset.sample_info else None,
        )
        xgb_path = xgb_trainer.save()

        # 4. Train MLP (optional — requires torch, only available in trainer container)
        logger.info("Step 4/5: Training MLP model...")
        mlp_metrics = {}
        mlp_path = None
        try:
            from ml.src.models.mlp_model import MLPTrainer  # lazy import (needs torch)
            mlp_trainer = MLPTrainer(version=ver)
            mlp_metrics = mlp_trainer.train(
                X_train=split_data["X_train"],
                y_win_train=split_data["y_win_train"],
                y_spread_train=split_data["y_spread_train"],
                y_total_train=split_data["y_total_train"],
                X_val=split_data["X_val"],
                y_win_val=split_data["y_win_val"],
                y_spread_val=split_data["y_spread_val"],
                y_total_val=split_data["y_total_val"],
            )
            mlp_path = mlp_trainer.save()
        except ImportError:
            logger.info("MLP training skipped (torch not available) — XGBoost only")
        except ValueError as e:
            # Sanity check failure or other data/model validation error
            logger.warning(
                "MLP training rejected: %s — model NOT saved, continuing with XGBoost only", e
            )
        except Exception as e:
            logger.warning("MLP training failed: %s — continuing with XGBoost", e)

        # 5. Calibration
        logger.info("Step 5/5: Computing calibration coefficients...")
        calibration = None
        if calibrate and split_data["X_val"].shape[0] >= ml_settings.min_games_for_calibration:
            calibrator = Calibrator()
            xgb_preds = xgb_trainer.predict(split_data["X_val"])
            if not isinstance(xgb_preds, list):
                xgb_preds = [xgb_preds]
            calibration = calibrator.compute_coefficients(
                val_predictions=xgb_preds,
                y_win=split_data["y_win_val"],
                y_spread=split_data["y_spread_val"],
                y_total=split_data["y_total_val"],
            )
            calibrator.save()  # Persist to disk so predictor can load it

        result = TrainingResult(
            version=ver,
            trained_at=datetime.now().isoformat(),
            xgboost_metrics=xgb_metrics,
            mlp_metrics=mlp_metrics,
            calibration=calibration,
            dataset_info={
                "n_samples": len(dataset.X),
                "n_features": dataset.X.shape[1],
                "win_rate": float(dataset.y_win.mean()),
                "seasons": seasons,
            },
            model_paths={
                "xgboost": xgb_path,
                "mlp": mlp_path,
            },
        )

        logger.info("=== Training pipeline v%s complete! ===", ver)
        logger.info(
            "XGBoost: Win acc=%.3f, Spread MAE=%.2f, Total MAE=%.2f",
            xgb_metrics.win_accuracy,
            xgb_metrics.spread_mae,
            xgb_metrics.total_mae,
        )
        logger.info(
            "MLP:     Win acc=%.3f, Spread MAE=%.2f, Total MAE=%.2f",
            mlp_metrics.get("win_accuracy", 0.0),
            mlp_metrics.get("spread_mae", 0.0),
            mlp_metrics.get("total_mae", 0.0),
        )

        return result

    async def train_cover_over_pipeline(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        version: str | None = None,
        min_games_before: int = 5,
    ) -> tuple[CoverOverTrainingMetrics, str]:
        """Train the cover/over (盘口) binary classifiers and save them.

        Uses the reduced cross-season-safe feature subset and games that have a
        betting line (2012-2019).  Saves cover_model.json / over_model.json into
        the given version directory (or a fresh one), alongside any regression
        models already there.

        Returns:
            (CoverOverTrainingMetrics, version string).
        """
        ver = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=== Training cover/over classifiers v%s ===", ver)

        dataset: CoverOverDataset = await self.feature_builder.build_cover_over_dataset(
            seasons=seasons,
            lookback_games=lookback_games,
            min_games_before=min_games_before,
        )

        if len(dataset.X) == 0:
            raise ValueError(
                "No cover/over training samples found. Check betting_lines import "
                "(scripts/import_historical_odds.py)."
            )

        logger.info(
            "Cover/over dataset: %d samples, %d features, "
            "cover_balance=%.2f, over_balance=%.2f",
            len(dataset.X), dataset.X.shape[1],
            dataset.y_cover.mean(), dataset.y_over.mean(),
        )

        # Chronological split (last 15% for validation)
        n = len(dataset.X)
        split_idx = int(n * 0.85)
        split_idx = max(split_idx, 1)
        split_idx = min(split_idx, n - 1)

        xgb = XGBoostTrainer(version=ver)
        metrics = xgb.train_cover_over(
            X_train=dataset.X[:split_idx],
            y_cover_train=dataset.y_cover[:split_idx],
            y_over_train=dataset.y_over[:split_idx],
            X_val=dataset.X[split_idx:],
            y_cover_val=dataset.y_cover[split_idx:],
            y_over_val=dataset.y_over[split_idx:],
        )
        xgb.save()

        logger.info(
            "=== Cover/over training complete: Cover acc=%.3f, Over acc=%.3f ===",
            metrics.cover_accuracy, metrics.over_accuracy,
        )
        return metrics, ver

    @staticmethod
    def _chronological_split(dataset: TrainingDataset, val_ratio: float = 0.15) -> dict:
        """Split dataset chronologically (last val_ratio for validation).

        This mirrors real-world usage: we train on past data, validate on more recent data.
        """
        n = len(dataset.X)
        split_idx = int(n * (1 - val_ratio))
        split_idx = max(split_idx, 1)  # At least 1 validation sample
        split_idx = min(split_idx, n - 1)  # At least 1 training sample

        X_train = dataset.X[:split_idx]
        X_val = dataset.X[split_idx:]

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_win_train": dataset.y_win[:split_idx],
            "y_win_val": dataset.y_win[split_idx:],
            "y_spread_train": dataset.y_spread[:split_idx],
            "y_spread_val": dataset.y_spread[split_idx:],
            "y_total_train": dataset.y_total[:split_idx],
            "y_total_val": dataset.y_total[split_idx:],
        }


async def train_and_save(
    seasons: list[str] | None = None,
    lookback: int = 10,
    version: str | None = None,
) -> TrainingResult:
    """Convenience function: train models and save to disk."""
    trainer = Trainer()
    return await trainer.train_full_pipeline(
        seasons=seasons,
        lookback_games=lookback,
        version=version,
    )
