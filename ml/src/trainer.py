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
from ml.src.features.feature_builder import FeatureBuilder, TrainingDataset, CoverOverDataset, FEATURE_NAMES
from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
from ml.src.models.xgboost_model import XGBoostTrainer, TrainingMetrics, CoverOverTrainingMetrics
from ml.src.models.calibration import Calibrator
from ml.src.training_policy import policy_mask_and_weights, subset_arrays
from ml.src.evaluation import complete_day_split, nba_day

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
    release_status: str = "candidate"


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
        use_cache: bool = True,
        activate: bool = False,
        feature_cache_key: str | None = None,
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
        if activate:
            raise ValueError('Training only stages candidates; promotion requires release-gate evidence')
        self._check_output_directories()
        ver = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=== Starting training pipeline v%s ===", ver)

        # 1. Build training dataset
        logger.info("Step 1/5: Building training dataset...")
        if feature_cache_key:
            from ml.src.features.dataset_cache import DatasetCache

            if seasons is not None:
                raise ValueError("Pinned cache already defines seasons; do not also specify seasons")
            dataset = DatasetCache().load(feature_cache_key, len(FEATURE_NAMES))
            if dataset is None:
                raise ValueError("Pinned feature cache is missing or incompatible")
        else:
            dataset = await self.feature_builder.build_training_dataset(
                seasons=seasons, lookback_games=lookback_games, min_games_before=5,
                use_historical=use_historical, use_cache=use_cache,
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

        base_policy = getattr(ml_settings, "training_policy", "T4")
        policy_names = {
            "xgboost_win": getattr(ml_settings, "xgboost_win_policy", None) or base_policy,
            "xgboost_spread": getattr(ml_settings, "xgboost_spread_policy", None) or base_policy,
            "xgboost_total": getattr(ml_settings, "xgboost_total_policy", None) or base_policy,
            "mlp_shared": getattr(ml_settings, "mlp_training_policy", None) or base_policy,
        }
        full_weights: dict[str, np.ndarray] = {}
        reports: dict[str, dict] = {}
        union_mask = np.zeros(len(dataset.X), dtype=bool)
        for target, policy_name in policy_names.items():
            mask, compact_weights, report = policy_mask_and_weights(
                dataset.sample_info, policy_name
            )
            aligned = np.zeros(len(dataset.X), dtype=np.float32)
            aligned[mask] = compact_weights
            full_weights[target] = aligned
            reports[target] = report
            union_mask |= mask
        dataset = subset_arrays(dataset, union_mask)
        target_weights = {
            target: values[union_mask] for target, values in full_weights.items()
        }
        weight_report = {"targets": reports, "feature_cache_key": feature_cache_key}
        if len(dataset.X) < 100:
            raise ValueError(
                f"Training policies retained only {len(dataset.X)} union rows"
            )
        logger.info("Training policies: %s", weight_report)

        # 2. Chronological train/val split
        logger.info("Step 2/5: Chronological split...")
        split_data = self._chronological_split(dataset)
        split_idx = split_data["split_idx"]
        selection_weights = {}
        selection_reports = {}
        for target, policy_name in policy_names.items():
            mask, weights, summary = policy_mask_and_weights(
                dataset.sample_info[:split_idx], policy_name
            )
            aligned = np.zeros(split_idx, dtype=np.float32)
            aligned[mask] = weights
            selection_weights[target] = aligned
            selection_reports[target] = summary
        weight_report["selection_targets"] = selection_reports
        baseline_index = FEATURE_NAMES.index("shrunk_expected_total")
        total_baseline = dataset.X[:, baseline_index]

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
            feature_names=FEATURE_NAMES,
            sample_weight_win=selection_weights["xgboost_win"],
            sample_weight_spread=selection_weights["xgboost_spread"],
            sample_weight_total=selection_weights["xgboost_total"],
            total_baseline_train=split_data["X_train"][:, baseline_index],
            total_baseline_val=split_data["X_val"][:, baseline_index],
            total_baseline_feature_index=baseline_index,
        )

        # Calibration, when explicitly enabled, is fitted only on predictions
        # from the held-out selection model.  A full refit would make these
        # rows in-sample and invalidate calibration.
        calibration = None
        if (
            calibrate
            and getattr(ml_settings, "calibration_enabled", False)
            and split_data["X_val"].shape[0] >= ml_settings.min_games_for_calibration
        ):
            calibrator = Calibrator()
            selection_preds = xgb_trainer.predict(split_data["X_val"])
            if not isinstance(selection_preds, list):
                selection_preds = [selection_preds]
            calibration = calibrator.compute_coefficients(
                val_predictions=selection_preds,
                y_win=split_data["y_win_val"],
                y_spread=split_data["y_spread_val"],
                y_total=split_data["y_total_val"],
            )
            calibrator.save()

        # Refit on ALL data: the split above only chose the stopping point.
        # Without this the shipped model never sees the most recent seasons.
        if getattr(ml_settings, "refit_on_full", True):
            logger.info("Refitting XGBoost on the full dataset...")
            xgb_trainer.refit_full(
                dataset.X, dataset.y_win, dataset.y_spread, dataset.y_total,
                sample_weight_win=target_weights["xgboost_win"],
                sample_weight_spread=target_weights["xgboost_spread"],
                sample_weight_total=target_weights["xgboost_total"],
                total_baseline=total_baseline,
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
                sample_weight=selection_weights["mlp_shared"],
            )
            if getattr(ml_settings, "refit_on_full", True):
                selection_metrics = dict(mlp_metrics)
                mlp_trainer.refit_full(
                    dataset.X,
                    dataset.y_win,
                    dataset.y_spread,
                    dataset.y_total,
                    sample_weight=target_weights["mlp_shared"],
                    best_epoch=int(selection_metrics["best_epoch"]),
                    lr_schedule=selection_metrics.get("lr_schedule"),
                )
                mlp_metrics = selection_metrics
            mlp_path = mlp_trainer.save()
            if not self._validate_mlp_bundle(mlp_trainer, mlp_path, dataset.X):
                logger.warning("MLP bundle failed ONNX parity/completeness checks; publishing XGBoost only")
                mlp_path = None
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
        # Calibration was handled before the production refit above.

        # 6. Publish the release atomically (model + scaler + metadata pin)
        self._write_release_manifest(
            version=ver,
            xgb_path=xgb_path,
            mlp_path=mlp_path,
            dataset=dataset,
            weight_report=weight_report,
            split_data=split_data,
            xgb_trainer=xgb_trainer,
            mlp_metrics=mlp_metrics,
            activate=activate,
        )

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
                "weight_report": weight_report,
                "feature_cache_key": feature_cache_key,
            },
            model_paths={
                "xgboost": xgb_path,
                "mlp": mlp_path,
            },
            release_status="active" if activate else "candidate",
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

    @staticmethod
    def _check_output_directories() -> None:
        """Reject unwritable output paths before computing features or fitting."""
        from pathlib import Path
        from tempfile import TemporaryFile

        for folder in (Path(ml_settings.models_dir) / 'candidates',
                       Path(ml_settings.xgboost_dir), Path(ml_settings.mlp_dir)):
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with TemporaryFile(dir=folder) as probe:
                    probe.write(b'output-check')
                    probe.flush()
            except OSError as exc:
                raise RuntimeError(
                    f'Training output directory is not writable: {folder}. '
                    'Run training in the trainer container with writable model storage.'
                ) from exc

    @staticmethod
    def _write_release_manifest(
        version, xgb_path, mlp_path, dataset, weight_report=None,
        split_data=None, xgb_trainer=None, mlp_metrics=None, activate=False,
    ) -> None:
        """Write a complete candidate manifest; promotion uses the gated CLI.

        Predictors read this to pin the exact versions that were validated
        together, instead of picking "whatever is newest" mid-training.
        ``mlp_version: null`` means the MLP failed its checks and must be
        treated as unavailable rather than falling back to an older model.
        """
        if activate:
            raise ValueError('Direct activation disabled; use gated promotion with evidence')
        import json
        import os
        import hashlib
        from pathlib import Path

        dates = [nba_day(s) for s in dataset.sample_info]
        def _range(rows):
            values = [nba_day(row) for row in rows]
            starts = [row['game_date'] for row in rows]
            return {"from": min(values), "until": max(values),
                    "first_start_utc":min(starts),"last_start_utc":max(starts)} if values else None

        split_at = int(split_data.get("split_idx", len(dataset.X))) if split_data else len(dataset.X)
        xgb_iterations = None
        if xgb_trainer is not None:
            xgb_iterations = {
                name: int(getattr(model, "n_estimators", 0))
                for name, model in (
                    ("win", xgb_trainer.model_win),
                    ("spread", xgb_trainer.model_spread),
                    ("total", xgb_trainer.model_total),
                )
            }
        def _file_hashes(folder):
            if not folder:
                return None
            root = Path(folder)
            return {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(root.iterdir()) if file.is_file()
            }

        required_xgb = {"win_model.json", "spread_model.json", "total_model.json", "metadata.json"}
        if not xgb_path or not required_xgb.issubset({p.name for p in Path(xgb_path).iterdir()}):
            raise RuntimeError("Incomplete XGBoost bundle; active release was not changed")
        if mlp_path:
            required_mlp = {"model.onnx", "scaler.npz", "metadata.json"}
            if not required_mlp.issubset({p.name for p in Path(mlp_path).iterdir()}):
                raise RuntimeError("Incomplete MLP bundle; active release was not changed")

        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "xgboost_version": os.path.basename(xgb_path) if xgb_path else None,
            "mlp_version": os.path.basename(mlp_path) if mlp_path else None,
            "feature_dim": int(dataset.X.shape[1]),
            "feature_names": FEATURE_NAMES,
            "feature_schema_sha256": hashlib.sha256(
                "|".join(FEATURE_NAMES).encode("utf-8")
            ).hexdigest(),
            "sample_manifest_sha256": hashlib.sha256(json.dumps(
                dataset.sample_info, sort_keys=True, default=str
            ).encode("utf-8")).hexdigest(),
            "n_samples": int(len(dataset.X)),
            "training_data_until": max(dates) if dates else None,
            "calibration_enabled": bool(getattr(ml_settings, "calibration_enabled", False)),
            "feature_semantic_version": CACHE_FORMAT_VERSION,
            "training_policy": weight_report,
            "xgboost": {
                "fit_range": _range(dataset.sample_info),
                "target_policies": (weight_report or {}).get("targets", {}),
                "selection_train_range": _range(dataset.sample_info[:split_at]),
                "selection_validation_range": _range(dataset.sample_info[split_at:]),
                "n_estimators": xgb_iterations,
            },
            "mlp": {
                "fit_range": _range(dataset.sample_info) if mlp_path else None,
                "training_policy": (weight_report or {}).get("targets", {}).get("mlp_shared"),
                "selection_validation_range": _range(dataset.sample_info[split_at:]) if mlp_path else None,
                "best_epoch": (mlp_metrics or {}).get("best_epoch"),
                "lr_schedule": (mlp_metrics or {}).get("lr_schedule"),
            },
            "file_hashes": {
                "xgboost": _file_hashes(xgb_path),
                "mlp": _file_hashes(mlp_path),
            },
        }

        root = Path(ml_settings.models_dir)
        candidate_path = root / "candidates" / f"{version}.json"
        candidate_tmp = candidate_path.with_name(candidate_path.name + ".tmp")
        try:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            os.replace(candidate_tmp, candidate_path)
            logger.info("Candidate release %s saved to %s; gated promotion required", version, candidate_path)
        except Exception as e:
            logger.warning("Could not write release manifest: %s", e)
            raise

    @staticmethod
    def _validate_mlp_bundle(mlp_trainer, model_dir: str, X: np.ndarray) -> bool:
        """Compare PyTorch and ONNX on 100 varied rows before publication."""
        import os

        onnx_path = os.path.join(model_dir, "model.onnx")
        scaler_path = os.path.join(model_dir, "scaler.npz")
        if not os.path.exists(onnx_path) or not os.path.exists(scaler_path):
            return False
        try:
            import onnxruntime as ort

            n = min(100, len(X))
            indices = np.linspace(0, len(X) - 1, n, dtype=int)
            sample = X[indices]
            mlp_trainer._onnx_session = None
            torch_predictions = mlp_trainer.predict(sample)
            if not isinstance(torch_predictions, list):
                torch_predictions = [torch_predictions]
            mlp_trainer._onnx_session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            mlp_trainer._onnx_input_name = mlp_trainer._onnx_session.get_inputs()[0].name
            onnx_predictions = mlp_trainer.predict(sample)
            if not isinstance(onnx_predictions, list):
                onnx_predictions = [onnx_predictions]
            single_predictions = [mlp_trainer.predict(row) for row in sample]
            for expected, actual, single in zip(torch_predictions, onnx_predictions, single_predictions):
                if abs(expected.win_prob_home - actual.win_prob_home) > 1e-5:
                    return False
                if abs(expected.predicted_spread - actual.predicted_spread) > 1e-3:
                    return False
                if abs(expected.predicted_total - actual.predicted_total) > 1e-3:
                    return False
                if abs(actual.win_prob_home - single.win_prob_home) > 1e-5:
                    return False
                if abs(actual.predicted_spread - single.predicted_spread) > 1e-3:
                    return False
                if abs(actual.predicted_total - single.predicted_total) > 1e-3:
                    return False
            return True
        except Exception:
            logger.exception("MLP ONNX parity validation failed")
            return False

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
    def _recent_subset(dataset: TrainingDataset, min_date: str) -> TrainingDataset | None:
        """Return only samples whose game date is on/after ``min_date``.

        Used to train the MLP on a homogeneous recent window: older seasons have
        structurally different features (advanced stats and player box scores
        are absent before ~2012, so those columns are constant zero) and the
        network extrapolates wildly when they are mixed with modern games.

        Returns None when the window would be too small to train on.
        """
        if not dataset.sample_info:
            return None

        dates = [nba_day(s) for s in dataset.sample_info]
        keep = np.array([d >= min_date for d in dates])
        n = int(keep.sum())

        if n < 1000:
            return None
        if n == len(keep):
            return dataset

        return TrainingDataset(
            X=dataset.X[keep],
            y_win=dataset.y_win[keep],
            y_spread=dataset.y_spread[keep],
            y_total=dataset.y_total[keep],
            sample_info=[s for s, k in zip(dataset.sample_info, keep) if k],
        )

    @staticmethod
    def _chronological_split(
        dataset: TrainingDataset,
        val_ratio: float = 0.15,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        """Split dataset chronologically (last val_ratio for validation).

        This mirrors real-world usage: we train on past data, validate on more recent data.
        """
        n = len(dataset.X)
        split_idx = complete_day_split(dataset.sample_info,val_ratio)

        X_train = dataset.X[:split_idx]
        X_val = dataset.X[split_idx:]

        weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else np.ones(n, dtype=np.float32)
        )
        return {
            "split_idx": split_idx,
            "X_train": X_train,
            "X_val": X_val,
            "y_win_train": dataset.y_win[:split_idx],
            "y_win_val": dataset.y_win[split_idx:],
            "y_spread_train": dataset.y_spread[:split_idx],
            "y_spread_val": dataset.y_spread[split_idx:],
            "y_total_train": dataset.y_total[:split_idx],
            "y_total_val": dataset.y_total[split_idx:],
            "sample_weight_train": weights[:split_idx],
            "sample_weight_val": weights[split_idx:],
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
