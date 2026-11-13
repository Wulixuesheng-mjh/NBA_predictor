"""Fast synthetic fit/refit/persistence smoke test for the training pipeline."""

import tempfile

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_DIM, FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer


def main() -> None:
    rng = np.random.default_rng(20261023)
    rows = 360
    X = rng.normal(size=(rows, FEATURE_DIM)).astype(np.float32)
    baseline_index = FEATURE_NAMES.index("shrunk_expected_total")
    baseline = rng.normal(225.0, 8.0, size=rows).astype(np.float32)
    X[:, baseline_index] = baseline
    spread = (3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 4, rows)).astype(np.float32)
    win = (spread > 0).astype(np.float32)
    total = (baseline + 2.0 * X[:, 2] + rng.normal(0, 3, rows)).astype(np.float32)
    weights = np.linspace(0.25, 1.75, rows, dtype=np.float32)
    split = 300

    original = {
        "xgboost_n_estimators": ml_settings.xgboost_n_estimators,
        "xgboost_early_stopping_rounds": ml_settings.xgboost_early_stopping_rounds,
        "xgboost_max_depth": ml_settings.xgboost_max_depth,
    }
    ml_settings.xgboost_n_estimators = 30
    ml_settings.xgboost_early_stopping_rounds = 5
    ml_settings.xgboost_max_depth = 3
    try:
        with tempfile.TemporaryDirectory() as directory:
            trainer = XGBoostTrainer(models_dir=directory, version="smoke")
            metrics = trainer.train(
                X[:split], win[:split], spread[:split], total[:split],
                X[split:], win[split:], spread[split:], total[split:],
                feature_names=FEATURE_NAMES,
                sample_weight_win=weights[:split],
                sample_weight_spread=weights[:split][::-1].copy(),
                sample_weight_total=np.ones(split, dtype=np.float32),
                total_baseline_train=baseline[:split],
                total_baseline_val=baseline[split:],
                total_baseline_feature_index=baseline_index,
            )
            assert metrics.val_samples == rows - split
            assert np.isfinite(metrics.total_mae)
            trainer.refit_full(
                X, win, spread, total,
                sample_weight_win=weights,
                sample_weight_spread=weights[::-1].copy(),
                sample_weight_total=np.ones(rows, dtype=np.float32),
                total_baseline=baseline,
            )
            before = trainer.predict(X[:5])
            model_dir = trainer.save()
            restored = XGBoostTrainer(models_dir=directory, version="smoke")
            restored.load(model_dir)
            after = restored.predict(X[:5])
            if not isinstance(before, list):
                before = [before]
            if not isinstance(after, list):
                after = [after]
            for expected, actual in zip(before, after):
                assert abs(expected.win_prob_home - actual.win_prob_home) < 1e-6
                assert abs(expected.predicted_spread - actual.predicted_spread) < 1e-6
                assert abs(expected.predicted_total - actual.predicted_total) < 1e-6
            print({
                "status": "PASS",
                "feature_dim": FEATURE_DIM,
                "total_mode": trainer._total_mode,
                "total_mae": round(metrics.total_mae, 4),
            })
    finally:
        for name, value in original.items():
            setattr(ml_settings, name, value)


if __name__ == "__main__":
    main()
