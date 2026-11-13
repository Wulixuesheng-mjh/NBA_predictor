"""Fast weighted MLP refit and PyTorch/ONNX parity smoke test."""

import os
import tempfile

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_DIM
from ml.src.models.mlp_model import MLPTrainer
from ml.src.trainer import Trainer


def main() -> None:
    rng = np.random.default_rng(20261023)
    rows = 240
    X = rng.normal(size=(rows, FEATURE_DIM)).astype(np.float32)
    X[:, 5] = 0.0  # unseen feature must not gain a random effect after fitting
    spread = (4.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, 3, rows)).astype(np.float32)
    spread -= spread.mean()
    win = (spread + rng.normal(0, 2, rows) > 0).astype(np.float32)
    total = (225.0 + 6.0 * X[:, 2] + rng.normal(0, 4, rows)).astype(np.float32)
    weights = np.linspace(0.25, 1.75, rows, dtype=np.float32)
    weights[0] = 0.0
    X[0] = 10000.0  # excluded window must not contaminate standardization
    split = 200

    original_epochs = ml_settings.mlp_epochs
    original_patience = ml_settings.mlp_patience
    original_batch = ml_settings.mlp_batch_size
    ml_settings.mlp_epochs = 8
    ml_settings.mlp_patience = 8
    ml_settings.mlp_batch_size = 64
    try:
        with tempfile.TemporaryDirectory() as directory:
            trainer = MLPTrainer(models_dir=directory, version="smoke", device="cpu")
            metrics = trainer.train(
                X[:split], win[:split], spread[:split], total[:split],
                X[split:], win[split:], spread[split:], total[split:],
                sample_weight=weights[:split],
            )
            assert metrics["best_epoch"] >= 1
            assert trainer.model.input_normalization == "none"
            assert metrics["train_samples"] == split - 1
            assert np.max(np.abs(trainer._scaler_mean)) < 1.0
            expected_mean = np.average(X[1:split], axis=0, weights=weights[1:split])
            np.testing.assert_allclose(trainer._scaler_mean, expected_mean, atol=1e-5)
            assert len(metrics["lr_schedule"]) == metrics["best_epoch"]
            trainer.refit_full(
                X, win, spread, total,
                sample_weight=weights,
                best_epoch=metrics["best_epoch"],
                lr_schedule=metrics["lr_schedule"],
            )
            model_dir = trainer.save()
            assert os.path.exists(os.path.join(model_dir, "model.onnx"))
            assert Trainer._validate_mlp_bundle(trainer, model_dir, X)
            restored = MLPTrainer(models_dir=directory, version="smoke", device="cpu")
            restored.load(model_dir, input_dim=FEATURE_DIM)
            predictions = restored.predict(X[:10])
            assert len(predictions) == 10
            assert all(0.0 <= item.win_prob_home <= 1.0 for item in predictions)
            assert all(np.isfinite(item.predicted_total) for item in predictions)
            changed = X[1:11].copy()
            changed[:, 5] = 1000.0
            baseline = restored.predict(X[1:11])
            unseen = restored.predict(changed)
            np.testing.assert_allclose(
                [p.predicted_total for p in baseline], [p.predicted_total for p in unseen], atol=1e-6
            )
            print({
                "status": "PASS",
                "feature_dim": FEATURE_DIM,
                "best_epoch": metrics["best_epoch"],
                "onnx": True,
            })
    finally:
        ml_settings.mlp_epochs = original_epochs
        ml_settings.mlp_patience = original_patience
        ml_settings.mlp_batch_size = original_batch


if __name__ == "__main__":
    main()
