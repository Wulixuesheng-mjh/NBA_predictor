"""PyTorch MLP model for NBA predictions — auxiliary / comparison model.

Architecture:
    Input (FEATURE_DIM) → Linear(128) + BN + ReLU + Dropout(0.3)
                         → Linear(64) + BN + ReLU + Dropout(0.3)
                         → Linear(32) + BN + ReLU
                         → 3 heads:
                             Win Head:    Linear(32, 1) + Sigmoid
                             Spread Head: Linear(32, 1)
                             Total Head:  Linear(32, 1)

Training:
    - AdamW optimizer with weight decay
    - ReduceLROnPlateau scheduler
    - Early stopping (patience=30)
    - Combined loss: BCE(win) + Huber(spread) + Huber(total)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_DIM

logger = logging.getLogger(__name__)

# ── Lazy torch imports ─────────────────────────────────
# torch is ONLY needed for training and PyTorch inference.
# ONNX Runtime handles production inference without torch.
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug("torch not installed — MLP training and PyTorch inference disabled")
    # Create dummy placeholders so the module can be imported
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


@dataclass
class MLPPrediction:
    """Result from MLP inference."""

    win_prob_home: float
    win_prob_away: float
    predicted_spread: float
    spread_confidence: float
    predicted_total: float
    total_confidence: float
    model_version: str


if _TORCH_AVAILABLE:

    class NBA_MLP(nn.Module):
        """Three-head MLP for win/spread/total prediction.

        Architecture scales regularization by dataset size:
          - Small data (~2K): dropout=0.35, more aggressive BN
          - Full data (~30K): dropout=0.2, lighter regularization
        """

        def __init__(self, input_dim: int = FEATURE_DIM, dropout: float = 0.2):
            super().__init__()

            self.input_bn = nn.BatchNorm1d(input_dim)

            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(128, 96),
                nn.BatchNorm1d(96),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layer3 = nn.Sequential(
                nn.Linear(96, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
            )
            self.layer4 = nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )

            # Heads — no activation on regression heads
            self.win_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            self.spread_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(16, 1),
            )
            self.total_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Returns:
                (win_prob, spread_pred, total_pred)
            """
            x = self.input_bn(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            shared = self.layer4(x)

            win = self.win_head(shared)
            spread = self.spread_head(shared)
            total = self.total_head(shared)
            return win, spread, total

else:
    # Placeholder — ONNX Runtime doesn't need the PyTorch model class
    NBA_MLP = None  # type: ignore


class MLPTrainer:
    """Trains and manages the PyTorch MLP model."""

    def __init__(
        self,
        models_dir: str | None = None,
        version: str | None = None,
        device: str | None = None,
    ):
        self.models_dir = models_dir or ml_settings.mlp_dir
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        if _TORCH_AVAILABLE:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"  # ONNX-only mode — device is irrelevant

        self.model: NBA_MLP | None = None
        self._onnx_session = None  # ONNX Runtime inference session (production)
        self._onnx_input_name: str | None = None
        self._scaler_mean: np.ndarray | None = None  # Feature standardization
        self._scaler_std: np.ndarray | None = None
        logger.info("MLP using device: %s", self.device)

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
    ) -> dict:
        """Train the MLP model. Requires PyTorch.

        Returns dict with training metrics.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for MLP training. "
                "Install torch to train new models."
            )
        import time
        start_time = time.time()

        input_dim = X_train.shape[1]

        # ── Feature standardization (critical for neural nets) ──
        # XGBoost is scale-invariant, but MLP needs features ≈ N(0, 1).
        # Fit on training data; apply to both train and val.
        # Use robust statistics: clip extreme outliers (>5 sigma) before
        # computing mean/std — sentinel values like 999 would otherwise
        # corrupt the scaler and cause NaN/exploding outputs at inference.
        X_for_scaler = X_train.copy()
        # Clip values beyond 5 std from mean (iteratively, 2 passes)
        for _ in range(2):
            m = X_for_scaler.mean(axis=0)
            s = X_for_scaler.std(axis=0)
            s[s < 1e-8] = 1.0
            lower = m - 5 * s
            upper = m + 5 * s
            X_for_scaler = np.clip(X_for_scaler, lower, upper)

        self._scaler_mean = X_for_scaler.mean(axis=0).astype(np.float32)
        self._scaler_std = X_for_scaler.std(axis=0).astype(np.float32)
        # Prevent division by zero for constant features
        self._scaler_std[self._scaler_std < 1e-8] = 1.0

        X_train_scaled = (X_train - self._scaler_mean) / self._scaler_std
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = (X_val - self._scaler_mean) / self._scaler_std

        # Initialize model (use config dropout for full-dataset training)
        self.model = NBA_MLP(
            input_dim=input_dim,
            dropout=ml_settings.mlp_dropout,
        ).to(self.device)

        # Convert data to tensors (use scaled features)
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_win_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_spread_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_total_train, dtype=torch.float32).unsqueeze(1),
        )

        batch_size = min(ml_settings.mlp_batch_size, max(16, len(X_train) // 2))
        batch_size = max(batch_size, 4)  # Ensure minimum batch size for BatchNorm
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Validation tensors (scaled)
        val_data = None
        if X_val_scaled is not None:
            val_data = (
                torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device),
                torch.tensor(y_win_val, dtype=torch.float32).unsqueeze(1).to(self.device),
                torch.tensor(y_spread_val, dtype=torch.float32).unsqueeze(1).to(self.device),
                torch.tensor(y_total_val, dtype=torch.float32).unsqueeze(1).to(self.device),
            )

        # Optimizer and schedulers — lighter regularization for full dataset
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=ml_settings.mlp_learning_rate,
            weight_decay=ml_settings.mlp_weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15,
        )

        # Loss functions — wider Huber deltas for smoother gradients on large data
        win_criterion = nn.BCELoss()
        spread_criterion = nn.HuberLoss(delta=ml_settings.mlp_huber_delta_spread)
        total_criterion = nn.HuberLoss(delta=ml_settings.mlp_huber_delta_total)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        epochs = ml_settings.mlp_epochs
        train_losses: list[float] = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_win, batch_spread, batch_total in train_loader:
                batch_X = batch_X.to(self.device)
                batch_win = batch_win.to(self.device)
                batch_spread = batch_spread.to(self.device)
                batch_total = batch_total.to(self.device)

                optimizer.zero_grad()

                pred_win, pred_spread, pred_total = self.model(batch_X)

                loss_win = win_criterion(pred_win, batch_win)
                loss_spread = spread_criterion(pred_spread, batch_spread)
                loss_total = total_criterion(pred_total, batch_total)

                # Combined loss (win has lower weight since it's easier)
                loss = 0.3 * loss_win + 0.35 * loss_spread + 0.35 * loss_total

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if val_data is not None:
                val_loss = self._compute_val_loss(val_data, win_criterion, spread_criterion, total_criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self._best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= ml_settings.mlp_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
            else:
                scheduler.step(avg_train_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f, val_loss=%s",
                    epoch + 1, epochs, avg_train_loss,
                    f"{best_val_loss:.4f}" if val_data is not None else "N/A",
                )

        # Restore best model state
        if val_data is not None and hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        training_time = time.time() - start_time

        # Final evaluation
        metrics = {}
        if val_data is not None:
            self.model.eval()
            with torch.no_grad():
                Xv, yw, ysp, yt = val_data
                pw, psp, pt = self.model(Xv)

                win_preds = (pw.cpu().numpy() > 0.5).astype(float)
                metrics["win_accuracy"] = float((win_preds == yw.cpu().numpy()).mean())

                sp_preds = psp.cpu().numpy()
                metrics["spread_mae"] = float(np.mean(np.abs(sp_preds - ysp.cpu().numpy())))
                metrics["spread_rmse"] = float(np.sqrt(np.mean((sp_preds - ysp.cpu().numpy()) ** 2)))

                t_preds = pt.cpu().numpy()
                metrics["total_mae"] = float(np.mean(np.abs(t_preds - yt.cpu().numpy())))
                metrics["total_rmse"] = float(np.sqrt(np.mean((t_preds - yt.cpu().numpy()) ** 2)))

        metrics["train_samples"] = len(X_train)
        metrics["val_samples"] = len(X_val) if X_val is not None else 0
        metrics["training_time_seconds"] = training_time
        metrics["epochs_completed"] = epoch + 1
        metrics["best_val_loss"] = float(best_val_loss) if val_data is not None else float(train_losses[-1])

        logger.info(
            "MLP training complete in %.1fs (%d epochs). "
            "Win acc=%.3f, Spread MAE=%.2f, Total MAE=%.2f",
            training_time, metrics["epochs_completed"],
            metrics.get("win_accuracy", 0.0),
            metrics.get("spread_mae", 0.0),
            metrics.get("total_mae", 0.0),
        )

        # ── Post-training sanity check ──
        # Prevents deploying models with weight explosion (e.g. v20260812_050800
        # which had 63% validation accuracy but produced Spread=-161, Total=4535).
        if X_val is not None and len(X_val) > 0:
            self._run_sanity_check(X_val)

        return metrics

    def predict(self, X: np.ndarray) -> MLPPrediction:
        """Run inference via ONNX Runtime (production) or PyTorch (dev).

        Args:
            X: Feature vector(s), shape (n_samples, FEATURE_DIM) or (FEATURE_DIM,).

        Returns:
            MLPPrediction (if single sample) or list of them.
        """
        # Ensure 2D — derive single-sample flag AFTER reshape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        single = X.shape[0] == 1

        # ── Apply feature standardization ──
        if self._scaler_mean is not None and self._scaler_std is not None:
            X = (X - self._scaler_mean) / self._scaler_std
            # Clip to reasonable range (prevents extreme values from
            # sentinel/defaults that differ between train and inference)
            X = np.clip(X, -5.0, 5.0)

        # ── Ensure float32 for ONNX Runtime ──
        X = X.astype(np.float32)

        # ONNX Runtime path (production)
        if self._onnx_session is not None:
            win_np, spread_np, total_np = self._predict_onnx(X)
        elif self.model is not None:
            # PyTorch path (development / training container)
            win_np, spread_np, total_np = self._predict_torch(X)
        else:
            raise RuntimeError(
                "MLP model not loaded. Call load() first. "
                "Neither ONNX session nor PyTorch model available."
            )

        results = []
        for i in range(len(X)):
            results.append(
                MLPPrediction(
                    win_prob_home=float(win_np[i, 0]),
                    win_prob_away=float(1.0 - win_np[i, 0]),
                    predicted_spread=float(spread_np[i, 0]),
                    spread_confidence=float(abs(win_np[i, 0] - 0.5) * 2),
                    predicted_total=float(total_np[i, 0]),
                    total_confidence=float(abs(win_np[i, 0] - 0.5) * 2),
                    model_version=self.version,
                )
            )

        return results[0] if single else results

    def _predict_onnx(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ONNX inference."""
        ort_inputs = {self._onnx_input_name: X.astype(np.float32)}
        outputs = self._onnx_session.run(None, ort_inputs)
        # outputs order matches: win_prob, predicted_spread, predicted_total
        return outputs[0], outputs[1], outputs[2]

    def _predict_torch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PyTorch inference."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            win, spread, total = self.model(X_tensor)
        return win.cpu().numpy(), spread.cpu().numpy(), total.cpu().numpy()

    def _compute_val_loss(
        self,
        val_data: tuple,
        win_criterion: nn.Module,
        spread_criterion: nn.Module,
        total_criterion: nn.Module,
    ) -> float:
        """Compute combined validation loss."""
        self.model.eval()
        with torch.no_grad():
            Xv, yw, ysp, yt = val_data
            pw, psp, pt = self.model(Xv)
            loss = (
                0.3 * win_criterion(pw, yw)
                + 0.35 * spread_criterion(psp, ysp)
                + 0.35 * total_criterion(pt, yt)
            )
        return loss.item()

    def _run_sanity_check(self, X_val: np.ndarray) -> None:
        """Validate model outputs are within reasonable ranges.

        Catches weight explosion / collapse before models are persisted.
        Thresholds are deliberately generous — a healthy model will never
        come close to these bounds.
        """
        predictions = self.predict(X_val)
        if not isinstance(predictions, list):
            predictions = [predictions]

        spreads = [p.predicted_spread for p in predictions]
        totals = [p.predicted_total for p in predictions]
        win_probs = [p.win_prob_home for p in predictions]

        issues: list[str] = []

        # 1. Spread should be within ±50 (NBA record is 73; typical range ±30)
        max_abs_spread = max(abs(s) for s in spreads)
        if max_abs_spread > 50:
            issues.append(
                f"max |spread| = {max_abs_spread:.1f} exceeds 50 "
                f"(range [{min(spreads):.1f}, {max(spreads):.1f}])"
            )

        # 2. Total should be in [180, 350] (NBA range ~150–370)
        t_min, t_max = min(totals), max(totals)
        if t_min < 180 or t_max > 350:
            issues.append(
                f"total range = [{t_min:.0f}, {t_max:.0f}] "
                f"outside [180, 350]"
            )

        # 3. All spreads having the same sign → model is biased or broken
        if all(s > 0 for s in spreads) or all(s < 0 for s in spreads):
            sign_str = "positive" if all(s > 0 for s in spreads) else "negative"
            issues.append(f"all {len(spreads)} spreads are {sign_str} (model biased)")

        # 4. Win probabilities should have reasonable variance
        unique_probs = len(set(round(w, 3) for w in win_probs))
        if unique_probs < 3:
            issues.append(
                f"only {unique_probs} unique win probabilities "
                f"(model outputs may be collapsed)"
            )

        if issues:
            raise ValueError(
                f"MLP sanity check FAILED:\n  - " + "\n  - ".join(issues)
            )

        logger.info(
            "MLP sanity check PASSED: %d predictions — "
            "spread [%.1f, %.1f], total [%.0f, %.0f], %d unique win probs",
            len(predictions), min(spreads), max(spreads),
            t_min, t_max, unique_probs,
        )

    def save(self, version: str | None = None) -> str:
        """Save model to disk (PyTorch state_dict + ONNX for production).

        ONNX export allows the backend Docker image to run MLP inference
        without installing PyTorch (~800 MB). ONNX Runtime is ~10 MB.
        """
        ver = version or self.version
        model_dir = os.path.join(self.models_dir, ver)
        os.makedirs(model_dir, exist_ok=True)

        if self.model:
            # 1. Save PyTorch state_dict (training container only)
            torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pt"))

            # 2. Export to ONNX for production inference
            try:
                self._export_onnx(model_dir)
                logger.info("ONNX model exported to %s/model.onnx", model_dir)
            except Exception as e:
                logger.warning("ONNX export failed (non-fatal): %s", e)

        # 3. Save feature scaler (mean + std for standardization)
        if self._scaler_mean is not None and self._scaler_std is not None:
            scaler_path = os.path.join(model_dir, "scaler.npz")
            np.savez(scaler_path, mean=self._scaler_mean, std=self._scaler_std)
            logger.info("Scaler saved to %s", scaler_path)

        metadata = {
            "version": ver,
            "created_at": datetime.now().isoformat(),
            "device": str(self.device),
            "input_dim": self.model.layer1[0].in_features if self.model else FEATURE_DIM,
            "has_onnx": os.path.exists(os.path.join(model_dir, "model.onnx")),
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved MLP model to %s", model_dir)
        return model_dir

    def _export_onnx(self, model_dir: str) -> None:
        """Export the trained model to ONNX format. Requires PyTorch.

        Uses a dummy input matching the expected feature dimension.
        The exported model has 3 outputs: win, spread, total.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to export ONNX model.")
        self.model.eval()
        input_dim = self.model.layer1[0].in_features
        dummy_input = torch.randn(1, input_dim, device=self.device)

        onnx_path = os.path.join(model_dir, "model.onnx")
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["win_prob", "predicted_spread", "predicted_total"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "win_prob": {0: "batch_size"},
                "predicted_spread": {0: "batch_size"},
                "predicted_total": {0: "batch_size"},
            },
        )

    def load(self, version: str | None = None, input_dim: int = FEATURE_DIM) -> str:
        """Load model from disk. Prefers ONNX Runtime, falls back to PyTorch.

        ONNX Runtime is the production path — no PyTorch dependency needed.
        PyTorch is the fallback for training/development environments.
        """
        ver = version or self._find_latest_version()
        if ver is None:
            raise FileNotFoundError(f"No MLP model found in {self.models_dir}")

        model_dir = os.path.join(self.models_dir, ver)
        onnx_path = os.path.join(model_dir, "model.onnx")

        # Try ONNX Runtime first (production path)
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                self._onnx_input_name = self._onnx_session.get_inputs()[0].name
                self.model = None  # Not using PyTorch
                self.version = ver
                self._load_scaler(model_dir)
                logger.info(
                    "Loaded MLP model v%s via ONNX Runtime (%s)",
                    ver, self._onnx_session.get_providers()[0],
                )
                return ver
            except ImportError:
                logger.debug("onnxruntime not installed, trying PyTorch...")
            except Exception as e:
                logger.warning("ONNX Runtime failed: %s, trying PyTorch...", e)

        # Fall back to PyTorch
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "Cannot load MLP model: neither onnxruntime nor torch is installed. "
                "Install `onnxruntime` for production or `torch` for development."
            )
        try:
            self.model = NBA_MLP(input_dim=input_dim).to(self.device)
            state = torch.load(
                os.path.join(model_dir, "model.pt"),
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state)
            self.model.eval()
            self._onnx_session = None
            self.version = ver
            self._load_scaler(model_dir)
            logger.info("Loaded MLP model v%s via PyTorch", ver)
            return ver
        except Exception as e:
            raise RuntimeError(f"Failed to load MLP model v{ver}: {e}")

    def _load_scaler(self, model_dir: str) -> None:
        """Load feature scaler from disk if available."""
        scaler_path = os.path.join(model_dir, "scaler.npz")
        if os.path.exists(scaler_path):
            data = np.load(scaler_path)
            self._scaler_mean = data["mean"]
            self._scaler_std = data["std"]
            logger.info("Loaded scaler from %s", scaler_path)
        else:
            self._scaler_mean = None
            self._scaler_std = None
            logger.debug("No scaler found at %s — using raw features", scaler_path)

    def _find_latest_version(self) -> str | None:
        """Find the most recent model version on disk.

        Checks for model.onnx (preferred) or model.pt files.
        """
        if not os.path.isdir(self.models_dir):
            return None
        versions = [
            d for d in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, d))
            and (
                os.path.exists(os.path.join(self.models_dir, d, "model.onnx"))
                or os.path.exists(os.path.join(self.models_dir, d, "model.pt"))
            )
        ]
        if not versions:
            return None
        return sorted(versions)[-1]
