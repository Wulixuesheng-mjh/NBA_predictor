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

        def __init__(self, input_dim: int = FEATURE_DIM, dropout: float = 0.2,
                     input_normalization: str = "batchnorm"):
            super().__init__()

            if input_normalization not in ("batchnorm", "none"):
                raise ValueError("Unknown MLP input normalization")
            self.input_normalization = input_normalization
            self.input_bn = nn.BatchNorm1d(input_dim) if input_normalization == "batchnorm" else nn.Identity()

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
        self._active_features: np.ndarray | None = None
        # Target standardization (spread ≈ ±10, total ≈ 220 — mixing those
        # scales in one loss lets the total head dominate the gradients)
        self._y_spread_mean: float | None = None
        self._y_spread_std: float | None = None
        self._y_total_mean: float | None = None
        self._y_total_std: float | None = None
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
        sample_weight: np.ndarray | None = None,
        epochs_override: int | None = None,
        lr_schedule: list[float] | None = None,
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
        torch.manual_seed(ml_settings.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ml_settings.random_seed)

        # Targets may use different windows in the orchestrator's union. Zero
        # weight rows must not affect the scaler or BatchNorm running state.
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (len(X_train),) or not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite, non-negative and aligned")
            keep = sample_weight > 0
            if not keep.any():
                raise ValueError("sample_weight has no positive training rows")
            X_train = X_train[keep]
            y_win_train = y_win_train[keep]
            y_spread_train = y_spread_train[keep]
            y_total_train = y_total_train[keep]
            from ml.src.training_policy import normalize_fit_weights
            sample_weight = normalize_fit_weights(sample_weight[keep], int(keep.sum()))

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
            m = np.average(X_for_scaler, axis=0, weights=sample_weight)
            s = np.sqrt(np.average(np.square(X_for_scaler - m), axis=0, weights=sample_weight))
            s[s < 1e-8] = 1.0
            lower = m - 5 * s
            upper = m + 5 * s
            X_for_scaler = np.clip(X_for_scaler, lower, upper)

        scaler_mean = np.average(X_for_scaler, axis=0, weights=sample_weight)
        self._scaler_mean = scaler_mean.astype(np.float32)
        self._scaler_std = np.sqrt(np.average(
            np.square(X_for_scaler - scaler_mean), axis=0, weights=sample_weight
        )).astype(np.float32)
        # Unseen/constant training features have no learned effect. Freeze them
        # in validation and inference instead of exposing random input weights
        # when a later injury report or new season first changes their value.
        self._active_features = self._scaler_std >= 1e-8
        # Prevent division by zero for constant features
        self._scaler_std[self._scaler_std < 1e-8] = 1.0

        X_train_scaled = (X_train - self._scaler_mean) / self._scaler_std
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = (X_val - self._scaler_mean) / self._scaler_std

        # Match inference: predict() clips standardised inputs to ±5σ. Without
        # the same clip here, a handful of extreme rows (e.g. rest_days = 295
        # after the off-season, starters/bench ratio = 392) reach the network
        # during training and the loss diverges — the model then emits totals
        # of 5 or 1000 and gets rejected by the sanity check.
        X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
        X_train_scaled[:, ~self._active_features] = 0.0
        if X_val_scaled is not None:
            X_val_scaled = np.clip(X_val_scaled, -5.0, 5.0)
            X_val_scaled[:, ~self._active_features] = 0.0

        # ── Target standardization (spread/total only; win is already 0-1) ──
        # The three heads share one loss: raw spread (±10) and raw total (~220)
        # have wildly different scales, so the total head dominates the gradient
        # and the regression heads drift. Train on z-scored targets and invert
        # in predict().
        target_weights = (
            np.asarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else np.ones(len(y_spread_train), dtype=np.float64)
        )

        def _weighted_stats(values: np.ndarray) -> tuple[float, float]:
            values = np.asarray(values, dtype=np.float64)
            mean = float(np.average(values, weights=target_weights))
            variance = float(np.average(np.square(values - mean), weights=target_weights))
            return mean, float(np.sqrt(variance)) or 1.0

        self._y_spread_mean, self._y_spread_std = _weighted_stats(y_spread_train)
        self._y_total_mean, self._y_total_std = _weighted_stats(y_total_train)

        y_spread_train = (np.asarray(y_spread_train, dtype=np.float64) - self._y_spread_mean) / self._y_spread_std
        y_total_train = (np.asarray(y_total_train, dtype=np.float64) - self._y_total_mean) / self._y_total_std
        if y_spread_val is not None:
            y_spread_val = (np.asarray(y_spread_val, dtype=np.float64) - self._y_spread_mean) / self._y_spread_std
        if y_total_val is not None:
            y_total_val = (np.asarray(y_total_val, dtype=np.float64) - self._y_total_mean) / self._y_total_std

        # Initialize model (use config dropout for full-dataset training)
        self.model = NBA_MLP(
            input_dim=input_dim,
            dropout=ml_settings.mlp_dropout,
            input_normalization=ml_settings.mlp_input_normalization,
        ).to(self.device)

        # Convert data to tensors (use scaled features)
        if sample_weight is None:
            sample_weight = np.ones(len(X_train), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != (len(X_train),):
            raise ValueError("sample_weight must have one value per training row")
        if np.any(sample_weight < 0) or not np.isfinite(sample_weight).all():
            raise ValueError("sample_weight must be finite and non-negative")

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_win_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_spread_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_total_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(sample_weight, dtype=torch.float32).unsqueeze(1),
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

        # Loss functions — Huber deltas are expressed in raw points, so convert
        # them to the z-scored target space we train on.
        win_criterion = nn.BCELoss(reduction="none")
        spread_criterion = nn.HuberLoss(
            delta=ml_settings.mlp_huber_delta_spread / self._y_spread_std,
            reduction="none",
        )
        total_criterion = nn.HuberLoss(
            delta=ml_settings.mlp_huber_delta_total / self._y_total_std,
            reduction="none",
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        epochs = int(epochs_override or ml_settings.mlp_epochs)
        train_losses: list[float] = []
        lr_history: list[float] = []
        best_epoch = epochs

        for epoch in range(epochs):
            if lr_schedule:
                lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
                for group in optimizer.param_groups:
                    group["lr"] = lr
            lr_history.append(float(optimizer.param_groups[0]["lr"]))
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_win, batch_spread, batch_total, batch_weight in train_loader:
                batch_X = batch_X.to(self.device)
                batch_win = batch_win.to(self.device)
                batch_spread = batch_spread.to(self.device)
                batch_total = batch_total.to(self.device)
                batch_weight = batch_weight.to(self.device)

                optimizer.zero_grad()

                pred_win, pred_spread, pred_total = self.model(batch_X)

                denom = torch.clamp(batch_weight.sum(), min=1e-8)
                loss_win = (win_criterion(pred_win, batch_win) * batch_weight).sum() / denom
                loss_spread = (spread_criterion(pred_spread, batch_spread) * batch_weight).sum() / denom
                loss_total = (total_criterion(pred_total, batch_total) * batch_weight).sum() / denom

                # Combined loss (win has lower weight since it's easier)
                loss = 0.3 * loss_win + 0.35 * loss_spread + 0.35 * loss_total

                loss.backward()
                # Guard against the occasional explosive batch (a single
                # mis-scaled feature used to diverge the whole run)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    best_epoch = epoch + 1
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
                if not lr_schedule:
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

                # Targets are z-scored during training — invert before metrics
                sp_preds = psp.cpu().numpy() * self._y_spread_std + self._y_spread_mean
                sp_true = ysp.cpu().numpy() * self._y_spread_std + self._y_spread_mean
                metrics["spread_mae"] = float(np.mean(np.abs(sp_preds - sp_true)))
                metrics["spread_rmse"] = float(np.sqrt(np.mean((sp_preds - sp_true) ** 2)))

                t_preds = pt.cpu().numpy() * self._y_total_std + self._y_total_mean
                t_true = yt.cpu().numpy() * self._y_total_std + self._y_total_mean
                metrics["total_mae"] = float(np.mean(np.abs(t_preds - t_true)))
                metrics["total_rmse"] = float(np.sqrt(np.mean((t_preds - t_true) ** 2)))

        metrics["train_samples"] = len(X_train)
        metrics["val_samples"] = len(X_val) if X_val is not None else 0
        metrics["training_time_seconds"] = training_time
        metrics["epochs_completed"] = epoch + 1
        metrics["best_epoch"] = int(best_epoch)
        metrics["lr_schedule"] = lr_history[:best_epoch]
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

    def refit_full(
        self,
        X: np.ndarray,
        y_win: np.ndarray,
        y_spread: np.ndarray,
        y_total: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        best_epoch: int,
        lr_schedule: list[float] | None = None,
    ) -> dict:
        """Reinitialize and fit the production MLP on the full allowed window."""
        if best_epoch < 1:
            raise ValueError("best_epoch must be positive")
        return self.train(
            X_train=X,
            y_win_train=y_win,
            y_spread_train=y_spread,
            y_total_train=y_total,
            sample_weight=sample_weight,
            epochs_override=best_epoch,
            lr_schedule=lr_schedule,
        )

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
            if self._active_features is not None:
                X[:, ~self._active_features] = 0.0

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

        # Invert target standardization (no-op for models trained on raw targets)
        sp_std = self._y_spread_std if self._y_spread_std is not None else 1.0
        sp_mean = self._y_spread_mean if self._y_spread_mean is not None else 0.0
        t_std = self._y_total_std if self._y_total_std is not None else 1.0
        t_mean = self._y_total_mean if self._y_total_mean is not None else 0.0

        results = []
        for i in range(len(X)):
            results.append(
                MLPPrediction(
                    win_prob_home=float(win_np[i, 0]),
                    win_prob_away=float(1.0 - win_np[i, 0]),
                    predicted_spread=float(spread_np[i, 0] * sp_std + sp_mean),
                    spread_confidence=float(abs(win_np[i, 0] - 0.5) * 2),
                    predicted_total=float(total_np[i, 0] * t_std + t_mean),
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
        # Must be in eval mode: BatchNorm would otherwise use the batch's own
        # statistics and Dropout would stay active, which produced wildly
        # out-of-range outputs whenever predict() was called between epochs.
        self.model.eval()
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
            ).mean()
        return loss.item()

    def _run_sanity_check(self, X_val: np.ndarray) -> None:
        """Validate model outputs are within reasonable ranges.

        Catches weight explosion / collapse before models are persisted.

        Two tiers, because the validation set is thousands of games and a
        handful of out-of-distribution rows (the first games after the
        off-season carry rest_days ≈ 100) are normal:

          * absolute bounds — any single prediction beyond these means the
            weights really did explode;
          * typical bounds — applied to the 1st/99th percentile, so a couple of
            odd rows cannot veto an otherwise healthy model.
        """
        predictions = self.predict(X_val)
        if not isinstance(predictions, list):
            predictions = [predictions]

        spreads = np.array([p.predicted_spread for p in predictions])
        totals = np.array([p.predicted_total for p in predictions])
        win_probs = [p.win_prob_home for p in predictions]

        issues: list[str] = []

        s_min, s_max = float(spreads.min()), float(spreads.max())
        t_min, t_max = float(totals.min()), float(totals.max())
        s_p01, s_p99 = (float(v) for v in np.percentile(np.abs(spreads), [1, 99]))
        t_p01, t_p99 = (float(v) for v in np.percentile(totals, [1, 99]))

        # 1. Absolute blow-up (weight explosion)
        if s_max > 80 or s_min < -80:
            issues.append(f"spread range [{s_min:.1f}, {s_max:.1f}] exceeds ±80")
        if t_min < 50 or t_max > 500:
            issues.append(f"total range [{t_min:.0f}, {t_max:.0f}] outside [50, 500]")

        # 2. Typical range: 98% of predictions should look like basketball
        if s_p99 > 50:
            issues.append(f"|spread| p99 = {s_p99:.1f} exceeds 50")
        if t_p01 < 180 or t_p99 > 350:
            issues.append(
                f"total p1-p99 = [{t_p01:.0f}, {t_p99:.0f}] outside [180, 350]"
            )

        # 3. All spreads having the same sign → model is biased or broken
        # Same-sign output is meaningful evidence only on a reasonably sized,
        # representative validation set. Tiny smoke folds can legitimately be
        # one-sided and used to make this guard reject a healthy export path.
        if len(spreads) >= 100 and (all(s > 0 for s in spreads) or all(s < 0 for s in spreads)):
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

        # 3. Save feature scaler (mean + std) and target stats for de-scaling
        if self._scaler_mean is not None and self._scaler_std is not None:
            scaler_path = os.path.join(model_dir, "scaler.npz")
            np.savez(
                scaler_path,
                mean=self._scaler_mean,
                std=self._scaler_std,
                active_features=self._active_features if self._active_features is not None else np.ones_like(self._scaler_std, dtype=bool),
                y_spread_mean=np.float32(self._y_spread_mean or 0.0),
                y_spread_std=np.float32(self._y_spread_std or 1.0),
                y_total_mean=np.float32(self._y_total_mean or 0.0),
                y_total_std=np.float32(self._y_total_std or 1.0),
            )
            logger.info("Scaler saved to %s", scaler_path)

        metadata = {
            "version": ver,
            "created_at": datetime.now().isoformat(),
            "device": str(self.device),
            "input_dim": self.model.layer1[0].in_features if self.model else FEATURE_DIM,
            "has_onnx": os.path.exists(os.path.join(model_dir, "model.onnx")),
            "input_normalization": self.model.input_normalization if self.model else "batchnorm",
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
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            self.model = NBA_MLP(
                input_dim=metadata.get("input_dim", input_dim),
                input_normalization=metadata.get("input_normalization", "batchnorm"),
            ).to(self.device)
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
            self._active_features = data["active_features"] if "active_features" in data else None
            # Target stats are optional (older models trained on raw targets)
            if "y_spread_mean" in data:
                self._y_spread_mean = float(data["y_spread_mean"])
                self._y_spread_std = float(data["y_spread_std"])
                self._y_total_mean = float(data["y_total_mean"])
                self._y_total_std = float(data["y_total_std"])
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
