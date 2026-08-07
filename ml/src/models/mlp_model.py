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

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_DIM

logger = logging.getLogger(__name__)


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

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: NBA_MLP | None = None
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
        """Train the MLP model.

        Returns dict with training metrics.
        """
        import time
        start_time = time.time()

        input_dim = X_train.shape[1]

        # Initialize model (use config dropout for full-dataset training)
        self.model = NBA_MLP(
            input_dim=input_dim,
            dropout=ml_settings.mlp_dropout,
        ).to(self.device)

        # Convert data to tensors
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_win_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_spread_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_total_train, dtype=torch.float32).unsqueeze(1),
        )

        batch_size = min(ml_settings.mlp_batch_size, max(16, len(X_train) // 2))
        batch_size = max(batch_size, 4)  # Ensure minimum batch size for BatchNorm
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Validation tensors
        val_data = None
        if X_val is not None:
            val_data = (
                torch.tensor(X_val, dtype=torch.float32).to(self.device),
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

        return metrics

    def predict(self, X: np.ndarray) -> MLPPrediction:
        """Run inference.

        Args:
            X: Feature vector(s), shape (n_samples, FEATURE_DIM) or (FEATURE_DIM,).

        Returns:
            MLPPrediction (if single sample) or list of them.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load().")

        self.model.eval()

        # Ensure 2D — derive single-sample flag AFTER reshape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        single = X.shape[0] == 1

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            win, spread, total = self.model(X_tensor)

        win_np = win.cpu().numpy()
        spread_np = spread.cpu().numpy()
        total_np = total.cpu().numpy()

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

    def save(self, version: str | None = None) -> str:
        """Save model to disk."""
        ver = version or self.version
        model_dir = os.path.join(self.models_dir, ver)
        os.makedirs(model_dir, exist_ok=True)

        if self.model:
            torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pt"))

        metadata = {
            "version": ver,
            "created_at": datetime.now().isoformat(),
            "device": str(self.device),
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved MLP model to %s", model_dir)
        return model_dir

    def load(self, version: str | None = None, input_dim: int = FEATURE_DIM) -> str:
        """Load model from disk."""
        ver = version or self._find_latest_version()
        model_dir = os.path.join(self.models_dir, ver)

        self.model = NBA_MLP(input_dim=input_dim).to(self.device)
        state = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.version = ver

        logger.info("Loaded MLP model v%s from %s", ver, model_dir)
        return ver

    def _find_latest_version(self) -> str | None:
        """Find the most recent model version on disk."""
        if not os.path.isdir(self.models_dir):
            return None
        versions = [
            d for d in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, d))
            and os.path.exists(os.path.join(self.models_dir, d, "model.pt"))
        ]
        if not versions:
            return None
        return sorted(versions)[-1]
