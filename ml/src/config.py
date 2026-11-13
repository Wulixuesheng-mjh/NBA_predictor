"""ML module configuration loaded from environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class MLSettings(BaseSettings):
    """ML-specific settings, loaded from environment or .env file."""

    # Database (same as backend, but ML module is independent)
    database_url: str = "postgresql+asyncpg://nba_admin:nba_predictor_2024@localhost:5432/nba_predictor"

    # Model paths
    models_dir: str = str(Path(__file__).parent.parent / "saved_models")
    xgboost_dir: str = str(Path(__file__).parent.parent / "saved_models" / "xgboost")
    mlp_dir: str = str(Path(__file__).parent.parent / "saved_models" / "mlp")

    # Training — Optuna-tuned on 30,658 games (80 trials, 2026-08-07)
    # Best trial #52: combined MAE 13.31, Spread 11.53, Total 15.09, Win 63.3%
    xgboost_n_estimators: int = 400
    xgboost_max_depth: int = 5
    xgboost_learning_rate: float = 0.0153
    xgboost_early_stopping_rounds: int = 80
    xgboost_subsample: float = 0.6378
    xgboost_colsample_bytree: float = 0.8974
    xgboost_reg_alpha: float = 0.3418
    xgboost_reg_lambda: float = 1.3977
    xgboost_min_child_weight: int = 4
    xgboost_gamma: float = 0.7319
    mlp_epochs: int = 400
    mlp_batch_size: int = 128
    mlp_learning_rate: float = 0.0005
    mlp_patience: int = 50
    mlp_dropout: float = 0.2
    mlp_weight_decay: float = 3e-4
    # Inputs are already standardized and clipped by the persisted scaler.
    # A second BatchNorm can amplify newly varying opener features by hundreds
    # when their training variance was zero. Keep it only for legacy loading.
    mlp_input_normalization: str = "none"
    mlp_huber_delta_spread: float = 3.0
    mlp_huber_delta_total: float = 4.0

    # Feature defaults
    default_lookback_games: int = 10
    min_lookback_games: int = 5
    max_lookback_games: int = 15

    # MLP training window (ISO date). The MLP is trained only on games from
    # this date onwards: older seasons have structurally different features
    # (advanced stats / player box scores are absent before ~2012, so those
    # columns are constant 0), and mixing eras makes the network extrapolate
    # wildly on the validation split. XGBoost still uses the full history.
    mlp_training_start: str = "2021-07-01"

    # Temporal freshness policy.  T4 keeps the current season plus the three
    # previous completed seasons and halves sample weight every 365 days.
    # T0--T5 are implemented in ``ml.src.training_policy`` for controlled
    # experiments without rebuilding features.
    training_policy: str = "T4"
    # Optional per-target overrides selected by the T0--T5 experiment report.
    # Empty values inherit ``training_policy``.
    xgboost_win_policy: str | None = None
    xgboost_spread_policy: str | None = None
    xgboost_total_policy: str | None = None
    mlp_training_policy: str | None = None

    # Evaluation targets (gambling lines typically within 5 points)
    spread_mae_target: float = 5.0
    total_mae_target: float = 5.0
    win_accuracy_target: float = 0.65

    # Calibration
    calibration_window_days: int = 30
    min_games_for_calibration: int = 10
    # The variance-inflating calibration (scale = std(actual)/std(pred)) made
    # MAE worse on the validation set, so it is OFF by default. Re-enable only
    # after a rolling out-of-sample check shows it helps.
    calibration_enabled: bool = False

    # Refit the shipped model on ALL available data after early stopping picked
    # the iteration count. Without this the production model only ever saw the
    # first 85% of history (e.g. games up to 2022).
    refit_on_full: bool = True

    # Plausibility guards: outputs outside these bounds are treated as model
    # failures instead of being shown as real predictions.
    output_total_min: float = 150.0
    output_total_max: float = 300.0
    output_max_abs_spread: float = 45.0

    # Feedback loop
    min_new_games_for_retrain: int = 3
    auto_retrain: bool = True

    # Random seed
    random_seed: int = 42

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


ml_settings = MLSettings()

# Ensure model directories exist
os.makedirs(ml_settings.xgboost_dir, exist_ok=True)
os.makedirs(ml_settings.mlp_dir, exist_ok=True)
