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
    mlp_huber_delta_spread: float = 3.0
    mlp_huber_delta_total: float = 4.0

    # Feature defaults
    default_lookback_games: int = 10
    min_lookback_games: int = 5
    max_lookback_games: int = 15

    # Evaluation targets (gambling lines typically within 5 points)
    spread_mae_target: float = 5.0
    total_mae_target: float = 5.0
    win_accuracy_target: float = 0.65

    # Calibration
    calibration_window_days: int = 30
    min_games_for_calibration: int = 10

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
