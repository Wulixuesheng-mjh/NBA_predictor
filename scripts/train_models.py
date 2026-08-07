"""CLI script to train the NBA prediction models.

Usage:
    python scripts/train_models.py [--seasons 2024-25 2025-26] [--lookback 10] [--backtest]

This script:
  1. Builds the training dataset from historical game data
  2. Trains XGBoost (primary) and MLP (auxiliary) models
  3. Runs a walk-forward backtest
  4. Saves models and reports metrics

Requirements:
  - PostgreSQL database must be running and populated (run seed_data.py first)
  - PYTHONPATH should include the project root
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root AND backend directory to Python path.
# Both are needed: /app for 'ml' package, /app/backend for 'app' ORM models.
# IMPORTANT: ML modules now import models as "app.models.*" (not "backend.app.models.*").
# Using consistent import paths prevents SQLAlchemy double-registration errors.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.src.trainer import Trainer
from ml.src.backtest import Backtester
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.models.mlp_model import MLPTrainer
from ml.src.features.feature_builder import FeatureBuilder, FEATURE_DIM


async def main(
    seasons: list[str] | None = None,
    lookback: int = 10,
    run_backtest: bool = True,
    version: str | None = None,
    tune: bool = False,
    tune_trials: int = 100,
) -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("NBA Predictor — Model Training Pipeline")
    print("=" * 60)
    print(f"  Seasons:      {seasons or 'all available'}")
    print(f"  Lookback:     {lookback} games")
    print(f"  Backtest:     {'yes' if run_backtest else 'no'}")
    print(f"  Feature dim:  {FEATURE_DIM}")
    print("=" * 60)

    # 0. Optional hyperparameter tuning
    if tune:
        print("\n[0/2] Running Optuna hyperparameter tuning...")
        print(f"       Trials: {tune_trials}")

        from ml.src.models.tuner import XGBoostTuner

        tuner = XGBoostTuner()
        await tuner.load_data(
            seasons=seasons,
            lookback_games=lookback,
            min_games_before=5,
        )
        tune_result = await tuner.tune(n_trials=tune_trials)

        print(f"\n  Tuning complete ({tune_result.duration_seconds:.0f}s, {tune_result.n_trials} trials)")
        print(f"  Best combined MAE:  {tune_result.best_value:.2f}")
        print(f"  Best Spread MAE:    {tune_result.best_spread_mae:.2f}")
        print(f"  Best Total MAE:     {tune_result.best_total_mae:.2f}")
        print(f"  Best Win Accuracy:  {tune_result.best_win_accuracy:.3f}")
        print(f"\n  Best params:")
        for k, v in sorted(tune_result.best_params.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # 1. Train models
    print("\n[1/2] Training models...")
    trainer = Trainer()
    result = await trainer.train_full_pipeline(
        seasons=seasons,
        lookback_games=lookback,
        version=version,
        calibrate=True,
    )

    print(f"\n  Version: {result.version}")
    print(f"  Samples: {result.dataset_info['n_samples']}")
    print(f"\n  XGBoost Metrics:")
    print(f"    Win Accuracy:  {result.xgboost_metrics.win_accuracy:.3f}")
    print(f"    Win LogLoss:   {result.xgboost_metrics.win_logloss:.3f}")
    print(f"    Spread MAE:    {result.xgboost_metrics.spread_mae:.2f}")
    print(f"    Spread RMSE:   {result.xgboost_metrics.spread_rmse:.2f}")
    print(f"    Total MAE:     {result.xgboost_metrics.total_mae:.2f}")
    print(f"    Total RMSE:    {result.xgboost_metrics.total_rmse:.2f}")
    print(f"    Train time:    {result.xgboost_metrics.training_time_seconds:.1f}s")

    print(f"\n  MLP Metrics:")
    print(f"    Win Accuracy:  {result.mlp_metrics.get('win_accuracy', 'N/A'):.3f}"
          if isinstance(result.mlp_metrics.get('win_accuracy'), float) else
          f"    Win Accuracy:  {result.mlp_metrics.get('win_accuracy', 'N/A')}")
    if isinstance(result.mlp_metrics.get('spread_mae'), float):
        print(f"    Spread MAE:    {result.mlp_metrics['spread_mae']:.2f}")
        print(f"    Total MAE:     {result.mlp_metrics['total_mae']:.2f}")
    print(f"    Epochs:        {result.mlp_metrics.get('epochs_completed', 'N/A')}")
    print(f"    Train time:    {result.mlp_metrics.get('training_time_seconds', 0):.1f}s")

    if result.calibration:
        cal = result.calibration
        n_games = getattr(cal, 'n_games_used', 0) if not isinstance(cal, dict) else cal.get('n_games_used', 0)
        win_b = getattr(cal, 'win_bias', 0) if not isinstance(cal, dict) else cal.get('win_bias', 0)
        spread_b = getattr(cal, 'spread_bias', 0) if not isinstance(cal, dict) else cal.get('spread_bias', 0)
        total_b = getattr(cal, 'total_bias', 0) if not isinstance(cal, dict) else cal.get('total_bias', 0)
        print(f"\n  Calibration (from {n_games} games):")
        print(f"    Win bias:     {win_b:+.3f}")
        print(f"    Spread bias:  {spread_b:+.2f}")
        print(f"    Total bias:   {total_b:+.2f}")

    print(f"\n  Models saved to: {result.model_paths}")

    # 2. Backtest
    if run_backtest:
        print("\n[2/2] Running walk-forward backtest...")

        xgb = XGBoostTrainer()
        xgb.load(result.version)

        mlp = MLPTrainer()
        try:
            mlp.load(result.version)
        except Exception as e:
            print(f"  MLP load skipped: {e}")
            mlp = None

        backtester = Backtester()
        metrics = await backtester.run(
            xgb_trainer=xgb,
            mlp_trainer=mlp,
            seasons=seasons,
            lookback_games=lookback,
            min_games_before=5,
        )

        print(f"\n  Backtest Results ({metrics.n_games} games):")
        print(f"    Win Accuracy:      {metrics.win_accuracy:.3f}")
        print(f"    Spread MAE:        {metrics.spread_mae:.2f}  (target < 5.0)")
        print(f"    Spread RMSE:       {metrics.spread_rmse:.2f}")
        print(f"    Spread within ±3:  {metrics.spread_within_3:.1%}  (stretch)")
        print(f"    Spread within ±5:  {metrics.spread_within_5:.1%}  (target)")
        print(f"    Total MAE:         {metrics.total_mae:.2f}  (target < 5.0)")
        print(f"    Total RMSE:        {metrics.total_rmse:.2f}")
        print(f"    Total within ±5:   {metrics.total_within_5:.1%}  (target)")
        print(f"    Total within ±8:   {metrics.total_within_8:.1%}")

        if metrics.win_accuracy_by_season:
            print(f"\n  By Season:")
            for season, acc in sorted(metrics.win_accuracy_by_season.items()):
                print(f"    {season}:  {acc:.3f}")

    print("\n" + "=" * 60)
    print("Training pipeline complete! 🏀")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument(
        "--seasons", type=str, nargs="+", default=None,
        help="Seasons to train on (e.g. 2024-25 2025-26)",
    )
    parser.add_argument(
        "--lookback", type=int, default=10,
        help="Games to look back for team features (5-15, default: 10)",
    )
    parser.add_argument(
        "--no-backtest", action="store_true",
        help="Skip the backtest step",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Custom model version string",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before training",
    )
    parser.add_argument(
        "--tune-trials", type=int, default=100,
        help="Number of Optuna trials (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        seasons=args.seasons,
        lookback=args.lookback,
        run_backtest=not args.no_backtest,
        version=args.version,
        tune=args.tune,
        tune_trials=args.tune_trials,
    ))
