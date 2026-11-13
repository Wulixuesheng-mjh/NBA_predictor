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
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

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


def _save_backtest_report(
    metrics,
    version: str,
    walk_forward: bool = False,
    seasons: list[str] | None = None,
    lookback: int = 10,
) -> str:
    """Save backtest metrics as a JSON report."""
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode = "walkforward" if walk_forward else "frozen"
    filename = f"backtest_{mode}_{timestamp}.json"
    filepath = reports_dir / filename

    report = {
        "report_type": f"backtest_{mode}",
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "seasons": seasons or ["all"],
            "lookback": lookback,
        },
        "results": {
            "n_games": metrics.n_games,
            "win_accuracy": metrics.win_accuracy,
            "spread_mae": metrics.spread_mae,
            "spread_rmse": metrics.spread_rmse,
            "spread_within_3": metrics.spread_within_3,
            "spread_within_5": metrics.spread_within_5,
            "total_mae": metrics.total_mae,
            "total_rmse": metrics.total_rmse,
            "total_within_5": metrics.total_within_5,
            "total_within_8": metrics.total_within_8,
            "win_accuracy_by_season": metrics.win_accuracy_by_season,
            "per_game_last_50": metrics.per_game,
        },
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  📄 Report saved to: {filepath}")
    return str(filepath)


async def main(
    seasons: list[str] | None = None,
    lookback: int = 10,
    run_backtest: bool = True,
    version: str | None = None,
    tune: bool = False,
    tune_trials: int = 100,
    walk_forward: bool = False,
    initial_train_games: int = 200,
    retrain_every: int = 50,
    max_games: int | None = None,
    cover_over: bool = False,
    cover_over_initial: int = 1000,
    cover_over_retrain: int = 250,
    use_cache: bool = True,
    clear_cache: bool = False,
    activate: bool = False,
    feature_cache_key: str | None = None,
) -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("NBA Predictor — Model Training Pipeline")
    print("=" * 60)
    print(f"  Seasons:      {seasons or 'all available'}")
    print(f"  Lookback:     {lookback} games")
    print(f"  Backtest:     {'yes' if run_backtest else 'no'}")
    print(f"  Feature dim:  {FEATURE_DIM}")
    print(f"  Feature cache:{' off' if not use_cache else (' cleared' if clear_cache else ' on')}")
    print("=" * 60)

    # 0a. Optional cache maintenance
    if clear_cache:
        from ml.src.features.dataset_cache import DatasetCache

        removed = DatasetCache().clear()
        print(f"  Cleared {removed} cached feature dataset(s)")

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
        use_cache=use_cache,
        activate=activate,
        feature_cache_key=feature_cache_key,
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
    print(f"  Release status: {result.release_status}")
    reports_dir = Path("/app/ml/saved_models/training_reports") if Path("/app").exists() else Path("ml/saved_models/training_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{result.version}.json"
    report_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    print(f"  Training report: {report_path}")

    # 2. Backtest
    if run_backtest:
        if walk_forward:
            # ── Walk-forward with incremental retraining ──
            print(f"\n[2/2] Running walk-forward backtest "
                  f"(initial={initial_train_games}, retrain_every={retrain_every})...")

            backtester = Backtester()
            metrics = await backtester.run_walk_forward(
                seasons=seasons,
                lookback_games=lookback,
                min_games_before=5,
                initial_train_games=initial_train_games,
                retrain_every=retrain_every,
                max_games=max_games,
            )
        else:
            # ── Frozen-model backtest (legacy) ──
            print("\n[2/2] Running frozen-model backtest...")

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

        if metrics.n_games == 0:
            print("\n  ⚠️  Backtest produced no results (not enough data).")
        else:
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

            # Save JSON report
            _save_backtest_report(
                metrics,
                version=result.version,
                walk_forward=walk_forward,
                seasons=seasons,
                lookback=lookback,
            )

    # 3. Cover/over (盘口) training + walk-forward backtest
    if cover_over:
        print("\n[3/3] Cover/over (盘口) binary classifiers...")

        try:
            cov_metrics, cov_version = await trainer.train_cover_over_pipeline(
                version=result.version,
            )
            print(f"\n  Cover/over models saved (version {cov_version}):")
            print(f"    Cover (让分胜负) accuracy: {cov_metrics.cover_accuracy:.3f}")
            print(f"    Over  (大小分)   accuracy: {cov_metrics.over_accuracy:.3f}")
            print(f"    Cover logloss:   {cov_metrics.cover_logloss:.3f}")
            print(f"    Over logloss:    {cov_metrics.over_logloss:.3f}")
        except Exception as e:
            print(f"\n  ⚠️  Cover/over training failed: {e}")

        print(f"\n  Running cover/over walk-forward backtest "
              f"(initial={cover_over_initial}, retrain={cover_over_retrain})...")
        backtester = Backtester()
        cov_bt = await backtester.run_cover_over_walk_forward(
            initial_train_games=cover_over_initial,
            retrain_every=cover_over_retrain,
        )
        if cov_bt.n_games == 0:
            print("  ⚠️  Cover/over backtest produced no results (not enough data).")
        else:
            print(f"\n  Cover/Over Walk-Forward Results ({cov_bt.n_games} games):")
            print(f"    Cover accuracy:  {cov_bt.cover_accuracy:.3f}  (break-even 52.4%)")
            print(f"    Over accuracy:   {cov_bt.over_accuracy:.3f}  (break-even 52.4%)")
            print(f"    Cover logloss:   {cov_bt.cover_logloss:.3f}")
            print(f"    Over logloss:    {cov_bt.over_logloss:.3f}")
            if cov_bt.n_confident:
                print(f"    Confident cover: {cov_bt.cover_accuracy_confident:.3f}")
                print(f"    Confident over:  {cov_bt.over_accuracy_confident:.3f}")
            if cov_bt.cover_accuracy_by_season:
                print(f"\n  Cover accuracy by season:")
                for s, a in sorted(cov_bt.cover_accuracy_by_season.items()):
                    print(f"    {s}: {a:.3f}")
            if cov_bt.over_accuracy_by_season:
                print(f"\n  Over accuracy by season:")
                for s, a in sorted(cov_bt.over_accuracy_by_season.items()):
                    print(f"    {s}: {a:.3f}")

    print("\n" + "=" * 60)
    print("Training pipeline complete! 🏀")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward backtest with incremental retraining instead of frozen-model backtest",
    )
    parser.add_argument(
        "--initial-train-games", type=int, default=200,
        help="Number of games for the initial training window in walk-forward mode (default: 200)",
    )
    parser.add_argument(
        "--retrain-every", type=int, default=50,
        help="Retrain model every N games in walk-forward mode (default: 50)",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Limit number of games in walk-forward backtest (for quick testing)",
    )
    parser.add_argument(
        "--cover-over", action="store_true",
        help="Train cover/over (盘口) binary classifiers and run their walk-forward backtest",
    )
    parser.add_argument(
        "--cover-over-initial", type=int, default=1000,
        help="Initial training games for cover/over walk-forward (default: 1000)",
    )
    parser.add_argument(
        "--cover-over-retrain", type=int, default=250,
        help="Retrain cover/over every N games (default: 250)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore the feature dataset cache (always rebuild, and don't write it)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete cached feature datasets before training",
    )
    parser.add_argument(
        "--activate", action="store_true",
        help="Atomically activate this bundle; omit for a shadow candidate",
    )
    parser.add_argument("--feature-cache-key", help="Use an immutable, already built feature cache")
    args = parser.parse_args()

    run_backtest = not args.no_backtest

    asyncio.run(main(
        seasons=args.seasons,
        lookback=args.lookback,
        run_backtest=run_backtest,
        version=args.version,
        tune=args.tune,
        tune_trials=args.tune_trials,
        walk_forward=args.walk_forward,
        initial_train_games=args.initial_train_games,
        retrain_every=args.retrain_every,
        max_games=args.max_games,
        cover_over=args.cover_over,
        cover_over_initial=args.cover_over_initial,
        cover_over_retrain=args.cover_over_retrain,
        use_cache=not args.no_cache,
        clear_cache=args.clear_cache,
        activate=args.activate,
        feature_cache_key=args.feature_cache_key,
    ))
