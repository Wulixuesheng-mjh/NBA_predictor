"""Data-volume experiment: 3-season vs 5-season training window.

Trains the XGBoost (primary) model twice, both with historical injury features
(use_historical=True), on a short and a long season window, and reports the
validation metrics side by side.  Answers whether a longer lookback horizon
(adds older seasons) measurably improves the primary model.

Usage (inside the trainer container):
    python scripts/experiment_data_volume.py \
        --small 2023-24 2024-25 2025-26 \
        --large 2021-22 2022-23 2023-24 2024-25 2025-26 \
        --lookback 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.src.features.feature_builder import FeatureBuilder  # noqa: E402
from ml.src.models.xgboost_model import XGBoostTrainer  # noqa: E402


def _chronological_split(X, y_win, y_spread, y_total, val_ratio=0.15):
    n = len(X)
    split_idx = int(n * (1 - val_ratio))
    split_idx = max(split_idx, 1)
    split_idx = min(split_idx, n - 1)
    return {
        "X_train": X[:split_idx],
        "X_val": X[split_idx:],
        "y_win_train": y_win[:split_idx],
        "y_win_val": y_win[split_idx:],
        "y_spread_train": y_spread[:split_idx],
        "y_spread_val": y_spread[split_idx:],
        "y_total_train": y_total[:split_idx],
        "y_total_val": y_total[split_idx:],
    }


async def _build_and_train(label: str, seasons, lookback, min_games_before):
    builder = FeatureBuilder()
    print(f"\n[{label}] building dataset for {seasons} ...")
    ds = await builder.build_training_dataset(
        seasons=seasons,
        lookback_games=lookback,
        min_games_before=min_games_before,
        use_historical=True,
    )
    print(f"[{label}] {ds.X.shape[0]} samples, {ds.X.shape[1]} features")

    split = _chronological_split(ds.X, ds.y_win, ds.y_spread, ds.y_total)
    xgb = XGBoostTrainer(version=f"vol_{label}")
    metrics = xgb.train(
        X_train=split["X_train"],
        y_win_train=split["y_win_train"],
        y_spread_train=split["y_spread_train"],
        y_total_train=split["y_total_train"],
        X_val=split["X_val"],
        y_win_val=split["y_win_val"],
        y_spread_val=split["y_spread_val"],
        y_total_val=split["y_total_val"],
    )
    return ds, metrics


async def main(small, large, lookback, min_games_before):
    print("=" * 70)
    print("Data-volume experiment — 3-season vs 5-season (both use_historical=True)")
    print("=" * 70)
    print(f"  small window: {small}")
    print(f"  large window: {large}")
    print(f"  lookback: {lookback}, min_games_before: {min_games_before}")
    print("=" * 70)

    ds_s, m_s = await _build_and_train("small", small, lookback, min_games_before)
    ds_l, m_l = await _build_and_train("large", large, lookback, min_games_before)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'metric':<22}{'3-season':>12}{'5-season':>12}{'Δ (5-3)':>14}")
    print("-" * 60)
    rows = [
        ("win_accuracy", m_s.win_accuracy, m_l.win_accuracy, 4),
        ("win_logloss", m_s.win_logloss, m_l.win_logloss, 4),
        ("spread_mae", m_s.spread_mae, m_l.spread_mae, 3),
        ("spread_rmse", m_s.spread_rmse, m_l.spread_rmse, 3),
        ("total_mae", m_s.total_mae, m_l.total_mae, 3),
        ("total_rmse", m_s.total_rmse, m_l.total_rmse, 3),
    ]
    for name, s, l, prec in rows:
        print(f"{name:<22}{s:>12.{prec}f}{l:>12.{prec}f}{l - s:>+14.{prec}f}")
    print("-" * 60)
    print(f"{'total_samples':<22}{ds_s.X.shape[0]:>12}{ds_l.X.shape[0]:>12}")
    print(f"{'train_samples':<22}{m_s.train_samples:>12}{m_l.train_samples:>12}")
    print(f"{'val_samples':<22}{m_s.val_samples:>12}{m_l.val_samples:>12}")

    report = {
        "small_seasons": small,
        "large_seasons": large,
        "lookback": lookback,
        "min_games_before": min_games_before,
        "small": {
            "n_samples": int(ds_s.X.shape[0]),
            "win_accuracy": m_s.win_accuracy,
            "win_logloss": m_s.win_logloss,
            "spread_mae": m_s.spread_mae,
            "spread_rmse": m_s.spread_rmse,
            "total_mae": m_s.total_mae,
            "total_rmse": m_s.total_rmse,
        },
        "large": {
            "n_samples": int(ds_l.X.shape[0]),
            "win_accuracy": m_l.win_accuracy,
            "win_logloss": m_l.win_logloss,
            "spread_mae": m_l.spread_mae,
            "spread_rmse": m_l.spread_rmse,
            "total_mae": m_l.total_mae,
            "total_rmse": m_l.total_rmse,
        },
    }
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", type=str, nargs="+", default=["2023-24", "2024-25", "2025-26"])
    ap.add_argument("--large", type=str, nargs="+",
                    default=["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--min-games-before", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main(args.small, args.large, args.lookback, args.min_games_before))
