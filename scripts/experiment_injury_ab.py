"""A/B experiment: does historical injury data improve XGBoost?

Runs the XGBoost (primary) model twice on the SAME season window:
  - baseline  : injury feature columns zeroed (equivalent to use_historical=False,
                which the live ``player_injuries`` table cannot provide — it holds
                only a handful of manual records with zero variance).
  - injected  : injury features populated from ``historical_player_injuries``
                (use_historical=True).

Because the only difference between the two paths is the 12 injury feature
columns, we build the dataset once (use_historical=True) and derive the
baseline by zeroing those columns — avoiding a second, identical, expensive
feature-build pass.

Reports, side by side:
  - XGBoost validation metrics (win accuracy, spread MAE/RMSE, total MAE/RMSE)
  - aggregated + per-feature XGBoost "gain" importance of the injury features

Usage (inside the trainer container):
    python scripts/experiment_injury_ab.py --seasons 2023-24 2024-25 2025-26 --lookback 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

# Mirror train_models.py's import preamble (app + ml packages).
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.src.features.feature_builder import FeatureBuilder, FEATURE_NAMES  # noqa: E402
from ml.src.features.player_features import PLAYER_FEATURE_NAMES  # noqa: E402
from ml.src.models.xgboost_model import XGBoostTrainer  # noqa: E402


# The 12 injury feature columns (6 per team), in FEATURE_NAMES order.
INJURY_FEATURE_INDICES = [
    i for i, n in enumerate(FEATURE_NAMES)
    if n[5:] in PLAYER_FEATURE_NAMES  # strips "home_"/"away_" prefix
]


def _chronological_split(X, y_win, y_spread, y_total, val_ratio=0.15):
    """Mirror Trainer._chronological_split (last val_ratio → validation)."""
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


def _train_arm(X, y_win, y_spread, y_total, arm: str):
    split = _chronological_split(X, y_win, y_spread, y_total)
    xgb = XGBoostTrainer(version=f"ab_{arm}")
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
    return xgb, metrics


def _injury_importance(xgb: XGBoostTrainer):
    """Aggregate XGBoost 'gain' importance across the 12 injury features."""
    out = {}
    for model_name, model in (
        ("win", xgb.model_win),
        ("spread", xgb.model_spread),
        ("total", xgb.model_total),
    ):
        imp = np.asarray(model.feature_importances_, dtype=float)
        injury_imp = imp[INJURY_FEATURE_INDICES]
        out[model_name] = {
            "sum": float(injury_imp.sum()),
            "share": float(injury_imp.sum() / (imp.sum() + 1e-12)),
            "top": sorted(
                [
                    (FEATURE_NAMES[i], float(imp[i]))
                    for i in INJURY_FEATURE_INDICES
                    if imp[i] > 0
                ],
                key=lambda kv: -kv[1],
            )[:8],
        }
    return out


async def main(
    seasons: list[str] | None,
    lookback: int,
    min_games_before: int,
) -> None:
    builder = FeatureBuilder()

    print("=" * 70)
    print("Injury A/B experiment — baseline (injury=0) vs injected (historical)")
    print("=" * 70)
    print(f"  Seasons:           {seasons or 'all'}")
    print(f"  Lookback:          {lookback}")
    print(f"  Min games before:  {min_games_before}")
    print(f"  Injury feature cols: {len(INJURY_FEATURE_INDICES)}")
    print("=" * 70)

    # Build once with historical injuries, then derive baseline by zeroing.
    print("\n[1/3] Building dataset (use_historical=True)...")
    dataset = await builder.build_training_dataset(
        seasons=seasons,
        lookback_games=lookback,
        min_games_before=min_games_before,
        use_historical=True,
    )
    X = dataset.X
    y_win, y_spread, y_total = dataset.y_win, dataset.y_spread, dataset.y_total
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Report how much injury signal is actually present.
    injury_cols = X[:, INJURY_FEATURE_INDICES]
    nonzero = int((injury_cols != 0).any(axis=1).sum())
    print(
        f"  Injury signal: {nonzero}/{X.shape[0]} games ({100*nonzero/X.shape[0]:.1f}%) "
        f"have ≥1 nonzero injury feature"
    )
    for name, col in zip(
        [FEATURE_NAMES[i] for i in INJURY_FEATURE_INDICES], injury_cols.T
    ):
        nz = int((col != 0).sum())
        if nz:
            print(f"    {name:34s} nonzero={nz:5d}  mean={col.mean():.3f}  max={col.max():.2f}")

    # Baseline = zero out injury columns.
    X_base = X.copy()
    X_base[:, INJURY_FEATURE_INDICES] = 0.0

    print("\n[2/3] Training baseline (injury=0)...")
    base_xgb, base_m = _train_arm(X_base, y_win, y_spread, y_total, "baseline")

    print("\n[3/3] Training injected (historical injuries)...")
    inj_xgb, inj_m = _train_arm(X, y_win, y_spread, y_total, "injected")

    # ── Side-by-side report ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'metric':<22}{'baseline':>12}{'injected':>12}{'Δ (inj-bas)':>14}")
    print("-" * 60)
    rows = [
        ("win_accuracy", base_m.win_accuracy, inj_m.win_accuracy, 4),
        ("win_logloss", base_m.win_logloss, inj_m.win_logloss, 4),
        ("spread_mae", base_m.spread_mae, inj_m.spread_mae, 3),
        ("spread_rmse", base_m.spread_rmse, inj_m.spread_rmse, 3),
        ("total_mae", base_m.total_mae, inj_m.total_mae, 3),
        ("total_rmse", base_m.total_rmse, inj_m.total_rmse, 3),
    ]
    for name, b, i, prec in rows:
        print(f"{name:<22}{b:>12.{prec}f}{i:>12.{prec}f}{i - b:>+14.{prec}f}")
    print("-" * 60)
    print(f"{'train_samples':<22}{base_m.train_samples:>12}{inj_m.train_samples:>12}")
    print(f"{'val_samples':<22}{base_m.val_samples:>12}{inj_m.val_samples:>12}")

    print("\nInjury feature importance (XGBoost 'gain'):")
    print(f"{'model':<8}{'sum':>10}{'share':>10}")
    for model_name in ("win", "spread", "total"):
        d = _injury_importance(inj_xgb)[model_name]
        print(f"{model_name:<8}{d['sum']:>10.4f}{d['share']:>10.3%}")
        for name, imp in d["top"]:
            print(f"    {name:<34s} {imp:.4f}")

    # Emit a JSON blob for easy machine capture.
    report = {
        "seasons": seasons,
        "lookback": lookback,
        "min_games_before": min_games_before,
        "n_samples": int(X.shape[0]),
        "games_with_injury_signal": nonzero,
        "metrics": {
            "baseline": {
                "win_accuracy": base_m.win_accuracy,
                "win_logloss": base_m.win_logloss,
                "spread_mae": base_m.spread_mae,
                "spread_rmse": base_m.spread_rmse,
                "total_mae": base_m.total_mae,
                "total_rmse": base_m.total_rmse,
            },
            "injected": {
                "win_accuracy": inj_m.win_accuracy,
                "win_logloss": inj_m.win_logloss,
                "spread_mae": inj_m.spread_mae,
                "spread_rmse": inj_m.spread_rmse,
                "total_mae": inj_m.total_mae,
                "total_rmse": inj_m.total_rmse,
            },
        },
        "injury_importance": _injury_importance(inj_xgb),
    }
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, nargs="+", default=["2023-24", "2024-25", "2025-26"])
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--min-games-before", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main(args.seasons, args.lookback, args.min_games_before))
