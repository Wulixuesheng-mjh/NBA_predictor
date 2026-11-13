"""Run controlled T0--T5 temporal-policy experiments on one feature cache.

All candidates train on policy-specific slices that end before one common,
complete-day validation interval. Validation metrics are deliberately
unweighted so every policy is compared on the same games.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ml.src.evaluation import classification_metrics, complete_day_split, regression_metrics
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder, TrainingDataset
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.training_policy import POLICIES, policy_mask_and_weights, subset_arrays


def _slice(dataset: TrainingDataset, start: int, stop: int | None = None) -> TrainingDataset:
    return TrainingDataset(
        X=dataset.X[start:stop],
        y_win=dataset.y_win[start:stop],
        y_spread=dataset.y_spread[start:stop],
        y_total=dataset.y_total[start:stop],
        sample_info=dataset.sample_info[start:stop],
    )


def _identity(mask: np.ndarray, weights: np.ndarray) -> str:
    payload = mask.tobytes() + np.round(weights, 8).tobytes()
    return hashlib.sha256(payload).hexdigest()


async def run(args) -> dict:
    dataset = await FeatureBuilder().build_training_dataset(min_games_before=5, use_cache=True)
    split = complete_day_split(dataset.sample_info, 0.15)
    development = _slice(dataset, 0, split)
    validation = _slice(dataset, split)
    if len(validation.X) < 50:
        raise ValueError("common validation interval has fewer than 50 games")

    reports: dict[str, dict] = {}
    duplicate_of: dict[str, str] = {}
    seen: dict[str, str] = {}
    baseline_index = FEATURE_NAMES.index("shrunk_expected_total")
    for name in args.policies:
        mask, weights, weight_report = policy_mask_and_weights(
            development.sample_info,
            name,
            cutoff=development.sample_info[-1]["game_date"],
        )
        identity = _identity(mask, weights)
        if identity in seen:
            duplicate_of[name] = seen[identity]
            reports[name] = {"duplicate_of": seen[identity], "weights": weight_report}
            continue
        seen[identity] = name
        selected = subset_arrays(development, mask)
        trainer = XGBoostTrainer(version=f"experiment-{name}")
        trainer.train(
            selected.X, selected.y_win, selected.y_spread, selected.y_total,
            validation.X, validation.y_win, validation.y_spread, validation.y_total,
            sample_weight=weights,
            total_baseline_train=selected.X[:, baseline_index],
            total_baseline_val=validation.X[:, baseline_index],
            total_baseline_feature_index=baseline_index,
        )
        predictions = trainer.predict(validation.X)
        if not isinstance(predictions, list):
            predictions = [predictions]
        win_p = np.array([p.win_prob_home for p in predictions])
        spread_p = np.array([p.predicted_spread for p in predictions])
        total_p = np.array([p.predicted_total for p in predictions])
        entry = {
            "weights": weight_report,
            "fit_from": selected.sample_info[0]["game_date"],
            "fit_until": selected.sample_info[-1]["game_date"],
            "validation_from": validation.sample_info[0]["game_date"],
            "validation_until": validation.sample_info[-1]["game_date"],
            "validation_samples": len(validation.X),
            "xgboost": {
                "win": classification_metrics(validation.y_win, win_p),
                "spread": regression_metrics(validation.y_spread, spread_p),
                "total": regression_metrics(validation.y_total, total_p),
                "total_mode": trainer._total_mode,
            },
        }

        if args.include_mlp:
            from ml.src.models.mlp_model import MLPTrainer

            mlp = MLPTrainer(version=f"experiment-{name}", device="cpu")
            metrics = mlp.train(
                selected.X, selected.y_win, selected.y_spread, selected.y_total,
                validation.X, validation.y_win, validation.y_spread, validation.y_total,
                sample_weight=weights,
            )
            mlp_p = mlp.predict(validation.X)
            if not isinstance(mlp_p, list):
                mlp_p = [mlp_p]
            win_metrics = classification_metrics(
                validation.y_win, np.array([p.win_prob_home for p in mlp_p])
            )
            spread_metrics = regression_metrics(
                validation.y_spread, np.array([p.predicted_spread for p in mlp_p])
            )
            total_metrics = regression_metrics(
                validation.y_total, np.array([p.predicted_total for p in mlp_p])
            )
            composite = float(np.mean([
                win_metrics["logloss"],
                spread_metrics["mae"] / max(float(np.std(validation.y_spread)), 1e-6),
                total_metrics["mae"] / max(float(np.std(validation.y_total)), 1e-6),
            ]))
            entry["mlp"] = {
                "win": win_metrics,
                "spread": spread_metrics,
                "total": total_metrics,
                "standardized_composite_loss": composite,
                "best_epoch": metrics.get("best_epoch"),
            }
        reports[name] = entry

    unique = {name: value for name, value in reports.items() if "xgboost" in value}
    selected_policies = {
        "xgboost_win": min(unique, key=lambda n: unique[n]["xgboost"]["win"]["brier"]),
        "xgboost_spread": min(unique, key=lambda n: unique[n]["xgboost"]["spread"]["mae"]),
        "xgboost_total": min(unique, key=lambda n: unique[n]["xgboost"]["total"]["mae"]),
    }
    mlp_candidates = {n: v for n, v in unique.items() if "mlp" in v}
    if mlp_candidates:
        selected_policies["mlp_shared"] = min(
            mlp_candidates,
            key=lambda n: mlp_candidates[n]["mlp"]["standardized_composite_loss"],
        )
    return {
        "protocol": {
            "validation_weighted": False,
            "common_validation": True,
            "duplicate_candidates_skipped": duplicate_of,
        },
        "selected_policies": selected_policies,
        "policies": reports,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES), default=sorted(POLICIES))
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--output", default="ml/saved_models/experiment_report.json")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
