"""Build one real season and report v6 feature coverage without training."""

import asyncio
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ml.src.features.feature_builder import FEATURE_DIM, FEATURE_NAMES, FeatureBuilder


async def main(args) -> None:
    dataset = await FeatureBuilder().build_training_dataset(
        seasons=None if args.all_seasons else args.seasons,
        lookback_games=10,
        min_games_before=5,
        use_historical=True,
        use_cache=True,
    )
    assert dataset.X.shape[1] == FEATURE_DIM
    assert len(dataset.X) > 0
    assert np.isfinite(dataset.X).all()
    baseline = dataset.X[:, FEATURE_NAMES.index("shrunk_expected_total")]
    home_net = dataset.X[:, FEATURE_NAMES.index("home_injury_net_available")]
    away_net = dataset.X[:, FEATURE_NAMES.index("away_injury_net_available")]
    report = {
        "status": "PASS",
        "samples": len(dataset.X),
        "feature_dim": FEATURE_DIM,
        "feature_cache_key": dataset.cache_key,
        "first_game_date": dataset.sample_info[0]["game_date"],
        "last_game_date": dataset.sample_info[-1]["game_date"],
        "baseline_positive_rate": float(np.mean(baseline > 0)),
        "home_injury_net_available_rate": float(np.mean(home_net > 0)),
        "away_injury_net_available_rate": float(np.mean(away_net > 0)),
        "by_season": {},
    }
    for season in sorted({row["season"] for row in dataset.sample_info}):
        mask = np.array([row["season"] == season for row in dataset.sample_info])
        report["by_season"][season] = {
            "samples": int(mask.sum()),
            "home_net_coverage": float(np.mean(home_net[mask] > 0)),
            "away_net_coverage": float(np.mean(away_net[mask] > 0)),
            "aggregate_nonzero_team_games": int(sum(np.sum(dataset.X[mask, FEATURE_NAMES.index(f'{side}_injury_firepower_loss')] > 0)
                for side in ('home','away'))),
            "net_nonzero_team_games": int(sum(np.sum(np.abs(dataset.X[mask, FEATURE_NAMES.index(f'{side}_injury_net_offense_delta')]) > 1e-8)
                for side in ('home','away'))),
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2025-26"])
    parser.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--output", default="/app/ml/saved_models/feature_coverage.json")
    asyncio.run(main(parser.parse_args()))
