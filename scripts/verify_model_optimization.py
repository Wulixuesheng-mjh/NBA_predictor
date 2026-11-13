"""Dependency-light invariant checks for the model-optimization rollout."""

from datetime import datetime, timedelta, timezone

import numpy as np

from app.schemas.prediction import PredictionRequest
from ml.src.evaluation import complete_day_split
from ml.src.features.injury_net_features import RotationPlayer, project_rotation, summarize_rotation
from ml.src.features.matchup_features import compute_matchup_features
from ml.src.features.roster_continuity_features import RosterContinuityFeatures
from ml.src.features.season_prior_features import blend_current, shrink_prior
from ml.src.training_policy import policy_mask_and_weights, season_start_year
from ml.src.uncertainty import prediction_interval, threshold_probabilities


def close(left: float, right: float, tolerance: float = 1e-6) -> None:
    assert abs(left - right) <= tolerance, (left, right)


def main() -> None:
    cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows = [{"game_date": (cutoff - timedelta(days=d)).isoformat()} for d in (730, 365, 0)]
    mask, weights, report = policy_mask_and_weights(rows, "T4", cutoff=cutoff)
    assert mask.all() and weights[0] < weights[1] < weights[2]
    close(float(weights.mean()), 1.0)
    assert report["effective_samples"] < 3
    assert season_start_year(datetime(2026, 8, 1)) == 2025
    assert season_start_year(datetime(2026, 10, 1)) == 2026

    days = [{"game_date": f"2026-01-{d:02d}T01:00:00+00:00"} for d in (1, 1, 1, 2, 2, 3, 3, 3, 4, 4)]
    assert complete_day_split(days, 0.25) == 5

    matchup = compute_matchup_features(
        {"avg_pace": 100, "avg_off_rtg": 112, "avg_def_rtg": 109},
        {"avg_pace": 100, "avg_off_rtg": 108, "avg_def_rtg": 111},
    )
    close(matchup["expected_total"], 220.0)
    unknown = RosterContinuityFeatures._compute_continuity({}, {1: 1000})
    assert unknown["roster_data_missing"] == 1.0 and unknown["roster_continuity"] == 0.5

    prior = shrink_prior(115.0, 82, 112.0)
    zero = blend_current(None, 0, prior, 10)
    one = blend_current(125.0, 1, prior, 10)
    ten = blend_current(125.0, 10, prior, 10)
    close(zero, prior)
    assert prior < one < ten < 125.0

    players = [
        RotationPlayer(i, "G", 30.0, 30.0, 0.0 if i == 0 else 1.0, 0.5, 0.05)
        for i in range(7)
    ] + [RotationPlayer(7, "C", 0.0, 20.0, 1.0, 0.4, 0.04)]
    projected = project_rotation(players)
    assert projected is not None and projected[0] == 0.0
    close(sum(projected.values()), 240.0)
    summary = summarize_rotation(players, projected)
    assert summary["injury_replacement_minutes"] > 0
    close(summary["injury_newly_lost_minutes"], 30.0)

    absent = RotationPlayer(1, "F", 0.0, 30.0, 0.0, 0.7, 0.04)
    healthy = [RotationPlayer(i, "G", 40.0, 40.0, 1.0, 0.4, 0.03) for i in range(2, 8)]
    projected = project_rotation([absent, *healthy])
    assert projected is not None
    close(summarize_rotation([absent, *healthy], projected)["injury_net_offense_delta"], 0.0)
    returning = RotationPlayer(1, "F", 0.0, 30.0, 1.0, 0.8, 0.04)
    projected = project_rotation([returning, *healthy])
    assert projected is not None
    assert summarize_rotation([returning, *healthy], projected)["injury_net_offense_delta"] > 0

    residuals = [float(value) for value in range(-12, 13)] * 10
    easy = threshold_probabilities(7.0, 3.5, residuals, integer_outcome=True, fallback_sigma=12)
    hard = threshold_probabilities(7.0, 8.5, residuals, integer_outcome=True, fallback_sigma=12)
    integer = threshold_probabilities(7.0, 7.0, residuals, integer_outcome=True, fallback_sigma=12)
    assert easy.win > hard.win and integer.push > 0
    assert prediction_interval(220.0, residuals, 0.8) is not None

    try:
        PredictionRequest(
            home_team_id=1,
            away_team_id=2,
            spread_line=6.5,
            market_home_handicap=-6.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy and new handicap fields must be mutually exclusive")

    print("model optimization invariant checks: PASS")


if __name__ == "__main__":
    main()
