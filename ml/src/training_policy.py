"""Training-window and temporal sample-weight policies.

The feature cache intentionally contains every causally valid row.  Windowing
and weighting are applied after cache loading so T0--T5 experiments can reuse
the same feature build.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import log

import numpy as np


@dataclass(frozen=True)
class TrainingPolicy:
    name: str
    completed_seasons: int | None
    half_life_days: int | None


POLICIES: dict[str, TrainingPolicy] = {
    "T0": TrainingPolicy("T0", None, None),
    "T1": TrainingPolicy("T1", 2, None),
    "T2": TrainingPolicy("T2", 3, None),
    "T3": TrainingPolicy("T3", 3, 180),
    "T4": TrainingPolicy("T4", 3, 365),
    "T5": TrainingPolicy("T5", 3, 730),
}


def normalize_fit_weights(values, n_samples):
    """Mean one among actual fitted rows, preserving excluded zero rows.

    Slicing a mean-one full window leaves a smaller mean in the early-stopping
    training segment. XGB regularization depends on the absolute weight scale,
    so every fit must normalize at its own boundary, including residual subsets.
    """
    if values is None:
        return None
    weights=np.asarray(values,dtype=np.float64).copy()
    if weights.shape!=(n_samples,) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Fit weights must be finite, non-negative and aligned')
    positive=weights>0
    if not positive.any():
        raise ValueError('Fit weights have no positive samples')
    weights[positive]/=weights[positive].mean()
    return weights


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def season_start_year(value: str | date | datetime) -> int:
    """Return the NBA season start year for a game date."""
    day = _as_date(value)
    # NBA regular seasons start in October.  Treating July--September as the
    # next season would silently discard an extra historical season when an
    # experiment is run during the off-season.
    return day.year if day.month >= 10 else day.year - 1


def policy_mask_and_weights(
    sample_info: list[dict],
    policy: str | TrainingPolicy,
    *,
    cutoff: str | date | datetime | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a row mask, mean-one weights and an auditable summary.

    ``cutoff`` defaults to the newest sample.  The current season plus N
    completed seasons are retained.  Validation metrics must remain unweighted;
    callers use the returned weights only for fitting.
    """
    selected = POLICIES[policy] if isinstance(policy, str) else policy
    dates = np.array([_as_date(row.get("source_game_date") or row["game_date"]) for row in sample_info], dtype=object)
    if len(dates) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64), {}

    cutoff_day = _as_date(cutoff) if cutoff is not None else max(dates)
    cutoff_season = season_start_year(cutoff_day)
    if selected.completed_seasons is None:
        mask = dates <= cutoff_day
    else:
        earliest = cutoff_season - selected.completed_seasons
        mask = np.array(
            [d <= cutoff_day and season_start_year(d) >= earliest for d in dates],
            dtype=bool,
        )

    kept_dates = dates[mask]
    if selected.half_life_days is None:
        weights = np.ones(len(kept_dates), dtype=np.float64)
    else:
        ages = np.array([(cutoff_day - d).days for d in kept_dates], dtype=np.float64)
        weights = np.exp(-log(2.0) * ages / float(selected.half_life_days))
        weights /= weights.mean()

    by_season: dict[str, dict[str, float]] = {}
    for d, w in zip(kept_dates, weights):
        start = season_start_year(d)
        key = f"{start}-{str(start + 1)[-2:]}"
        bucket = by_season.setdefault(key, {"samples": 0, "weight": 0.0})
        bucket["samples"] += 1
        bucket["weight"] += float(w)

    total_w = float(weights.sum())
    effective_n = total_w * total_w / float(np.square(weights).sum()) if total_w else 0.0
    summary = {
        "policy": selected.name,
        "half_life_days": selected.half_life_days,
        "cutoff": cutoff_day.isoformat(),
        "samples": int(mask.sum()),
        "fit_from": min(kept_dates).isoformat() if len(kept_dates) else None,
        "fit_until": max(kept_dates).isoformat() if len(kept_dates) else None,
        "effective_samples": effective_n,
        "by_season": by_season,
    }
    return mask, weights.astype(np.float32), summary


def subset_arrays(dataset, mask: np.ndarray):
    """Create a TrainingDataset without coupling policy code to its module."""
    from ml.src.features.feature_builder import TrainingDataset

    return TrainingDataset(
        X=dataset.X[mask],
        y_win=dataset.y_win[mask],
        y_spread=dataset.y_spread[mask],
        y_total=dataset.y_total[mask],
        sample_info=[row for row, keep in zip(dataset.sample_info, mask) if keep],
    )
