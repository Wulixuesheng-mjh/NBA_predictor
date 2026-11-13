"""Shared rolling evaluation protocol and release-gate metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
    }


def classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "brier": float(np.mean(np.square(p - y))),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def nba_day(row: dict) -> str:
    """Use source calendar days; UTC starts can cross a day or month boundary."""
    return str(row.get('source_game_date') or row['game_date'])[:10]


def complete_day_split(sample_info: list[dict], validation_ratio: float = 0.15) -> int:
    dates = [nba_day(row) for row in sample_info]
    if len(dates) < 2:
        raise ValueError('At least two complete NBA days are required')
    target = max(1, min(len(dates) - 1, int(len(dates) * (1 - validation_ratio))))
    split_day = dates[target]
    split = next((i for i, value in enumerate(dates) if value >= split_day), target)
    if split == 0:
        split = next((i for i, value in enumerate(dates) if value > dates[0]), 0)
    if split == 0:
        raise ValueError('Cannot split a single NBA day')
    return split


def monthly_folds(sample_info: list[dict], audit_seasons: set[str] | None = None):
    """Yield expanding-window month folds, never splitting an NBA day."""
    dates = [date.fromisoformat(nba_day(row)) for row in sample_info]
    months = sorted({(day.year, day.month) for day in dates})
    for year, month in months:
        val = np.array([day.year == year and day.month == month for day in dates])
        if not val.any():
            continue
        first = min(day for day, keep in zip(dates, val) if keep)
        train = np.array([day < first for day in dates])
        if train.sum() < 1000:
            continue
        if audit_seasons is not None:
            season_values = {sample_info[i].get("season") for i in np.flatnonzero(val)}
            if not season_values.intersection(audit_seasons):
                continue
        yield f"{year:04d}-{month:02d}", train, val


def paired_day_bootstrap(
    sample_info: list[dict],
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% interval for mean paired loss difference, resampled by game day."""
    dates = np.array([nba_day(row) for row in sample_info])
    unique = np.unique(dates)
    rng = np.random.default_rng(seed)
    differences = np.asarray(loss_b) - np.asarray(loss_a)
    by_day = {day: differences[dates == day] for day in unique}
    draws = []
    for _ in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([by_day[day] for day in chosen])
        draws.append(float(values.mean()))
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def release_gate(baseline: dict, candidate: dict, *, slice_worst_change: float = 0.0) -> dict:
    improvement = (baseline["mae"] - candidate["mae"]) / baseline["mae"]
    rmse_change = (candidate["rmse"] - baseline["rmse"]) / baseline["rmse"]
    passed = improvement >= 0.02 and rmse_change <= 0.01 and slice_worst_change <= 0.05
    return {
        "passed": passed,
        "mae_improvement": improvement,
        "rmse_change": rmse_change,
        "slice_worst_change": slice_worst_change,
    }
