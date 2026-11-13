"""Empirical out-of-sample residual intervals and market probabilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class DistributionResult:
    win: float
    push: float
    lose: float
    calibrated: bool
    sample_size: int


def prediction_interval(point: float, residuals: list[float], coverage: float) -> tuple[float, float] | None:
    if len(residuals) < 200:
        return None
    alpha = (1.0 - coverage) / 2.0
    low, high = np.quantile(np.asarray(residuals), [alpha, 1.0 - alpha])
    return float(point + low), float(point + high)


def threshold_probabilities(
    point: float,
    threshold: float,
    residuals: list[float],
    *,
    integer_outcome: bool,
    fallback_sigma: float,
) -> DistributionResult:
    """P(point + residual >/=/< threshold), with explicit push mass."""
    calibrated = len(residuals) >= 200
    if calibrated:
        outcomes = point + np.asarray(residuals, dtype=np.float64)
        if integer_outcome:
            outcomes = np.floor(outcomes + .5)
        win = float(np.mean(outcomes > threshold))
        push = float(np.mean(outcomes == threshold)) if integer_outcome else 0.0
        lose = float(np.mean(outcomes < threshold)) if integer_outcome else 1.0-win
        return DistributionResult(win, push, lose, True, len(residuals))

    # Explicitly marked fallback; used only to keep the UI informative until
    # 200 genuine OOS residuals have accumulated.
    def cdf(value):
        z = (value - point) / max(float(fallback_sigma), 1e-6)
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    boundary = math.floor(threshold) + .5 if integer_outcome else threshold
    high = cdf(boundary)
    low = cdf(threshold-.5) if integer_outcome and float(threshold).is_integer() else high
    return DistributionResult(float(1.-high), float(high-low), float(low), False, len(residuals))


class ResidualStore:
    """Read-only inference view of rolling residuals saved by evaluation jobs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload: dict = {}
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                self.payload = value if isinstance(value,dict) else {}
            except (OSError, ValueError):
                self.payload = {}

    def get(self, model: str, target: str, stratum: str = "global", *,
            version: str | None = None, as_of: datetime | None = None) -> list[float]:
        # Unversioned legacy residuals cannot certify a new model's uncertainty.
        if not version or self.payload.get('format_version') != 2:
            return []
        cutoff = as_of or datetime.now(timezone.utc)
        if cutoff.tzinfo is None:
            raise ValueError('Residual information cutoff must be timezone-aware')
        group = self.payload.get('models', {}).get(model, {}).get(version, {}).get(target, {})
        def available(label):
            values=[]
            for row in group.get(label, []):
                try:
                    timestamp=datetime.fromisoformat(row['available_at'])
                    value=float(row['residual'])
                    if timestamp.tzinfo is not None and timestamp <= cutoff and np.isfinite(value):
                        values.append(value)
                except (KeyError, TypeError, ValueError):
                    continue
            return values[-1000:]
        values = available(stratum)
        return available('global') if stratum != 'global' and len(values) < 200 else values


def build_residual_payload(pairs):
    """Rebuild from settled certified rows; one earliest forecast per game/version.

    Missing snapshots, in-training labels and calibrated/legacy records never
    acquire fabricated provenance. Rebuilding makes retries idempotent.
    """
    from ml.src.inference_snapshot import validate_snapshot
    def aware(value):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    payload = dict(format_version=2, models={})
    seen=set()
    for pred,game,result in sorted(pairs,key=lambda item:(
            aware(item[0].as_of) if item[0].as_of else datetime.min.replace(tzinfo=timezone.utc), item[0].id)):
        metadata=pred.features_json or {}; manifest=metadata.get('_release_manifest')
        if (pred.prediction_mode!='pregame' or pred.calibrated or not manifest
                or game.status!='Final' or not game.score_complete or game.time_precision!='exact'
                or not game.scheduled_start_utc or not pred.as_of or not pred.created_at
                or not result.evaluated_at or not manifest.get('training_data_until')
                or not game.source_game_date
                or str(game.source_game_date)<=str(manifest['training_data_until'])[:10]):
            continue
        start=aware(game.scheduled_start_utc)
        if aware(pred.as_of)>=start or aware(pred.created_at)>=start or aware(result.evaluated_at)<start:
            continue
        expected_version = manifest.get(f'{pred.model_name}_version')
        if pred.model_name == 'mlp' and manifest.get('mlp_input_compatibility') is not None:
            from ml.src.legacy_mlp_compatibility import contract_spec, VERSION, CONTRACT_ID
            if (manifest['mlp_input_compatibility'] != contract_spec()
                    or expected_version != VERSION):
                continue
            expected_version = VERSION + '+' + CONTRACT_ID
        if expected_version != pred.model_version:
            continue
        try:
            validate_snapshot(metadata.get('_inference_snapshot'), model_name=pred.model_name,
                model_version=pred.model_version, manifest=manifest)
        except (TypeError, ValueError):
            continue
        key=(pred.model_name,pred.model_version,game.id)
        if key in seen:
            continue
        seen.add(key)
        model=payload['models'].setdefault(pred.model_name,{}).setdefault(pred.model_version,{})
        for target in ('spread','total'):
            actual=getattr(result,f'actual_{target}'); predicted=getattr(pred,f'predicted_{target}')
            if actual is None or predicted is None or not np.isfinite([actual,predicted]).all():
                continue
            values=model.setdefault(target,{}).setdefault('global',[])
            values.append(dict(game_id=game.id,prediction_id=pred.id,residual=float(actual-predicted),
                               available_at=aware(result.evaluated_at).isoformat()))
    for versions in payload['models'].values():
        for model in versions.values():
            for group in model.values():
                group['global']=sorted(group['global'],key=lambda r:(r['available_at'],r['prediction_id']))[-1000:]
    return payload
