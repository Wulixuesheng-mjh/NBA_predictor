"""Replay a stored prediction from its saved feature snapshot.

Exact replay requires an ordered, preprocessed feature vector and checksum.
Legacy rows without that snapshot are explicitly reported as unreproducible.

Usage (inside the backend or trainer container):

    python scripts/replay_prediction.py --prediction-id 26

It prints:
  * the stored prediction (what the user saw),
  * the raw model outputs replayed from the saved features,
  * whether each output passes the plausibility guard,
  * whether calibration was applied back then vs. today.

Exit code is non-zero when the replay disagrees with the stored row beyond
rounding, which makes it usable in CI.
"""

import argparse
import asyncio
import json
import hashlib
import os
import sys
from copy import deepcopy
from pathlib import Path

for _p in ("/app", "/app/backend"):
    if os.path.isdir(os.path.join(_p, "app")):
        sys.path.insert(0, _p)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sqlalchemy import select

from app.database import async_session_factory
from app.models.prediction import Prediction
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor, round_to_half
from ml.src.inference_snapshot import validate_snapshot


def replay_processed_row(row, predictor):
    """Do not send processed inputs through the builder/release adapter twice."""
    archived = row.features_json or {}
    manifest = archived.get('_release_manifest')
    if not isinstance(manifest, dict) or manifest != predictor.release_manifest:
        raise ValueError('Pinned release manifest is absent or differs')
    if row.calibrated:
        raise ValueError('Exact calibrated replay requires archived calibration coefficients')
    snapshot = archived.get('_inference_snapshot')
    X = validate_snapshot(snapshot, model_name=row.model_name,
                          model_version=row.model_version, manifest=manifest)
    if snapshot['feature_names'] != row.feature_names or snapshot['feature_vector'] != row.feature_vector:
        raise ValueError('Ledger columns differ from processed snapshot')
    if row.feature_names != FEATURE_NAMES[:len(row.feature_names)]:
        raise ValueError('Feature order differs; matching code image is required')
    trainer = predictor.mlp_trainer if row.model_name == 'mlp' else predictor.xgb_trainer
    if trainer is None or trainer.version != row.model_version:
        raise ValueError('Pinned model is unavailable')
    values = trainer.predict(X)
    raw = values[0] if isinstance(values, list) else values
    rounded = deepcopy(raw)
    rounded.predicted_spread = round_to_half(raw.predicted_spread)
    rounded.predicted_total = round_to_half(raw.predicted_total)
    return raw, rounded


async def main(prediction_id: int) -> int:
    async with async_session_factory() as session:
        row = (
            await session.execute(
                select(Prediction).where(Prediction.id == prediction_id)
            )
        ).scalar_one_or_none()

    if row is None:
        print(f"prediction #{prediction_id} not found")
        return 2

    print(f"prediction #{row.id}  model={row.model_name} v{row.model_version}")
    print(f"  stored : p_home={row.win_prob_home:.4f} spread={row.predicted_spread} total={row.predicted_total} "
          f"calibrated={row.calibrated}")

    if not row.feature_names or row.feature_vector is None or not row.feature_hash:
        print("  exact ordered feature snapshot absent — legacy row cannot be reproduced exactly")
        return 3
    if row.feature_names != FEATURE_NAMES[:len(row.feature_names)] or len(row.feature_vector) != len(row.feature_names):
        print("  feature order differs — replay requires the matching code image")
        return 3
    checksum = hashlib.sha256(
        ("|".join(row.feature_names) + "|" + ",".join(f"{v:.12g}" for v in row.feature_vector)).encode()
    ).hexdigest()
    if checksum != row.feature_hash:
        print("  feature snapshot checksum mismatch")
        return 3
    if row.calibrated:
        print("  archived calibration coefficients absent — exact calibrated replay unavailable")
        return 3
    manifest = (row.features_json or {}).get('_release_manifest')
    if manifest is None or not (row.features_json or {}).get('_inference_snapshot'):
        print('  legacy row has no certified preprocessing stage; exact replay is unavailable')
        return 3
    try:
        from app.services.prospective_service import verify_bundle
        verify_bundle(Path(ml_settings.models_dir), manifest)
        predictor = load_predictor(release_manifest=manifest)
        raw, replayed = replay_processed_row(row, predictor)
    except (ValueError, OSError) as exc:
        print(f'  exact replay unavailable: {exc}')
        return 3

    def _fmt(p):
        if p is None:
            return "rejected/unavailable"
        ok, reason = predictor._is_plausible(p)
        return (
            f"p_home={p.win_prob_home:.4f} spread={p.predicted_spread} "
            f"total={p.predicted_total} plausible={ok}" + ("" if ok else f" ({reason})")
        )

    print(f"  replayed {row.model_name}: {_fmt(replayed)}")
    print('  calibration_applied_now=False')

    status = 0
    if replayed is None:
        print("  saved model output is unavailable or rejected")
        status = 1
    if row.raw_output:
        for field, tolerance in (('win_prob_home', 1e-5), ('predicted_spread', 1e-3), ('predicted_total', 1e-3)):
            if field not in row.raw_output or abs(getattr(raw, field)-row.raw_output[field]) > tolerance:
                print('  raw replay differs from the archived model output')
                status = 1
    elif any(abs(actual - expected) > tolerance for actual, expected, tolerance in (
        (replayed.win_prob_home, row.win_prob_home, 1e-5),
        (replayed.predicted_spread, row.predicted_spread, 1e-3),
        (replayed.predicted_total, row.predicted_total, 1e-3),
    ) if expected is not None):
        print("  replay differs from the stored model output")
        status = 1
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a stored prediction")
    parser.add_argument("--prediction-id", type=int, required=True)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.prediction_id)))
