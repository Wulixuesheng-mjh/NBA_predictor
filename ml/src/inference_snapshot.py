"""Checksummed inputs at the trainer.predict boundary, before its saved scaler.

These values have already received release unit conversions, masks and column
selection. Replays must call the pinned trainer directly, never adapt them twice.
"""
import hashlib
import json

import numpy as np


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def input_snapshot(values, names, *, model_name, model_version, manifest):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(names) or not np.isfinite(array).all():
        raise ValueError('Invalid processed model input')
    payload = dict(stage='trainer_predict_input', preprocessing_version='release_input_v1',
        feature_names=list(names), feature_vector=array.tolist(), model_name=model_name,
        model_version=model_version, feature_semantic_version=manifest.get('feature_semantic_version', 5),
        disabled_features=list(manifest.get('disabled_features', [])),
        release_manifest_sha256=checksum(manifest))
    return {**payload, 'sha256': checksum(payload)}


def validate_snapshot(snapshot, *, model_name, model_version, manifest):
    if not isinstance(snapshot, dict):
        raise ValueError('Exact processed snapshot is absent')
    payload = {k: v for k, v in snapshot.items() if k != 'sha256'}
    if checksum(payload) != snapshot.get('sha256'):
        raise ValueError('Processed snapshot checksum mismatch')
    if (payload.get('stage') != 'trainer_predict_input'
            or payload.get('preprocessing_version') != 'release_input_v1'
            or payload.get('model_name') != model_name
            or payload.get('model_version') != model_version
            or payload.get('release_manifest_sha256') != checksum(manifest)):
        raise ValueError('Processed snapshot provenance mismatch')
    array = np.asarray(payload.get('feature_vector'), dtype=np.float64)
    names = payload.get('feature_names')
    if (array.ndim != 1 or not isinstance(names, list) or len(array) != len(names)
            or len(names) != len(set(names)) or not np.isfinite(array).all()):
        raise ValueError('Invalid processed snapshot values/order')
    return array.reshape(1, -1)
