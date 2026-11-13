"""Verify a candidate bundle without changing active.json or prediction ledger."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureVector
from ml.src.predictor import load_predictor
from ml.src.trainer import Trainer


def array(predictions):
    if not isinstance(predictions, list):
        predictions = [predictions]
    return np.asarray([[p.win_prob_home, p.predicted_spread, p.predicted_total] for p in predictions])


def main(args):
    root = Path(ml_settings.models_dir)
    active = root / 'active.json'
    active_before = active.read_bytes() if active.exists() else None
    manifest = json.loads((root / 'candidates' / f'{args.version}.json').read_text())
    assert manifest['feature_dim'] == 213
    assert manifest['feature_names'] == FEATURE_NAMES
    assert manifest['feature_schema_sha256'] == hashlib.sha256('|'.join(FEATURE_NAMES).encode()).hexdigest()
    hashes_checked = 0
    for group, version in (('xgboost', manifest['xgboost_version']), ('mlp', manifest['mlp_version'])):
        if version is None:
            continue
        for filename, expected in manifest['file_hashes'][group].items():
            assert hashlib.sha256((root / group / version / filename).read_bytes()).hexdigest() == expected
            hashes_checked += 1
    dataset = DatasetCache().load(args.feature_cache_key, 213)
    if dataset is None or len(dataset.X) < 100:
        raise ValueError('Need at least 100 real cached samples')
    # Deliberately include opening weeks, missing player stats and ordinary rows.
    opening = np.flatnonzero(dataset.X[:, FEATURE_NAMES.index('home_season_games_played')] < 5)
    indices = np.unique(np.concatenate([
        np.linspace(0, len(dataset.X)-1, 100, dtype=int), opening[:30]
    ]))
    sample = dataset.X[indices]
    predictor = load_predictor(xgb_version=args.version)
    assert predictor.release_manifest['version'] == args.version
    report = {
        'version': args.version, 'status': 'PASS', 'feature_dim': 213,
        'feature_cache_key': args.feature_cache_key, 'file_hashes_checked': hashes_checked,
        'parity_samples': len(sample), 'mlp_pytorch_onnx_batch_single_parity': None,
    }
    xgb = predictor.xgb_trainer
    batch = array(xgb.predict(sample))
    single = np.vstack([array(xgb.predict(row)) for row in sample])
    np.testing.assert_allclose(batch, single, rtol=0, atol=1e-5)
    report['xgboost_batch_single_max_error'] = np.max(np.abs(batch-single), axis=0).tolist()
    if predictor.mlp_trainer is not None:
        import torch
        from ml.src.models.mlp_model import NBA_MLP

        mlp = predictor.mlp_trainer
        folder = root / 'mlp' / manifest['mlp_version']
        metadata = json.loads((folder / 'metadata.json').read_text())
        mlp.model = NBA_MLP(
            input_dim=213, input_normalization=metadata.get('input_normalization', 'batchnorm')
        ).to(mlp.device)
        mlp.model.load_state_dict(torch.load(folder / 'model.pt', weights_only=True, map_location=mlp.device))
        assert Trainer._validate_mlp_bundle(mlp, str(folder), sample)
        report['mlp_pytorch_onnx_batch_single_parity'] = True
        report['mlp_parity_samples'] = min(100, len(sample))
        report['parity_tolerances'] = {'win_probability': 1e-5, 'points': 1e-3}
        report['mlp_input_normalization'] = metadata.get('input_normalization')
        report['mlp_active_feature_count'] = int(mlp._active_features.sum()) if mlp._active_features is not None else 213
    rejected = {'xgboost': 0, 'mlp': 0}
    for row in sample:
        result = predictor.predict(FeatureVector(features=row, home_games_available=10, away_games_available=10), apply_calibration=False)
        rejected['xgboost'] += int(result.xgboost is None)
        if manifest['mlp_version']:
            rejected['mlp'] += int(result.mlp is None)
    report['rejected_outputs_on_parity_samples'] = rejected
    assert rejected['xgboost'] == 0
    assert rejected['mlp'] == 0
    assert (active.read_bytes() if active.exists() else None) == active_before
    report['active_manifest_unchanged'] = True
    path = root / 'training_reports' / f'{args.version}_validation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version')
    parser.add_argument('--feature-cache-key', required=True)
    main(parser.parse_args())
