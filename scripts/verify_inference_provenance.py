"""Read-only real-model compatibility check; never update a selector or ledger."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureVector
from ml.src.inference_snapshot import validate_snapshot
from ml.src.predictor import load_predictor


def main(args):
    root = Path(ml_settings.models_dir)
    before = (root/'active.json').read_bytes()
    manifest = json.loads(Path(args.baseline_manifest).read_text()) if args.baseline_manifest else json.loads(before)
    if args.baseline_manifest:
        assert manifest['origin_active_sha256']==hashlib.sha256(before).hexdigest()
        from app.services.prospective_service import verify_bundle
        verify_bundle(root,manifest)
    assert int(manifest.get('feature_semantic_version', 5)) == 5
    assert not manifest.get('disabled_features') and not manifest.get('calibration_enabled')
    data = DatasetCache().load(args.feature_cache_key, len(FEATURE_NAMES))
    assert data is not None and len(data.X) >= 100
    sample = data.X[np.linspace(0, len(data.X)-1, 100, dtype=int)]
    prepared = sample.astype(np.float64).copy()
    prepared[:, FEATURE_NAMES.index('expected_total')] /= 2.
    predictor = load_predictor(release_manifest=manifest)
    assert predictor.xgb_trainer.is_trained
    if manifest.get('mlp_version'):
        assert predictor.mlp_trainer is not None
    report = dict(status='PASS', version=manifest['version'], samples=100,
        feature_cache_key=args.feature_cache_key, source_feature_dim=len(FEATURE_NAMES), models={},
        scope='raw inference compatibility and input provenance; not candidate precision approval',
        archived_artifact_hashes_verified=bool(args.baseline_manifest))
    logging.getLogger('ml.src.predictor').setLevel(logging.ERROR)
    raw_results = [predictor.predict(FeatureVector(features=row, home_games_available=10,
                   away_games_available=10), apply_calibration=False) for row in sample]
    for name, trainer in (('xgboost',predictor.xgb_trainer), ('mlp',predictor.mlp_trainer)):
        if trainer is None:
            continue
        dim = int(trainer.model_win.n_features_in_) if name=='xgboost' else len(trainer._scaler_mean)
        manual = prepared[:, :dim]
        batch = trainer.predict(manual)
        fields = ('win_prob_home','predicted_spread','predicted_total')
        expected = np.array([[getattr(p,f) for f in fields] for p in batch])
        actual = np.array([[r.raw_outputs[name][f] for f in fields] for r in raw_results])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        for i, result in enumerate(raw_results):
            value = validate_snapshot(result.processed_inputs[name], model_name=name,
                model_version=trainer.version, manifest=manifest)
            np.testing.assert_array_equal(value[0], manual[i])
            assert not result.calibration_applied
        report['models'][name] = dict(processed_dim=dim,
            max_single_batch_difference=np.abs(actual-expected).max(axis=0).tolist(),
            snapshots_verified=100,
            plausibility_rejections=sum(getattr(r,name) is None for r in raw_results))
    assert (root/'active.json').read_bytes() == before
    report['active_manifest_sha256'] = hashlib.sha256(before).hexdigest()
    report['active_unchanged'] = True
    Path(args.output).write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-cache-key',required=True)
    parser.add_argument('--baseline-manifest')
    parser.add_argument('--output',default='/tmp/inference_provenance.json')
    main(parser.parse_args())
