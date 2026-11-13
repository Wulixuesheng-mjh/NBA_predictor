"""Validate a totals-only shadow package and atomically expose its catalog."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from ml.src.config import ml_settings
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from app.services.shadow_total_service import prepare_shadow_features
from app.services.prospective_service import feature_code_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root=Path(ml_settings.models_dir).resolve()
    active=root/'active.json';before=active.read_bytes()
    bundle=(root/args.bundle).resolve()
    assert bundle.is_relative_to(root)
    dataset=DatasetCache().load(args.feature_cache_key,len(FEATURE_NAMES))
    assert dataset is not None and len(dataset.X)>=100
    opening=np.flatnonzero(dataset.X[:,FEATURE_NAMES.index('home_season_games_played')]<5)
    indexes=np.unique(np.r_[np.linspace(0,len(dataset.X)-1,100,dtype=int),opening[:25]])
    report=dict(status='PASS',version=args.version,role='totals_only_diagnostic_shadow',
        created_at=datetime.now(timezone.utc).isoformat(),parity_samples=len(indexes),
        feature_code_sha256=feature_code_hash(),models={},primary_promotion=False)
    catalog=dict(version=args.version,bundle=args.bundle,status='diagnostic_shadow',
                 manifest_sha256={},primary_promotion=False)
    for mode in ('direct','residual'):
        path=bundle/f'{mode}_manifest.json'
        manifest=json.loads(path.read_text())
        assert manifest['feature_dim']==213 and manifest['feature_cache_key']==args.feature_cache_key
        for name,expected in manifest['file_hashes'].items():
            assert Path(name).name==name and digest(bundle/'models'/mode/name)==expected
        model=XGBoostTrainer(models_dir=str(bundle/'models'));model.load(mode)
        X=np.vstack([prepare_shadow_features(dataset.X[i],manifest) for i in indexes])
        expected=model._predict_totals(X)
        single=np.array([model._predict_totals(x.reshape(1,-1))[0] for x in X])
        np.testing.assert_allclose(expected,single,rtol=0,atol=1e-3)
        assert np.isfinite(expected).all() and ((expected>=150)&(expected<=300)).all()
        report['models'][mode]=dict(training_data_until=manifest['training_data_until'],
            file_hashes_verified=True,preprocessing_verified=True,
            batch_single_max_error=float(np.max(abs(expected-single))),
            mlp_included=False,offline_release_gate_passed=False)
        catalog['manifest_sha256'][mode]=digest(path)
    folder=root/'shadow_catalogs';folder.mkdir(exist_ok=True)
    (folder/f'{args.version}_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    catalog_path=folder/f'{args.version}.json'
    with catalog_path.open('x',encoding='utf-8') as handle:
        json.dump(catalog,handle,indent=2)
    if args.activate_shadow:
        pointer=root/'shadow_totals.json'
        if pointer.exists():
            (root/'shadow_totals_previous.json').write_bytes(pointer.read_bytes())
        temporary=pointer.with_suffix('.tmp')
        temporary.write_bytes(catalog_path.read_bytes());temporary.replace(pointer)
    assert active.read_bytes()==before
    report['active_primary_unchanged']=True
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--feature-cache-key',required=True)
    parser.add_argument('--activate-shadow',action='store_true')
    main(parser.parse_args())
