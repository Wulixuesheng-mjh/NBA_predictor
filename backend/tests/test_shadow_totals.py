from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pytest
from app.services.shadow_total_service import prepare_shadow_features, shadow_totals
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION


def manifest():
    return dict(feature_names=FEATURE_NAMES,feature_semantic_version=CACHE_FORMAT_VERSION,
                disabled_features=['home_injury_firepower_loss'],training_data_until='2026-10-21')


def test_mask_is_applied_without_mutating_primary_features():
    X=np.ones(len(FEATURE_NAMES))
    prepared=prepare_shadow_features(X,manifest())
    assert X.sum()==len(X)
    assert prepared[0,FEATURE_NAMES.index('home_injury_firepower_loss')]==0
    assert prepared.sum()==len(X)-1


def test_wrong_feature_order_and_nan_are_rejected():
    wrong=manifest();wrong['feature_names']=FEATURE_NAMES[::-1]
    with pytest.raises(ValueError):
        prepare_shadow_features(np.ones(len(FEATURE_NAMES)),wrong)
    X=np.ones(len(FEATURE_NAMES));X[0]=np.nan
    with pytest.raises(ValueError):
        prepare_shadow_features(X,manifest())


def test_missing_bundle_does_not_fabricate_a_prediction(tmp_path):
    assert shadow_totals([],as_of=datetime.now(timezone.utc),lookback=10,root=tmp_path) is None


def test_corrupt_manifest_is_rejected_before_model_load(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'direct_manifest.json').write_text('{}')
    (tmp_path/'shadow_totals.json').write_text(json.dumps(dict(version='test',bundle='bundle',
        manifest_sha256={'direct':'invalid','residual':'invalid'})))
    with pytest.raises(ValueError,match='manifest changed'):
        shadow_totals([],as_of=datetime.now(timezone.utc),lookback=10,root=tmp_path)


def test_training_games_cannot_be_presented_as_shadow_forecasts(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    raw=json.dumps(manifest()).encode()
    (bundle/'direct_manifest.json').write_bytes(raw)
    (tmp_path/'shadow_totals.json').write_text(json.dumps(dict(version='test',bundle='bundle',
        manifest_sha256={'direct':hashlib.sha256(raw).hexdigest()})))
    result=shadow_totals([],as_of=datetime(2026,10,22,tzinfo=timezone.utc),
                         target_day='2026-10-21',lookback=10,root=tmp_path)
    assert result['status']=='unavailable' and result['candidates']==[]
