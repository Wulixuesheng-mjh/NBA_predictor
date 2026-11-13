from datetime import datetime,timezone,timedelta
import json,hashlib
import numpy as np
import pytest
from ml.src.features.feature_builder import FEATURE_NAMES,TrainingDataset
from ml.src.win_shadow import schema,prepare_win_input,fit_win_policy,predict_win_bundle,DISABLED


def test_win_shadow_has_an_explicit_isolated_schema_and_never_mutates_primary_input():
    raw=np.ones((1,len(FEATURE_NAMES)))
    value=prepare_win_input(raw,schema())
    assert raw.sum()==len(FEATURE_NAMES) and value.sum()==len(FEATURE_NAMES)-len(DISABLED)
    for invalid in (schema()|dict(targets=['win','total']),schema()|dict(feature_names=FEATURE_NAMES[::-1]),
                    schema()|dict(disabled_features=[])):
        with pytest.raises(ValueError): prepare_win_input(raw,invalid)


def test_win_only_bundle_rejects_in_training_targets_and_changed_weights(tmp_path):
    rng=np.random.default_rng(2);X=rng.normal(size=(160,len(FEATURE_NAMES))).astype(np.float32)
    start=datetime(2024,1,1,tzinfo=timezone.utc)
    rows=[dict(nba_game_id=str(i),season='2023-24',game_date=(start+timedelta(days=i)).isoformat()) for i in range(len(X))]
    y=(X[:,0]+rng.normal(size=len(X))>0).astype(float)
    data=TrainingDataset(X=X,y_win=y,y_spread=np.zeros(len(X)),y_total=np.full(len(X),225.),sample_info=rows)
    model,details=fit_win_policy(data,params=dict(n_estimators=8,max_depth=2,n_jobs=1,random_state=42))
    assert details['selection_fit_until']<details['validation_from']<=details['fit_until']
    path=tmp_path/'win_model.json';model.save_model(path)
    manifest=schema()|dict(version='test',training_data_until=details['fit_until'],
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    output=predict_win_bundle(tmp_path,X[:3],target_day='2025-01-01',as_of=datetime.now(timezone.utc))
    assert len(output)==3 and all('predicted_total' not in r and 'predicted_spread' not in r for r in output)
    assert all(r['win_prob_home']+r['win_prob_away']==pytest.approx(1) for r in output)
    with pytest.raises(ValueError,match='out-of-training'):
        predict_win_bundle(tmp_path,X[:1],target_day=details['fit_until'],as_of=datetime.now(timezone.utc))
    path.write_text('corrupted')
    with pytest.raises(ValueError,match='changed'):
        predict_win_bundle(tmp_path,X[:1],target_day='2025-01-01',as_of=datetime.now(timezone.utc))
