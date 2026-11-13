"""Real OOS errors cannot be pooled across releases or duplicated by retries."""
from datetime import date, datetime, timedelta, timezone
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.inference_snapshot import input_snapshot
from ml.src.uncertainty import ResidualStore, build_residual_payload, prediction_interval, threshold_probabilities


def pair(game_id=1, prediction_id=1, version='new'):
    as_of=datetime(2026,10,20,10,tzinfo=timezone.utc)
    manifest=dict(xgboost_version=version,feature_semantic_version=9,training_data_until='2026-10-19')
    snapshot=input_snapshot(np.ones(len(FEATURE_NAMES)), FEATURE_NAMES, model_name='xgboost',
                            model_version=version,manifest=manifest)
    pred=SimpleNamespace(id=prediction_id,model_name='xgboost',model_version=version,
        prediction_mode='pregame',as_of=as_of,created_at=as_of,calibrated=False,
        predicted_total=220.,predicted_spread=5.,
        features_json={'_release_manifest':manifest,'_inference_snapshot':snapshot})
    game=SimpleNamespace(id=game_id,status='Final',score_complete=True,time_precision='exact',
        scheduled_start_utc=as_of+timedelta(hours=1), source_game_date=date(2026,10,20))
    result=SimpleNamespace(evaluated_at=as_of+timedelta(hours=4),actual_total=225.,actual_spread=8.)
    return pred,game,result


def test_residuals_are_versioned_unique_per_game_and_information_time_filtered(tmp_path):
    pairs=[pair()]
    for number in range(2,251):
        duplicate=pair(prediction_id=number);duplicate[0].as_of+=timedelta(minutes=1)
        pairs.append(duplicate)
    pairs.append(pair(game_id=2,prediction_id=300,version='old'))
    payload=build_residual_payload(pairs)
    assert payload==build_residual_payload(list(reversed(pairs)))
    path=tmp_path/'residuals.json';path.write_text(json.dumps(payload))
    store=ResidualStore(path)
    assert store.get('xgboost','total',version='new')==[5.]
    assert store.get('xgboost','total',version='old')==[5.]
    assert store.get('xgboost','total',version='unseen')==[]
    assert store.get('xgboost','total')==[]
    assert store.get('xgboost','total',version='new',as_of=datetime(2026,10,20,12,tzinfo=timezone.utc))==[]
    assert prediction_interval(220,store.get('xgboost','total',version='new'),.8) is None


@pytest.mark.parametrize('invalid', ['training', 'after_tipoff', 'created_after_tipoff',
    'legacy', 'calibrated', 'preview', 'unknown_as_of', 'unsettled', 'corrupt'])
def test_invalid_predictions_never_certify_intervals(invalid):
    pred,game,result=pair()
    if invalid=='training':
        pred.features_json['_release_manifest']['training_data_until']='2026-10-20'
    elif invalid=='after_tipoff': pred.as_of=game.scheduled_start_utc
    elif invalid=='created_after_tipoff': pred.created_at=game.scheduled_start_utc
    elif invalid=='legacy': pred.features_json.pop('_inference_snapshot')
    elif invalid=='calibrated': pred.calibrated=True
    elif invalid=='preview': pred.prediction_mode='reconstructed_replay'
    elif invalid=='unknown_as_of': pred.as_of=None
    elif invalid=='unsettled': result.evaluated_at=pred.as_of
    else: pred.features_json['_inference_snapshot']['feature_vector'][0]=99
    assert build_residual_payload([(pred,game,result)])['models']=={}


def test_legacy_unversioned_values_are_not_promoted_to_calibrated_errors(tmp_path):
    path=tmp_path/'residuals.json'
    path.write_text(json.dumps({'xgboost':{'total':{'global':[0]*250}}}))
    assert ResidualStore(path).get('xgboost','total',version='new')==[]


@pytest.mark.parametrize('tampered',[False,True])
def test_repair_behavior_version_is_separate_from_original_weight_version(tampered):
    from ml.src.legacy_mlp_compatibility import VERSION,CONTRACT_ID,contract_spec
    pred,game,result=pair()
    pred.model_name='mlp';pred.model_version=VERSION+'+'+CONTRACT_ID
    manifest=dict(mlp_version=VERSION,mlp_input_compatibility=contract_spec(),
        feature_semantic_version=5,training_data_until='2026-10-19')
    if tampered:manifest['mlp_input_compatibility']['coordinates']=[18,50]
    snap=input_snapshot(np.zeros(166),FEATURE_NAMES[:166],model_name='mlp',
        model_version=pred.model_version,manifest=manifest)
    pred.features_json={'_release_manifest':manifest,'_inference_snapshot':snap}
    payload=build_residual_payload([(pred,game,result)])
    if tampered:assert payload['models']=={}
    else:
        assert set(payload['models']['mlp'])=={pred.model_version}
        assert payload['models']['mlp'][pred.model_version]['total']['global'][0]['residual']==5


@pytest.mark.asyncio
async def test_residual_refresh_is_recoverable_and_idempotent_from_database(seeded_db,tmp_path,monkeypatch):
    from ml.src import feedback
    from app.models.prediction import Prediction
    from app.models.prediction_results import PredictionResult
    from app.models.game import Game
    pred,game,result=pair()
    stored_game=await seeded_db.get(Game,seeded_db._game_ids['final'])
    stored_game.scheduled_start_utc=game.scheduled_start_utc
    stored_game.time_precision='exact';stored_game.source_game_date=game.source_game_date
    stored_game.score_complete=True
    row=Prediction(game_id=stored_game.id,model_name=pred.model_name,model_version=pred.model_version,
        home_team_id=stored_game.home_team_id,away_team_id=stored_game.away_team_id,
        as_of=pred.as_of,created_at=pred.created_at,predicted_at=pred.as_of,
        prediction_mode='pregame',win_prob_home=.6,win_prob_away=.4,
        predicted_spread=pred.predicted_spread,predicted_total=pred.predicted_total,
        features_json=pred.features_json,calibrated=False)
    seeded_db.add(row);await seeded_db.flush()
    seeded_db.add(PredictionResult(prediction_id=row.id,actual_total=result.actual_total,
        actual_spread=result.actual_spread,evaluated_at=result.evaluated_at))
    await seeded_db.commit()
    monkeypatch.setattr(feedback,'ml_settings',SimpleNamespace(models_dir=str(tmp_path)))
    (tmp_path/'residuals.json').write_text('interrupted old file')
    await feedback.FeedbackLoop._refresh_residuals(seeded_db)
    before=(tmp_path/'residuals.json').read_bytes()
    await feedback.FeedbackLoop._refresh_residuals(seeded_db)
    assert before==(tmp_path/'residuals.json').read_bytes()
    assert ResidualStore(tmp_path/'residuals.json').get('xgboost','total',version='new')==[5.]


@pytest.mark.parametrize('residuals', [[], list(np.arange(-10,11,dtype=float))*10])
def test_integer_push_and_threshold_monotonicity_with_and_without_residuals(residuals):
    integer=threshold_probabilities(7,7,residuals,integer_outcome=True,fallback_sigma=12)
    assert integer.push>0 and integer.win+integer.push+integer.lose==pytest.approx(1)
    values=[threshold_probabilities(7,t,residuals,integer_outcome=True,fallback_sigma=12)
            for t in (-4.5,3.5,4,4.5,7,7.5,10)]
    assert all(a.win>=b.win for a,b in zip(values,values[1:]))
    assert all(a.lose<=b.lose for a,b in zip(values,values[1:]))
    assert threshold_probabilities(7,7.5,residuals,integer_outcome=True,fallback_sigma=12).push==0
