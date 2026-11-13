"""Release preprocessing, immutable inputs and exact replay at the same boundary."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from sqlalchemy import select

from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureVector
from ml.src.inference_snapshot import validate_snapshot
from ml.src.models.xgboost_model import XGBoostPrediction
from ml.src.predictor import Predictor


def release(legacy=False):
    return dict(version='pinned', xgboost_version='pinned', mlp_version=None,
        feature_dim=166 if legacy else len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES[:166] if legacy else FEATURE_NAMES.copy(),
        feature_semantic_version=5 if legacy else CACHE_FORMAT_VERSION,
        disabled_features=[] if legacy else ['home_injury_firepower_loss'],
        training_data_until='2026-01-01', calibration_enabled=False)


def predictor(manifest):
    expected = manifest['feature_dim']
    trainer = SimpleNamespace(is_trained=True, version='pinned',
        model_win=SimpleNamespace(n_features_in_=expected))
    def predict(X):
        assert X.shape == (1, expected)
        return [XGBoostPrediction(win_prob_home=.6, win_prob_away=.4,
            predicted_spread=4.23, spread_confidence=.5, predicted_total=227.23,
            total_confidence=.5, model_version='pinned')]
    trainer.predict = Mock(side_effect=predict)
    return Predictor(xgb_trainer=trainer, release_manifest=manifest)


@pytest.mark.parametrize('legacy', [True, False])
def test_model_receives_release_units_and_masks_and_snapshot_replays_once(legacy):
    original = release(legacy)
    p = predictor(original)
    # Caller mutations must not change this loaded release.
    original['disabled_features'].append('away_injury_firepower_loss')
    fv = FeatureVector(features=np.full(len(FEATURE_NAMES), 220.),
                       home_games_available=10, away_games_available=10)
    result = p.predict(fv, apply_calibration=False)
    snapshot = result.processed_inputs['xgboost']
    X = validate_snapshot(snapshot, model_name='xgboost', model_version='pinned', manifest=p.release_manifest)
    np.testing.assert_array_equal(X, p.xgb_trainer.predict.call_args.args[0])
    assert X.shape[1] == (166 if legacy else len(FEATURE_NAMES))
    assert X[0, FEATURE_NAMES.index('expected_total')] == (110 if legacy else 220)
    assert X[0, FEATURE_NAMES.index('home_injury_firepower_loss')] == (220 if legacy else 0)
    assert X[0, FEATURE_NAMES.index('away_injury_firepower_loss')] == 220
    assert (fv.features == 220).all()
    assert 'processed_inputs' not in result.to_dict()

    source = Path(__file__).resolve().parents[2]/'scripts/replay_prediction.py'
    spec = importlib.util.spec_from_file_location('snapshot_replay', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    row = SimpleNamespace(features_json={'_release_manifest': p.release_manifest,
        '_inference_snapshot': snapshot}, feature_names=snapshot['feature_names'],
        feature_vector=snapshot['feature_vector'], calibrated=False,
        model_name='xgboost', model_version='pinned')
    p._prepare_release_features = Mock(side_effect=AssertionError('Double preprocessing'))
    raw, final = module.replay_processed_row(row, p)
    np.testing.assert_array_equal(X, p.xgb_trainer.predict.call_args.args[0])
    assert raw.predicted_total == 227.23 and final.predicted_total == 227.0
    corrupted = deepcopy(snapshot); corrupted['feature_vector'][0] += 1
    with pytest.raises(ValueError, match='checksum'):
        validate_snapshot(corrupted, model_name='xgboost', model_version='pinned', manifest=p.release_manifest)
    with pytest.raises(ValueError, match='provenance'):
        validate_snapshot(snapshot, model_name='mlp', model_version='pinned', manifest=p.release_manifest)

    from app.services.prospective_service import validate_record_inputs
    output = dict(release_manifest=p.release_manifest, processed_inputs=result.processed_inputs)
    paired = SimpleNamespace(feature_names=FEATURE_NAMES, feature_vector=fv.features.tolist(),
                             baseline_output=output, candidate_output=deepcopy(output))
    protocol = dict(baseline=p.release_manifest, candidate=p.release_manifest)
    validate_record_inputs(paired, protocol)
    paired.candidate_output['processed_inputs'] = {}
    with pytest.raises(ValueError, match='absent'):
        validate_record_inputs(paired, protocol)


@pytest.mark.parametrize('problem', ['order', 'semantic', 'mask', 'nan', 'dimension'])
def test_incompatible_preprocessing_rejected_before_model_call(problem):
    manifest = release()
    X = np.ones((1, len(FEATURE_NAMES)))
    if problem == 'order': manifest['feature_names'].reverse()
    elif problem == 'semantic': manifest['feature_semantic_version'] += 1
    elif problem == 'mask': manifest['disabled_features'] = ['unknown_feature']
    elif problem == 'nan': X[0, 0] = np.nan
    else: manifest['feature_dim'] -= 1
    p = predictor(manifest)
    with pytest.raises(ValueError): p._prepare_release_features(X)
    p.xgb_trainer.predict.assert_not_called()


def test_manifest_disabled_calibration_wins_over_global_switch(monkeypatch):
    from ml.src.config import ml_settings
    monkeypatch.setattr(ml_settings, 'calibration_enabled', True)
    p = predictor(release())
    p.calibrator = SimpleNamespace(is_fitted=True,
        apply=Mock(side_effect=AssertionError('Release disabled calibration')))
    result = p.predict(FeatureVector(features=np.ones(len(FEATURE_NAMES)),
        home_games_available=10, away_games_available=10))
    assert not result.calibration_applied and result.xgboost.predicted_total == 227
    p.calibrator.apply.assert_not_called()


@pytest.mark.asyncio
async def test_pregame_ledger_stores_processed_legacy_vector_separately_from_builder(seeded_db, monkeypatch):
    from app.models.game import Game
    from app.models.prediction import Prediction
    from app.services.prediction_service import PredictionService
    from app.services import shadow_total_service
    from ml.src.features.feature_builder import FeatureBuilder
    p = predictor(release(legacy=True))
    monkeypatch.setattr(PredictionService, '_get_predictor', lambda: p)
    monkeypatch.setattr(p, 'explain_prediction', lambda **kwargs: None)
    monkeypatch.setattr(shadow_total_service, 'shadow_totals', lambda *args, **kwargs: None)
    fv = FeatureVector(features=np.full(len(FEATURE_NAMES), 220.),
                       home_games_available=10, away_games_available=10)
    monkeypatch.setattr(FeatureBuilder, 'build_features', AsyncMock(return_value=fv))
    start = datetime.now(timezone.utc) + timedelta(days=1)
    game = Game(nba_game_id='future-ledger-test', season='2026-27', home_team_id=1,
        away_team_id=2, game_date=start, scheduled_start_utc=start, time_precision='exact',
        source_game_date=start.date(), status='Scheduled', game_type='regular')
    seeded_db.add(game); await seeded_db.flush()
    output = await PredictionService.generate_prediction(seeded_db, game_id=game.id,
        prediction_mode='pregame', persist=True)
    row = await seeded_db.scalar(select(Prediction).where(Prediction.game_id == game.id))
    assert row is not None and len(row.feature_vector) == 166
    assert row.feature_vector[FEATURE_NAMES.index('expected_total')] == 110
    assert row.features_json['_builder_snapshot']['feature_vector'][FEATURE_NAMES.index('expected_total')] == 220
    assert row.feature_semantic_version == '5'
    assert 'processed_inputs' not in output
    validate_snapshot(row.features_json['_inference_snapshot'], model_name='xgboost',
        model_version='pinned', manifest=row.features_json['_release_manifest'])
