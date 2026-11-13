from contextlib import asynccontextmanager
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json
import numpy as np
import pytest
from app.services import win_observation_service as service
from app.models.game import Game
from ml.src.features.feature_builder import FeatureVector,FEATURE_NAMES,FeatureBuilder


def test_atomic_observation_publication_never_overwrites(tmp_path):
    path=tmp_path/'records/1.json'
    assert service.publish_once(path,{'first':True})
    original=path.read_bytes()
    assert not service.publish_once(path,{'changed':True})
    assert path.read_bytes()==original and len(list(path.parent.glob('*')))==1


def test_win_capture_window_is_real_time_only():
    now=datetime.now(timezone.utc)
    game=SimpleNamespace(status='Scheduled',time_precision='exact',scheduled_start_utc=now+timedelta(minutes=55))
    assert service.in_window(game,now)
    for delta in (-1,49,61):
        game.scheduled_start_utc=now+timedelta(minutes=delta)
        assert not service.in_window(game,now)
    game.scheduled_start_utc=now+timedelta(minutes=55);game.time_precision='date'
    assert not service.in_window(game,now)


@pytest.mark.asyncio
async def test_win_capture_is_unique_and_tampering_blocks_further_capture(seeded_db,tmp_path,monkeypatch):
    from ml.src import win_shadow
    from ml.src.models import xgboost_model
    from app.services import injury_snapshot_reader
    now=datetime.now(timezone.utc)
    game=await seeded_db.get(Game,seeded_db._game_ids['scheduled'])
    game.status='Scheduled';game.time_precision='exact';game.scheduled_start_utc=now+timedelta(minutes=55)
    game.source_game_date=now.date();await seeded_db.flush()
    baseline=dict(feature_dim=166,feature_semantic_version=5,training_data_until='2020-01-01',xgboost_version='old')
    manifest=win_shadow.schema()|dict(training_data_until='2020-01-01')
    protocol=dict(kind='win_only_diagnostic',baseline=baseline,version='test')
    monkeypatch.setattr(service,'checked_protocol',lambda *args:(protocol,manifest))
    @asynccontextmanager
    async def factory(): yield seeded_db
    monkeypatch.setattr(service,'async_session_factory',factory)
    fv=FeatureVector(features=np.ones(len(FEATURE_NAMES)),home_games_available=10,away_games_available=10)
    monkeypatch.setattr(FeatureBuilder,'build_features',AsyncMock(return_value=fv))
    monkeypatch.setattr(win_shadow,'predict_win_bundle',lambda *a,**k:[{'win_prob_home':.7}])
    class Old:
        def __init__(self):
            self.version='old';self.model_win=SimpleNamespace(n_features_in_=166,predict_proba=lambda X:np.array([[.4,.6]]))
        def load(self,*args): pass
    monkeypatch.setattr(xgboost_model,'XGBoostTrainer',Old)
    monkeypatch.setattr(injury_snapshot_reader,'latest_injury_snapshot',AsyncMock(return_value=None))
    (tmp_path/'records').mkdir()
    assert await service.capture_due(tmp_path,tmp_path)==1
    path=tmp_path/'records'/f'{game.id}.json';before=path.read_bytes()
    assert await service.capture_due(tmp_path,tmp_path)==0
    assert before==path.read_bytes()
    record=json.loads(before)['payload']
    assert record['acceptance_run'] is False and set(record['win_probabilities'])=={'baseline','candidate'}
    assert len(record['baseline_input']['feature_vector'])==166
    assert len(record['candidate_input']['feature_vector'])==213
    assert record['injury_snapshots']=={'home':None,'away':None}
    broken=json.loads(before);broken['payload']['win_probabilities']['candidate']=.99
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='integrity'): await service.capture_due(tmp_path,tmp_path)
