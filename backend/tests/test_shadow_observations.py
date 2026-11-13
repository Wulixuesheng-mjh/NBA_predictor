from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import pytest
from app.services import shadow_observation_service as service
from app.services.prospective_service import digest
from app.models.shadow_observation import ShadowObservationRun,ShadowObservation
from app.models.game import Game


def test_capture_window_rejects_early_late_started_and_imprecise_games():
    now=datetime.now(timezone.utc)
    game=SimpleNamespace(status='Scheduled',time_precision='exact',scheduled_start_utc=now+timedelta(minutes=55))
    assert service.in_window(game,now)
    for minutes in (-1,49,61,120):
        game.scheduled_start_utc=now+timedelta(minutes=minutes)
        assert not service.in_window(game,now)
    game.scheduled_start_utc=now+timedelta(minutes=55);game.status='InProgress'
    assert not service.in_window(game,now)
    game.status='Scheduled';game.time_precision='date'
    assert not service.in_window(game,now)


@pytest.mark.asyncio
async def test_settlement_uses_only_finished_scores_and_keeps_forecast_immutable(seeded_db,monkeypatch):
    session=seeded_db
    monkeypatch.setattr(service,'verify_protocol',lambda run:True)
    game=await session.get(Game,session._game_ids['final'])
    game.score_complete=True
    run=ShadowObservationRun(name='diagnostic',protocol={'catalog_version':'test'},protocol_hash='test',enabled=True)
    session.add(run);await session.flush()
    payload={'protocol_hash':'test','totals':{'baseline':220.,'direct':222.,'residual':223.},'market':None}
    before=digest(payload)
    row=ShadowObservation(run_id=run.id,game_id=game.id,as_of=datetime.now(timezone.utc),
        scheduled_start_utc=datetime.now(timezone.utc)+timedelta(hours=1),payload=payload,payload_hash=before)
    session.add(row);await session.flush()
    result=await service.progress(session,run)
    assert result['settled']==1 and result['acceptance_run'] is False
    assert result['metrics']['baseline']['mae']==3
    assert result['metrics']['residual']['mae']==0
    assert result['market_samples']==0 and row.payload_hash==before and digest(row.payload)==before
    game.status='InProgress';await session.flush()
    result=await service.progress(session,run)
    assert result['settled']==0 and result['metrics']=={}
    row.payload_hash='corrupt';await session.flush()
    result=await service.progress(session,run)
    assert result['integrity_errors']==1 and result['settled']==0


@pytest.mark.asyncio
async def test_status_route_is_not_shadowed_by_prediction_id(client):
    response=await client.get('/api/v1/predictions/shadow-status')
    assert response.status_code==200 and response.json()['runs']==[]
