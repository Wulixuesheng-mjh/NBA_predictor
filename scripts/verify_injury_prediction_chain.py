"""Synthetic API chain verification, refusing production database mutation."""
import asyncio,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import httpx
from sqlalchemy import select
from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.game import Game
from app.models.player import Player
from app.models.prediction import Prediction
from app.services.injury_snapshot_reader import latest_injury_snapshot,snapshot_statuses


async def main():
    assert 'nba-mlp-preflight-db-20261110' in settings.database_url and not settings.scheduler_enabled
    async with async_session_factory() as db:
        game=await db.get(Game,32798)
        game.status='Scheduled';game.scheduled_start_utc=datetime.now(timezone.utc)+timedelta(hours=2)
        game.game_date=game.scheduled_start_utc;game.time_precision='exact';game.score_complete=False
        game.home_score=game.away_score=0
        player=await db.scalar(select(Player).where(Player.full_name=='Zion Williamson'))
        pid=player.id;tid=game.away_team_id;home=game.home_team_id
        await db.commit()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers={'X-API-Key':settings.admin_api_key}) as c:
            paths=app.openapi()['paths'];base=next(p for p in paths if p.endswith('/injuries/'))
            pred=next(p for p in paths if p.endswith('/predictions/'))
            bad=await c.post(base,json=dict(player_id=pid,game_id=32798,team_id=999999,injury_status='Out'))
            assert bad.status_code==400,bad.text
            res=await c.post(base,json=dict(player_id=pid,game_id=32798,team_id=tid,injury_status='Out'))
            assert res.status_code==201,res.text
            row=res.json();assert row['game_id']==32798 and row['team_id']==tid
            listing=await c.get(base,params=dict(game_id=32798,team_id=tid))
            assert any(r['id']==row['id'] and r['game_id']==32798 for r in listing.json()['injuries'])
            async with async_session_factory() as db:
                snapshot=await latest_injury_snapshot(db,32798,tid,datetime.now(timezone.utc))
                assert snapshot_statuses(snapshot)[pid]=='Out' and not snapshot.is_complete
                assert await latest_injury_snapshot(db,32798,home,datetime.now(timezone.utc)) is None
                before=snapshot.observed_at
            report=await c.post(base+'reports',json=dict(game_id=32798,team_id=tid,is_complete=True))
            assert report.status_code==201 and report.json()['is_complete'],report.text
            response=await c.post(pred,json=dict(game_id=32798,mode='pregame',persist=True))
            assert response.status_code==200,response.text
            async with async_session_factory() as db:
                rows=list((await db.scalars(select(Prediction).where(Prediction.game_id==32798,Prediction.created_at>=before))).all())
                assert {p.model_name for p in rows}=={'xgboost','mlp'}
                assert all(p.feature_vector and p.features_json.get('_inference_snapshot') and p.injury_snapshot for p in rows)
                assert all(p.features_json['away_players_out_count']>=1 for p in rows)
                for p in rows:
                    i=p.feature_names.index('away_players_out_count')
                    assert p.feature_vector[i]>=1, 'Submitted Out must reach each actual model input'
                assert any('injury' in k and v for k,v in rows[0].injury_snapshot.items()),rows[0].injury_snapshot
            removed=await c.delete(base+str(row['id']));assert removed.status_code==200
            async with async_session_factory() as db:
                after=await latest_injury_snapshot(db,32798,tid,datetime.now(timezone.utc))
                assert not after.is_complete and pid not in snapshot_statuses(after)
                old=await latest_injury_snapshot(db,32798,tid,before)
                assert snapshot_statuses(old)[pid]=='Out'
    result=dict(status='PASS',scope='synthetic isolated PostgreSQL API test',wrong_team_rejected=True,
        create_response_binding=True,list_binding=True,partial_out_read=True,unknown_not_healthy=True,
        complete_confirmation=True,both_models_persisted=True,inference_snapshots_saved=True,
        injury_features_present=True,out_count_in_both_model_inputs=True,delete_appends_partial=True,past_snapshot_unchanged=True,
        production_mutations=0,real_acceptance_games=0)
    Path('/out/chain_api_strict.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':asyncio.run(main())
