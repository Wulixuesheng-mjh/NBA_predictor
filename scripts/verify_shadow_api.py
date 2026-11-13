"""Exercise the running HTTP API without writing prediction or acceptance rows."""
import asyncio
import argparse
import json
from pathlib import Path
import requests
from sqlalchemy import select,func
from app.config import settings
from app.database import async_session_factory
from app.models.game import Game
from app.models.prediction import Prediction
from datetime import datetime,timezone


async def main(base_url):
    active=Path(settings.model_path)/'active.json'
    before=active.read_bytes()
    async with async_session_factory() as session:
        count=await session.scalar(select(func.count()).select_from(Prediction))
        game=await session.scalar(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
            Game.scheduled_start_utc>datetime.now(timezone.utc)).order_by(Game.scheduled_start_utc).limit(1))
        assert game is not None
        gid=game.id
    response=requests.post(base_url.rstrip('/')+'/api/v1/predictions/',json=dict(
        game_id=gid,mode='hypothetical_preview',persist=False,lookback_games=10,total_line=234.5),
        headers={'X-API-Key':settings.admin_api_key},timeout=120)
    response.raise_for_status()
    result=response.json()
    assert result['model_release']['xgboost_version']=='20261022_050952'
    shadow=result['shadow_totals']
    assert shadow['status']=='diagnostic_shadow' and len(shadow['candidates'])==2
    assert all(c['training_data_until']=='2026-10-28' for c in shadow['candidates'])
    assert all(c['preprocessing_verified'] and len(c['processed_feature_vector'])==213 for c in shadow['candidates'])
    assert shadow['as_of']==result['as_of']
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Prediction))==count
    assert active.read_bytes()==before
    path=Path('/tmp/shadow_api_validation_20261029.json')
    path.write_text(json.dumps(dict(status='PASS',game_id=gid,primary_unchanged=True,
        ledger_unchanged=True,response=result),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',game_id=gid,primary=result['model_release']['xgboost_version'],
        shadow=shadow['version'],totals=[c['predicted_total'] for c in shadow['candidates']],ledger_unchanged=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',default='http://localhost:8000')
    asyncio.run(main(parser.parse_args().base_url))
