"""Export diagnostic payload/protocol evidence, never create forecasts."""
import argparse,asyncio,json
from pathlib import Path
from sqlalchemy import select,text
from app.database import async_session_factory
from app.models.shadow_observation import ShadowObservation,ShadowObservationRun
from app.models.game import Game
from app.services.prospective_service import digest


async def main(args):
    result=dict(kind='read_only_authentic_evidence_export',runs=[],rows=[])
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        for run in (await session.scalars(select(ShadowObservationRun))).all():
            assert digest(run.protocol)==run.protocol_hash
            result['runs'].append(dict(id=run.id,name=run.name,protocol=run.protocol,protocol_hash=run.protocol_hash))
        for row in (await session.scalars(select(ShadowObservation).order_by(ShadowObservation.id))).all():
            game=await session.get(Game,row.game_id)
            assert digest(row.payload)==row.payload_hash
            result['rows'].append(dict(id=row.id,run_id=row.run_id,game_id=row.game_id,
                as_of=row.as_of.isoformat(),created_at=row.created_at.isoformat(),
                captured_start=row.scheduled_start_utc.isoformat(),payload=row.payload,payload_hash=row.payload_hash,
                current_game=dict(source_day=str(game.source_game_date),scheduled_start_utc=game.scheduled_start_utc.isoformat(),
                    status=game.status,home_score=game.home_score,away_score=game.away_score)))
    p=Path(args.output)
    if p.exists():raise ValueError('Never overwrite source evidence')
    p.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(exported=len(result['rows']),writes=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
