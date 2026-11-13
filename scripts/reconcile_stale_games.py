"""Plan from configured real source with explicit 429 handling, then apply checked rows."""
import argparse,asyncio,hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import httpx
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from app.collector.balldontlie_collector import BallDontLieCollector,BALLDONTLIE_BASE_URL


def state(g):
    return dict(id=g.id,nba_game_id=g.nba_game_id,status=g.status,home_team_id=g.home_team_id,
        away_team_id=g.away_team_id,home_score=g.home_score,away_score=g.away_score,
        source_game_date=str(g.source_game_date),scheduled_start_utc=str(g.scheduled_start_utc))


async def main(args):
    path=Path(args.plan);collector=BallDontLieCollector()
    if args.action=='plan':
        if path.exists(): raise ValueError('Plan is immutable')
        if not collector._api_key: raise ValueError('Configured source key is unavailable')
        async with async_session_factory() as session:
            games=list((await session.scalars(select(Game).where(Game.status=='Scheduled',
                Game.scheduled_start_utc<datetime.now(timezone.utc)-timedelta(hours=6)))).all())
            team_names={t.id:t.abbreviation for t in (await session.scalars(select(Team))).all()}
        report=dict(created_at=datetime.now(timezone.utc).isoformat(),responses=[],updates=[],unresolved=[])
        async with httpx.AsyncClient(timeout=30) as client:
            for day in sorted({str(g.source_game_date) for g in games}):
                response=None
                for attempt in range(3):
                    response=await client.get(f'{BALLDONTLIE_BASE_URL}/games',params={'dates[]':day,'per_page':100},
                        headers={'Authorization':collector._api_key})
                    if response.status_code!=429: break
                    wait=int(response.headers.get('Retry-After','60'))
                    print(json.dumps(dict(day=day,rate_limited=True,retry_after=wait)),flush=True)
                    if wait>60 or attempt==2: break
                    await asyncio.sleep(max(60,wait))
                if response.status_code!=200:
                    report['unresolved'].append(dict(day=day,http_status=response.status_code))
                    continue
                payload=response.json()
                report['responses'].append(dict(day=day,observed_at=datetime.now(timezone.utc).isoformat(),data=payload))
                lookup={f'bdl-{r["id"]}':r for r in payload.get('data',[])}
                for game in [g for g in games if str(g.source_game_date)==day]:
                    raw=lookup.get(game.nba_game_id)
                    if not raw or raw.get('status')!='Final':
                        report['unresolved'].append(dict(game_id=game.id,reason='no_confirmed_final'));continue
                    home=(raw.get('home_team') or {}).get('abbreviation')
                    away=(raw.get('visitor_team') or {}).get('abbreviation')
                    h,a=raw.get('home_team_score'),raw.get('visitor_team_score')
                    if (home!=team_names[game.home_team_id] or away!=team_names[game.away_team_id]
                            or str(raw.get('date'))[:10]!=day or not h or not a or h==a or not 120<=h+a<=400):
                        report['unresolved'].append(dict(game_id=game.id,reason='binding_or_score_conflict'));continue
                    report['updates'].append(dict(before=state(game),source=raw))
                path.parent.mkdir(parents=True,exist_ok=True)
                await asyncio.sleep(15)
        with path.open('x') as stream: json.dump(report,stream,indent=2)
        print(json.dumps(dict(updates=len(report['updates']),unresolved=report['unresolved'])),flush=True)
    else:
        report=json.loads(path.read_text());applied=[];skipped=[]
        async with async_session_factory() as session:
            teams=await collector._get_team_map(session)
            for row in report['updates']:
                game=await session.get(Game,row['before']['id'])
                if state(game)!=row['before']: skipped.append(game.id);continue
                if not await collector._upsert_game(session,row['source'],teams,'Final'):
                    raise ValueError('Previously checked source update was refused')
                applied.append(game.id)
            await session.commit()
        result=dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),applied=applied,skipped=skipped,
                    statistics_not_assumed_complete=True)
        Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','apply']);p.add_argument('--plan',required=True)
    p.add_argument('--output',default='/tmp/stale_apply.json');asyncio.run(main(p.parse_args()))
