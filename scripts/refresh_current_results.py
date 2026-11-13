"""Bounded score/schedule refresh; archive real API responses before DB writes."""
import argparse
import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import httpx
from app.collector.balldontlie_collector import BallDontLieCollector, BALLDONTLIE_BASE_URL


async def main(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    collector=BallDontLieCollector()
    if not collector._api_key: raise ValueError('Configured source key unavailable')
    day,end=date.fromisoformat(args.start),date.fromisoformat(args.end)
    report=dict(start=str(day),end=str(end),days=[],failed_days=[],new_finals=0,invalid=[])
    async with httpx.AsyncClient(timeout=40) as client:
        async with collector.db_session_factory() as session:
            teams=await collector._get_team_map(session)
            while day<=end:
                payload=None
                for attempt in range(3):
                    response=await client.get(BALLDONTLIE_BASE_URL+'/games',
                        params={'dates[]':str(day),'per_page':100},
                        headers={'Authorization':collector._api_key})
                    if response.status_code==200:
                        payload=response.json();break
                    if response.status_code!=429:break
                    delay=min(60,max(15,int(response.headers.get('Retry-After','60'))))
                    print(json.dumps(dict(day=str(day),rate_limited=True,retry=attempt)),flush=True)
                    if attempt<2:await asyncio.sleep(delay)
                if payload is None:
                    report['failed_days'].append(dict(day=str(day),status=response.status_code))
                else:
                    # Pin observed source first. Missing/paginated responses never imply zero games.
                    source=output/(str(day)+'.json')
                    if source.exists():raise ValueError('Refuse to overwrite observed source')
                    source.write_text(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),
                        payload=payload),indent=2),encoding='utf-8')
                    if (payload.get('meta') or {}).get('next_cursor'):
                        raise ValueError('Unexpected pagination; refuse incomplete date')
                    finals=0
                    for game in payload.get('data',[]):
                        h,a=game.get('home_team_score'),game.get('visitor_team_score')
                        final=game.get('status')=='Final'
                        valid=(str(game.get('date'))[:10]==str(day)
                            and (game.get('home_team') or {}).get('abbreviation') in teams
                            and (game.get('visitor_team') or {}).get('abbreviation') in teams)
                        if final:valid=valid and isinstance(h,int) and isinstance(a,int) and h>0 and a>0 and h!=a and 120<=h+a<=400
                        if not valid:
                            report['invalid'].append(dict(day=str(day),id=game.get('id')));continue
                        changed=await collector._upsert_game(session,game,teams,'Final' if final else 'Scheduled')
                        finals+=int(final);report['new_finals']+=int(changed)
                    await session.commit()
                    report['days'].append(dict(day=str(day),games=len(payload.get('data',[])),finals=finals))
                    print(json.dumps(report['days'][-1]),flush=True)
                (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                day+=timedelta(days=1)
                if day<=end:await asyncio.sleep(15)
    print(json.dumps(report),flush=True)
    if report['failed_days'] or report['invalid']:raise ValueError('Refresh incomplete; inspect archived summary')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
