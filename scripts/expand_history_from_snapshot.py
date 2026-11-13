"""Plan then atomically bind one season to an existing checksummed source.

No scores or team identities are overwritten on conflict. Only source-provided
tipoffs are installed; no injury publication times are inferred.
"""
import argparse
import asyncio
from collections import defaultdict
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path

from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from app.collector.kaggle_collector import KaggleCollector
from app.collector.source_snapshot import checksum
from app.collector.stat_backfill import backfill_csv_stats,source_id,source_start
from app.services.data_quality_service import bump_data_revision


def state(game):
    return dict(id=game.id,nba_game_id=game.nba_game_id,season=game.season,
        source_game_date=str(game.source_game_date),game_date=game.game_date.isoformat(),
        scheduled_start_utc=game.scheduled_start_utc.isoformat() if game.scheduled_start_utc else None,
        time_precision=game.time_precision,game_type=game.game_type,status=game.status,
        home_team_id=game.home_team_id,away_team_id=game.away_team_id,
        home_score=game.home_score,away_score=game.away_score)


def source_games(folder,season,team_map):
    result={}
    for row in csv.DictReader((folder/'Games.csv').open(encoding='utf-8-sig')):
        sid=source_id(row['gameId']); code=sid.zfill(10)
        kind=KaggleCollector._game_type(sid)
        if len(code)!=10 or code[3:5]!=season[2:4] or kind not in ('regular','play_in','playoffs'):
            continue
        if sid in result:
            raise ValueError('Duplicate game in source: '+sid)
        start=source_start(row['gameDateTimeEst'])
        if start.year not in (int(season[:4]),int(season[:4])+1):
            raise ValueError('Source season ID and calendar year conflict')
        home,away=int(row['homeScore']),int(row['awayScore'])
        if min(home,away)<=0 or not 120<=home+away<=400:
            raise ValueError('Invalid score in expansion source')
        result[sid]=dict(nba_game_id=sid,season=season,source_game_date=str(start.date()),
            game_date=start.astimezone(timezone.utc).isoformat(),
            scheduled_start_utc=start.astimezone(timezone.utc).isoformat(),time_precision='exact',
            game_type=kind,status='Final',home_team_id=team_map[int(row['hometeamId'])],
            away_team_id=team_map[int(row['awayteamId'])],home_score=home,away_score=away)
    return result


def verified_source(folder):
    path=folder/'source_manifest.json';manifest=json.loads(path.read_text())
    for name in ('Games.csv','TeamStatistics.csv','PlayerStatistics.csv','Players.csv'):
        if checksum(folder/name)!=manifest['files'][name]['sha256']:
            raise ValueError('Source checksum mismatch: '+name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def plan(args):
    folder=Path(args.data_dir);source_sha=verified_source(folder)
    path=Path(args.plan)
    if path.exists():
        raise ValueError('Expansion plan is immutable; choose a new path')
    async with async_session_factory() as session:
        team_map=dict((await session.execute(select(Team.nba_team_id,Team.id))).all())
        targets=source_games(folder,args.season,team_map)
        by_id=defaultdict(list);by_match=defaultdict(list)
        for game in (await session.scalars(select(Game))).all():
            by_id[source_id(game.nba_game_id)].append(game)
            by_match[(str(game.source_game_date),game.home_team_id,game.away_team_id)].append(game)
        rows=[];issues=[]
        for sid,target in targets.items():
            matches=by_id.get(sid) or by_match.get((target['source_game_date'],target['home_team_id'],target['away_team_id']),[])
            if len(matches)>1:
                issues.append(dict(nba_game_id=sid,reason='ambiguous existing games'));continue
            game=matches[0] if matches else None
            if game and any(getattr(game,k)!=target[k] for k in ('home_team_id','away_team_id','home_score','away_score')):
                issues.append(dict(nba_game_id=sid,reason='score or team conflict'));continue
            rows.append(dict(target=target,before=state(game) if game else None))
    report=dict(season=args.season,source_dir=str(folder),source_manifest_sha256=source_sha,
        executor_sha256=checksum(Path(__file__)),created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_games=len(targets),new_games=sum(r['before'] is None for r in rows),
        existing_games=sum(r['before'] is not None for r in rows),issues=issues,rows=rows)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


async def apply(args):
    path=Path(args.plan);report=json.loads(path.read_text());folder=Path(args.data_dir)
    assert not report['issues'] and report['season']==args.season
    assert report['source_manifest_sha256']==verified_source(folder)
    assert report['executor_sha256']==checksum(Path(__file__))
    changed=created=already=0
    async with async_session_factory() as session:
        live=defaultdict(list)
        for game in (await session.scalars(select(Game))).all():
            live[source_id(game.nba_game_id)].append(game)
        for row in report['rows']:
            target=row['target'];before=row['before']
            if before:
                game=await session.get(Game,before['id'])
            else:
                matches=live.get(target['nba_game_id'],[])
                assert len(matches)<=1
                game=matches[0] if matches else None
            fields={k:v for k,v in target.items() if k!='nba_game_id'}
            if game:
                current=state(game)
                if all(current[k]==v for k,v in fields.items()):
                    already+=1;continue
                assert before is not None and current==before,'Database changed since expansion plan'
                changed+=1
            else:
                assert before is None
                game=Game(nba_game_id=target['nba_game_id']);session.add(game);created+=1
            for key,value in fields.items():
                if key in ('game_date','scheduled_start_utc'):
                    value=datetime.fromisoformat(value)
                elif key=='source_game_date':
                    value=datetime.fromisoformat(value).date()
                setattr(game,key,value)
            game.score_complete=True
        if changed or created:
            await bump_data_revision(session,'checked-history-expansion:'+args.season)
        await session.commit()
    starts=[datetime.fromisoformat(r['target']['game_date']) for r in report['rows']]
    stats=await backfill_csv_stats(async_session_factory,folder,
        start_date=str(min(starts).date()),end_date=str(max(starts).date()+timedelta(days=1)),
        seasons=[args.season]) if not args.metadata_only else None
    result=dict(season=args.season,plan_sha256=checksum(path),metadata_created=created,
        metadata_updated=changed,metadata_already_current=already,stats=stats,
        injury_times_modified=False)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','apply'])
    p.add_argument('--season',default='2022-23');p.add_argument('--data-dir',default='/app/ml/saved_models/data_sources/kaggle/v525')
    p.add_argument('--plan',required=True);p.add_argument('--output',default='/tmp/history_expansion_result.json')
    p.add_argument('--metadata-only',action='store_true');args=p.parse_args()
    asyncio.run(plan(args) if args.action=='plan' else apply(args))
