"""Read-only coverage and identity checks before expanding a historical season."""
import argparse
import asyncio
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.collector.kaggle_collector import KaggleCollector
from app.collector.stat_backfill import source_id,source_start


async def main(args):
    folder=Path(args.data_dir); target={}
    for r in csv.DictReader((folder/'Games.csv').open(encoding='utf-8-sig')):
        sid=source_id(r['gameId']); code=sid.zfill(10)
        if len(code)!=10 or code[3:5]!=args.season[2:4] or KaggleCollector._game_type(sid) not in ('regular','play_in','playoffs'):
            continue
        target[sid]=r
    team_counts=Counter();player_counts=Counter()
    for r in csv.DictReader((folder/'TeamStatistics.csv').open(encoding='utf-8-sig')):
        sid=source_id(r['gameId'])
        if sid in target and float(r.get('fieldGoalsAttempted') or 0)>0:
            team_counts[sid]+=1
    for r in csv.DictReader((folder/'PlayerStatistics.csv').open(encoding='utf-8-sig')):
        sid=source_id(r['gameId'])
        if sid in target:
            player_counts[sid]+=1
    async with async_session_factory() as session:
        existing=defaultdict(list)
        for g in (await session.scalars(select(Game))).all():
            if source_id(g.nba_game_id) in target:
                existing[source_id(g.nba_game_id)].append(g)
    report=dict(season=args.season,source_games=len(target),source_two_team_rows=sum(n==2 for n in team_counts.values()),
        source_player_games=len(player_counts),source_player_rows=sum(player_counts.values()),
        existing_games=sum(map(len,existing.values())),missing_games=sum(s not in existing for s in target),
        duplicate_canonical_ids={s:[g.id for g in rows] for s,rows in existing.items() if len(rows)>1},
        existing_season_labels=dict(Counter(g.season for rows in existing.values() for g in rows)),
        source_day_differences=sum(str(g.source_game_date)!=str(source_start(target[s]['gameDateTimeEst']).date())
                                   for s,rows in existing.items() for g in rows),
        score_conflicts=[dict(game_id=g.id,nba_game_id=s) for s,rows in existing.items() for g in rows
                        if (g.home_score,g.away_score)!=(int(target[s]['homeScore']),int(target[s]['awayScore']))],
        examples=[dict(nba_game_id=s,date=r['gameDateTimeEst'],home_score=r['homeScore'],away_score=r['awayScore'])
                  for s,r in list(target.items())[:3]],writes=0)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--season',default='2022-23')
    p.add_argument('--data-dir',default='/app/ml/saved_models/data_sources/kaggle/v525')
    p.add_argument('--output',default='/app/ml/saved_models/data_quality/history_expansion_probe_20261030.json')
    asyncio.run(main(p.parse_args()))
