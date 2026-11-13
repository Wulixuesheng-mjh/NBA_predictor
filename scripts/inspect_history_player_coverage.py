"""Read-only source identity coverage; do not equate imported row counts with completeness."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from app.collector.stat_backfill import source_id


def main(args):
    folder=Path(args.data_dir); games={}
    for row in csv.DictReader((folder/'Games.csv').open(encoding='utf-8-sig')):
        sid=source_id(row['gameId'])
        if sid.zfill(10)[3:5]==args.season[2:4]:
            games[sid]=row
    counts=Counter(); mismatches=[]; teams=Counter()
    examples=[]
    for row in csv.DictReader((folder/'PlayerStatistics.csv').open(encoding='utf-8-sig')):
        sid=source_id(row['gameId'])
        if sid not in games: continue
        game=games[sid];counts['rows']+=1
        if len(examples)<2: examples.append(row)
        team=int(row.get('playerteamId') or 0);teams[team]+=1
        expected={int(game['hometeamId']),int(game['awayteamId'])}
        if team in expected: counts['team_matches']+=1
        else:
            counts['team_mismatches']+=1
            if len(mismatches)<8:
                mismatches.append({k:row[k] for k in ('gameId','personId','playerteamId','opponentteamId') if k in row}
                    |dict(source_home=game['hometeamId'],source_away=game['awayteamId']))
    report=dict(season=args.season,counts=dict(counts),player_team_ids=dict(teams),mismatches=mismatches,
                source_row_examples=examples,writes=0)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--season',required=True)
    p.add_argument('--data-dir',default='/app/ml/saved_models/data_sources/kaggle/v525')
    p.add_argument('--output',required=True);main(p.parse_args())
