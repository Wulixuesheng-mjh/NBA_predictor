"""Apply validated official tables atomically; retain immutable source evidence."""
import argparse
import asyncio
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from app.database import async_session_factory
from app.collector.gamebook_parser import normalize,FIELDS
from app.collector.player_aliases import PLAYER_ALIASES, SCORER_NAME_IDS
from app.collector.advanced_stats import derive_game_advanced_stats
from app.models.game import Game
from app.models.player import Player
from app.models.team import Team
from app.models.game_player_stats import GamePlayerStats
from app.models.team_game_stats import TeamGameStats
from app.models.box_score_snapshot import BoxScoreSnapshot
from app.services.data_quality_service import refresh_game_quality,bump_data_revision


# Scorer-only identities verified against the NBA profile, independent of the
# frozen inference alias table. Never infer a player's historical team here.
SCORER_ONLY_IDENTITIES = {
    'Karim López': (1643510, 'https://www.nba.com/player/1643510/karim-lopez'),
    'Moritz Wagner': (1629021, 'https://www.nba.com/player/1629021/moritz-wagner'),
    'Hugo González': (1642864, 'https://www.nba.com/player/1642864/hugo-gonzalez'),
    'Luka Dončić': (1629029, 'https://www.nba.com/player/1629029/luka-doncic'),
    'Tacko Fall': (1629605, 'https://www.nba.com/player/1629605/tacko-fall'),
}


def name_keys(value):
    key=normalize(value).replace('’',"'")
    return {key,re.sub(r'\s+(?:jr|sr|ii|iii|iv|v)$','',key)}


def scorer_player_ids(name,local_names,names,by_id,official_ids):
    exact=normalize(name).replace('’',"'")
    official=official_ids.get(exact)
    if official is not None:
        # A verified official ID also handles accented spellings absent from
        # the name index. Missing IDs fail closed instead of selecting a legacy
        # same-name record merely because it is the only name candidate.
        return {pid for pid,player in by_id.items() if player.nba_player_id==official}
    keys=name_keys(name)
    ids=local_names.get(exact) or names.get(exact)
    if not ids:
        ids=set().union(*(local_names.get(k,set()) for k in keys))
    if not ids:
        ids=set().union(*(names.get(k,set()) for k in keys))
    return ids


async def run(args):
    folder=Path(args.input)
    destination=Path('/app/ml/saved_models/data_sources/official_gamebooks')
    report=dict(applied=0,already_complete=0,rejected=0,player_rows=0,issues=[])
    async with async_session_factory() as session:
        players=(await session.execute(select(Player.id,Player.full_name,Player.first_name,Player.last_name,Player.nba_player_id))).all()
        teams={t.id:t for t in (await session.execute(select(Team))).scalars()}
    names=defaultdict(set)
    for pid,full,first,last,nba_pid in players:
        for value in (full or '',f'{first or ""} {last or ""}',*PLAYER_ALIASES.get(nba_pid,())):
            for key in name_keys(value):
                names[key].add(pid)
    by_id={r.id:r for r in players}
    official_ids={normalize(name):nba_id for name,nba_id in SCORER_NAME_IDS.items()}
    official_ids.update({normalize(name):value[0] for name,value in SCORER_ONLY_IDENTITIES.items()})
    paths=[p for p in sorted(folder.glob('*.json')) if p.name not in ('download_manifest.json','download_progress.json')]
    if args.limit:
        paths=paths[:args.limit]
    for index,path in enumerate(paths):
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload.get('status')!='verified':
            continue
        gid=payload['game_id']
        try:
            async with async_session_factory() as session:
                game=await session.get(Game,gid)
                if not game or game.season not in args.seasons or game.status!='Final':
                    raise ValueError('Not a completed game in the explicitly allowed seasons')
                if payload['parsed']['game_date'] != str(game.source_game_date or game.game_date.date()):
                    raise ValueError('Game date changed since source validation')
                existing_source=await session.scalar(select(BoxScoreSnapshot.id).where(
                    BoxScoreSnapshot.game_id==gid,BoxScoreSnapshot.sha256==payload['sha256']))
                if existing_source and game.player_stats_complete:
                    report['already_complete']+=1
                    continue
                filename=payload['filename']
                if Path(filename).name!=filename or not payload['url'].startswith('https://statsdmz.nba.com/pdfs/'):
                    raise ValueError('Invalid official source path')
                source=folder/filename
                if hashlib.sha256(source.read_bytes()).hexdigest()!=payload['sha256']:
                    raise ValueError('Raw PDF hash mismatch')
                existing=list((await session.execute(select(GamePlayerStats).where(GamePlayerStats.game_id==gid))).scalars())
                team_values,player_values=[],[]
                mapped_ids=set()
                for table in payload['parsed']['teams']:
                    tid=game.home_team_id if table['is_home'] else game.away_team_id
                    expected=game.home_score if table['is_home'] else game.away_score
                    if table['team_id']!=tid or normalize(table['name'])!=normalize(teams[tid].full_name) or table['totals']['pts']!=expected:
                        raise ValueError('Team/score binding mismatch')
                    local_names=defaultdict(set)
                    for row in existing:
                        if row.team_id==tid:
                            player=by_id[row.player_id]
                            for key in name_keys(player.full_name or ''):
                                local_names[key].add(row.player_id)
                    ids_in_table=set()
                    for row in table['players']:
                        ids=scorer_player_ids(row['name'],local_names,names,by_id,official_ids)
                        if len(ids)!=1:
                            raise ValueError('Ambiguous/unmatched player: '+row['name'])
                        pid=next(iter(ids))
                        if pid in mapped_ids:
                            raise ValueError('Player assigned twice')
                        mapped_ids.add(pid);ids_in_table.add(pid)
                        values={k:row[k] for k in FIELDS if k not in ('oreb','dreb')}
                        values.update(game_id=gid,player_id=pid,team_id=tid,
                                      minutes=f"{row['seconds']/60:.6f}" if row['seconds'] else '0')
                        player_values.append(values)
                    for old in existing:
                        if old.team_id==tid and old.player_id not in ids_in_table and any(
                            getattr(old,k) or 0 for k in ('pts','reb','ast','stl','blk','tov','fgm','fga','ftm','fta')):
                            raise ValueError('Nonzero existing player absent from official table')
                    values={k:v for k,v in table['totals'].items() if k!='plus_minus'}
                    values.update(game_id=gid,team_id=tid,is_home=table['is_home'],
                        game_minutes=table['total_seconds']/300,possessions_estimated=False)
                    team_values.append(values)
                home=next(t for t in team_values if t['is_home'])
                away=next(t for t in team_values if not t['is_home'])
                h_adv,a_adv=derive_game_advanced_stats(SimpleNamespace(**home),SimpleNamespace(**away),home['game_minutes'])
                home.update(h_adv);away.update(a_adv)
                for model,values,keys in ((GamePlayerStats,player_values,['game_id','player_id']),
                                           (TeamGameStats,team_values,['game_id','team_id'])):
                    stmt=insert(model).values(values)
                    changes={key:getattr(stmt.excluded,key) for key in values[0] if key not in keys}
                    changes['updated_at']=datetime.now(timezone.utc)
                    await session.execute(stmt.on_conflict_do_update(index_elements=keys,set_=changes))
                await session.flush()
                quality=await refresh_game_quality(session,gid)
                if not quality['player_stats_complete']:
                    raise ValueError('Final DB completeness rejected: '+str(quality['issues']))
                target=destination/payload['sha256'][:12]/filename
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source,target)
                snapshot=insert(BoxScoreSnapshot).values(game_id=gid,source='nba_official_gamebook',
                    source_url=payload['url'],source_file=str(target),sha256=payload['sha256'],
                    observed_at=datetime.fromisoformat(payload['observed_at']),published_at=None,
                    payload=payload['parsed'],validation=dict(quality=quality,minutes_precision='seconds',
                        game_date_verified=True,score_verified=True,
                        scorer_identity_sources={name:dict(nba_player_id=value[0],source=value[1])
                            for name,value in SCORER_ONLY_IDENTITIES.items()
                            if any(normalize(row['name'])==normalize(name)
                                for team in payload['parsed']['teams'] for row in team['players'])}))
                await session.execute(snapshot.on_conflict_do_nothing(index_elements=['game_id','sha256']))
                await bump_data_revision(session,f'official-gamebook:{gid}')
                await session.commit()
                report['applied']+=1;report['player_rows']+=len(player_values)
        except Exception as exc:
            report['rejected']+=1
            report['issues'].append(dict(game_id=gid,error=str(exc)[:300]))
        if (index+1)%50==0:
            print(json.dumps({k:v for k,v in report.items() if k!='issues'}),flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='/app/gamebooks')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--seasons',nargs='+',default=['2023-24','2024-25'])
    parser.add_argument('--output',default='/app/ml/saved_models/data_quality/dev_gamebook_import.json')
    asyncio.run(run(parser.parse_args()))
