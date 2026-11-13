import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from types import SimpleNamespace


def module():
    path=Path('/app/scripts/expand_history_from_snapshot.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/expand_history_from_snapshot.py'
    spec=importlib.util.spec_from_file_location('expand_snapshot',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_source_season_and_real_eastern_tipoff_are_bound_together(tmp_path):
    m=module()
    row=dict(gameId='22200001',gameDateTimeEst='2022-10-18 19:30:00',homeScore='126',awayScore='117',
             hometeamId='10',awayteamId='20')
    with (tmp_path/'Games.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        writer.writerow(row|dict(gameId='22300001'))
    result=m.source_games(tmp_path,'2022-23',{10:1,20:2})
    assert list(result)==['22200001']
    assert result['22200001']['source_game_date']=='2022-10-18'
    assert result['22200001']['scheduled_start_utc']=='2022-10-18T23:30:00+00:00'
    assert result['22200001']['season']=='2022-23'


def test_corrupt_source_is_rejected_before_database_work(tmp_path):
    m=module();files={}
    for name in ('Games.csv','TeamStatistics.csv','PlayerStatistics.csv','Players.csv'):
        (tmp_path/name).write_bytes(b'original')
        files[name]={'sha256':hashlib.sha256(b'original').hexdigest()}
    (tmp_path/'source_manifest.json').write_text(json.dumps({'files':files}))
    assert m.verified_source(tmp_path)
    (tmp_path/'Games.csv').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        m.verified_source(tmp_path)


def test_missing_numeric_team_id_requires_explicit_source_home_and_both_team_names():
    from app.collector.stat_backfill import source_player_team
    game=SimpleNamespace(home_team_id=1,away_team_id=2)
    teams={1:SimpleNamespace(name='Celtics'),2:SimpleNamespace(name='Warriors')}
    source=dict(playerteamId='',home='1',playerteamName='Celtics',opponentteamName='Warriors')
    assert source_player_team(source,game,{10:1,20:2},teams)==(1,'source_home_and_team_names')
    assert source_player_team(source|dict(home='0',playerteamName='Warriors',opponentteamName='Celtics'),
                              game,{10:1,20:2},teams)[0]==2
    for bad in (dict(home=''),dict(home='unknown'),dict(home='0'),dict(opponentteamName=''),dict(playerteamId='999')):
        assert source_player_team(source|bad,game,{10:1,20:2},teams)[0] is None
    assert source_player_team(source|dict(playerteamId='20'),game,{10:1,20:2},teams)==(2,'source_team_id')


@pytest.mark.asyncio
async def test_statistics_backfill_can_be_limited_to_one_season(seeded_db,tmp_path):
    from contextlib import asynccontextmanager
    from app.collector.stat_backfill import backfill_csv_stats
    from app.models.game import Game
    base=await seeded_db.get(Game,seeded_db._game_ids['final'])
    for name,season,day in (('a','2022-23','2023-01-02'),('b','2023-24','2023-01-03'),('c','2022-23','2024-01-02')):
        seeded_db.add(Game(nba_game_id='scope-'+name,season=season,
            game_date=datetime.fromisoformat(day).replace(tzinfo=timezone.utc),
            home_team_id=base.home_team_id,away_team_id=base.away_team_id,status='Final',home_score=110,away_score=100))
    await seeded_db.flush()
    for name,header in (('Games.csv','gameId,gameDateTimeEst'),('TeamStatistics.csv','gameId'),('PlayerStatistics.csv','gameId')):
        (tmp_path/name).write_text(header+'\n')
    @asynccontextmanager
    async def factory():
        yield seeded_db
    result=await backfill_csv_stats(factory,tmp_path,'2022-01-01',end_date='2023-07-01',seasons=['2022-23'])
    assert result['pending_games']==1 and result['matched_games']==0
