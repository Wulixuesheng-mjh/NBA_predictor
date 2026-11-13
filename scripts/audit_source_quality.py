"""Read-only source range and box-score quality diagnosis."""
import asyncio
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select, text
from app.database import async_session_factory
from app.models.game import Game
from app.services.data_quality_service import assess_game_data


async def main():
    output = {'sources': {}, 'issues': {}, 'examples': []}
    pointer = Path('/app/ml/saved_models/data_sources/kaggle/current.json')
    folder = Path(json.loads(pointer.read_text(encoding='utf-8'))['dataset_dir'])
    output['source_folder'] = str(folder)
    for name in ('Games.csv', 'TeamStatistics.csv', 'PlayerStatistics.csv'):
        lo, hi, count = None, None, 0
        with (folder/name).open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                day = row.get('gameDateTimeEst', '')[:10]
                if day:
                    lo = min(lo, day) if lo else day
                    hi = max(hi, day) if hi else day
                count += 1
        output['sources'][name] = dict(rows=count, first=lo, last=hi)
    async with async_session_factory() as session:
        output['latest_db_games'] = [dict(r) for r in (await session.execute(text('''
            SELECT g.nba_game_id, g.source_game_date::text AS source_day, h.abbreviation home,
            a.abbreviation away, g.home_score, g.away_score FROM games g
            JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
            WHERE g.season='2026-27' ORDER BY g.game_date LIMIT 15
        '''))).mappings()]
        output['historical_raw_minutes'] = []
        with (folder/'PlayerStatistics.csv').open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row['gameId'].lstrip('0') == '42300405':
                    output['historical_raw_minutes'].append({k:row.get(k) for k in (
                        'personId', 'playerteamId', 'numMinutes', 'points', 'comment')})
        output['coverage'] = [dict(r) for r in (await session.execute(text('''
            SELECT season, count(*) games, count(*) FILTER (WHERE team_stats_complete) team_complete,
            count(*) FILTER (WHERE player_stats_complete) player_complete,
            count(*) FILTER (WHERE time_precision='exact') exact_times
            FROM games WHERE status='Final' AND season >= '2023-24' GROUP BY season ORDER BY season
        '''))).mappings()]
        # Count actual causes, rather than treating every failed flag as absent rows.
        games = list((await session.execute(select(Game).where(
            Game.status == 'Final', Game.season >= '2023-24',
            Game.player_stats_complete.is_(False),
        ).order_by(Game.game_date.desc()))).scalars())
        reasons = defaultdict(Counter)
        for game in games:
            state = await assess_game_data(session, game.id)
            for reason in state['issues']:
                reasons[game.season][reason.split(':')[0]] += 1
            if len(output['examples']) < 12 or (
                game.season == '2023-24' and not any(e['season'] == '2023-24' for e in output['examples'])
            ):
                rows = (await session.execute(text('''
                    SELECT team_id, count(*) players, sum(pts) points,
                    sum(CASE WHEN minutes ~ '^[0-9]+([.][0-9]+)?$' THEN minutes::float ELSE 0 END) minutes
                    FROM game_player_stats WHERE game_id=:gid GROUP BY team_id
                '''), {'gid': game.id})).mappings().all()
                output['examples'].append(dict(id=game.id, nba_game_id=game.nba_game_id,
                    season=game.season, day=str(game.source_game_date),
                    score=[game.home_score, game.away_score], issues=state['issues'],
                    teams=[dict(r) for r in rows]))
        output['issues'] = {s: dict(c) for s,c in reasons.items()}
    path = Path('/app/ml/saved_models/data_quality/source_audit_20261028.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
