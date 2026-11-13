"""Repair exact game times using configured API; preserve score-only quality."""
import argparse
import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from app.collector.balldontlie_collector import BallDontLieCollector


async def main(args):
    collector = BallDontLieCollector()
    day, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    evidence = []
    async with collector.db_session_factory() as session:
        team_map = await collector._get_team_map(session)
        while day <= end:
            games = await collector._fetch_games(day.isoformat())
            for game in games:
                evidence.append({k:game.get(k) for k in ('id','date','datetime','status','home_team_score','visitor_team_score')})
                if not args.read_only:
                    await collector._upsert_game(session, game, team_map,
                        'Final' if game.get('status') == 'Final' else 'Scheduled')
            if not args.read_only:
                await session.commit()
            day += timedelta(days=1)
            await asyncio.sleep(13)  # remain within configured source's basic rate limit
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(read_only=args.read_only, games=evidence), indent=2), encoding='utf-8')
    print(json.dumps(dict(games=len(evidence), read_only=args.read_only)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--output', default='/app/ml/saved_models/data_quality/exact_schedule_20261028.json')
    asyncio.run(main(parser.parse_args()))
