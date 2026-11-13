"""Read-only schedule/result snapshot for a bounded results refresh."""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import text
from app.database import async_session_factory


async def main(args):
    async with async_session_factory() as session:
        result = dict(checked_at=datetime.now(timezone.utc).isoformat())
        queries = dict(
            coverage="""SELECT season,count(*) games,count(*) FILTER (WHERE status='Final') finals,
              count(*) FILTER (WHERE status='Final' AND score_complete) scores,
              count(*) FILTER (WHERE status='Final' AND team_stats_complete) team_complete,
              count(*) FILTER (WHERE status='Final' AND player_stats_complete) player_complete,
              max(source_game_date) FILTER (WHERE status='Final')::text latest_final_day
              FROM games WHERE season='2026-27' GROUP BY season""",
            stale="""SELECT id,nba_game_id,source_game_date::text AS source_day,status,
              scheduled_start_utc::text start_utc FROM games WHERE status!='Final'
              AND scheduled_start_utc < now()-interval '6 hours' ORDER BY scheduled_start_utc""",
            recent="""SELECT g.id,g.nba_game_id,g.source_game_date::text AS source_day,h.abbreviation home,
              a.abbreviation away,g.status,g.home_score,g.away_score,g.score_complete,
              g.team_stats_complete,g.player_stats_complete FROM games g
              JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
              WHERE g.source_game_date>='2026-11-04' ORDER BY g.source_game_date,g.id""",
            revision="SELECT revision,reason,updated_at::text FROM data_revisions")
        for key, sql in queries.items():
            result[key] = [dict(r) for r in (await session.execute(text(sql))).mappings()]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='recent'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    asyncio.run(main(parser.parse_args()))
