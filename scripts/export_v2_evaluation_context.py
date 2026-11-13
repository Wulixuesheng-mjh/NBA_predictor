"""Read-only current score and data-chain evidence for a versioned re-review."""
import asyncio,json
from pathlib import Path
from datetime import datetime,timezone
from sqlalchemy import text
from app.database import async_session_factory


async def main():
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),writes=0)
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        queries=dict(games="""SELECT id,nba_game_id,source_game_date::text source_day,status,
            scheduled_start_utc::text start_utc,home_score,away_score,score_complete,
            team_stats_complete,player_stats_complete FROM games WHERE source_game_date>='2026-10-22' ORDER BY id""",
            injury="""SELECT i.id,p.full_name,i.game_id,i.team_id,i.published_at::text,i.observed_at::text
                FROM player_injuries i JOIN players p ON p.id=i.player_id WHERE i.id IN (35,36)""",
            reports="SELECT count(*) count FROM injury_reports WHERE game_id=32798",
            prediction_inventory="SELECT model_name,prediction_mode,count(*) FROM predictions GROUP BY model_name,prediction_mode",
            revision="SELECT revision,reason FROM data_revisions")
        for name,sql in queries.items():result[name]=[dict(r) for r in (await session.execute(text(sql))).mappings()]
    Path('/tmp/v2_evaluation_context_20261110.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(games=len(result['games']),revision=result['revision'],writes=0)))


if __name__=='__main__':asyncio.run(main())
