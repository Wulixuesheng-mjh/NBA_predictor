"""Trace manual injury, bound reports and saved predictions without changing them."""
import asyncio,json
from pathlib import Path
from sqlalchemy import text
from app.database import async_session_factory


async def main():
    result={}
    queries=dict(
        injury_rows="""SELECT i.id,p.full_name,i.game_id,i.team_id,i.source,i.confirmation,i.injury_status,
          i.is_active,i.reported_date::text,i.published_at::text,i.observed_at::text,
          i.created_at::text,i.updated_at::text FROM player_injuries i JOIN players p ON p.id=i.player_id
          WHERE lower(p.full_name) LIKE '%zion%' OR lower(p.full_name) LIKE '%murphy%'
          ORDER BY i.created_at DESC LIMIT 30""",
        bound_reports="""SELECT id,game_id,team_id,source,published_at::text,observed_at::text,
          is_complete,publication_verified,entries_json,created_at::text FROM injury_reports
          WHERE game_id=32798 ORDER BY observed_at,id""",
        predictions="""SELECT id,game_id,home_team_id,away_team_id,model_name,model_version,prediction_mode,
          as_of::text,created_at::text,predicted_spread,predicted_total,injury_snapshot,data_quality,
          feature_names,feature_vector,features_json,raw_output,final_output
          FROM predictions WHERE game_id=32798 OR created_at>='2026-11-09'
          ORDER BY id DESC LIMIT 30""",
        totals="SELECT model_name,prediction_mode,count(*) FROM predictions GROUP BY model_name,prediction_mode")
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        for name,sql in queries.items():result[name]=[dict(row) for row in (await session.execute(text(sql))).mappings()]
    result['writes']=0
    Path('/tmp/manual_injury_trace_20261110.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    compact=dict(result)
    compact['predictions']=[{k:v for k,v in row.items() if k not in ('features_json','feature_vector','feature_names')} for row in result['predictions']]
    print(json.dumps(compact,indent=2,default=str))


if __name__=='__main__':asyncio.run(main())
