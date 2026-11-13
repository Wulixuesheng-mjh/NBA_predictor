"""Read-only audit of displayed accuracy, serving MLP, and schedule binding."""
import asyncio,json,inspect
from pathlib import Path
from sqlalchemy import text
from app.database import async_session_factory
from app.api.predictions import accuracy_trends
from app.services.prediction_service import PredictionService
from ml.src.feedback import FeedbackLoop


async def main():
    result={}
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        for name,sql in dict(
            predictions="SELECT model_name,prediction_mode,count(*) rows FROM predictions GROUP BY model_name,prediction_mode",
            results="""SELECT p.model_name,p.prediction_mode,count(*) rows,
                count(*) FILTER (WHERE r.win_correct) correct,
                min(r.evaluated_at)::text first_evaluated,max(r.evaluated_at)::text last_evaluated
                FROM prediction_results r JOIN predictions p ON p.id=r.prediction_id
                GROUP BY p.model_name,p.prediction_mode""",
            matchups="""SELECT g.id,g.source_game_date::text source_day,g.scheduled_start_utc::text start_utc,
                g.status,g.home_score,g.away_score,h.abbreviation home,a.abbreviation away
                FROM games g JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
                WHERE h.abbreviation='GSW' AND a.abbreviation='NOP' ORDER BY g.game_date DESC LIMIT 4""",
            version_width="""SELECT column_name,character_maximum_length FROM information_schema.columns
                WHERE table_name='predictions' AND column_name='model_version'"""
        ).items():
            result[name]=[dict(r) for r in (await session.execute(text(sql))).mappings()]
        trends=await accuracy_trends(days=30,window=7,db=session)
        result['public_accuracy_overall']=trends.overall.model_dump()
        game=next((r for r in result['matchups'] if r['status']=='Scheduled'),None)
        if game:
            # A read-only present-time preview, not a backdated or persisted prediction.
            pred=await PredictionService.generate_prediction(session,game_id=game['id'],persist=False)
            result['present_time_preview']={k:pred.get(k) for k in ('game_id','prediction_mode','as_of','xgboost','mlp','raw_outputs','data_quality','model_release','release_manifest')}
    result['runtime_feedback_signature']=str(inspect.signature(FeedbackLoop.run_cycle))
    result['runtime_evaluation_only_branch']='if evaluate_only:' in inspect.getsource(FeedbackLoop.run_cycle)
    result['writes']=0
    Path('/tmp/live_feedback_audit_20261110.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':asyncio.run(main())
