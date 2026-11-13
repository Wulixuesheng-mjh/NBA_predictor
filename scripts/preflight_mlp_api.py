"""Synthetic end-to-end test, restricted to the disposable preflight database."""
import asyncio,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import httpx
from sqlalchemy import text,select
from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.game import Game
from app.models.prediction import Prediction
from app.models.prediction_results import PredictionResult
from app.services.prediction_service import PredictionService
from ml.src.feedback import FeedbackLoop
from ml.src.legacy_mlp_compatibility import CONTRACT_ID
from ml.src.config import ml_settings
from scripts.replay_prediction import replay_processed_row


async def main():
    if 'nba-mlp-preflight-db-20261110' not in settings.database_url or settings.scheduler_enabled:
        raise ValueError('Refuse to mutate a real database or enable scheduled jobs')
    async with async_session_factory() as session:
        revision=await session.scalar(text('SELECT version_num FROM alembic_version'))
        assert revision=='010_mlp_compat_version'
        width=await session.scalar(text("SELECT character_maximum_length FROM information_schema.columns WHERE table_name='predictions' AND column_name='model_version'"))
        assert width==128
        before=await session.scalar(text('SELECT count(*) FROM predictions'))
        g=await session.get(Game,32798)
        g.status='Scheduled';g.scheduled_start_utc=datetime.now(timezone.utc)+timedelta(hours=1)
        g.game_date=g.scheduled_start_utc;g.time_precision='exact';g.home_score=0;g.away_score=0
        g.score_complete=False
        await session.commit()
    async with app.router.lifespan_context(app):
        endpoint=next(path for path,methods in app.openapi()['paths'].items() if path.endswith('/predictions/') and 'post' in methods)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://preflight',
            headers={'X-API-Key':settings.admin_api_key}) as client:
            health=await client.get('/health');assert health.status_code==200
            result=await client.post(endpoint,json=dict(game_id=32798,mode='pregame',persist=True))
            if result.status_code!=200:raise ValueError('Prediction API failed: '+result.text)
            response=result.json();assert CONTRACT_ID in response['mlp']['model_version']
            assert response['xgboost']['model_version']=='20261022_050952'
            bad=await client.post(endpoint,json=dict(game_id=32798,mode='pregame',persist=True,as_of='2026-01-01T00:00:00Z'))
            assert bad.status_code==400
        async with async_session_factory() as session:
            predictions=list((await session.scalars(select(Prediction).where(Prediction.id>before).order_by(Prediction.id))).all())
            assert len(predictions)==2 and {p.model_name for p in predictions}=={'mlp','xgboost'}
            fixed=PredictionService._get_predictor()
            for p in predictions:
                assert p.feature_vector and p.feature_names and p.features_json.get('_inference_snapshot')
                raw,rounded=replay_processed_row(p,fixed)
                assert abs(rounded.predicted_total-p.predicted_total)<1e-3
            mlp=next(p for p in predictions if p.model_name=='mlp')
            assert len(mlp.model_version)>20
            last_created=max(p.created_at for p in predictions)
            g=await session.get(Game,32798)
            g.scheduled_start_utc=datetime.now(timezone.utc)
            assert last_created<g.scheduled_start_utc
            g.status='Final';g.home_score=120;g.away_score=113;g.score_complete=True
            await session.commit()
        loop=FeedbackLoop();original_models_dir=ml_settings.models_dir
        try:
            ml_settings.models_dir='/out/residual_store_isolated'
            settled=await loop._evaluate_predictions();again=await loop._evaluate_predictions()
        finally:
            ml_settings.models_dir=original_models_dir
        assert settled==2 and again==0
        residuals=json.loads(Path('/out/residual_store_isolated/residuals.json').read_text())
        assert mlp.model_version in residuals['models']['mlp']
        assert '20261022_050952' not in residuals['models']['mlp']
        async with async_session_factory() as session:
            ids=[p.id for p in predictions]
            outcomes=list((await session.scalars(select(PredictionResult).where(PredictionResult.prediction_id.in_(ids)))).all())
            assert len(outcomes)==2 and all(r.actual_total==233 and r.actual_spread==7 for r in outcomes)
            after=list((await session.scalars(select(Prediction).where(Prediction.id.in_(ids)))).all())
            assert {p.id:p.feature_hash for p in after}=={p.id:p.feature_hash for p in predictions}
    report=dict(status='PASS',scope='synthetic_API_and_database_preflight_not_real_predictions',
        migration=revision,version_width=width,health=True,pregame_api=True,both_models_persisted=True,
        full_MLP_version_stored=mlp.model_version,snapshots_replayed=2,backdated_write_rejected=True,
        settlement_rows=settled,repeated_settlement_rows=again,original_forecasts_immutable=True,
        repaired_residuals_separately_versioned=True,
        real_database_touched=False,formal_acceptance_added=0,production_promoted=False)
    Path('/out/api_database_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':asyncio.run(main())
