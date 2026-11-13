"""Old serving image must read widened schema without truncating repair IDs."""
import asyncio,json
from pathlib import Path
import httpx
from sqlalchemy import text
from app.config import settings
from app.main import app
from app.database import async_session_factory
from ml.src.predictor import load_predictor


async def main():
    assert 'nba-mlp-preflight-db-20261110' in settings.database_url
    assert not settings.scheduler_enabled
    old=load_predictor();assert old.mlp_trainer.version=='20261022_050952'
    assert getattr(old.mlp_trainer,'input_compatibility_id',None) is None
    async with async_session_factory() as session:
        version=await session.scalar(text('SELECT version_num FROM alembic_version'))
        assert version=='010_mlp_compat_version'
        length=await session.scalar(text('SELECT max(length(model_version)) FROM predictions'))
        assert length>20
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://rollback') as client:
            assert (await client.get('/health')).status_code==200
            history=await client.get('/api/v1/predictions/',params={'limit':10})
            assert history.status_code==200 and 'legacy_mlp_1022_constant_ast_v1' in history.text
    report=dict(status='PASS',old_code_with_widened_schema=True,full_version_retained=True,
        active_model='20261022_050952',schema=version,real_production_changed=False)
    Path('/out/rollback_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':asyncio.run(main())
