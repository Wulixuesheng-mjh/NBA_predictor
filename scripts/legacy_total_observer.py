"""Run the original frozen total capture code with its original selector."""
import asyncio,logging
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from sqlalchemy import select
from app.database import async_session_factory
from app.models.shadow_observation import ShadowObservationRun
from app.services.shadow_observation_service import capture_due,progress


async def cycle():
    while True:
        try:
            async with async_session_factory() as db:
                count=await capture_due(db);await db.commit()
                if count:logging.warning('Frozen total observations captured: %s',count)
        except Exception:logging.exception('Frozen total capture failed; never backfill')
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app):
    task=asyncio.create_task(cycle())
    try:yield
    finally:
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass


app=FastAPI(lifespan=lifespan)


@app.get('/status')
async def status():
    async with async_session_factory() as db:
        runs=(await db.scalars(select(ShadowObservationRun).order_by(ShadowObservationRun.id))).all()
        reports=[await progress(db,r) for r in runs]
    if not reports or any(not r['configuration_matches'] or r['integrity_errors'] for r in reports):
        raise HTTPException(status_code=503,detail='Frozen observer protocol failed verification')
    return dict(purpose='diagnostic_only',acceptance_run=False,runs=reports)
