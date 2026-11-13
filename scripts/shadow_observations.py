"""Lock, run or inspect diagnostic capture without starting formal acceptance."""
import argparse
import asyncio
import json
from sqlalchemy import select
from app.database import async_session_factory
from app.models.shadow_observation import ShadowObservationRun
from app.services.shadow_observation_service import lock_diagnostic_run,capture_due,progress


async def main(action):
    async with async_session_factory() as session:
        if action=='lock':
            await lock_diagnostic_run(session)
            await session.commit()
        if action=='capture':
            count=await capture_due(session)
            await session.commit()
            print(json.dumps(dict(captured=count,formal_acceptance_added=0)))
        runs=(await session.execute(select(ShadowObservationRun))).scalars().all()
        print(json.dumps(dict(runs=[await progress(session,r) for r in runs],formal_acceptance_started=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['lock','capture','status'])
    asyncio.run(main(parser.parse_args().action))
