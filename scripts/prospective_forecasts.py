"""Inspect/capture a locked protocol; does not choose a candidate or promote."""
import argparse
import asyncio
import json
from pathlib import Path
from sqlalchemy import select
from app.database import async_session_factory
from app.models.prospective import ProspectiveRun
from app.services.prospective_service import capture_due, lock_run, progress_report


async def main(args):
    async with async_session_factory() as session:
        if args.action == 'lock':
            if not all((args.name,args.baseline,args.candidate,args.evidence)):
                raise ValueError('Lock requires name, baseline/candidate manifests and evidence files')
            read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
            await lock_run(session, name=args.name, baseline=read(args.baseline),
                           candidate=read(args.candidate), gate_evidence=read(args.evidence))
            await session.commit()
        if args.action == 'capture':
            print(json.dumps({'captured':await capture_due(session)}))
            await session.commit()
        runs = (await session.execute(select(ProspectiveRun))).scalars().all()
        reports = [await progress_report(session, run) for run in runs]
        print(json.dumps({'runs':reports, 'state':'no_locked_final_candidate' if not runs else 'locked'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status','lock','capture'], default='status', nargs='?')
    for flag in ('name','baseline','candidate','evidence'):
        parser.add_argument('--'+flag)
    asyncio.run(main(parser.parse_args()))
