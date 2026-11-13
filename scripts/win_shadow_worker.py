"""Separate win-only scheduler; no training, promotion, or retrospective capture."""
import argparse,asyncio,json,logging
from pathlib import Path
from app.services.win_observation_service import lock_run,capture_due,status
from ml.src.config import ml_settings


async def main(args):
    root=Path(ml_settings.models_dir);folder=root/'win_observations'/args.run
    if args.action=='lock':
        lock_run(root,args.run,args.version);print(json.dumps(await status(root,folder)));return
    if args.action=='status':
        report=await status(root,folder)
        if report['integrity_errors']: raise ValueError('Win record integrity errors')
        print(json.dumps(report));return
    while True:
        try:
            count=await capture_due(root,folder)
            if count: print(json.dumps(dict(captured=count,acceptance_run=False)),flush=True)
        except Exception: logging.exception('Win-only capture failed; no backfill or promotion')
        if args.action=='once': break
        await asyncio.sleep(300)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','status','once','serve'])
    p.add_argument('--run',required=True);p.add_argument('--version',default='win_T5_20261104')
    asyncio.run(main(p.parse_args()))
