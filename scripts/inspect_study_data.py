"""Read-only cache integrity and live data-completeness context for research."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from sqlalchemy import text
from app.database import async_session_factory
from ml.src.features.dataset_cache import DatasetCache


async def main(args):
    path=DatasetCache().path_for('bbeb026e149ba7b9')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=args.expected_sha: raise ValueError('Pinned volume cache checksum differs from saved artifact')
    async with async_session_factory() as session:
        coverage=[dict(r) for r in (await session.execute(text('''
            SELECT season,count(*) games,
            count(*) FILTER (WHERE score_complete) score_complete,
            count(*) FILTER (WHERE team_stats_complete) team_complete,
            count(*) FILTER (WHERE player_stats_complete) player_complete
            FROM games WHERE status='Final' AND season IN
            ('2020-21','2021-22','2022-23','2023-24','2024-25','2025-26','2026-27')
            GROUP BY season ORDER BY season
        '''))).mappings()]
        revision=[dict(r) for r in (await session.execute(text(
            'SELECT revision,reason,updated_at::text FROM data_revisions'))).mappings()]
        stale=await session.scalar(text("SELECT count(*) FROM games WHERE status='Scheduled' AND scheduled_start_utc < now()-interval '6 hours'"))
    report=dict(cache_key='bbeb026e149ba7b9',cache_sha256=actual,cache_unchanged=True,
                coverage=coverage,data_revision=revision,stale_scheduled_games=stale,writes=0)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-sha',required=True);p.add_argument('--output',required=True)
    asyncio.run(main(p.parse_args()))
