"""Fetch official PDFs and verify publication from the document header.

Stores immutable payloads; does not rewrite legacy injury rows or invent
publication/collection times. Run in the existing Java-enabled importer image.
"""
import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyPDF2 import PdfReader
from nbainjuries import injury
from sqlalchemy import select
from app.database import async_session_factory
from app.models.injury_report import InjuryReport
from app.models.game import Game
from app.services.report_snapshot_service import ingest_report


def published_from_header(path):
    text = PdfReader(str(path)).pages[0].extract_text()
    match = re.search(r'Injury\s*Report:\s*(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*(AM|PM)', text, re.I)
    if not match:
        raise ValueError('PDF has no verifiable publication header')
    raw = ' '.join(match.groups())
    fmt = '%m/%d/%Y %I:%M %p' if len(match.group(1).split('/')[-1]) == 4 else '%m/%d/%y %I:%M %p'
    return datetime.strptime(raw, fmt).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)


async def main(args):
    client = requests.Session()
    client.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.5, status_forcelist=(429,500,502,503,504),
        allowed_methods=('GET',),
    )))
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    day = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    summary = {'reports': 0, 'missed': 0, 'snapshots': 0, 'complete': 0, 'partial': 0, 'unbound': 0, 'errors': []}
    game_days = None
    if args.game_days_only:
        async with async_session_factory() as session:
            game_days = set((await session.execute(select(Game.source_game_date).where(
                Game.source_game_date >= day.date(), Game.source_game_date <= end.date()+timedelta(days=1),
                Game.game_type.in_(('regular','play_in','playoffs')),
            ).distinct())).scalars())
    while day <= end:
        if game_days is not None and day.date() not in game_days and day.date()+timedelta(days=1) not in game_days:
            day += timedelta(days=1)
            continue
        for hour, minute in [(11,30),(17,30)]:
            nominal = day.replace(hour=hour, minute=minute)
            if nominal.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc) > datetime.now(timezone.utc):
                continue
            url = injury.gen_url(nominal)
            path = root / url.rsplit('/',1)[-1]
            try:
                if not path.exists():
                    response = client.get(url, timeout=(10,25))
                    if response.status_code == 404:
                        summary['missed'] += 1
                        continue
                    response.raise_for_status()
                    if not response.content.startswith(b'%PDF'):
                        raise ValueError('Source did not return a PDF')
                    path.write_bytes(response.content)
                observed_at = datetime.now(timezone.utc)
                published_at = published_from_header(path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                metadata_path = path.with_suffix('.json')
                if metadata_path.exists():
                    cached = json.loads(metadata_path.read_text(encoding='utf-8'))
                    if cached.get('sha256') == sha and cached.get('observed_at'):
                        observed_at = datetime.fromisoformat(cached['observed_at'])
                async with async_session_factory() as session:
                    existing = await session.scalar(select(InjuryReport.id).where(InjuryReport.source_sha256 == sha).limit(1))
                    if existing and not args.reparse:
                        continue
                df = injury.get_reportdata(nominal, local=True, localdir=str(root), return_df=True)
                if df is None or df.empty:
                    raise ValueError('PDF parser returned no rows')
                rows = df.fillna('').to_dict(orient='records')
                (path.with_suffix('.json')).write_text(json.dumps({
                    'url':url,'sha256':sha,'published_at':published_at.isoformat(),
                    'observed_at':observed_at.isoformat(),'rows':rows,
                },indent=2),encoding='utf-8')
                async with async_session_factory() as session:
                    counts = await ingest_report(session, rows, published_at=published_at,
                        observed_at=observed_at, source_url=url, sha256=sha)
                    await session.commit()
                summary['reports'] += 1
                for key in counts:
                    summary[key] += counts[key]
                print(day.date(), hour, json.dumps(counts), flush=True)
            except Exception as exc:
                summary['errors'].append({'url':url,'error':str(exc)[:300]})
                print('report_error',url,str(exc)[:200],flush=True)
        day += timedelta(days=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',required=True)
    parser.add_argument('--end',required=True)
    parser.add_argument('--cache-dir',default='/app/ml/saved_models/data_sources/official_injuries')
    parser.add_argument('--output',default='/app/ml/saved_models/data_quality/verified_injury_import.json')
    parser.add_argument('--reparse',action='store_true')
    parser.add_argument('--game-days-only',action='store_true')
    asyncio.run(main(parser.parse_args()))
