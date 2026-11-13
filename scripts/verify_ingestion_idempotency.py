"""Use cached real payloads in a transaction that is always rolled back."""
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from app.database import async_session_factory
from app.services.report_snapshot_service import ingest_report


async def main():
    payload = json.loads(next(Path('/app/ml/saved_models/data_sources/official_injuries').glob('*.json')).read_text(encoding='utf-8'))
    sha = hashlib.sha256(('transaction-test:'+payload['sha256']).encode()).hexdigest()
    async with async_session_factory() as session:
        try:
            kwargs = dict(published_at=datetime.fromisoformat(payload['published_at']),
                observed_at=datetime.now(timezone.utc),source_url=payload['url'],sha256=sha)
            first = await ingest_report(session,payload['rows'],**kwargs)
            second = await ingest_report(session,payload['rows'],**kwargs)
            assert first['snapshots'] > 0 and second['snapshots'] == 0
            print('PASS: verified payload insert, repeat import adds zero snapshots; all test changes rolled back')
        finally:
            await session.rollback()


if __name__ == '__main__':
    asyncio.run(main())
