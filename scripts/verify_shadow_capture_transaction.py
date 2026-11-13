"""Exercise real capture and settlement inside a transaction that always rolls back.

This is a synthetic engineering check, never a real pregame observation.
No clock is backdated and no artificial schedule/result/forecast is committed.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from app.database import async_session_factory
from app.models.game import Game
from app.models.shadow_observation import ShadowObservation, ShadowObservationRun
from app.services.prospective_service import digest
from app.services.shadow_observation_service import capture_one, model_root, progress, verify_protocol


async def main():
    active = (model_root() / 'active.json').read_bytes()
    async with async_session_factory() as session:
        before = await session.scalar(select(func.count()).select_from(ShadowObservation))
        run = await session.scalar(select(ShadowObservationRun).where(ShadowObservationRun.enabled.is_(True)))
        assert run is not None and verify_protocol(run)
        game = await session.scalar(select(Game).where(Game.status == 'Scheduled',
            Game.time_precision == 'exact', Game.scheduled_start_utc > datetime.now(timezone.utc),
            Game.source_game_date > datetime(2026, 10, 28).date())
            .order_by(Game.scheduled_start_utc).limit(1))
        assert game is not None
        original_start, game_id = game.scheduled_start_utc, game.id
        try:
            game.scheduled_start_utc = datetime.now(timezone.utc) + timedelta(minutes=55)
            await session.flush()
            assert await capture_one(session, run, game)
            assert not await capture_one(session, run, game), 'Duplicate capture was accepted'
            row = await session.scalar(select(ShadowObservation).where(
                ShadowObservation.run_id == run.id, ShadowObservation.game_id == game.id))
            assert row.payload_hash == digest(row.payload)
            assert row.as_of < row.scheduled_start_utc
            assert set(row.payload['totals']) == {'baseline', 'direct', 'residual'}
            assert all(np.isfinite(v) and 150 <= v <= 300 for v in row.payload['totals'].values())
            assert len(row.payload['baseline_processed_vector']) == 166
            assert all(len(candidate['processed_feature_vector']) == 213
                       for candidate in row.payload['shadow']['candidates'])
            payload_before = row.payload_hash
            game.status, game.score_complete = 'Final', True
            game.home_score, game.away_score = 120, 118
            await session.flush()
            settled = await progress(session, run)
            assert settled['settled'] >= 1 and settled['acceptance_run'] is False
            assert payload_before == row.payload_hash == digest(row.payload)
            totals = row.payload['totals']
        finally:
            await session.rollback()
    async with async_session_factory() as session:
        after = await session.scalar(select(func.count()).select_from(ShadowObservation))
        game = await session.get(Game, game_id)
        assert after == before and game.scheduled_start_utc == original_start and game.status == 'Scheduled'
    assert active == (model_root() / 'active.json').read_bytes()
    report = dict(status='PASS', kind='synthetic_transaction_rollback_check',
        real_pregame_observation=False, game_id=game_id, database_rows_before=before,
        database_rows_after=after, original_schedule_restored=True,
        duplicate_rejected=True, settlement_preserves_forecast=True,
        primary_unchanged=True, synthetic_totals=totals, formal_acceptance_added=0)
    Path('/tmp/shadow_capture_transaction_20261030.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
