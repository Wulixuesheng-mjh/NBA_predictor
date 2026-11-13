"""Pinned real weights and PostgreSQL check; synthetic state always rolls back.

Run as the worker user with its final read-only model mount and isolated /tmp.
Synthetic forecasts stay in a temporary directory and never enter a real ledger.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

from sqlalchemy import select, text
from app.database import async_session_factory
from app.models.game import Game
from app.services import win_observation_service as service
from ml.src.config import ml_settings


async def main():
    root = Path(ml_settings.models_dir)
    real = root / 'win_observations/win_T5_20261104_tminus60_v1'
    service.checked_protocol(root, real)
    active = (root / 'active.json').read_bytes()
    originals = {p.name: p.read_bytes() for p in (real / 'records').glob('*.json')}
    denied = []
    # Opening in append mode never truncates even if an incorrect mount permits it.
    for path in (root / 'active.json', real / 'protocol.json',
                 root / 'win_shadows/win_T5_20261104/win_model.json'):
        try:
            with path.open('ab'):
                pass
        except (PermissionError, OSError):
            denied.append(path.name)
        else:
            raise AssertionError('Protected model/protocol is writable: ' + str(path))
    with tempfile.NamedTemporaryFile(prefix='.permission_probe_', dir=real / 'records') as probe:
        probe.write(b'worker permissions check'); probe.flush()
    with tempfile.TemporaryDirectory(prefix='win_synthetic_') as directory:
        isolated = Path(directory)
        (isolated / 'protocol.json').write_bytes((real / 'protocol.json').read_bytes())
        (isolated / 'records').mkdir()
        async with async_session_factory() as session:
            await session.connection(execution_options={'isolation_level': 'REPEATABLE READ'})
            counts = dict((await session.execute(text('SELECT '
                '(SELECT count(*) FROM prospective_forecasts) formal_forecasts, '
                '(SELECT count(*) FROM shadow_observations) total_observations'))).mappings().one())
            game = await session.scalar(select(Game).where(Game.status == 'Scheduled',
                Game.time_precision == 'exact', Game.scheduled_start_utc > datetime.now(timezone.utc),
                Game.game_type.in_(('regular', 'play_in', 'playoffs')))
                .order_by(Game.scheduled_start_utc).limit(1))
            assert game is not None, 'No future game for rollback-only engineering check'
            original_start, game_id = game.scheduled_start_utc, game.id
            factory = service.async_session_factory
            @asynccontextmanager
            async def borrowed_session():
                yield session
            try:
                game.scheduled_start_utc = datetime.now(timezone.utc) + timedelta(minutes=55)
                await session.flush()
                service.async_session_factory = borrowed_session
                assert await service.capture_due(root, isolated) >= 1
                row_path = isolated / 'records' / f'{game_id}.json'
                before = row_path.read_bytes()
                row = json.loads(before)['payload']
                assert not row['acceptance_run']
                assert set(row['win_probabilities']) == {'baseline', 'candidate'}
                assert all(0 <= p <= 1 for p in row['win_probabilities'].values())
                assert len(row['baseline_input']['feature_vector']) == 166
                assert len(row['candidate_input']['feature_vector']) == 213
                assert await service.capture_due(root, isolated) == 0
                game.status, game.score_complete = 'Final', True
                game.home_score, game.away_score = 120, 118
                await session.flush()
                result = await service.status(root, isolated)
                assert result['settled'] >= 1 and result['integrity_errors'] == 0
                assert row_path.read_bytes() == before
                probabilities = row['win_probabilities']
            finally:
                service.async_session_factory = factory
                await session.rollback()
        async with async_session_factory() as session:
            game = await session.get(Game, game_id)
            assert game.scheduled_start_utc == original_start and game.status == 'Scheduled'
            after = dict((await session.execute(text('SELECT '
                '(SELECT count(*) FROM prospective_forecasts) formal_forecasts, '
                '(SELECT count(*) FROM shadow_observations) total_observations'))).mappings().one())
            assert counts == after, (counts, after)
    assert active == (root / 'active.json').read_bytes()
    assert originals == {p.name: p.read_bytes() for p in (real / 'records').glob('*.json')}
    report = dict(status='PASS', kind='synthetic_win_runtime_rollback', real_pregame=False,
        protected_writes_denied=denied, record_directory_writable=True, game_id=game_id,
        pinned_real_win_probabilities=probabilities, duplicate_rejected=True,
        original_schedule_restored=True, settlement_preserves_forecast=True,
        real_win_records_unchanged=True, other_ledgers_unchanged=counts,
        primary_unchanged=True, formal_acceptance_added=0)
    Path('/tmp/win_runtime_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
