"""No forecast writes: validate lock guards and pregame clock rules."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.database import async_session_factory
from app.services.prospective_service import assert_pregame, lock_run
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Invalid capture was accepted')


async def main():
    now = datetime.now(timezone.utc)
    game = SimpleNamespace(status='Scheduled', time_precision='exact',
        scheduled_start_utc=now+timedelta(hours=1), source_game_date=(now+timedelta(days=1)).date())
    manifest = dict(training_data_until=(now-timedelta(days=2)).isoformat())
    assert_pregame(game,now,[manifest],now=now)
    rejects(lambda:assert_pregame(game,now-timedelta(days=1),[manifest],now=now))
    rejects(lambda:assert_pregame(game,now+timedelta(seconds=1),[manifest],now=now))
    rejects(lambda:assert_pregame(game,now,[manifest],now=now+timedelta(hours=2)))
    rejects(lambda:assert_pregame(game,now,[dict(training_data_until=str(game.source_game_date))],now=now))
    root = Path(ml_settings.models_dir)
    active = root/'active.json'
    before = active.read_bytes()
    baseline = json.loads(before)
    predictor = load_predictor(release_manifest=baseline)
    assert predictor.release_manifest == baseline and predictor.xgb_trainer.is_trained
    candidate = json.loads((root/'candidates/20261027_213v7.json').read_text(encoding='utf-8'))
    async with async_session_factory() as session:
        try:
            try:
                await lock_run(session,name='test_should_not_exist',baseline=baseline,
                               candidate=candidate,gate_evidence={})
            except ValueError as exc:
                assert 'semantics' in str(exc)
            else:
                raise AssertionError('Old candidate was incorrectly locked')
        finally:
            await session.rollback()
    assert active.read_bytes() == before
    print('PASS: clock/backdating/training cutoff, pinned manifest, old candidate rejected, active unchanged')


if __name__ == '__main__':
    asyncio.run(main())
