"""Check the deployed evaluation-only path without issuing any training."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sqlalchemy import func, select
from app.database import async_session_factory
from app.models.game import Game
from app.models.shadow_observation import ShadowObservation
from app.services.shadow_observation_service import model_root
from ml.src.feedback import FeedbackLoop
from ml.src.trainer import Trainer


async def main():
    root = model_root()
    files = {name: (root / name).read_bytes() for name in ('active.json', 'shadow_totals.json')}
    folders = sorted(p.name for p in (root / 'xgboost').iterdir() if p.is_dir())
    cutoff = datetime.fromisoformat(json.loads(files['active.json'])['training_data_until']).replace(tzinfo=timezone.utc)
    async with async_session_factory() as session:
        before = await session.scalar(select(func.count()).select_from(ShadowObservation))
        legacy = await session.scalar(select(func.count(Game.id)).where(Game.status == 'Final',
            Game.score_complete.is_(True), Game.team_stats_complete.is_(True),
            Game.player_stats_complete.is_(True), Game.game_date > cutoff))
    result = await FeedbackLoop().run_cycle()
    assert not result.retrained and not result.calibration_updated and not result.errors
    assert sorted(p.name for p in (root / 'xgboost').iterdir() if p.is_dir()) == folders
    assert all((root / name).read_bytes() == data for name, data in files.items())
    async with async_session_factory() as session:
        after = await session.scalar(select(func.count()).select_from(ShadowObservation))
    assert before == after
    try:
        Trainer._check_output_directories()
        output_check = dict(writable=True)
    except RuntimeError as exc:
        output_check = dict(writable=False, reason=str(exc))
    report = dict(status='PASS', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        feedback=asdict(result), legacy_utc_midnight_ready_count=legacy,
        primary_and_shadow_selectors_unchanged=True, training_folders_unchanged=True,
        diagnostic_rows_unchanged=True, api_training_output_check=output_check,
        selector_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    Path('/tmp/feedback_runtime_20261030.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
