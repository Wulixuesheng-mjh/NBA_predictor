"""Production-code paired capture in an isolated folder and rollback-only DB."""
import asyncio,json,tempfile
from contextlib import asynccontextmanager
from datetime import datetime,timedelta,timezone
from pathlib import Path
from sqlalchemy import select,text
from app.database import async_session_factory
from app.models.game import Game
from ml.src.config import ml_settings
from scripts import paired_models_worker as worker


async def main():
    root=Path(ml_settings.models_dir);real=root/'paired_model_observations'/worker.RUN
    worker.checked(root,real);active=worker.sha(root/'active.json')
    old={p.name:worker.sha(p) for p in (real/'records').glob('*.json')}
    for path in (root/'active.json',real/'protocol.json',root/'mlp/20261022_050952/scaler.npz'):
        try:
            with path.open('ab'):pass
        except OSError:pass
        else:raise ValueError('Protected file writable')
    with tempfile.NamedTemporaryFile(prefix='.permissions_',dir=real/'records') as probe:
        probe.write(b'check');probe.flush()
    with tempfile.TemporaryDirectory(prefix='paired_rollback_') as temporary:
        folder=Path(temporary);(folder/'records').mkdir()
        (folder/'protocol.json').write_bytes((real/'protocol.json').read_bytes())
        async with async_session_factory() as session:
            await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
            before=dict((await session.execute(text('SELECT (SELECT count(*) FROM predictions) predictions, '
                '(SELECT count(*) FROM prospective_forecasts) formal, (SELECT count(*) FROM shadow_observations) totals'))).mappings().one())
            game=await session.scalar(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
                Game.game_type.in_(('regular','play_in','playoffs')),Game.scheduled_start_utc>datetime.now(timezone.utc))
                .order_by(Game.scheduled_start_utc).limit(1))
            assert game is not None
            original_start,game_id=game.scheduled_start_utc,game.id
            factory=worker.async_session_factory
            @asynccontextmanager
            async def borrow():yield session
            try:
                game.scheduled_start_utc=datetime.now(timezone.utc)+timedelta(minutes=55);await session.flush()
                worker.async_session_factory=borrow
                assert await worker.capture(root,folder)>=1
                file=folder/'records'/f'{game.id}.json';saved=file.read_bytes();row=json.loads(saved)['payload']
                assert set(row['models'])=={'xgboost','mlp'}
                assert all(row['models'][m]['raw'] is not None for m in row['models'])
                assert len(row['processed_input'])==len(row['mlp_scaled_clipped_input'])==166
                assert not row['acceptance_run'] and not row['calibration_applied']
                assert await worker.capture(root,folder)==0
                game.status='Final';game.score_complete=True;game.home_score=120;game.away_score=118;await session.flush()
                settled=await worker.status(root,folder)
                assert settled['same_as_of_paired_settled']>=1
                assert file.read_bytes()==saved
            finally:
                worker.async_session_factory=factory;await session.rollback()
        async with async_session_factory() as session:
            game=await session.get(Game,game_id)
            assert game.status=='Scheduled' and game.scheduled_start_utc==original_start
            after=dict((await session.execute(text('SELECT (SELECT count(*) FROM predictions) predictions, '
                '(SELECT count(*) FROM prospective_forecasts) formal, (SELECT count(*) FROM shadow_observations) totals'))).mappings().one())
            assert before==after
    assert old=={p.name:worker.sha(p) for p in (real/'records').glob('*.json')}
    assert active==worker.sha(root/'active.json')
    print(json.dumps(dict(status='PASS',kind='synthetic_paired_runtime_rollback',real_pregame=False,
        game_id=game_id,duplicate_rejected=True,both_raw_models_saved=True,exact_processed_dim=166,
        settlement_preserves_prediction=True,protected_files_read_only=True,real_records_unchanged=True,
        primary_unchanged=True,other_ledgers_unchanged=before,formal_acceptance_added=0),indent=2))


if __name__=='__main__':asyncio.run(main())
