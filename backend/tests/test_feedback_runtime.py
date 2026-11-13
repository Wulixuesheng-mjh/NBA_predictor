"""Online feedback must not train, miscount cutoff days, or hide failures."""
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ml.src import feedback


@pytest.mark.asyncio
async def test_automatic_feedback_settles_without_training_or_calibration(monkeypatch):
    monkeypatch.setattr(feedback, 'ml_settings', SimpleNamespace(
        auto_retrain=True, min_new_games_for_retrain=3, calibration_enabled=True))
    trainer = Mock(side_effect=AssertionError('Automatic feedback must not train'))
    monkeypatch.setattr(feedback, 'Trainer', trainer)
    loop = feedback.FeedbackLoop()
    monkeypatch.setattr(loop, '_evaluate_predictions', AsyncMock(return_value=2))
    monkeypatch.setattr(loop, '_count_ready_games_since_release', AsyncMock(return_value=66))
    calibrate = AsyncMock(side_effect=AssertionError('Automatic feedback must not calibrate'))
    monkeypatch.setattr(loop, '_update_calibration_only', calibrate)
    result = await loop.run_cycle(force_retrain=True)
    assert result.predictions_evaluated == 2 and result.new_games_found == 66
    assert not result.retrained and not result.calibration_updated and result.errors == []
    trainer.assert_not_called()
    calibrate.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_training_preserves_seasons_and_calibration_policy(monkeypatch):
    monkeypatch.setattr(feedback, 'ml_settings', SimpleNamespace(auto_retrain=True,
        min_new_games_for_retrain=3, calibration_enabled=False, default_lookback_games=10))
    train = AsyncMock(return_value=SimpleNamespace(version='candidate', calibration=None))
    monkeypatch.setattr(feedback, 'Trainer', Mock(return_value=SimpleNamespace(train_full_pipeline=train)))
    loop = feedback.FeedbackLoop()
    monkeypatch.setattr(loop, '_evaluate_predictions', AsyncMock(return_value=0))
    monkeypatch.setattr(loop, '_count_ready_games_since_release', AsyncMock(return_value=4))
    result = await loop.run_cycle(seasons=['2025-26'], evaluate_only=False)
    train.assert_awaited_once_with(seasons=['2025-26'], lookback_games=10, calibrate=False)
    assert result.retrained and not result.calibration_updated


@pytest.mark.asyncio
async def test_ready_games_exclude_entire_fitted_nba_day_and_partial_stats(seeded_db, tmp_path, monkeypatch):
    from app.models.game import Game
    session = seeded_db
    (tmp_path / 'active.json').write_text('{"training_data_until":"2026-01-15"}')
    monkeypatch.setattr(feedback, 'ml_settings', SimpleNamespace(models_dir=str(tmp_path)))
    original = await session.get(Game, session._game_ids['final'])
    original.source_game_date = date(2026, 1, 15)
    original.game_date = datetime(2026, 1, 16, 2, tzinfo=timezone.utc)
    original.score_complete = original.team_stats_complete = original.player_stats_complete = True
    for number, source_day, complete in ((1, date(2026, 1, 16), True),
                                         (2, date(2026, 1, 17), False), (3, None, True)):
        session.add(Game(nba_game_id=f'ready-test-{number}', season='2025-26',
            source_game_date=source_day, game_date=datetime(2026, 1, 18, tzinfo=timezone.utc),
            home_team_id=original.home_team_id, away_team_id=original.away_team_id,
            status='Final', home_score=110, away_score=105, score_complete=True,
            team_stats_complete=complete, player_stats_complete=complete))
    await session.flush()

    @asynccontextmanager
    async def factory():
        yield session

    assert await feedback.FeedbackLoop(factory)._count_ready_games_since_release() == 1


@pytest.mark.asyncio
async def test_unwritable_candidate_directory_fails_before_feature_build(tmp_path, monkeypatch):
    from ml.src import trainer
    (tmp_path / 'candidates').write_text('not a directory')
    monkeypatch.setattr(trainer, 'ml_settings', SimpleNamespace(models_dir=str(tmp_path),
        xgboost_dir=str(tmp_path / 'xgboost'), mlp_dir=str(tmp_path / 'mlp')))
    instance = trainer.Trainer()
    build = AsyncMock(side_effect=AssertionError('Feature building must not start'))
    monkeypatch.setattr(instance.feature_builder, 'build_training_dataset', build)
    with pytest.raises(RuntimeError, match='trainer container'):
        await instance.train_full_pipeline()
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_training_returns_failure_and_forwards_requested_seasons(monkeypatch):
    from app.api.admin import trigger_model_training
    run = AsyncMock(return_value=SimpleNamespace(errors=['Output directory is not writable']))
    monkeypatch.setattr(feedback, 'run_feedback_cycle', run)
    result = await trigger_model_training(seasons='2024-25,2025-26', force=True)
    assert result.status == 'failed' and 'not writable' in result.message
    run.assert_awaited_once_with(force_retrain=True, seasons=['2024-25', '2025-26'])
