"""Export read-only rollout evidence without creating or backdating forecasts."""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select, text

from app.database import async_session_factory
from app.models.game import Game
from app.models.prospective import ProspectiveForecast, ProspectiveRun
from app.models.shadow_observation import ShadowObservation, ShadowObservationRun
from app.services.shadow_observation_service import model_root, progress


def game_summary(game):
    return dict(game_id=game.id, source_game_date=str(game.source_game_date),
                start_utc=game.scheduled_start_utc.isoformat(), status=game.status,
                score_complete=game.score_complete,
                team_stats_complete=game.team_stats_complete,
                player_stats_complete=game.player_stats_complete)


async def inspect(output):
    now = datetime.now(timezone.utc)
    root = model_root()
    active_bytes = (root / 'active.json').read_bytes()
    report = dict(checked_at_utc=now.isoformat(),
                  active_version=json.loads(active_bytes)['version'],
                  active_sha256=hashlib.sha256(active_bytes).hexdigest(), runs=[])
    async with async_session_factory() as session:
        report['database_migration'] = await session.scalar(text('SELECT version_num FROM alembic_version'))
        report['formal_runs'] = await session.scalar(select(func.count()).select_from(ProspectiveRun))
        report['formal_forecasts'] = await session.scalar(select(func.count()).select_from(ProspectiveForecast))
        runs = (await session.scalars(select(ShadowObservationRun))).all()
        for run in runs:
            result = await progress(session, run)
            created = run.created_at
            rows = (await session.scalars(select(Game).where(
                Game.time_precision == 'exact',
                Game.game_type.in_(('regular', 'play_in', 'playoffs')),
                Game.scheduled_start_utc >= created,
                Game.scheduled_start_utc <= now + timedelta(days=7),
            ).order_by(Game.scheduled_start_utc, Game.id))).all()
            observed = set((await session.scalars(select(ShadowObservation.game_id).where(
                ShadowObservation.run_id == run.id))).all())
            eligible = [g for g in rows if str(g.source_game_date) > run.protocol['baseline']['training_data_until'][:10]]
            result['closed_windows_without_capture'] = [game_summary(g) for g in eligible
                if g.scheduled_start_utc - timedelta(minutes=50) < now and g.id not in observed]
            upcoming = [g for g in eligible if g.status == 'Scheduled' and
                        g.scheduled_start_utc - timedelta(minutes=50) >= now]
            result['upcoming_games_next_7_days'] = len(upcoming)
            result['next_capture_windows'] = [dict(**game_summary(g),
                window_start_utc=(g.scheduled_start_utc - timedelta(minutes=60)).isoformat(),
                window_end_utc=(g.scheduled_start_utc - timedelta(minutes=50)).isoformat())
                for g in upcoming[:20]]
            result['protocol_hash'] = run.protocol_hash
            result['protocol'] = run.protocol
            report['runs'].append(result)
    assert active_bytes == (root / 'active.json').read_bytes()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    compact = {k: v for k, v in report.items() if k != 'runs'}
    compact['runs'] = [{k: v for k, v in run.items() if k != 'protocol'} for run in report['runs']]
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/shadow_rollout_status.json')
    asyncio.run(inspect(parser.parse_args().output))
