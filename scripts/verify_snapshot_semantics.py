"""Transactional assertions against the real schema; changes are rolled back."""
import asyncio
from datetime import timedelta
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.injury_report import InjuryReport
from app.services.injury_snapshot_reader import latest_injury_snapshot, snapshot_statuses
from ml.src.features.injury_net_features import RotationPlayer, project_rotation, summarize_rotation
from ml.src.features.player_features import PlayerFeatures, InjuryInfo
from ml.src.evaluation import complete_day_split, monthly_folds
from ml.src.features.context_features import ContextFeatures


async def main():
    async with async_session_factory() as session:
        try:
            game = (await session.execute(select(Game).where(Game.time_precision == 'exact').limit(1))).scalar_one()
            cutoff = game.scheduled_start_utc - timedelta(seconds=1)
            async def add(offset, complete, entries):
                row = InjuryReport(game_id=game.id, team_id=game.home_team_id, source='transaction_test',
                    published_at=cutoff-timedelta(seconds=offset), observed_at=cutoff+timedelta(days=1, seconds=offset),
                    publication_verified=True, is_complete=complete, entries_json=entries)
                session.add(row)
                await session.flush()
                return row
            older = await add(3, True, [{'player_id':1, 'status':'Out'}])
            partial = await add(2, False, [])
            selected = await latest_injury_snapshot(session, game.id, game.home_team_id, cutoff, reconstructed=True)
            assert selected.id == partial.id and not selected.is_complete, 'Partial must mask older complete'
            live = await latest_injury_snapshot(session, game.id, game.home_team_id, cutoff)
            assert live is None or live.id not in (older.id, partial.id), 'Late collection leaked into live'
            available = await add(1, True, [{'player_id':1, 'status':'Available'}])
            selected = await latest_injury_snapshot(session, game.id, game.home_team_id, cutoff, reconstructed=True)
            assert snapshot_statuses(selected) == {1:'Available'}, 'Available must replace Out'
            await add(-1, True, [{'player_id':1, 'status':'Out'}])
            assert (await latest_injury_snapshot(session, game.id, game.home_team_id, cutoff,
                reconstructed=True)).id == available.id, 'Future publication leaked'
            from app.services.injury_service import InjuryService
            from app.models.player import Player
            pid = await session.scalar(select(Player.id).limit(1))
            manual = await InjuryService.create_injury(session, player_id=pid, injury_status='Out',
                game_id=game.id, team_id=game.home_team_id)
            initial = (await session.execute(select(InjuryReport).where(
                InjuryReport.source == 'manual', InjuryReport.game_id == game.id,
                InjuryReport.team_id == game.home_team_id,
            ).order_by(InjuryReport.observed_at.desc()).limit(1))).scalar_one()
            assert not initial.is_complete
            await InjuryService.update_injury(session, manual.id, injury_status='Available')
            assert next(e for e in initial.entries_json if e['player_id'] == pid)['status'] == 'Out'
            current = await latest_injury_snapshot(session, game.id, game.home_team_id,
                manual.updated_at + timedelta(seconds=1))
            assert snapshot_statuses(current)[pid] == 'Available'
        finally:
            await session.rollback()
    healthy = PlayerFeatures._compute_from_injuries([InjuryInfo(1,'Available',4,30,36,2,1)])
    assert all(v == 0 for v in healthy.values())
    players = [RotationPlayer(i, 'G', 30, 30, 0 if i == 0 else 1, .5, .03) for i in range(9)]
    projected = project_rotation(players)
    assert projected[0] == 0 and abs(sum(projected.values())-240) < 1e-7
    unknown = [RotationPlayer(i, 'G', 24, 24, 1, .5, .03, production_observed=False) for i in range(10)]
    assert summarize_rotation(unknown, project_rotation(unknown))['injury_estimated_minutes_share'] == 0
    # These starts cross UTC month boundaries; grouping must use the NBA day.
    info = [dict(source_game_date='2025-01-31', game_date='2025-02-01T03:00:00Z')]*1000
    info += [dict(source_game_date='2025-02-01', game_date='2025-02-02T01:00:00Z')]*20
    assert complete_day_split(info) == 1000
    folds = list(monthly_folds(info))
    assert folds[0][0] == '2025-02' and int(folds[0][1].sum()) == 1000
    print('PASS: snapshot cutoff, partial/Available precedence, rollback, rotation capacity, coverage, NBA days')


if __name__ == '__main__':
    asyncio.run(main())
