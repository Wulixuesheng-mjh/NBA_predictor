"""Leakage acceptance check: a future game must not change past features.

Picks a completed game, builds its feature vector, inserts a *synthetic* game
for the same home team dated after it (inside a transaction that is rolled
back), rebuilds the vector and compares. Any difference means the feature
pipeline can see the future.

Usage (inside the trainer/backend container):

    python scripts/check_no_leakage.py --game-id 27000

Exit code 0 = identical, 1 = leak detected.
"""

import argparse
import asyncio
import os
import sys
from datetime import timedelta

for _p in ("/app", "/app/backend"):
    if os.path.isdir(os.path.join(_p, "app")):
        sys.path.insert(0, _p)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sqlalchemy import select, desc

from app.database import async_session_factory
from app.models.game import Game
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder


async def main(game_id: int | None) -> int:
    async with async_session_factory() as session:
        query = select(Game)
        if game_id is not None:
            query = query.where(Game.id == game_id)
        else:
            query = query.where(Game.team_stats_complete.is_(True), Game.status == 'Final').order_by(desc(Game.game_date)).limit(1)
        game = (await session.execute(query)).scalar_one_or_none()
        if game is None or game.status != "Final":
            print(f"game {game_id} not found or not Final")
            return 2

        builder = FeatureBuilder()
        game_id = game.id

        async def build():
            fv = await builder.build_features(
                home_team_id=game.home_team_id,
                away_team_id=game.away_team_id,
                game_date=game.game_date,
                lookback_games=10,
                session=session,
                use_historical=True,
                target_game_id=game.id,
            )
            return np.asarray(fv.features, dtype=np.float64)

        before = await build()

        # A synthetic result that would leak into season aggregates / roster
        # continuity if the pipeline were not as-of-date.
        future = Game(
            nba_game_id=f"leakcheck-{game.id}",
            season=game.season,
            game_date=game.game_date + timedelta(days=7),
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            status="Final",
            home_score=123,
            away_score=111,
            game_type="regular",
            score_complete=True,
            team_stats_complete=True,
            player_stats_complete=True,
        )
        session.add(future)
        await session.flush()

        from app.models.team_game_stats import TeamGameStats
        from app.models.game_player_stats import GamePlayerStats
        from app.models.historical_player_injuries import HistoricalPlayerInjury
        from app.models.injury_report import InjuryReport
        from app.models.player import Player

        for team_id, points, home in ((game.home_team_id, 123, True), (game.away_team_id, 111, False)):
            session.add(TeamGameStats(
                game_id=future.id, team_id=team_id, is_home=home, pts=points,
                fga=100, fgm=45, fg3a=50, fg3m=25, fta=20, ftm=15,
                pace=130.0, off_rtg=150.0, def_rtg=80.0, game_minutes=48.0,
            ))
        player_id = await session.scalar(select(Player.id).order_by(Player.id).limit(1))
        if player_id is not None:
            session.add(GamePlayerStats(
                game_id=future.id, player_id=player_id, team_id=game.home_team_id,
                minutes='48', pts=99, stl=20, blk=20,
            ))
            # Bind a later injury to the OLD target: matching game_id alone
            # must never bypass the information cutoff.
            session.add(HistoricalPlayerInjury(
                player_id=player_id, team_id=game.home_team_id, game_id=game.id,
                game_date=game.game_date.date(), injury_status='Out', source='leakcheck',
                published_at=game.game_date + timedelta(days=2),
                observed_at=game.game_date + timedelta(days=2),
            ))
        session.add(InjuryReport(
            game_id=game.id, team_id=game.home_team_id, source='leakcheck', is_complete=True,
            published_at=game.game_date + timedelta(days=2),
            observed_at=game.game_date + timedelta(days=2),
        ))
        await session.flush()

        after = await build()
        await session.rollback()  # never persist the synthetic game

    diff = np.abs(before - after)
    changed = np.where(diff > 1e-9)[0]
    print(f"game {game_id}: {len(before)} features, changed={len(changed)}")
    if changed.size:
        for i in changed[:10]:
            print(f"  {FEATURE_NAMES[i]}: {before[i]:.6f} -> {after[i]:.6f}")
        print("LEAK DETECTED")
        return 1

    print("OK — future scores, team/player statistics and later target-game injuries do not change past features")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check feature pipeline for future leakage")
    parser.add_argument("--game-id", type=int)
    sys.exit(asyncio.run(main(parser.parse_args().game_id)))
