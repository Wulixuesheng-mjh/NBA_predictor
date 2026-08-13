"""Backfill advanced stats into team_game_stats from basic box score data.

Derives pace, off_rtg, def_rtg, ts_pct, efg_pct, tov_pct, and ast_pct from the
basic box score columns we already store (see app.collector.advanced_stats for
the formulas). This fills the advanced-stat columns that are currently NULL
because neither the Kaggle dataset nor nba_collector ever provided them.

Usage (run inside the trainer container, which mounts scripts/ + backend/app):
    docker compose --profile trainer run --rm --entrypoint python \
        trainer scripts/compute_advanced_stats.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

# Add project root AND backend directory to Python path (same as train_models.py):
# /app for the 'ml' package, /app/backend so 'app' resolves to /app/backend/app.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import select, update

from app.collector.advanced_stats import derive_game_advanced_stats, REGULATION_MINUTES
from app.database import async_session_factory
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats


def _parse_minutes(m) -> float:
    """Parse a minutes value (decimal string '33.5' or 'mm:ss') to float minutes."""
    if m is None:
        return 0.0
    if isinstance(m, str):
        if ":" in m:
            parts = m.split(":")
            try:
                return float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                return 0.0
        try:
            return float(m)
        except ValueError:
            return 0.0
    return float(m)


async def main() -> None:
    """Compute advanced stats for all games with two team-stat rows and update them."""
    print("=" * 60)
    print("NBA Advanced Stats Backfill")
    print("=" * 60)

    async with async_session_factory() as session:
        # 1. Bulk-load all team box scores (small — ~20 columns x ~60k rows).
        stats_result = await session.execute(
            select(TeamGameStats).order_by(TeamGameStats.game_id)
        )
        all_stats = stats_result.scalars().all()
        print(f"Loaded {len(all_stats)} team game stat rows")

        by_game: dict[int, list[TeamGameStats]] = defaultdict(list)
        for ts in all_stats:
            by_game[ts.game_id].append(ts)

        # 2. Bulk-load player minutes, aggregated per game for OT detection.
        minutes_result = await session.execute(
            select(GamePlayerStats.game_id, GamePlayerStats.minutes)
        )
        minutes_by_game: dict[int, float] = defaultdict(float)
        for game_id, m in minutes_result.all():
            minutes_by_game[game_id] += _parse_minutes(m)

        # 3. Compute + update each game's two team rows.
        updated = 0
        skipped = 0
        for game_id, team_stats in by_game.items():
            if len(team_stats) != 2:
                skipped += 1
                continue

            home = next((ts for ts in team_stats if ts.is_home), None)
            away = next((ts for ts in team_stats if not ts.is_home), None)
            if home is None or away is None:
                skipped += 1
                continue

            total_player_minutes = minutes_by_game.get(game_id, 0.0)
            game_minutes = (
                total_player_minutes / 10.0
                if total_player_minutes > 480.0
                else REGULATION_MINUTES
            )

            home_adv, away_adv = derive_game_advanced_stats(
                home, away, game_minutes
            )

            await session.execute(
                update(TeamGameStats)
                .where(TeamGameStats.id == home.id)
                .values(**home_adv)
            )
            await session.execute(
                update(TeamGameStats)
                .where(TeamGameStats.id == away.id)
                .values(**away_adv)
            )

            updated += 2
            if updated % 5000 == 0:
                await session.commit()
                print(f"  ... {updated} team stat rows updated")

        await session.commit()

    print(f"\nUpdated {updated} team game stat rows with advanced metrics")
    print(f"Skipped {skipped} games with missing/incomplete team stats")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
