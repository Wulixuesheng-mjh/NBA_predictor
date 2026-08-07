"""Seed script to populate initial NBA data.

Usage:
    python scripts/seed_data.py [--season 2025-26]

Handles off-season gracefully: if the requested season has no data,
falls back to the most recent completed season.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.collector.nba_collector import get_collector


async def main(season: str | None = None) -> None:
    """Run the seed process using NBACollector."""
    print(f"Seeding NBA data for season: {season or 'current'}")

    collector = get_collector()

    print("  → Syncing teams...")
    teams_count = await collector.sync_teams()
    print(f"    {teams_count} teams synced")

    print("  → Syncing players...")
    players_count = await collector.sync_players()
    print(f"    {players_count} players synced")

    print("  → Syncing schedule...")
    games_count = await collector.sync_schedule(season)
    print(f"    {games_count} games synced")

    print("  → Syncing recent game results...")
    results_count = await collector.sync_game_results()
    print(f"    {results_count} game results synced")

    print("  → Syncing injury reports...")
    injuries_count = await collector.sync_injuries()
    print(f"    {injuries_count} injuries synced")

    print("\nSeed complete! 🏀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed NBA data into the database")
    parser.add_argument("--season", type=str, default=None, help="Season to seed (e.g. 2025-26)")
    args = parser.parse_args()

    asyncio.run(main(args.season))
