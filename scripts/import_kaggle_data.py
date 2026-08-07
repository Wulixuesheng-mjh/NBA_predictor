"""Import NBA data from Kaggle CSV dataset into PostgreSQL.

Downloads data via kagglehub (if not already cached) and bulk-imports into
the NBA predictor database tables: players, games, team_game_stats, game_player_stats.

Usage:
    docker compose exec backend python scripts/import_kaggle_data.py [--seasons 2024-25 2025-26]

The Kaggle dataset used:
    eoinamoore/historical-nba-data-and-player-box-scores
    https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add backend to Python path
sys.path.insert(0, "/app")

import kagglehub
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import async_session_factory
from app.models.team import Team
from app.models.player import Player
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats

BATCH_SIZE = 500
KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"


def _get_data_dir() -> str:
    """Download (if needed) and return path to the Kaggle dataset."""
    return kagglehub.dataset_download(KAGGLE_DATASET)


def _determine_season(game_date_str: str) -> str:
    """Convert game date to NBA season string.

    Example: '2024-10-25' → '2024-25', '2025-03-15' → '2024-25'
    """
    dt = datetime.fromisoformat(game_date_str[:10])
    if dt.month >= 10:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    else:
        return f"{dt.year - 1}-{str(dt.year)[-2:]}"


async def import_players(data_dir: str) -> dict[int, int]:
    """Import players from Players.csv.

    Returns:
        Mapping of nba_player_id (personId) → our internal player.id
    """
    players_path = os.path.join(data_dir, "Players.csv")
    if not os.path.exists(players_path):
        print("  Players.csv not found, skipping")
        return {}

    # Read all players
    rows = []
    with open(players_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("personId", "").strip()
            if not pid:
                continue
            rows.append({
                "nba_player_id": int(pid),
                "first_name": row.get("firstName", "").strip() or None,
                "last_name": row.get("lastName", "").strip() or None,
                "full_name": f"{row.get('firstName', '').strip()} {row.get('lastName', '').strip()}".strip() or None,
                "position": None,  # Players.csv doesn't have position
            })

    print(f"  Read {len(rows)} players from CSV")

    count = 0
    async with async_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            for p in batch:
                stmt = pg_insert(Player).values(**p).on_conflict_do_update(
                    index_elements=["nba_player_id"],
                    set_={
                        "first_name": p["first_name"],
                        "last_name": p["last_name"],
                        "full_name": p["full_name"],
                                            },
                )
                await session.execute(stmt)
                count += 1
            await session.commit()
            if (i + BATCH_SIZE) % 5000 == 0:
                print(f"    ... {count} players imported")

    print(f"  Imported {count} players")

    # Build and return mapping
    mapping = {}
    async with async_session_factory() as session:
        result = await session.execute(select(Player.nba_player_id, Player.id))
        for row in result:
            mapping[row[0]] = row[1]

    return mapping


async def import_games(data_dir: str, seasons: list[str]) -> dict[str, int]:
    """Import games from Games.csv for specified seasons.

    Returns:
        Mapping of nba_game_id → our internal game.id
    """
    games_path = os.path.join(data_dir, "Games.csv")
    if not os.path.exists(games_path):
        print("  Games.csv not found, skipping")
        return {}

    # Build team nba_id → internal id mapping
    team_map = {}
    async with async_session_factory() as session:
        result = await session.execute(select(Team.nba_team_id, Team.id))
        for row in result:
            team_map[str(row[0])] = row[1]

    season_set = set(seasons)

    # Read games and filter by season
    game_rows = []
    skipped = 0
    with open(games_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_date = (row.get("gameDate") or row.get("gameDateTimeEst", "")).strip()
            if not game_date:
                skipped += 1
                continue

            season = _determine_season(game_date)
            if season not in season_set:
                skipped += 1
                continue

            home_id = row.get("hometeamId", "").strip()
            away_id = row.get("awayteamId", "").strip()
            nba_game_id = row.get("gameId", "").strip()
            home_score = row.get("homeScore", "").strip()
            away_score = row.get("awayScore", "").strip()
            game_type = row.get("gameType", "").strip()

            if not home_id or not away_id or not nba_game_id:
                skipped += 1
                continue

            home_db_id = team_map.get(home_id)
            away_db_id = team_map.get(away_id)
            if not home_db_id or not away_db_id:
                skipped += 1
                continue

            # Determine status
            if home_score and away_score:
                status = "Final"
            else:
                status = "Scheduled"

            try:
                parsed_date = datetime.fromisoformat(game_date[:19])
            except ValueError:
                skipped += 1
                continue

            game_rows.append({
                "nba_game_id": nba_game_id,
                "season": season,
                "game_date": parsed_date,
                "home_team_id": home_db_id,
                "away_team_id": away_db_id,
                "status": status,
                "home_score": int(home_score) if (home_score and home_score.isdigit()) else None,
                "away_score": int(away_score) if (away_score and away_score.isdigit()) else None,
            })

    print(f"  Filtered {len(game_rows)} games for seasons {seasons} (skipped {skipped})")

    count = 0
    async with async_session_factory() as session:
        for i in range(0, len(game_rows), BATCH_SIZE):
            batch = game_rows[i : i + BATCH_SIZE]
            for g in batch:
                stmt = pg_insert(Game).values(**g).on_conflict_do_update(
                    index_elements=["nba_game_id"],
                    set_={
                        "status": g["status"],
                        "home_score": g["home_score"],
                        "away_score": g["away_score"],
                                            },
                )
                await session.execute(stmt)
                count += 1
            await session.commit()
            if (i + BATCH_SIZE) % 5000 == 0:
                print(f"    ... {count} games imported")

    print(f"  Imported {count} games")

    # Build mapping
    game_map = {}
    async with async_session_factory() as session:
        result = await session.execute(select(Game.nba_game_id, Game.id))
        for row in result:
            game_map[row[0]] = row[1]

    return game_map


async def import_team_stats(data_dir: str, game_map: dict[str, int]) -> int:
    """Import team game stats from TeamStatistics.csv."""
    stats_path = os.path.join(data_dir, "TeamStatistics.csv")
    if not os.path.exists(stats_path):
        print("  TeamStatistics.csv not found, skipping")
        return 0

    # Build team nba_id → internal id mapping
    team_map = {}
    async with async_session_factory() as session:
        result = await session.execute(select(Team.nba_team_id, Team.id))
        for row in result:
            team_map[str(row[0])] = row[1]

    rows = []
    skipped = 0
    with open(stats_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nba_game_id = row.get("gameId", "").strip()
            game_db_id = game_map.get(nba_game_id)
            if not game_db_id:
                skipped += 1
                continue

            team_nba_id = row.get("teamId", "").strip()
            team_db_id = team_map.get(team_nba_id)
            if not team_db_id:
                skipped += 1
                continue

            def _float(v, default=0):
                try:
                    return float(v) if v else default
                except (ValueError, TypeError):
                    return default

            def _int(v, default=0):
                try:
                    return int(v) if v else default
                except (ValueError, TypeError):
                    return default

            is_home = row.get("home", "").strip() == "1"

            rows.append({
                "game_id": game_db_id,
                "team_id": team_db_id,
                "is_home": is_home,
                "pts": _int(row.get("teamScore")),
                "fgm": _int(row.get("fieldGoalsMade")),
                "fga": _int(row.get("fieldGoalsAttempted")),
                "fg3m": _int(row.get("threePointersMade")),
                "fg3a": _int(row.get("threePointersAttempted")),
                "ftm": _int(row.get("freeThrowsMade")),
                "fta": _int(row.get("freeThrowsAttempted")),
                "reb": _int(row.get("reboundsTotal")),
                "ast": _int(row.get("assists")),
                "stl": _int(row.get("steals")),
                "blk": _int(row.get("blocks")),
                "tov": _int(row.get("turnovers")),
                "pf": _int(row.get("foulsPersonal")),
            })

    print(f"  Read {len(rows)} team stat rows (skipped {skipped})")

    count = 0
    async with async_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            for s in batch:
                # No ON CONFLICT — table has no unique constraint on (game_id, team_id).
                # Plain INSERT is safe since we're importing into an empty table.
                stmt = pg_insert(TeamGameStats).values(**s)
                await session.execute(stmt)
                count += 1
            await session.commit()
            if (i + BATCH_SIZE) % 10000 == 0:
                print(f"    ... {count} team stats imported")

    print(f"  Imported {count} team game stats")
    return count


async def import_player_stats(
    data_dir: str,
    game_map: dict[str, int],
    player_map: dict[int, int],
) -> int:
    """Import player game stats from PlayerStatistics.csv."""
    stats_path = os.path.join(data_dir, "PlayerStatistics.csv")
    if not os.path.exists(stats_path):
        print("  PlayerStatistics.csv not found, skipping")
        return 0

    # Build team nba_id → internal id mapping
    team_map = {}
    async with async_session_factory() as session:
        result = await session.execute(select(Team.nba_team_id, Team.id))
        for row in result:
            team_map[str(row[0])] = row[1]

    rows = []
    skipped = 0
    with open(stats_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nba_game_id = row.get("gameId", "").strip()
            game_db_id = game_map.get(nba_game_id)
            if not game_db_id:
                skipped += 1
                continue

            try:
                nba_player_id = int(row.get("personId", "").strip())
            except (ValueError, TypeError):
                skipped += 1
                continue

            player_db_id = player_map.get(nba_player_id)
            if not player_db_id:
                skipped += 1
                continue

            team_nba_id = row.get("playerteamId", "").strip()

            team_db_id = team_map.get(team_nba_id)
            if not team_db_id:
                skipped += 1
                continue

            def _float(v, default=None):
                try:
                    return float(v) if v else default
                except (ValueError, TypeError):
                    return default

            def _int(v, default=0):
                try:
                    return int(v) if v else default
                except (ValueError, TypeError):
                    return default

            minutes_str = row.get("numMinutes", "").strip()
            # Convert decimal minutes to string (model expects String(10))
            minutes = None
            if minutes_str:
                try:
                    minutes = str(round(float(minutes_str), 1))
                except (ValueError, TypeError):
                    minutes = minutes_str

            rows.append({
                "game_id": game_db_id,
                "player_id": player_db_id,
                "team_id": team_db_id,
                "minutes": minutes,
                "pts": _int(row.get("points")),
                "reb": _int(row.get("reboundsTotal")),
                "ast": _int(row.get("assists")),
                "stl": _int(row.get("steals")),
                "blk": _int(row.get("blocks")),
                "tov": _int(row.get("turnovers")),
                "pf": _int(row.get("foulsPersonal")),
                "fgm": _int(row.get("fieldGoalsMade")),
                "fga": _int(row.get("fieldGoalsAttempted")),
                "fg3m": _int(row.get("threePointersMade")),
                "fg3a": _int(row.get("threePointersAttempted")),
                "ftm": _int(row.get("freeThrowsMade")),
                "fta": _int(row.get("freeThrowsAttempted")),
                "plus_minus": _int(row.get("plusMinusPoints")),
            })

    print(f"  Read {len(rows)} player stat rows (skipped {skipped})")

    count = 0
    async with async_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            for s in batch:
                stmt = pg_insert(GamePlayerStats).values(**s).on_conflict_do_update(
                    index_elements=["game_id", "player_id"],
                    set_={
                        "pts": s["pts"],
                        "reb": s["reb"],
                        "ast": s["ast"],
                                            },
                )
                await session.execute(stmt)
                count += 1
            await session.commit()
            if (i + BATCH_SIZE) % 20000 == 0:
                print(f"    ... {count} player stats imported")

    print(f"  Imported {count} player game stats")
    return count


async def main(seasons: list[str]) -> None:
    """Run the full import pipeline."""
    print("=" * 60)
    print("NBA Kaggle Data Importer")
    print("=" * 60)
    print(f"  Seasons: {seasons}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Download / locate dataset
    print("\n[0/4] Locating Kaggle dataset...")
    data_dir = _get_data_dir()
    print(f"  Data directory: {data_dir}")

    # 1. Import players
    print("\n[1/4] Importing players...")
    player_map = await import_players(data_dir)

    # 2. Import games
    print("\n[2/4] Importing games...")
    game_map = await import_games(data_dir, seasons)

    # 3. Import team stats
    print("\n[3/4] Importing team game stats...")
    team_stats_count = await import_team_stats(data_dir, game_map)

    # 4. Import player stats
    print("\n[4/4] Importing player game stats...")
    player_stats_count = await import_player_stats(data_dir, game_map, player_map)

    # Summary
    print("\n" + "=" * 60)
    print("Import Complete!")
    print(f"  Players:         {len(player_map)}")
    print(f"  Games:           {len(game_map)}")
    print(f"  Team stats:      {team_stats_count}")
    print(f"  Player stats:    {player_stats_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle NBA data")
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=["2024-25", "2025-26"],
        help="Seasons to import (default: 2024-25 2025-26)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inserts (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    asyncio.run(main(args.seasons))
