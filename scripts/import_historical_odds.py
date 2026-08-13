"""Import historical betting lines (spread + over/under) from Kaggle.

Source: ``erichqiu/nba-odds-and-scores`` (seasons 2012-13 through 2018-19).

Each season folder has ``vegas.txt`` (regular season) and ``vegas_playoff.txt``
(playoffs). Two rows per game (home + away). Relevant columns:

  - ``Average_Line_Spread`` — per-team point spread (negative = favorite)
  - ``Average_Line_OU``      — game total line (same for both rows)
  - ``GameId``               — maps to our ``games.nba_game_id`` by dropping
                               the first two chars (vegas "0021200001" → db "21200001")

Convention written to ``betting_lines``:
  ``spread_line`` = expected home margin (home - away), positive = home favored
                  = -(home team's Average_Line_Spread)
                  = (away team's Average_Line_Spread)
  ``total_line``  = Average_Line_OU

Run inside the backend container (has kagglehub + app models):

    docker cp scripts/import_historical_odds.py nba-backend:/tmp/
    docker exec nba-backend python /tmp/import_historical_odds.py
"""

import asyncio
import csv
import glob
import os
import sys

# The ``app`` package lives at /app/app inside the backend image (WORKDIR /app),
# but running ``python /tmp/script.py`` puts /tmp — not /app — on sys.path.
# Make the import robust regardless of how it's invoked.
for _p in ("/app", "/app/backend"):
    if os.path.isdir(os.path.join(_p, "app")):
        sys.path.insert(0, _p)

import kagglehub
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import async_session_factory
from app.models.game import Game
from app.models.betting_line import BettingLine

KAGGLE_DATASET = "erichqiu/nba-odds-and-scores"


def _download() -> str:
    return kagglehub.dataset_download(KAGGLE_DATASET)


def _parse_season(folder_name: str) -> str:
    """Map dataset folder '2012-13' → DB season '2012'."""
    return folder_name.split("-")[0]


async def main() -> None:
    data_dir = await asyncio.to_thread(_download)
    print("Dataset at:", data_dir)

    # Build nba_game_id → games.id map (single query).
    async with async_session_factory() as session:
        result = await session.execute(select(Game.id, Game.nba_game_id))
        game_id_map = {row[1]: row[0] for row in result.all()}
    print("games loaded:", len(game_id_map))

    files = sorted(glob.glob(os.path.join(data_dir, "*", "vegas*.txt")))
    print("vegas files:", len(files))

    # Group rows by GameId → one entry per game.
    games: dict[str, dict] = {}
    for path in files:
        season = _parse_season(os.path.basename(os.path.dirname(path)))
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = row.get("GameId", "").strip()
                if not gid:
                    continue
                loc = row.get("Location", "").strip()
                try:
                    spread = float(row.get("Average_Line_Spread") or 0)
                    ou = float(row.get("Average_Line_OU") or 0)
                except (ValueError, TypeError):
                    continue
                if ou <= 0:
                    continue
                entry = games.setdefault(gid, {
                    "season": season,
                    "home_spread": None,
                    "away_spread": None,
                    "total_line": None,
                })
                entry["total_line"] = ou
                if loc == "home":
                    entry["home_spread"] = spread
                elif loc == "away":
                    entry["away_spread"] = spread

    print("unique games parsed:", len(games))

    inserted = 0
    skipped = 0
    async with async_session_factory() as session:
        for gid, e in games.items():
            db_nba_id = gid[2:]  # drop the leading "00" of the vegas game id
            db_game_id = game_id_map.get(db_nba_id)
            if db_game_id is None:
                skipped += 1
                continue

            home_spread = e["home_spread"]
            away_spread = e["away_spread"]
            # spread_line = expected home margin (home - away).
            if home_spread is not None:
                spread_line = -home_spread
            elif away_spread is not None:
                spread_line = away_spread  # away line already equals home margin
            else:
                skipped += 1
                continue

            stmt = pg_insert(BettingLine).values(
                game_id=db_game_id,
                season=e["season"],
                spread_line=spread_line,
                total_line=e["total_line"],
                home_spread=home_spread,
                away_spread=away_spread,
                source="erichqiu",
            ).on_conflict_do_update(
                index_elements=["game_id"],
                set_={
                    "season": e["season"],
                    "spread_line": spread_line,
                    "total_line": e["total_line"],
                    "home_spread": home_spread,
                    "away_spread": away_spread,
                    "source": "erichqiu",
                    "updated_at": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ),
                },
            )
            await session.execute(stmt)
            inserted += 1

            if inserted % 1000 == 0:
                await session.commit()
                print(f"  progress: {inserted} inserted")

        await session.commit()

    print(f"\nDone: inserted/upserted={inserted}, skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
