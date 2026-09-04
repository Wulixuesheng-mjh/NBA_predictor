"""Backfill ``historical_player_injuries`` from the NBA official injury reports.

Fetches the NBA daily injury report PDFs (via the ``nbainjuries`` package,
which reads ``ak-static.cms.nba.com/referee/injury/Injury-Report_*.pdf``) for
every game date in a season range, and upserts per-day player injury snapshots.

Mapping conventions
  - ``historical_player_injuries.team_id`` comes from the report's ``Team``
    column (not ``Player.team_id``), because players change teams.
  - ``historical_player_injuries.player_id`` is resolved by matching the
    report's ``Player Name`` ("Last, First") against ``players.last_name`` /
    ``players.first_name``.
  - ``game_date`` comes from the report's ``Game Date`` column (authoritative),
    not the report publication date.

Run as a one-off container (see ``ml/Dockerfile.injuries``):

    docker run --rm --network nba-network \
      -e DATABASE_URL=postgresql+asyncpg://nba_admin:nba_predictor_2024@postgres:5432/nba_predictor \
      nba-injury-importer --start 2021-10-15 --end 2026-06-15
"""

import argparse
import asyncio
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta

# The ``app`` package lives at /app/backend/app inside this image (mirrors the
# trainer layout). Make imports robust regardless of invocation directory.
for _p in ("/app", "/app/backend"):
    if os.path.isdir(os.path.join(_p, "app")):
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from app.database import async_session_factory  # noqa: E402
from app.models.team import Team  # noqa: E402
from app.models.player import Player  # noqa: E402
from app.models.historical_player_injuries import HistoricalPlayerInjury  # noqa: E402

# Imported once — starts the JVM (jpype) used by tabula.
import nbainjuries  # noqa: E402


# Statuses we persist. "Available" (healthy) carries no signal and is dropped.
VALID_STATUSES = {"Out", "Doubtful", "Questionable", "Probable"}

# Candidate report times to try per date, most-representative first.
# (17:30 = 5:30 PM is the package's own example time; legacy URLs truncate the
# minute to the hour, so distinct hours are what matter before 2025-12-19.)
CANDIDATE_TIMES = [
    (17, 30), (17, 0), (13, 30), (13, 0), (18, 30), (18, 0),
    (11, 30), (11, 0), (12, 0), (19, 0), (16, 30), (16, 0),
    (20, 0), (14, 0), (15, 0), (21, 0), (22, 0), (23, 0),
    (10, 0), (9, 0), (8, 0), (6, 0),
]


# NBA injury reports write the Clippers as "LA Clippers" but our DB stores
# "Los Angeles Clippers".  Normalize the city abbreviation for matching.
_CITY_ALIASES = {"la": "los angeles"}


def _norm(s: str) -> str:
    """Normalize a name: strip diacritics/periods, collapse whitespace, lowercase."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace(".", "")  # "Jr." vs "Jr" consistency
    s = re.sub(r"\s+", " ", s.strip().lower())
    # Apply city aliases on the leading token (e.g. "la clippers" → "los angeles clippers")
    for alias, full in _CITY_ALIASES.items():
        if s.startswith(alias + " "):
            s = full + s[len(alias):]
    return s


def _parse_player_name(raw: str) -> tuple[str, str]:
    """Split report "Last, First" → (last, first), normalized."""
    if not raw:
        return "", ""
    parts = raw.split(",", 1)
    last = parts[0].strip()
    first = parts[1].strip() if len(parts) > 1 else ""
    return _norm(last), _norm(first)


_SUFFIXES = {"iii", "ii", "iv", "v", "jr", "sr"}


def _strip_suffix(name: str) -> str:
    """Drop trailing generational suffixes ("III", "Jr", ...) from a name."""
    parts = name.split()
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _parse_game_date(raw):
    """Parse the report's Game Date column to a ``datetime.date`` (or None)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "nat", "none", ""):
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.split(" ")[0], fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


async def _load_maps() -> tuple[dict, dict, dict, dict]:
    """Load team_id and player_id lookup maps."""
    async with async_session_factory() as session:
        teams = (await session.execute(select(Team))).scalars().all()
        players = (await session.execute(
            select(Player.id, Player.first_name, Player.last_name, Player.full_name)
        )).all()

    # team map: many aliases → id (abbreviation, full_name, city+name, name)
    team_map: dict[str, int] = {}
    for t in teams:
        keys = {t.abbreviation, t.full_name, t.name, f"{t.city} {t.name}", t.city}
        for k in keys:
            if k:
                team_map[_norm(k)] = t.id

    # player maps: exact (last, first) → id; suffix-stripped (last, first) → [ids];
    # and full_name → id.
    player_map: dict[tuple[str, str], int] = {}
    player_map_nosuffix: dict[tuple[str, str], list[int]] = {}
    fullname_map: dict[str, int] = {}
    for pid, first, last, full in players:
        if last:
            player_map[(_norm(last), _norm(first))] = pid
            ns_key = (_strip_suffix(_norm(last)), _norm(first))
            player_map_nosuffix.setdefault(ns_key, []).append(pid)
        if full:
            fullname_map[_norm(full)] = pid

    return team_map, player_map, fullname_map, player_map_nosuffix


def _match_player(
    last: str, first: str, player_map: dict, fullname_map: dict,
    player_map_nosuffix: dict,
) -> int | None:
    if last and (last, first) in player_map:
        return player_map[(last, first)]
    # Suffix mismatch (report "Butler" vs DB "Butler III"): unambiguous only.
    if last:
        cands = player_map_nosuffix.get((_strip_suffix(last), first), [])
        if len(cands) == 1:
            return cands[0]
    # Fallback: reconstruct "First Last" full name.
    if last and first:
        cand = _norm(f"{first} {last}")
        if cand in fullname_map:
            return fullname_map[cand]
    return None


async def _fetch_report(dt: datetime):
    """Return the parsed DataFrame for the report at the given datetime, or None."""
    for (hh, mm) in CANDIDATE_TIMES:
        ts = dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        try:
            df = nbainjuries.injury.get_reportdata(ts, return_df=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2021-10-15")
    ap.add_argument("--end", default="2026-06-15")
    ap.add_argument("--limit", type=int, default=0, help="only process first N dates (test)")
    ap.add_argument("--inspect", default="", help="inspect one date (YYYY-MM-DD) and exit")
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    team_map, player_map, fullname_map, player_map_nosuffix = await _load_maps()
    print(f"teams={len(team_map)} players={len(player_map)}")

    if args.inspect:
        dt = datetime.strptime(args.inspect, "%Y-%m-%d")
        df = await _fetch_report(dt)
        if df is None:
            print("no report found")
            return
        print(df.head(30).to_string())
        print("\n--- columns:", list(df.columns))
        print("\n--- distinct Teams:", sorted(df["Team"].dropna().unique().tolist()))
        print("--- distinct Status:", sorted(df["Current Status"].dropna().unique().tolist()))
        return

    # Determine target dates that actually have games. The day-of report lists
    # that day's games (verified empirically), so game dates are sufficient.
    async with async_session_factory() as session:
        from app.models.game import Game
        from sqlalchemy import distinct, func
        rows = (await session.execute(
            select(distinct(func.date(Game.game_date)))
            .where(func.date(Game.game_date) >= start, func.date(Game.game_date) <= end)
        )).scalars().all()
    dates = sorted({d for d in rows})
    if args.limit:
        dates = dates[: args.limit]
    print(f"game dates in range: {len(dates)}")

    inserted = 0
    skipped_players = 0
    skipped_teams = 0
    days_fetched = 0
    days_missed = 0
    unmatched_players: dict[str, int] = {}
    unmatched_teams: dict[str, int] = {}

    # Keyed by (player_id, team_id, game_date) so the same game reported on
    # consecutive days (day-before + day-of) collapses to one row, and so a
    # single INSERT never contains two rows for the same unique constraint.
    pending: dict[tuple[int, int, object], dict] = {}

    async def flush(session):
        nonlocal inserted
        if not pending:
            return
        rows = list(pending.values())
        stmt = pg_insert(HistoricalPlayerInjury).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["player_id", "team_id", "game_date"],
            set_={
                "injury_status": stmt.excluded.injury_status,
                "injury_type": stmt.excluded.injury_type,
            },
        )
        await session.execute(stmt)
        inserted += len(rows)
        pending.clear()

    async with async_session_factory() as session:
        for i, d in enumerate(dates):
            dt = datetime(d.year, d.month, d.day)
            df = await _fetch_report(dt)
            if df is None:
                days_missed += 1
                continue
            days_fetched += 1

            for _, row in df.iterrows():
                status = str(row.get("Current Status", "") or "").strip()
                if status not in VALID_STATUSES:
                    continue
                game_date = _parse_game_date(row.get("Game Date"))
                if game_date is None:
                    continue
                team = _norm(row.get("Team"))
                tid = team_map.get(team)
                if tid is None:
                    skipped_teams += 1
                    unmatched_teams[team] = unmatched_teams.get(team, 0) + 1
                    continue
                last, first = _parse_player_name(row.get("Player Name"))
                pid = _match_player(last, first, player_map, fullname_map, player_map_nosuffix)
                if pid is None:
                    skipped_players += 1
                    key = f"{first} {last}".strip()
                    unmatched_players[key] = unmatched_players.get(key, 0) + 1
                    continue
                pending[(pid, tid, game_date)] = {
                    "player_id": pid,
                    "team_id": tid,
                    "game_date": game_date,
                    "injury_status": status,
                    "injury_type": (str(row.get("Reason", "") or "")[:100] or None),
                    "source": "nbainjuries",
                }

            if len(pending) >= args.batch:
                await flush(session)
                await session.commit()

            if (i + 1) % 50 == 0 or i == len(dates) - 1:
                print(
                    f"  [{i+1}/{len(dates)}] fetched={days_fetched} missed={days_missed} "
                    f"inserted={inserted} skip_player={skipped_players} skip_team={skipped_teams}"
                )

        await flush(session)
        await session.commit()

    print("\n=== Backfill complete ===")
    print(f"days_fetched={days_fetched} days_missed={days_missed}")
    print(f"inserted={inserted} skipped_players={skipped_players} skipped_teams={skipped_teams}")
    if unmatched_players:
        top = sorted(unmatched_players.items(), key=lambda kv: -kv[1])[:20]
        print(f"top unmatched players: {top}")
    if unmatched_teams:
        print(f"unmatched teams: {sorted(unmatched_teams.items(), key=lambda kv: -kv[1])[:20]}")


if __name__ == "__main__":
    asyncio.run(main())
