"""Kaggle-based NBA data collector.

Replaces NBACollector when nba_api is unavailable (blocked by Akamai/Cloudflare).
Uses the nightly-updated Kaggle dataset as the data source instead of live API calls.

Architecture:
  - sync_schedule():    Import upcoming games from LeagueSchedule CSV → "Scheduled"
  - sync_game_results(): Check Games.csv for newly completed games → update scores
  - sync_game_stats():   Import team + player box scores for newly Final games
  - sync_players():      Update player roster from Players.csv (nightly)
  - sync_teams():        No-op — teams are static and already seeded

KAGGLE DATASET (eoinamoore/historical-nba-data-and-player-box-scores):
  - Games.csv (11 MB)         → game results, scores, dates
  - Players.csv (512 KB)       → player metadata
  - TeamStatistics.csv (35 MB) → team-level box scores
  - PlayerStatistics.csv (372 MB) → player-level box scores
  - LeagueSchedule25_26.csv (187 KB) → full season schedule
  - LeagueSchedule24_25.csv (144 KB) → previous season schedule

Data freshness:
  The Kaggle dataset is updated NIGHTLY (US Eastern time). Same-day game results
  are NOT available through this collector. Use the admin manual-entry endpoint
  or the balldontlie.io integration for same-day results.
"""

import asyncio
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import kagglehub
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.collector.base import BaseCollector, CollectionResult
from app.database import async_session_factory
from app.models.team import Team
from app.models.player import Player
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats
from app.utils.logging import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
BATCH_SIZE = 500


class KaggleCollector(BaseCollector):
    """Collects NBA data from nightly-updated Kaggle dataset.

    Designed for environments where stats.nba.com is geo-blocked.
    All data reads go through Kaggle CSV files; no live API calls.
    """

    source = "kaggle"

    def __init__(self, session_factory=None):
        super().__init__(session_factory or async_session_factory)
        self._data_dir: Optional[str] = None
        self._team_map: Optional[dict[str, int]] = None

    # ── Public API ──────────────────────────────────────────────

    async def collect(self) -> CollectionResult:
        """Run the full Kaggle collection pipeline.

        Steps:
          1. Ensure Kaggle dataset is cached (download if needed)
          2. Sync current-season schedule
          3. Sync game results (update scores, detect new Final games)
          4. Sync box scores for newly completed games
          5. Sync player roster

        This is designed to be safe to run frequently — it only imports
        new or changed data. The Kaggle dataset is ~1 GB but cached;
        subsequent calls use the cached copy unless a new version exists.
        """
        result = CollectionResult(source=self.source, task_name="full_sync")
        total_records = 0

        try:
            # Step 0: Ensure dataset available
            await self._ensure_dataset()

            # Step 1: Sync schedule (fast — CSV is ~187 KB)
            total_records += await self.sync_schedule()

            # Step 2: Sync game results (fast — CSV is ~11 MB)
            total_records += await self.sync_game_results()

            # Step 3: Sync box scores for newly Final games
            total_records += await self.sync_game_stats()

            # Step 4: Sync player roster
            total_records += await self.sync_players()

            result.status = "success"
            result.records_collected = total_records

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error("Kaggle collection failed: %s", e, exc_info=True)
        finally:
            result.completed_at = datetime.now(timezone.utc)

        await self._log_collection(result)
        return result

    # ── Dataset Management ──────────────────────────────────────

    async def _ensure_dataset(self) -> str:
        """Download Kaggle dataset (if not cached) and return path.

        Uses asyncio.to_thread because kagglehub is synchronous and
        the initial download can take minutes (1 GB dataset).
        Subsequent calls return instantly from the local cache.
        """
        if self._data_dir is not None:
            return self._data_dir

        def _download():
            return kagglehub.dataset_download(KAGGLE_DATASET)

        self._data_dir = await asyncio.to_thread(_download)
        logger.info("Kaggle dataset at: %s", self._data_dir)
        return self._data_dir

    async def _get_team_map(self, session) -> dict[str, int]:
        """Build nba_team_id (str) → internal team.id mapping."""
        if self._team_map is None:
            result = await session.execute(select(Team.nba_team_id, Team.id))
            self._team_map = {str(row[0]): row[1] for row in result.all()}
        return self._team_map

    # ── Schedule Sync ───────────────────────────────────────────

    async def sync_schedule(self, season: Optional[str] = None) -> int:
        """Import upcoming games from LeagueSchedule CSV.

        Reads the schedule file for the specified season (or auto-detects
        the current season), inserting games with status='Scheduled'.
        Existing games are updated if their status has changed.
        """
        if season is None:
            season = self._determine_season()

        data_dir = await self._ensure_dataset()

        # Pick the right schedule file
        schedule_file = self._schedule_filename(season)
        schedule_path = os.path.join(data_dir, schedule_file)

        if not os.path.exists(schedule_path):
            logger.warning("Schedule file not found: %s", schedule_path)
            return 0

        logger.info("Syncing schedule from %s (season %s)", schedule_file, season)

        async with self.db_session_factory() as session:
            team_map = await self._get_team_map(session)

            count = 0
            skipped = 0
            with open(schedule_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    home_nba_id = row.get("homeTeamId", "").strip()
                    away_nba_id = row.get("awayTeamId", "").strip()
                    game_id = row.get("gameId", "").strip()
                    game_date_str = row.get("gameDateTimeEst", "").strip()

                    if not all([home_nba_id, away_nba_id, game_id, game_date_str]):
                        skipped += 1
                        continue

                    home_db_id = team_map.get(home_nba_id)
                    away_db_id = team_map.get(away_nba_id)
                    if not home_db_id or not away_db_id:
                        skipped += 1
                        continue

                    try:
                        game_date = datetime.fromisoformat(game_date_str[:19])
                    except ValueError:
                        skipped += 1
                        continue

                    # Check if game already exists
                    existing = await session.execute(
                        select(Game).where(Game.nba_game_id == game_id)
                    )
                    existing_game = existing.scalar_one_or_none()

                    if existing_game:
                        # Update if status changed (e.g., Scheduled → Final)
                        # Schedule CSV doesn't have scores, so only update
                        # if the game was somehow deleted and re-created
                        pass
                    else:
                        stmt = pg_insert(Game).values(
                            nba_game_id=game_id,
                            season=season,
                            game_date=game_date,
                            home_team_id=home_db_id,
                            away_team_id=away_db_id,
                            status="Scheduled",
                            home_score=None,
                            away_score=None,
                        ).on_conflict_do_update(
                            index_elements=["nba_game_id"],
                            set_={"game_date": game_date},
                        )
                        await session.execute(stmt)
                        count += 1

            await session.commit()

        logger.info("Schedule sync: %d new games, %d skipped", count, skipped)
        return count

    @staticmethod
    def _schedule_filename(season: str) -> str:
        """Map season string to schedule CSV filename.

        Example: '2024-25' → 'LeagueSchedule24_25.csv'
                 '2025-26' → 'LeagueSchedule25_26.csv'
        """
        # season = "2024-25" → parts = ["2024", "25"]
        parts = season.split("-")
        return f"LeagueSchedule{parts[0][-2:]}_{parts[1]}.csv"

    # ── Game Results Sync ───────────────────────────────────────

    async def sync_game_results(self) -> int:
        """Check Games.csv for newly completed games.

        Scans the full Games.csv (11 MB, fast) for games where:
          - The game exists in our DB with status='Scheduled'
          - The CSV shows a score (meaning the game is Final)

        Updates home_score, away_score, and status='Final'.
        Returns count of newly detected Final games.
        """
        data_dir = await self._ensure_dataset()
        games_path = os.path.join(data_dir, "Games.csv")

        if not os.path.exists(games_path):
            logger.warning("Games.csv not found")
            return 0

        # Build set of existing game IDs for quick lookup
        async with self.db_session_factory() as session:
            result = await session.execute(
                select(Game.nba_game_id, Game.status, Game.id)
                .where(Game.status == "Scheduled")
            )
            scheduled_games = {row[0]: row[2] for row in result.all()}

        if not scheduled_games:
            logger.info("No scheduled games to check — nothing to update")
            return 0

        logger.info(
            "Checking %d scheduled games against Games.csv...",
            len(scheduled_games),
        )

        count = 0
        async with self.db_session_factory() as session:
            with open(games_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nba_game_id = row.get("gameId", "").strip()

                    # Only process games we have as Scheduled
                    if nba_game_id not in scheduled_games:
                        continue

                    home_score_str = row.get("homeScore", "").strip()
                    away_score_str = row.get("awayScore", "").strip()

                    if not home_score_str or not away_score_str:
                        continue  # Still no result

                    if not home_score_str.isdigit() or not away_score_str.isdigit():
                        continue

                    home_score = int(home_score_str)
                    away_score = int(away_score_str)

                    db_game_id = scheduled_games[nba_game_id]

                    await session.execute(
                        pg_insert(Game).values(
                            nba_game_id=nba_game_id,
                            home_score=home_score,
                            away_score=away_score,
                            status="Final",
                        ).on_conflict_do_update(
                            index_elements=["nba_game_id"],
                            set_={
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": "Final",
                                "updated_at": datetime.now(timezone.utc),
                            },
                        )
                    )
                    count += 1

            await session.commit()

        logger.info("Updated %d games to Final", count)

        # Trigger retraining if enough new games
        if count >= 3:
            logger.info(
                "Detected %d new Final games — triggering model retraining", count
            )
            await self._trigger_model_retraining()

        return count

    # ── Box Score Sync ──────────────────────────────────────────

    async def sync_game_stats(self) -> int:
        """Import team + player box scores for newly Final games.

        Finds Final games that don't yet have team_game_stats records,
        then reads the corresponding rows from TeamStatistics.csv and
        PlayerStatistics.csv.
        """
        data_dir = await self._ensure_dataset()

        # Find Final games without team stats
        async with self.db_session_factory() as session:
            # Subquery: games that have team stats
            stats_query = select(TeamGameStats.game_id).distinct()
            # Games that are Final but have no stats
            result = await session.execute(
                select(Game.id, Game.nba_game_id)
                .where(
                    Game.status == "Final",
                    Game.id.not_in(stats_query),
                )
            )
            pending_games = [(row[0], row[1]) for row in result.all()]

        if not pending_games:
            logger.info("No pending games need box scores")
            return 0

        logger.info("Importing box scores for %d newly Final games", len(pending_games))

        # Build lookup: nba_game_id → internal game_id
        pending_nba_ids = {gid for _, gid in pending_games}
        pending_game_map = {gid: db_id for db_id, gid in pending_games}

        count_team = 0
        count_player = 0

        async with self.db_session_factory() as session:
            team_map = await self._get_team_map(session)

            # Process TeamStatistics.csv
            team_stats_path = os.path.join(data_dir, "TeamStatistics.csv")
            if os.path.exists(team_stats_path):
                with open(team_stats_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        nba_game_id = row.get("gameId", "").strip()
                        if nba_game_id not in pending_nba_ids:
                            continue

                        game_db_id = pending_game_map[nba_game_id]
                        team_nba_id = row.get("teamId", "").strip()
                        team_db_id = team_map.get(team_nba_id)
                        if not team_db_id:
                            continue

                        is_home = row.get("home", "").strip() == "1"

                        def _int(v, default=0):
                            try:
                                return int(v) if v else default
                            except (ValueError, TypeError):
                                return default

                        stmt = pg_insert(TeamGameStats).values(
                            game_id=game_db_id,
                            team_id=team_db_id,
                            is_home=is_home,
                            pts=_int(row.get("teamScore")),
                            fgm=_int(row.get("fieldGoalsMade")),
                            fga=_int(row.get("fieldGoalsAttempted")),
                            fg3m=_int(row.get("threePointersMade")),
                            fg3a=_int(row.get("threePointersAttempted")),
                            ftm=_int(row.get("freeThrowsMade")),
                            fta=_int(row.get("freeThrowsAttempted")),
                            reb=_int(row.get("reboundsTotal")),
                            ast=_int(row.get("assists")),
                            stl=_int(row.get("steals")),
                            blk=_int(row.get("blocks")),
                            tov=_int(row.get("turnovers")),
                            pf=_int(row.get("foulsPersonal")),
                        )
                        # TeamGameStats has no unique constraint — check first
                        existing = await session.execute(
                            select(TeamGameStats).where(
                                TeamGameStats.game_id == game_db_id,
                                TeamGameStats.team_id == team_db_id,
                            )
                        )
                        if not existing.first():
                            await session.execute(stmt)
                            count_team += 1

            await session.commit()

            # Process PlayerStatistics.csv
            player_stats_path = os.path.join(data_dir, "PlayerStatistics.csv")
            if os.path.exists(player_stats_path):
                # Build player nba_id → internal id mapping
                player_result = await session.execute(
                    select(Player.nba_player_id, Player.id)
                )
                player_map = {row[0]: row[1] for row in player_result.all()}

                with open(player_stats_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        nba_game_id = row.get("gameId", "").strip()
                        if nba_game_id not in pending_nba_ids:
                            continue

                        game_db_id = pending_game_map[nba_game_id]

                        try:
                            nba_player_id = int(row.get("personId", "").strip())
                        except (ValueError, TypeError):
                            continue

                        player_db_id = player_map.get(nba_player_id)
                        if not player_db_id:
                            continue

                        team_nba_id = row.get("playerteamId", "").strip()
                        team_db_id = team_map.get(team_nba_id)
                        if not team_db_id:
                            continue

                        def _int_ps(v, default=0):
                            try:
                                return int(v) if v else default
                            except (ValueError, TypeError):
                                return default

                        minutes_str = row.get("numMinutes", "").strip()
                        minutes = None
                        if minutes_str:
                            try:
                                minutes = str(round(float(minutes_str), 1))
                            except (ValueError, TypeError):
                                minutes = minutes_str[:10]

                        stmt = pg_insert(GamePlayerStats).values(
                            game_id=game_db_id,
                            player_id=player_db_id,
                            team_id=team_db_id,
                            minutes=minutes,
                            pts=_int_ps(row.get("points")),
                            reb=_int_ps(row.get("reboundsTotal")),
                            ast=_int_ps(row.get("assists")),
                            stl=_int_ps(row.get("steals")),
                            blk=_int_ps(row.get("blocks")),
                            tov=_int_ps(row.get("turnovers")),
                            pf=_int_ps(row.get("foulsPersonal")),
                            fgm=_int_ps(row.get("fieldGoalsMade")),
                            fga=_int_ps(row.get("fieldGoalsAttempted")),
                            fg3m=_int_ps(row.get("threePointersMade")),
                            fg3a=_int_ps(row.get("threePointersAttempted")),
                            ftm=_int_ps(row.get("freeThrowsMade")),
                            fta=_int_ps(row.get("freeThrowsAttempted")),
                            plus_minus=_int_ps(row.get("plusMinusPoints")),
                        ).on_conflict_do_update(
                            index_elements=["game_id", "player_id"],
                            set_={
                                "pts": _int_ps(row.get("points")),
                                "reb": _int_ps(row.get("reboundsTotal")),
                                "ast": _int_ps(row.get("assists")),
                            },
                        )
                        await session.execute(stmt)
                        count_player += 1

            await session.commit()

        logger.info(
            "Box score sync: %d team stats, %d player stats",
            count_team,
            count_player,
        )
        return count_team + count_player

    # ── Player Sync ─────────────────────────────────────────────

    async def sync_players(self) -> int:
        """Update player roster from Kaggle Players.csv.

        The Players.csv is small (~512 KB, ~4,500 players) and updated
        nightly. We import all players with upsert (ON CONFLICT DO UPDATE)
        so new players are added and existing ones get updated names.
        """
        data_dir = await self._ensure_dataset()
        players_path = os.path.join(data_dir, "Players.csv")

        if not os.path.exists(players_path):
            logger.warning("Players.csv not found")
            return 0

        count = 0
        async with self.db_session_factory() as session:
            with open(players_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("personId", "").strip()
                    if not pid:
                        continue

                    try:
                        nba_player_id = int(pid)
                    except ValueError:
                        continue

                    first_name = row.get("firstName", "").strip() or None
                    last_name = row.get("lastName", "").strip() or None
                    full_name = (
                        f"{first_name} {last_name}".strip()
                        if (first_name or last_name)
                        else None
                    )

                    # Derive position from boolean flags
                    position = None
                    is_guard = row.get("guard", "").strip() == "1"
                    is_forward = row.get("forward", "").strip() == "1"
                    is_center = row.get("center", "").strip() == "1"
                    if is_guard and is_forward:
                        position = "G-F"
                    elif is_forward and is_center:
                        position = "F-C"
                    elif is_guard:
                        position = "G"
                    elif is_forward:
                        position = "F"
                    elif is_center:
                        position = "C"

                    jersey = row.get("jersey", "").strip() or None
                    # Truncate to fit VARCHAR(5) — some CSV rows have malformed values
                    if jersey and len(jersey) > 5:
                        jersey = jersey[:5]

                    stmt = pg_insert(Player).values(
                        nba_player_id=nba_player_id,
                        first_name=first_name,
                        last_name=last_name,
                        full_name=full_name,
                        position=position,
                        jersey_number=jersey,
                    ).on_conflict_do_update(
                        index_elements=["nba_player_id"],
                        set_={
                            "first_name": first_name,
                            "last_name": last_name,
                            "full_name": full_name,
                            "position": position,
                            "jersey_number": jersey,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                    await session.execute(stmt)
                    count += 1

                    if count % 5000 == 0:
                        await session.commit()

            await session.commit()

        logger.info("Player sync: %d players upserted", count)
        return count

    # ── Team Sync (no-op) ───────────────────────────────────────

    async def sync_teams(self) -> int:
        """Teams are seeded once and rarely change — no-op."""
        return 0

    # ── Injury Sync (no-op) ─────────────────────────────────────

    async def sync_injuries(self) -> int:
        """Kaggle dataset doesn't include injury data.

        Injuries can be added manually via the admin endpoint.
        """
        return 0

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _determine_season() -> str:
        """Determine the current NBA season string.

        NBA season starts in October. Examples:
          - 2025-10-15 → "2025-26"
          - 2026-03-01 → "2025-26"
          - 2026-08-01 → "2026-27" (off-season → upcoming season)
        """
        today = datetime.now()
        year = today.year
        if today.month >= 7:
            # Jul-Sep: off-season → upcoming season
            # Oct-Dec: new season just started
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            # Jan-Jun: current season, started last year
            return f"{year - 1}-{str(year)[-2:]}"

    async def _trigger_model_retraining(self) -> None:
        """Trigger feedback loop when enough new games are detected."""
        logger.info("Model retraining trigger: >= 3 new Final games")
        try:
            import sys

            ml_path = str(Path(__file__).parent.parent.parent.parent / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)

            from ml.src.feedback import FeedbackLoop

            loop = FeedbackLoop(self.db_session_factory)
            result = await loop.run_cycle(force_retrain=False)
            logger.info(
                "Feedback cycle: %d evaluated, retrained=%s, v=%s",
                result.predictions_evaluated,
                result.retrained,
                result.retrain_version or "N/A",
            )
        except ImportError as e:
            logger.warning("ML module not available: %s", e)
        except Exception as e:
            logger.error("Model retraining failed: %s", e, exc_info=True)


# Singleton
_collector: Optional[KaggleCollector] = None


def get_kaggle_collector() -> KaggleCollector:
    """Get or create the singleton KaggleCollector instance."""
    global _collector
    if _collector is None:
        _collector = KaggleCollector(async_session_factory)
    return _collector
