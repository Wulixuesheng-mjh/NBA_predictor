"""NBA data collector using the nba_api library.

Syncs teams, players, schedule, game results, player stats, and injury reports.
All nba_api calls are wrapped in asyncio.to_thread since nba_api is synchronous.

Handles off-season / empty data gracefully — returns 0 records instead of erroring.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.collector.base import BaseCollector, CollectionResult
from app.database import async_session_factory
from app.models.team import Team
from app.models.player import Player
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats
from app.models.player_injuries import PlayerInjury
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── NBA API headers patch ───────────────────────────────────
# The nba_api library does not set proper request headers by default,
# causing stats.nba.com to return HTML error pages instead of JSON.
# We monkey-patch the HTTP layer to include required browser-mimic headers.

_NBA_API_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _patch_nba_api_headers() -> None:
    """Set proper HTTP headers on all nba_api endpoint classes.

    Called at module import time so every nba_api call includes the
    required headers regardless of which endpoint is used.
    """
    try:
        from nba_api.library.http import NBAHTTP

        # Always patch the base class — NBA API rejects requests without
        # proper browser-mimic headers. Unconditional assignment handles
        # cases where headers were set to empty dict during __init__.
        NBAHTTP.headers = _NBA_API_HEADERS
        logger.debug("NBA API headers patched on NBAHTTP base class")
    except ImportError:
        pass  # nba_api not installed — no-op


_patch_nba_api_headers()


class NBACollector(BaseCollector):
    """Collects data from NBA official API via nba_api library."""

    source = "nba_api"

    async def collect(self) -> CollectionResult:
        """Run all collection sub-tasks."""
        result = CollectionResult(source=self.source, task_name="full_sync")
        total_records = 0

        try:
            total_records += await self.sync_teams()
            total_records += await self.sync_players()
            total_records += await self.sync_schedule()
            total_records += await self.sync_game_results()
            # Injuries are collected manually (see frontend InjuryEntry).

            result.status = "success"
            result.records_collected = total_records
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error("NBA collection failed: %s", e, exc_info=True)
        finally:
            result.completed_at = datetime.now(timezone.utc)

        await self._log_collection(result)
        return result

    # ── Team Sync ─────────────────────────────────────────────

    async def sync_teams(self) -> int:
        """Sync all NBA teams from nba_api static data.

        Returns count of teams synced.
        """
        try:
            teams_data = await asyncio.to_thread(self._fetch_teams)
        except Exception as e:
            logger.warning("Failed to fetch teams: %s", e)
            return 0

        if not teams_data:
            logger.info("No teams returned from nba_api (off-season?)")
            return 0

        count = 0
        async with self.db_session_factory() as session:
            for t in teams_data:
                stmt = pg_insert(Team).values(
                    nba_team_id=t["id"],
                    abbreviation=t.get("abbreviation", ""),
                    city=t.get("city", ""),
                    name=t.get("nickname", ""),
                    full_name=t.get("full_name", ""),
                    conference=t.get("conference"),
                    division=t.get("division"),
                ).on_conflict_do_update(
                    index_elements=["nba_team_id"],
                    set_={
                        "abbreviation": t.get("abbreviation", ""),
                        "city": t.get("city", ""),
                        "name": t.get("nickname", ""),
                        "full_name": t.get("full_name", ""),
                        "conference": t.get("conference"),
                        "division": t.get("division"),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                await session.execute(stmt)
                count += 1
            await session.commit()

        logger.info("Synced %d teams", count)
        return count

    @staticmethod
    def _fetch_teams() -> list[dict]:
        """Synchronous: fetch all NBA teams."""
        from nba_api.stats.static import teams

        return teams.get_teams()

    # ── Player Sync ───────────────────────────────────────────

    async def sync_players(self) -> int:
        """Sync players for all active teams.

        For each team, fetches the current roster via CommonAllPlayers.
        Sets team_id on player records so each player is linked to their
        current NBA team.
        """
        try:
            from nba_api.stats.endpoints import CommonAllPlayers
        except ImportError:
            logger.warning("nba_api not installed")
            return 0

        async with self.db_session_factory() as session:
            # Build nba_team_id → internal team id mapping
            team_query = select(Team.nba_team_id, Team.id)
            result = await session.execute(team_query)
            nba_to_db_team: dict[int, int] = {row[0]: row[1] for row in result.all()}

        count = 0
        for nba_team_id, db_team_id in nba_to_db_team.items():
            try:
                players_data = await asyncio.to_thread(
                    lambda tid=nba_team_id: CommonAllPlayers(
                        team_id=tid, is_only_current_season=1
                    ).get_normalized_dict()
                )
            except Exception as e:
                logger.debug("Failed to fetch players for team %d: %s", nba_team_id, e)
                continue

            player_list = (
                players_data.get("CommonAllPlayers", []) if players_data else []
            )
            if not player_list:
                continue

            async with self.db_session_factory() as session:
                for p in player_list:
                    stmt = pg_insert(Player).values(
                        nba_player_id=p.get("PERSON_ID", 0),
                        first_name=p.get("FIRST_NAME"),
                        last_name=p.get("LAST_NAME"),
                        full_name=p.get("DISPLAY_FIRST_LAST"),
                        team_id=db_team_id,
                        jersey_number=p.get("JERSEY_NUMBER"),
                        position=p.get("POSITION"),
                    ).on_conflict_do_update(
                        index_elements=["nba_player_id"],
                        set_={
                            "first_name": p.get("FIRST_NAME"),
                            "last_name": p.get("LAST_NAME"),
                            "full_name": p.get("DISPLAY_FIRST_LAST"),
                            "team_id": db_team_id,
                            "jersey_number": p.get("JERSEY_NUMBER"),
                            "position": p.get("POSITION"),
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                    await session.execute(stmt)
                    count += 1
                await session.commit()

        logger.info("Synced %d player records", count)
        return count

    # ── Schedule Sync ─────────────────────────────────────────

    async def sync_schedule(self, season: str | None = None) -> int:
        """Sync game schedule for a season.

        If season not specified, determines the current season based on date.
        Handles off-season by checking if any games are returned.
        """
        if season is None:
            season = self._determine_season()

        logger.info("Syncing schedule for season %s", season)

        try:
            games_data = await asyncio.to_thread(self._fetch_schedule, season)
        except Exception as e:
            logger.warning("Failed to fetch schedule: %s", e)
            return 0

        if not games_data:
            logger.info("No games returned for season %s (off-season?)", season)
            return 0

        count = 0
        async with self.db_session_factory() as session:
            # Pre-load team id mappings
            team_map = await self._get_team_nba_id_map(session)

            for g in games_data:
                home_id = team_map.get(g.get("home_team_id"))
                away_id = team_map.get(g.get("away_team_id"))
                if not home_id or not away_id:
                    continue

                game_date = g.get("game_date")
                if isinstance(game_date, str):
                    game_date = datetime.fromisoformat(game_date.replace("Z", "+00:00"))

                stmt = pg_insert(Game).values(
                    nba_game_id=g.get("game_id", ""),
                    season=season,
                    game_date=game_date,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    status=g.get("status", "Scheduled"),
                    home_score=g.get("home_score"),
                    away_score=g.get("away_score"),
                ).on_conflict_do_update(
                    index_elements=["nba_game_id"],
                    set_={
                        "status": g.get("status", "Scheduled"),
                        "home_score": g.get("home_score"),
                        "away_score": g.get("away_score"),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                await session.execute(stmt)
                count += 1
            await session.commit()

        logger.info("Synced %d games for season %s", count, season)
        return count

    @staticmethod
    def _fetch_schedule(season: str) -> list[dict]:
        """Synchronous: fetch all games for a season via LeagueGameFinder.

        Each game appears twice in the API response (home + away row).
        We merge both rows to capture both team IDs and both scores.
        """
        from nba_api.stats.endpoints import LeagueGameFinder

        # LeagueGameFinder returns games for a season
        finder = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",  # NBA
        )
        data = finder.get_normalized_dict()
        games = data.get("LeagueGameFinderResults", []) if data else []

        # Merge home + away rows by game_id
        game_map: dict[str, dict] = {}
        for g in games:
            gid = str(g.get("GAME_ID"))
            if gid not in game_map:
                game_map[gid] = {
                    "game_id": gid,
                    "game_date": g.get("GAME_DATE"),
                    "home_team_id": None,
                    "away_team_id": None,
                    "status": "Scheduled",
                    "home_score": None,
                    "away_score": None,
                }

            matchup = g.get("MATCHUP", "")
            pts = g.get("PTS")
            if "vs." in matchup:
                game_map[gid]["home_team_id"] = g.get("TEAM_ID")
                game_map[gid]["home_score"] = pts
            elif "@" in matchup:
                game_map[gid]["away_team_id"] = g.get("TEAM_ID")
                game_map[gid]["away_score"] = pts

            # Mark as Final if any row has a result (WL is not None)
            if g.get("WL"):
                game_map[gid]["status"] = "Final"

        return list(game_map.values())

    # ── Game Results Sync ─────────────────────────────────────

    async def sync_game_results(self) -> int:
        """Sync box scores and stats for recently completed games.

        Finds games with status='Final' that don't yet have team/player stats,
        fetches box scores for them. After syncing, counts newly completed
        games for model retraining trigger.
        """
        try:
            from nba_api.stats.endpoints import BoxScoreTraditionalV2
        except ImportError:
            return 0

        async with self.db_session_factory() as session:
            # Find Final games without team stats
            query = (
                select(Game)
                .where(Game.status == "Final")
                .order_by(Game.game_date.desc())
            )
            result = await session.execute(query)
            games = result.scalars().all()

        count = 0
        for game in games:
            # Check if stats already exist
            async with self.db_session_factory() as session:
                stats_exists = await session.execute(
                    select(TeamGameStats).where(TeamGameStats.game_id == game.id)
                )
                if stats_exists.first():
                    continue

            try:
                stats_data = await asyncio.to_thread(
                    lambda gid=game.nba_game_id: BoxScoreTraditionalV2(
                        game_id=gid
                    ).get_normalized_dict()
                )
            except Exception as e:
                logger.debug("Failed to fetch box score for %s: %s", game.nba_game_id, e)
                continue

            await self._save_box_score(game, stats_data)
            count += 1

        # Check if >= 3 new Final games → trigger model retraining
        if count >= 3:
            logger.info("Detected %d new completed games. Triggering model retraining...", count)
            await self._trigger_model_retraining()

        logger.info("Synced box scores for %d games", count)
        return count

    async def _save_box_score(self, game: Game, data: dict) -> None:
        """Persist team and player box score data for a game."""
        if not data:
            return

        team_stats_list = data.get("TeamStats", [])
        player_stats_list = data.get("PlayerStats", [])

        async with self.db_session_factory() as session:
            team_map = await self._get_team_nba_id_map(session)

            # Save team stats
            for ts in team_stats_list:
                team_db_id = team_map.get(ts.get("TEAM_ID"))
                if not team_db_id:
                    continue
                is_home = team_db_id == game.home_team_id

                stmt = pg_insert(TeamGameStats).values(
                    game_id=game.id,
                    team_id=team_db_id,
                    is_home=is_home,
                    pts=ts.get("PTS", 0),
                    reb=ts.get("REB", 0),
                    ast=ts.get("AST", 0),
                    stl=ts.get("STL", 0),
                    blk=ts.get("BLK", 0),
                    tov=ts.get("TO", 0),
                    pf=ts.get("PF", 0),
                    fgm=ts.get("FGM", 0),
                    fga=ts.get("FGA", 0),
                    fg3m=ts.get("FG3M", 0),
                    fg3a=ts.get("FG3A", 0),
                    ftm=ts.get("FTM", 0),
                    fta=ts.get("FTA", 0),
                ).on_conflict_do_update(
                    index_elements=["game_id", "team_id"],
                    set_={"pts": ts.get("PTS", 0)},
                )
                await session.execute(stmt)

                # Update game scores from team stats
                if is_home:
                    game.home_score = ts.get("PTS")
                else:
                    game.away_score = ts.get("PTS")

            # Save player stats
            for ps in player_stats_list:
                team_db_id = team_map.get(ps.get("TEAM_ID"))
                if not team_db_id:
                    continue

                # Find player by nba_player_id
                player_query = select(Player).where(
                    Player.nba_player_id == ps.get("PLAYER_ID")
                )
                player_result = await session.execute(player_query)
                player = player_result.scalar_one_or_none()
                if not player:
                    continue

                minutes_raw = ps.get("MIN")
                minutes_decimal = self._parse_minutes(minutes_raw)

                stmt = pg_insert(GamePlayerStats).values(
                    game_id=game.id,
                    player_id=player.id,
                    team_id=team_db_id,
                    minutes=minutes_decimal,
                    pts=ps.get("PTS", 0),
                    reb=ps.get("REB", 0),
                    ast=ps.get("AST", 0),
                    stl=ps.get("STL", 0),
                    blk=ps.get("BLK", 0),
                    tov=ps.get("TO", 0),
                    pf=ps.get("PF", 0),
                    fgm=ps.get("FGM", 0),
                    fga=ps.get("FGA", 0),
                    fg3m=ps.get("FG3M", 0),
                    fg3a=ps.get("FG3A", 0),
                    ftm=ps.get("FTM", 0),
                    fta=ps.get("FTA", 0),
                    plus_minus=ps.get("PLUS_MINUS"),
                ).on_conflict_do_update(
                    index_elements=["game_id", "player_id"],
                    set_={
                        "pts": ps.get("PTS", 0),
                        "minutes": minutes_decimal,
                    },
                )
                await session.execute(stmt)

            await session.commit()

    # ── Injury Sync ───────────────────────────────────────────

    async def sync_injuries(self) -> int:
        """Sync injury reports from nba_api.

        All auto-fetched injuries are created with confirmation='pending'.
        If a pending record already exists for the same player + injury,
        skip to avoid duplicates.
        """
        try:
            from nba_api.stats.endpoints import PlayerInjuryReport
        except ImportError:
            return 0

        try:
            injury_data = await asyncio.to_thread(
                lambda: PlayerInjuryReport().get_normalized_dict()
            )
        except Exception as e:
            logger.debug("Failed to fetch injury report: %s", e)
            return 0

        if not injury_data:
            return 0

        injury_list = injury_data.get("PlayerInjuryReportResults", [])
        if not injury_list:
            return 0

        count = 0
        async with self.db_session_factory() as session:
            for inj in injury_list:
                # Find player by nba_player_id
                nba_pid = inj.get("PLAYER_ID")
                if not nba_pid:
                    continue

                player_query = select(Player).where(Player.nba_player_id == nba_pid)
                player_result = await session.execute(player_query)
                player = player_result.scalar_one_or_none()
                if not player:
                    continue

                # Check for existing active injury for same player + type
                existing = await session.execute(
                    select(PlayerInjury).where(
                        PlayerInjury.player_id == player.id,
                        PlayerInjury.injury_type == inj.get("INJURY_TYPE"),
                        PlayerInjury.is_active == True,
                    )
                )
                if existing.first():
                    continue

                injury = PlayerInjury(
                    player_id=player.id,
                    source="nba_api",
                    confirmation="pending",
                    injury_status=inj.get("INJURY_STATUS", "Out"),
                    injury_type=inj.get("INJURY_TYPE"),
                    description=inj.get("INJURY_DESCRIPTION"),
                    reported_date=datetime.now(timezone.utc),
                    is_active=True,
                )
                session.add(injury)
                count += 1
            await session.commit()

        logger.info("Synced %d pending injuries", count)
        return count

    # ── Model Retraining Trigger ──────────────────────────────

    async def _trigger_model_retraining(self) -> None:
        """Trigger the daily feedback loop when enough new games are detected.

        Runs in the background: evaluates pending predictions, retrains XGBoost,
        and updates calibration coefficients. Failures are logged but do not
        affect the collection pipeline.
        """
        logger.info("Model retraining trigger: >= 3 new games detected")
        try:
            # Defer import to avoid circular dependencies at module load
            import sys
            from pathlib import Path

            ml_path = str(Path(__file__).parent.parent.parent / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)

            from ml.src.feedback import FeedbackLoop

            loop = FeedbackLoop(self.db_session_factory)
            result = await loop.run_cycle(force_retrain=False)
            logger.info(
                "Feedback cycle: %d evaluated, retrained=%s, calib_updated=%s, v=%s",
                result.predictions_evaluated,
                result.retrained,
                result.calibration_updated,
                result.retrain_version or "N/A",
            )
            if result.errors:
                logger.warning("Feedback cycle errors: %s", result.errors)
        except ImportError as e:
            logger.warning("ML module not available for retraining: %s", e)
        except Exception as e:
            logger.error("Model retraining failed: %s", e, exc_info=True)

    # ── Helpers ───────────────────────────────────────────────

    async def _get_team_nba_id_map(self, session) -> dict[int, int]:
        """Build a mapping from nba_team_id to our internal team id."""
        query = select(Team.nba_team_id, Team.id)
        result = await session.execute(query)
        return {row[0]: row[1] for row in result.all()}

    @staticmethod
    def _determine_season() -> str:
        """Determine the current NBA season string based on today's date.

        The NBA season typically starts in October and ends in June.
        If today is July-September, return the upcoming season.
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

    @staticmethod
    def _parse_minutes(value: str | float | int | None) -> str | None:
        """Convert minutes from nba_api (MM:SS or decimal) to decimal string.

        BoxScoreTraditionalV2 returns MIN as "MM:SS" (e.g. "29:03").
        Other endpoints may return decimal floats.
        We store everything as decimal strings for consistent querying.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(round(float(value), 1))
        value = str(value).strip()
        if not value:
            return None
        if ":" in value:
            parts = value.split(":")
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return str(round(minutes + seconds / 60.0, 1))
            except (ValueError, IndexError):
                return None
        try:
            return str(round(float(value), 1))
        except ValueError:
            return None


# Singleton-style accessor
_collector: NBACollector | None = None


def get_collector() -> NBACollector:
    """Get or create the singleton NBACollector instance."""
    global _collector
    if _collector is None:
        _collector = NBACollector(async_session_factory)
    return _collector
