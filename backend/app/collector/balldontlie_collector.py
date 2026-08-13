"""BallDontLie.io real-time NBA data collector.

Provides SAME-DAY game results to complement the Kaggle collector
(which has a 24-hour delay due to nightly update cadence).

Key differences from KaggleCollector:
  - Real-time: game results available minutes after games end
  - Lightweight: only fetches today's games + scores (no big CSV downloads)
  - Free tier: 5 requests/minute — ample for daily score sync

Architecture:
  KaggleCollector  →  historical data + full box scores (nightly)
  BallDontLieCollector →  same-day game scores (every 30 min during season)
  Manual API       →  fallback for any missing data

API docs: https://api.balldontlie.io/v1
Free API key: https://app.balldontlie.io (no credit card required)
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.collector.base import BaseCollector, CollectionResult
from app.database import async_session_factory
from app.models.team import Team
from app.models.game import Game
from app.models.player import Player
from app.models.game_player_stats import GamePlayerStats
from app.models.player_injuries import PlayerInjury
from app.utils.logging import get_logger

logger = get_logger(__name__)

BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"


def _get_api_key() -> str:
    """Read the BallDontLie API key lazily from settings.

    NOT read at module import time — os.getenv would be frozen
    before pydantic-settings loads the .env file.
    """
    try:
        from app.config import settings
        return settings.balldontlie_api_key
    except Exception:
        return os.getenv("BALLDONTLIE_API_KEY", "")


class BallDontLieCollector(BaseCollector):
    """Collects same-day NBA game results from balldontlie.io.

    Designed for the "morning game → evening prediction" workflow.
    Fetches today's scoreboard and imports newly Final games.

    Usage:
        collector = BallDontLieCollector()
        result = await collector.collect()
    """

    source = "balldontlie"

    def __init__(self, session_factory=None, api_key: str | None = None):
        super().__init__(session_factory or async_session_factory)
        self._api_key = api_key or _get_api_key()
        self._team_map: Optional[dict[str, int]] = None

    # ── Public API ──────────────────────────────────────────────

    async def collect(self) -> CollectionResult:
        """Fetch today's games and import newly completed results.

        Steps:
          1. Fetch today's NBA games from balldontlie
          2. Find games that just finished (status=Final, score in API but not in DB)
          3. Update game scores + status in DB
          4. Optionally fetch player stats for newly Final games
        """
        if not self._api_key:
            return CollectionResult(
                source=self.source,
                task_name="full_sync",
                status="failed",
                error_message="BALLDONTLIE_API_KEY not set. Get a free key at https://app.balldontlie.io",
                completed_at=datetime.now(timezone.utc),
            )

        result = CollectionResult(source=self.source, task_name="full_sync")

        try:
            count = await self.sync_today_scores()
            injury_count = await self.sync_injuries()
            result.status = "success"
            result.records_collected = count + injury_count

            # Trigger retraining if enough new games
            if count >= 3:
                logger.info(
                    "Detected %d newly Final games via balldontlie — triggering retraining",
                    count,
                )
                await self._trigger_model_retraining()

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error("BallDontLie collection failed: %s", e, exc_info=True)
        finally:
            result.completed_at = datetime.now(timezone.utc)

        await self._log_collection(result)
        return result

    # ── Today's Score Sync ──────────────────────────────────────

    async def sync_today_scores(self, date_str: str | None = None) -> int:
        """Fetch today's games from balldontlie and update Final games.

        Args:
            date_str: ISO date string (e.g. "2026-08-06"). Defaults to today UTC.

        Returns:
            Number of games updated to Final.
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info("Fetching balldontlie games for %s", date_str)

        games = await self._fetch_games(date_str)
        if not games:
            logger.info("No games found on balldontlie for %s", date_str)
            return 0

        # Separate Final vs non-Final games
        final_games = [g for g in games if g.get("status") == "Final"]
        upcoming_games = [g for g in games if g.get("status") != "Final"]

        logger.info(
            "Balldontlie: %d total, %d Final, %d upcoming",
            len(games),
            len(final_games),
            len(upcoming_games),
        )

        count = 0

        async with self.db_session_factory() as session:
            team_map = await self._get_team_map(session)

            # 1. Create/update upcoming games as Scheduled
            for g in upcoming_games:
                await self._upsert_game(session, g, team_map, status="Scheduled")

            # 2. Update Final games with scores
            for g in final_games:
                updated = await self._upsert_game(
                    session, g, team_map, status="Final"
                )
                if updated:
                    count += 1

            await session.commit()

        logger.info("Updated %d games to Final via balldontlie", count)
        return count

    async def _upsert_game(
        self, session, api_game: dict, team_map: dict, status: str
    ) -> bool:
        """Insert or update a game from balldontlie data.

        Returns True if the game was newly set to Final (for retraining count).
        """
        home_abbr = (api_game.get("home_team") or {}).get("abbreviation", "")
        away_abbr = (api_game.get("visitor_team") or {}).get("abbreviation", "")
        balldontlie_game_id = str(api_game["id"])

        home_db_id = team_map.get(home_abbr.upper())
        away_db_id = team_map.get(away_abbr.upper())

        if not home_db_id or not away_db_id:
            logger.debug(
                "Skipping game %s — team not found: %s vs %s",
                balldontlie_game_id,
                home_abbr,
                away_abbr,
            )
            return False

        # Parse date
        date_str = api_game.get("date", "")
        try:
            game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            game_date = datetime.now(timezone.utc)

        # Scores
        home_score = api_game.get("home_team_score")
        away_score = api_game.get("visitor_team_score")
        season_year = api_game.get("season", game_date.year)
        season = f"{season_year}-{str(season_year + 1)[-2:]}"

        # Use balldontlie game ID as nba_game_id (prefixed to avoid collision)
        nba_game_id = f"bdl-{balldontlie_game_id}"

        # Check if game already exists by looking up scheduled games with matching teams + date
        existing = await session.execute(
            select(Game).where(
                Game.home_team_id == home_db_id,
                Game.away_team_id == away_db_id,
                Game.game_date >= game_date.replace(hour=0, minute=0, second=0),
                Game.game_date < game_date.replace(hour=23, minute=59, second=59),
            )
        )
        existing_game = existing.scalar_one_or_none()

        is_new_final = False

        if existing_game:
            # Update if now Final and we didn't have scores before
            was_scheduled = existing_game.status == "Scheduled"
            if was_scheduled and status == "Final" and home_score and away_score:
                existing_game.home_score = home_score
                existing_game.away_score = away_score
                existing_game.status = "Final"
                existing_game.updated_at = datetime.now(timezone.utc)
                is_new_final = True
                logger.info(
                    "Updated %s → Final: %s %d-%d %s",
                    existing_game.nba_game_id,
                    home_abbr,
                    home_score,
                    away_score,
                    away_abbr,
                )
        else:
            # Create new game
            stmt = pg_insert(Game).values(
                nba_game_id=nba_game_id,
                season=season,
                game_date=game_date,
                home_team_id=home_db_id,
                away_team_id=away_db_id,
                status=status,
                home_score=home_score,
                away_score=away_score,
            ).on_conflict_do_update(
                index_elements=["nba_game_id"],
                set_={
                    "status": status,
                    "home_score": home_score,
                    "away_score": away_score,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            await session.execute(stmt)

            if status == "Final" and home_score and away_score:
                is_new_final = True
                logger.info(
                    "Created Final game %s: %s %d-%d %s",
                    nba_game_id,
                    home_abbr,
                    home_score,
                    away_score,
                    away_abbr,
                )

        return is_new_final

    # ── Player Stats Sync (optional, uses extra API calls) ─────

    async def sync_game_stats(self, balldontlie_game_id: int) -> int:
        """Fetch player stats for a specific game.

        Note: balldontlie player IDs differ from NBA player IDs.
        We match by first_name + last_name, which is not 100% reliable.
        For complete box scores, use the Kaggle collector (nightly).
        """
        stats = await self._fetch_stats(balldontlie_game_id)
        if not stats:
            return 0

        # Find the game in our DB
        nba_game_id = f"bdl-{balldontlie_game_id}"

        count = 0
        async with self.db_session_factory() as session:
            game_result = await session.execute(
                select(Game).where(Game.nba_game_id == nba_game_id)
            )
            game = game_result.scalar_one_or_none()
            if not game:
                return 0

            team_map = await self._get_team_map(session)

            for s in stats:
                # Match team by abbreviation
                team_abbr = (s.get("team") or {}).get("abbreviation", "").upper()
                team_db_id = team_map.get(team_abbr)
                if not team_db_id:
                    continue

                # Match player by name (fragile but best we have with free tier)
                player_info = s.get("player") or {}
                first_name = player_info.get("first_name", "")
                last_name = player_info.get("last_name", "")

                from app.models.player import Player

                player_result = await session.execute(
                    select(Player).where(
                        Player.first_name == first_name,
                        Player.last_name == last_name,
                    )
                )
                player = player_result.scalar_one_or_none()
                if not player:
                    continue

                def _int(v, default=0):
                    try:
                        return int(v) if v else default
                    except (ValueError, TypeError):
                        return default

                stmt = pg_insert(GamePlayerStats).values(
                    game_id=game.id,
                    player_id=player.id,
                    team_id=team_db_id,
                    minutes=s.get("min"),
                    pts=_int(s.get("pts")),
                    reb=_int(s.get("reb")),
                    ast=_int(s.get("ast")),
                    stl=_int(s.get("stl")),
                    blk=_int(s.get("blk")),
                    tov=_int(s.get("turnover")),
                    pf=_int(s.get("pf")),
                    fgm=_int(s.get("fgm")),
                    fga=_int(s.get("fga")),
                    fg3m=_int(s.get("fg3m")),
                    fg3a=_int(s.get("fg3a")),
                    ftm=_int(s.get("ftm")),
                    fta=_int(s.get("fta")),
                ).on_conflict_do_update(
                    index_elements=["game_id", "player_id"],
                    set_={"pts": _int(s.get("pts"))},
                )
                await session.execute(stmt)
                count += 1

            await session.commit()

        return count

    # ── HTTP Methods ────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": self._api_key,
            "Accept": "application/json",
        }

    async def _fetch_games(self, date_str: str) -> list[dict]:
        """Fetch games for a specific date from balldontlie.

        Handles pagination — collects all pages of results.
        """
        all_games = []
        page = 1

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{BALLDONTLIE_BASE_URL}/games",
                        headers=self._headers(),
                        params={
                            "dates[]": date_str,
                            "per_page": 100,
                            "page": page,
                        },
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    logger.error("Balldontlie API error: %s", e)
                    break

                body = resp.json()
                data = body.get("data", [])
                all_games.extend(data)

                meta = body.get("meta", {})
                next_page = meta.get("next_page")
                if not next_page:
                    break
                page = next_page

        return all_games

    async def _fetch_stats(self, game_id: int) -> list[dict]:
        """Fetch player stats for a specific game."""
        all_stats = []
        page = 1

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{BALLDONTLIE_BASE_URL}/stats",
                        headers=self._headers(),
                        params={
                            "game_ids[]": game_id,
                            "per_page": 100,
                            "page": page,
                        },
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    logger.error("Balldontlie stats API error: %s", e)
                    break

                body = resp.json()
                data = body.get("data", [])
                all_stats.extend(data)

                meta = body.get("meta", {})
                next_page = meta.get("next_page")
                if not next_page:
                    break
                page = next_page

        return all_stats

    # ── Team Mapping ────────────────────────────────────────────

    async def _get_team_map(self, session) -> dict[str, int]:
        """Build abbreviation → internal team.id mapping."""
        if self._team_map is None:
            result = await session.execute(
                select(Team.abbreviation, Team.id)
            )
            self._team_map = {
                row[0].upper(): row[1] for row in result.all() if row[0]
            }
        return self._team_map

    # ── Injury Sync (BallDontLie All-Star tier only) ───────────

    async def sync_injuries(self) -> int:
        """Sync active NBA injuries from BallDontLie.

        Requires ALL-STAR tier or higher ($9.99/mo).
        Free tier will receive HTTP 402 Payment Required.

        Returns:
            Number of new injury records created.
        """
        url = f"{BALLDONTLIE_BASE_URL}/nba/player_injuries"
        headers = {
            "Authorization": self._api_key,
            "Accept": "application/json",
        }
        count = 0

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Fetch first page of active injuries
                resp = await client.get(url, headers=headers, params={"per_page": 100})
                resp.raise_for_status()
                data = resp.json()

                injuries_list = data.get("data", [])
                if not injuries_list:
                    logger.debug("BallDontLie: no active injuries found")
                    return 0

                async with self.db_session_factory() as session:
                    # Build player ID mapping: balldontlie player ID → internal player.id
                    player_result = await session.execute(
                        select(Player.nba_player_id, Player.id)
                    )
                    player_map = {
                        str(row[0]): row[1] for row in player_result.all()
                    }

                    seen = set()
                    now = datetime.now(timezone.utc)

                    for injury in injuries_list:
                        player_bdl = injury.get("player", {})
                        player_id = player_bdl.get("id")
                        if player_id is None:
                            continue

                        db_player_id = player_map.get(str(player_id))
                        if not db_player_id:
                            continue

                        status = injury.get("status", "Out")
                        injury_type = injury.get("comment") or injury.get("type") or "Unknown"

                        # Dedup: skip if same player + same status already active
                        dedup_key = (db_player_id, status)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        # Check if an active injury already exists for this player
                        existing = await session.execute(
                            select(PlayerInjury).where(
                                PlayerInjury.player_id == db_player_id,
                                PlayerInjury.is_active == True,
                                PlayerInjury.injury_status == status,
                            )
                        )
                        if existing.scalar_one_or_none():
                            continue

                        # Create new injury record
                        new_injury = PlayerInjury(
                            player_id=db_player_id,
                            source="balldontlie",
                            confirmation="pending",
                            injury_status=status,
                            injury_type=injury_type,
                            description=injury.get("comment") or "",
                            reported_date=now,
                            is_active=True,
                        )
                        session.add(new_injury)
                        count += 1

                    if count > 0:
                        await session.commit()
                        logger.info("BallDontLie: imported %d new injury records", count)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.debug(
                    "BallDontLie injuries skipped: requires All-Star tier ($9.99/mo). "
                    "Upgrade at https://app.balldontlie.io"
                )
            else:
                logger.warning("BallDontLie injury API error: HTTP %s", e.response.status_code)
        except httpx.RequestError as e:
            logger.debug("BallDontLie injury fetch failed (network): %s", e)
        except Exception as e:
            logger.warning("BallDontLie injury sync failed: %s", e)

        return count

    # ── Helpers ─────────────────────────────────────────────────

    async def _trigger_model_retraining(self) -> None:
        """Trigger feedback loop when enough new games are detected."""
        logger.info("Model retraining trigger: >= 3 new Final games from balldontlie")
        try:
            import sys
            from pathlib import Path

            ml_path = str(Path(__file__).parent.parent.parent.parent / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)

            from ml.src.feedback import FeedbackLoop

            loop = FeedbackLoop(self.db_session_factory)
            result = await loop.run_cycle(force_retrain=False)
            logger.info(
                "Feedback cycle: %d evaluated, retrained=%s",
                result.predictions_evaluated,
                result.retrained,
            )
        except ImportError:
            logger.debug("ML module not available for retraining")
        except Exception as e:
            logger.error("Model retraining failed: %s", e, exc_info=True)


# Singleton
_collector: Optional[BallDontLieCollector] = None


def get_balldontlie_collector() -> BallDontLieCollector:
    """Get or create the singleton BallDontLieCollector instance."""
    global _collector
    if _collector is None:
        _collector = BallDontLieCollector(async_session_factory)
    return _collector
