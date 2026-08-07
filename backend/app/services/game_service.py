"""Game service — business logic for game operations."""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game import Game
from app.models.team import Team
from app.models.team_game_stats import TeamGameStats
from app.models.game_player_stats import GamePlayerStats
from app.models.player import Player

logger = logging.getLogger(__name__)


class GameService:
    """Service for game-related operations."""

    @staticmethod
    async def list_games(
        session: AsyncSession,
        date: str | None = None,
        status: str | None = None,
        team_id: int | None = None,
        season: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict], int]:
        """List games with optional filters, enriched with team names."""
        query = select(Game)
        count_query = select(func.count(Game.id))

        if date:
            try:
                filter_date = datetime.fromisoformat(date)
                query = query.where(func.date(Game.game_date) == filter_date.date())
                count_query = count_query.where(func.date(Game.game_date) == filter_date.date())
            except ValueError:
                pass

        if status:
            query = query.where(Game.status == status)
            count_query = count_query.where(Game.status == status)

        if season:
            query = query.where(Game.season == season)
            count_query = count_query.where(Game.season == season)

        if team_id is not None:
            team_filter = (Game.home_team_id == team_id) | (Game.away_team_id == team_id)
            query = query.where(team_filter)
            count_query = count_query.where(team_filter)

        query = query.order_by(Game.game_date.desc()).offset(offset).limit(limit)

        result = await session.execute(query)
        games = list(result.scalars().all())

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        # Enrich with team names (batch lookup, same pattern as upcoming)
        team_map: dict[int, tuple[str | None, str | None]] = {}
        if games:
            team_ids: set[int] = set()
            for g in games:
                team_ids.add(g.home_team_id)
                team_ids.add(g.away_team_id)

            if team_ids:
                team_query = select(Team.id, Team.full_name, Team.abbreviation).where(
                    Team.id.in_(team_ids)
                )
                result = await session.execute(team_query)
                for row in result.all():
                    team_map[row.id] = (row.full_name, row.abbreviation)

        # Build enriched dicts
        enriched = []
        for g in games:
            ht = team_map.get(g.home_team_id, (None, None))
            at = team_map.get(g.away_team_id, (None, None))
            enriched.append({
                "id": g.id,
                "nba_game_id": g.nba_game_id,
                "season": g.season,
                "game_date": g.game_date,
                "home_team_id": g.home_team_id,
                "away_team_id": g.away_team_id,
                "home_team_name": ht[0],
                "away_team_name": at[0],
                "home_team_abbr": ht[1],
                "away_team_abbr": at[1],
                "status": g.status,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "quarter": g.quarter,
                "time_remaining": g.time_remaining,
                "created_at": g.created_at,
                "updated_at": g.updated_at,
            })

        return enriched, total

    @staticmethod
    async def get_game(
        session: AsyncSession, game_id: int
    ) -> Game | None:
        """Get a single game by ID."""
        query = select(Game).where(Game.id == game_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_game_detail(
        session: AsyncSession, game_id: int
    ) -> dict | None:
        """Get game with team and player box scores."""
        game = await GameService.get_game(session, game_id)
        if game is None:
            return None

        # Team names
        home_team_query = select(Team).where(Team.id == game.home_team_id)
        away_team_query = select(Team).where(Team.id == game.away_team_id)
        home_result = await session.execute(home_team_query)
        away_result = await session.execute(away_team_query)
        home_team = home_result.scalar_one_or_none()
        away_team = away_result.scalar_one_or_none()

        # Team stats
        team_stats_query = (
            select(TeamGameStats, Team.full_name, Team.abbreviation)
            .join(Team, TeamGameStats.team_id == Team.id)
            .where(TeamGameStats.game_id == game.id)
        )
        result = await session.execute(team_stats_query)
        team_stats_rows = result.all()

        team_stats = []
        for tgs, team_name, team_abbr in team_stats_rows:
            fg_pct = (tgs.fgm / tgs.fga) if tgs.fga and tgs.fga > 0 else None
            fg3_pct = (tgs.fg3m / tgs.fg3a) if tgs.fg3a and tgs.fg3a > 0 else None
            ft_pct = (tgs.ftm / tgs.fta) if tgs.fta and tgs.fta > 0 else None

            team_stats.append({
                "team_id": tgs.team_id,
                "team_name": team_name,
                "is_home": tgs.is_home,
                "pts": tgs.pts,
                "reb": tgs.reb,
                "ast": tgs.ast,
                "stl": tgs.stl,
                "blk": tgs.blk,
                "tov": tgs.tov,
                "fgm": tgs.fgm,
                "fga": tgs.fga,
                "fg3m": tgs.fg3m,
                "fg3a": tgs.fg3a,
                "ftm": tgs.ftm,
                "fta": tgs.fta,
                "fg_pct": round(fg_pct, 3) if fg_pct else None,
                "fg3_pct": round(fg3_pct, 3) if fg3_pct else None,
                "ft_pct": round(ft_pct, 3) if ft_pct else None,
            })

        # Player stats
        player_stats_query = (
            select(GamePlayerStats, Player.full_name)
            .join(Player, GamePlayerStats.player_id == Player.id)
            .where(GamePlayerStats.game_id == game.id)
            .order_by(GamePlayerStats.pts.desc())
        )
        result = await session.execute(player_stats_query)
        player_stats_rows = result.all()

        player_stats = [
            {
                "player_id": gps.player_id,
                "player_name": player_name,
                "team_id": gps.team_id,
                "minutes": gps.minutes,
                "pts": gps.pts,
                "reb": gps.reb,
                "ast": gps.ast,
                "stl": gps.stl,
                "blk": gps.blk,
                "tov": gps.tov,
                "pf": gps.pf,
                "fgm": gps.fgm,
                "fga": gps.fga,
                "fg3m": gps.fg3m,
                "fg3a": gps.fg3a,
                "ftm": gps.ftm,
                "fta": gps.fta,
                "plus_minus": gps.plus_minus,
            }
            for gps, player_name in player_stats_rows
        ]

        return {
            "game": game,
            "home_team_name": home_team.full_name if home_team else None,
            "away_team_name": away_team.full_name if away_team else None,
            "home_team_abbr": home_team.abbreviation if home_team else None,
            "away_team_abbr": away_team.abbreviation if away_team else None,
            "team_stats": team_stats,
            "player_stats": player_stats,
        }

    @staticmethod
    async def get_upcoming_games(
        session: AsyncSession, days: int = 7
    ) -> list[dict]:
        """Get upcoming scheduled games for the next N days."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days)

        query = (
            select(Game)
            .where(
                and_(
                    Game.status.in_(["Scheduled", "PreGame"]),
                    Game.game_date >= now,
                    Game.game_date <= future,
                )
            )
            .order_by(Game.game_date.asc())
        )
        result = await session.execute(query)
        games = result.scalars().all()

        # Enrich with team names
        team_ids = set()
        for g in games:
            team_ids.add(g.home_team_id)
            team_ids.add(g.away_team_id)

        team_map = {}
        if team_ids:
            team_query = select(Team.id, Team.full_name, Team.abbreviation).where(
                Team.id.in_(team_ids)
            )
            result = await session.execute(team_query)
            for row in result.all():
                team_map[row.id] = (row.full_name, row.abbreviation)

        return [
            {
                "id": g.id,
                "nba_game_id": g.nba_game_id,
                "game_date": g.game_date,
                "home_team_id": g.home_team_id,
                "away_team_id": g.away_team_id,
                "home_team_name": team_map.get(g.home_team_id, (None, None))[0],
                "away_team_name": team_map.get(g.away_team_id, (None, None))[0],
                "home_team_abbr": team_map.get(g.home_team_id, (None, None))[1],
                "away_team_abbr": team_map.get(g.away_team_id, (None, None))[1],
                "status": g.status,
            }
            for g in games
        ]
