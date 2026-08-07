"""Player service — business logic for player operations."""

import logging

from sqlalchemy import Float, select, func, and_, or_, not_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.player import Player
from app.models.game_player_stats import GamePlayerStats
from app.models.player_injuries import PlayerInjury
from app.models.team import Team
from app.models.game import Game

logger = logging.getLogger(__name__)


class PlayerService:
    """Service for player-related operations."""

    @staticmethod
    async def list_players(
        session: AsyncSession,
        team_id: int | None = None,
        search: str | None = None,
        position: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Player], int]:
        """List players with optional filters."""
        query = select(Player)
        count_query = select(func.count(Player.id))

        if team_id is not None:
            query = query.where(Player.team_id == team_id)
            count_query = count_query.where(Player.team_id == team_id)

        if search:
            search_filter = or_(
                Player.full_name.ilike(f"%{search}%"),
                Player.first_name.ilike(f"%{search}%"),
                Player.last_name.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        if position:
            query = query.where(Player.position == position)
            count_query = count_query.where(Player.position == position)

        query = query.order_by(Player.full_name).offset(offset).limit(limit)

        result = await session.execute(query)
        players = result.scalars().all()

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        return list(players), total

    @staticmethod
    async def get_player(
        session: AsyncSession, player_id: int
    ) -> Player | None:
        """Get a single player by ID."""
        query = select(Player).where(Player.id == player_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_player_detail(
        session: AsyncSession, player_id: int
    ) -> dict | None:
        """Get player with season stats and active injuries."""
        player = await PlayerService.get_player(session, player_id)
        if player is None:
            return None

        # Season stats — exclude rows with no actual stats (data from incomplete sources)
        stats_query = (
            select(
                func.count(GamePlayerStats.id).label("games"),
                func.avg(GamePlayerStats.pts).label("avg_pts"),
                func.avg(GamePlayerStats.reb).label("avg_reb"),
                func.avg(GamePlayerStats.ast).label("avg_ast"),
                func.avg(GamePlayerStats.stl).label("avg_stl"),
                func.avg(GamePlayerStats.blk).label("avg_blk"),
                func.avg(GamePlayerStats.tov).label("avg_tov"),
                func.avg(
                    func.nullif(GamePlayerStats.minutes, "").cast(Float)
                ).label("avg_minutes"),
                func.sum(GamePlayerStats.fgm).label("total_fgm"),
                func.sum(GamePlayerStats.fga).label("total_fga"),
                func.sum(GamePlayerStats.fg3m).label("total_fg3m"),
                func.sum(GamePlayerStats.fg3a).label("total_fg3a"),
                func.sum(GamePlayerStats.ftm).label("total_ftm"),
                func.sum(GamePlayerStats.fta).label("total_fta"),
            )
            .join(Game, GamePlayerStats.game_id == Game.id)
            .where(
                and_(
                    GamePlayerStats.player_id == player_id,
                    Game.status == "Final",
                    # Exclude rows where all key stats are 0 (incomplete data from old bulk load)
                    not_(and_(
                        GamePlayerStats.pts == 0,
                        GamePlayerStats.fgm == 0,
                        GamePlayerStats.fga == 0,
                        GamePlayerStats.reb == 0,
                        GamePlayerStats.ast == 0,
                    )),
                )
            )
        )
        result = await session.execute(stats_query)
        stats_row = result.one_or_none()

        season_stats = None
        if stats_row and stats_row.games:
            total_fgm = stats_row.total_fgm or 0
            total_fga = stats_row.total_fga or 1
            total_fg3m = stats_row.total_fg3m or 0
            total_fg3a = stats_row.total_fg3a or 1
            total_ftm = stats_row.total_ftm or 0
            total_fta = stats_row.total_fta or 1

            season_stats = {
                "games_played": stats_row.games,
                "avg_pts": round(float(stats_row.avg_pts or 0), 1),
                "avg_reb": round(float(stats_row.avg_reb or 0), 1),
                "avg_ast": round(float(stats_row.avg_ast or 0), 1),
                "avg_stl": round(float(stats_row.avg_stl or 0), 1),
                "avg_blk": round(float(stats_row.avg_blk or 0), 1),
                "avg_tov": round(float(stats_row.avg_tov or 0), 1),
                "avg_minutes": round(float(stats_row.avg_minutes or 0), 1),
                "fg_pct": round(total_fgm / total_fga, 3) if total_fga > 0 else 0.0,
                "fg3_pct": round(total_fg3m / total_fg3a, 3) if total_fg3a > 0 else 0.0,
                "ft_pct": round(total_ftm / total_fta, 3) if total_fta > 0 else 0.0,
            }

        # Active injuries
        injuries_query = (
            select(PlayerInjury)
            .where(
                and_(
                    PlayerInjury.player_id == player_id,
                    PlayerInjury.is_active == True,
                )
            )
            .order_by(PlayerInjury.reported_date.desc())
        )
        result = await session.execute(injuries_query)
        active_injuries = result.scalars().all()

        # Team abbreviation
        team_abbr = None
        if player.team_id:
            team_query = select(Team.abbreviation).where(Team.id == player.team_id)
            result = await session.execute(team_query)
            team_abbr = result.scalar_one_or_none()

        return {
            "player": player,
            "season_stats": season_stats,
            "active_injuries": list(active_injuries),
            "team_abbreviation": team_abbr,
        }
