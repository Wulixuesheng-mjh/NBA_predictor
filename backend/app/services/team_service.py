"""Team service — business logic for team operations."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.team import Team
from app.models.game import Game
from app.models.player import Player
from app.models.team_game_stats import TeamGameStats

logger = logging.getLogger(__name__)


class TeamService:
    """Service for team-related operations."""

    @staticmethod
    async def list_teams(
        session: AsyncSession,
        conference: str | None = None,
        division: str | None = None,
    ) -> tuple[list[Team], int]:
        """List teams with optional filters."""
        query = select(Team)
        count_query = select(func.count(Team.id))

        if conference:
            query = query.where(Team.conference == conference)
            count_query = count_query.where(Team.conference == conference)
        if division:
            query = query.where(Team.division == division)
            count_query = count_query.where(Team.division == division)

        query = query.order_by(Team.full_name)

        result = await session.execute(query)
        teams = result.scalars().all()

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        return list(teams), total

    @staticmethod
    async def get_team(
        session: AsyncSession, team_id: int
    ) -> Team | None:
        """Get a single team by ID."""
        query = select(Team).where(Team.id == team_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_team_detail(
        session: AsyncSession, team_id: int, n_recent: int = 10
    ) -> dict:
        """Get team with recent performance summary."""
        team = await TeamService.get_team(session, team_id)
        if team is None:
            return None

        # Recent record
        recent_query = (
            select(Game)
            .where(
                and_(
                    Game.status == "Final",
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                )
            )
            .order_by(desc(Game.game_date))
            .limit(n_recent)
        )
        result = await session.execute(recent_query)
        recent_games = result.scalars().all()

        wins = 0
        total_pts = 0
        total_pts_allowed = 0
        for g in recent_games:
            if g.home_team_id == team_id:
                if (g.home_score or 0) > (g.away_score or 0):
                    wins += 1
                total_pts += g.home_score or 0
                total_pts_allowed += g.away_score or 0
            else:
                if (g.away_score or 0) > (g.home_score or 0):
                    wins += 1
                total_pts += g.away_score or 0
                total_pts_allowed += g.home_score or 0

        n = len(recent_games)
        record = f"{wins}-{n - wins}" if n > 0 else "N/A"
        avg_pts = total_pts / n if n > 0 else 0.0
        avg_pts_allowed = total_pts_allowed / n if n > 0 else 0.0
        net_rating = avg_pts - avg_pts_allowed

        return {
            "team": team,
            "recent_record": record,
            "avg_pts": round(avg_pts, 1),
            "avg_pts_allowed": round(avg_pts_allowed, 1),
            "net_rating": round(net_rating, 1),
        }

    @staticmethod
    async def get_team_roster(
        session: AsyncSession, team_id: int
    ) -> dict | None:
        """Get current roster for a team."""
        team = await TeamService.get_team(session, team_id)
        if team is None:
            return None

        query = select(Player).where(Player.team_id == team_id).order_by(Player.full_name)
        result = await session.execute(query)
        players = result.scalars().all()

        return {
            "team_id": team.id,
            "team_name": team.full_name,
            "players": list(players),
            "total": len(players),
        }
