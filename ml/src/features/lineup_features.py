"""Lineup stability features.

Measures how consistent a team's starting lineup has been over recent games
using Jaccard similarity of the top-5-minutes players across consecutive games.

High stability → consistent starters, predictable system.
Low stability → injuries/trades/rotations disrupting chemistry.
"""

import logging
from datetime import datetime

from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

LINEUP_FEATURE_NAMES = [
    "lineup_stability",  # Jaccard similarity of starting 5, averaged across last 5 games
]


class LineupFeatures:
    """Computes lineup stability features for a team."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 5,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute lineup stability for a team.

        For each of the last N games, identifies the top 5 players by minutes
        (effective starters). Then computes Jaccard similarity between each
        consecutive pair of games and averages the results.

        Args:
            team_id: Internal team ID.
            before_date: Reference date.
            n_games: Number of recent games to analyze (default 5).
            session: Optional existing session.

        Returns:
            Dict with "lineup_stability" → float (0.0-1.0).
        """
        game_lineups = await self._query_game_lineups(
            team_id, before_date, n_games, session
        )
        return self._compute_stability(game_lineups)

    async def _query_game_lineups(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[set[int]]:
        """Query top-5 players by minutes for each of the last N games.

        Returns a list of sets of player_ids, one per game (chronological order).
        """
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[set[int]]:
            # Get the last N game IDs for this team
            game_query = (
                select(Game.id, Game.game_date)
                .where(
                    and_(
                        Game.status == "Final",
                        Game.game_date < before_date,
                        (Game.home_team_id == team_id)
                        | (Game.away_team_id == team_id),
                    )
                )
                .order_by(desc(Game.game_date))
                .limit(n_games)
            )
            result = await sess.execute(game_query)
            games = [(row[0], row[1]) for row in result.all()]
            # Reorder chronologically (oldest first) for pairwise comparison
            games = list(reversed(games))

            if len(games) < 2:
                return []

            game_lineups: list[set[int]] = []
            for game_id, _ in games:
                # Top 5 players by minutes for this game
                stats_query = (
                    select(GamePlayerStats.player_id)
                    .where(GamePlayerStats.game_id == game_id)
                    .order_by(
                        func.coalesce(
                            func.nullif(GamePlayerStats.minutes, ""),
                            "0:00",
                        ).desc()
                    )
                    .limit(5)
                )
                stats_result = await sess.execute(stats_query)
                top5 = {row[0] for row in stats_result.all()}
                if len(top5) >= 3:  # Need at least 3 players to be meaningful
                    game_lineups.append(top5)

            return game_lineups

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_stability(game_lineups: list[set[int]]) -> dict[str, float]:
        """Compute average Jaccard similarity between consecutive game lineups."""
        if len(game_lineups) < 2:
            return {"lineup_stability": 0.0}

        similarities: list[float] = []
        for i in range(len(game_lineups) - 1):
            a = game_lineups[i]
            b = game_lineups[i + 1]
            intersection = len(a & b)
            union = len(a | b)
            if union > 0:
                similarities.append(intersection / union)
            else:
                similarities.append(0.0)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return {"lineup_stability": round(avg_similarity, 3)}
