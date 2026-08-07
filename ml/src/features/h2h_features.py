"""Head-to-head (H2H) matchup features.

Queries the last 3 meetings between two teams and computes:
- Home team win count in those matchups
- Average point differential (from home team perspective)
- Average total points scored
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

H2H_FEATURE_NAMES = [
    "home_win_count",
    "avg_point_diff",
    "avg_total_pts",
]


@dataclass
class H2HGameRow:
    """Lightweight row for a head-to-head game result."""

    home_score: float
    away_score: float
    home_team_id: int


class H2HFeatures:
    """Computes head-to-head matchup features."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        home_team_id: int,
        away_team_id: int,
        before_date: datetime,
        n_games: int = 3,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute H2H features from the last N meetings.

        Args:
            home_team_id: Internal ID of the home team for the upcoming game.
            away_team_id: Internal ID of the away team.
            before_date: Only consider games before this date.
            n_games: Number of past H2H games to consider.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float.
        """
        rows = await self._query_h2h_games(
            home_team_id, away_team_id, before_date, n_games, session
        )
        return self._compute_from_rows(rows, home_team_id)

    async def _query_h2h_games(
        self,
        team_a_id: int,
        team_b_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[H2HGameRow]:
        """Query the last N completed games between two teams."""
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[H2HGameRow]:
            query = (
                select(Game)
                .where(
                    and_(
                        Game.status == "Final",
                        Game.game_date < before_date,
                        or_(
                            and_(
                                Game.home_team_id == team_a_id,
                                Game.away_team_id == team_b_id,
                            ),
                            and_(
                                Game.home_team_id == team_b_id,
                                Game.away_team_id == team_a_id,
                            ),
                        ),
                    )
                )
                .order_by(desc(Game.game_date))
                .limit(n_games)
            )
            result = await sess.execute(query)
            games = result.scalars().all()

            return [
                H2HGameRow(
                    home_score=float(g.home_score or 0),
                    away_score=float(g.away_score or 0),
                    home_team_id=g.home_team_id,
                )
                for g in games
            ]

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_rows(
        rows: list[H2HGameRow], home_team_id: int
    ) -> dict[str, float]:
        """Compute H2H features from game rows.

        All differentials are from the perspective of `home_team_id`.
        """
        n = len(rows)
        if n == 0:
            return {name: 0.0 for name in H2H_FEATURE_NAMES}

        home_wins = 0
        diffs = []
        totals = []

        for row in rows:
            totals.append(row.home_score + row.away_score)
            if row.home_team_id == home_team_id:
                # Our team was home in this past game
                diff = row.home_score - row.away_score
                if diff > 0:
                    home_wins += 1
            else:
                # Our team was away in this past game
                diff = row.away_score - row.home_score
                if diff > 0:
                    home_wins += 1
            diffs.append(diff)

        features = {
            "home_win_count": float(home_wins),
            "avg_point_diff": float(np.mean(diffs)),
            "avg_total_pts": float(np.mean(totals)),
        }

        return features
