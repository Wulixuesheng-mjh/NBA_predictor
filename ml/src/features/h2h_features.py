"""Head-to-head (H2H) matchup features.

Queries the last 10 meetings between two teams and computes
time-decay weighted features:
  - Within 1 year:   weight 0.6
  - 1-2 years ago:   weight 0.3
  - 2-3 years ago:   weight 0.1
  - Older / no data: zero weight

This ensures recent matchups (with current rosters) dominate
while still extracting signal from older franchise-level patterns.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

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
    game_date: datetime


class H2HFeatures:
    """Computes head-to-head matchup features."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        home_team_id: int,
        away_team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute H2H features from the last N meetings with time decay.

        Weights: ≤1 year = 0.6, 1-2 years = 0.3, 2-3 years = 0.1, >3 years = 0.
        This ensures recent matchups (with current rosters) dominate while
        still extracting signal from older franchise-level patterns.

        Args:
            home_team_id: Internal ID of the home team for the upcoming game.
            away_team_id: Internal ID of the away team.
            before_date: Only consider games before this date. Also used as the
                reference point for time-decay weighting.
            n_games: Number of past H2H games to consider (default 10).
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float, time-decay weighted.
        """
        rows = await self._query_h2h_games(
            home_team_id, away_team_id, before_date, n_games, session
        )
        return self._compute_from_rows(rows, home_team_id, before_date)

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
                    game_date=g.game_date,
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
        rows: list[H2HGameRow],
        home_team_id: int,
        before_date: datetime,
    ) -> dict[str, float]:
        """Compute time-decay weighted H2H features from game rows.

        Weights:
          - ≤1 year before `before_date`:   weight 0.6
          - 1-2 years before:               weight 0.3
          - 2-3 years before:               weight 0.1
          - >3 years before / missing date: weight 0.0

        All differentials are from the perspective of `home_team_id`.
        If all weights sum to 0, returns zero-filled features.
        """
        n = len(rows)
        if n == 0:
            return {name: 0.0 for name in H2H_FEATURE_NAMES}

        # Compute per-row time-decay weights
        weights = []
        diffs = []
        totals = []
        weighted_home_wins = 0.0

        # Ensure before_date is a naive datetime (can be date or TZ-aware)
        from datetime import datetime as dt_module, timezone
        if not isinstance(before_date, dt_module):
            before_date = dt_module.combine(before_date, dt_module.min.time())
        if before_date.tzinfo is not None:
            before_date = before_date.replace(tzinfo=None)

        one_year = 365.25 * 24 * 3600  # seconds for timedelta comparison

        for row in rows:
            # Time-decay weight based on how old the game is relative to before_date
            if row.game_date is not None:
                game_dt = row.game_date
                if game_dt.tzinfo is not None:
                    game_dt = game_dt.replace(tzinfo=None)
                age_seconds = (before_date - game_dt).total_seconds()
                years_ago = age_seconds / one_year
            else:
                years_ago = 99  # no date → skip

            if years_ago <= 1:
                w = 0.6
            elif years_ago <= 2:
                w = 0.3
            elif years_ago <= 3:
                w = 0.1
            else:
                w = 0.0  # too old, skip

            if w == 0.0:
                continue

            weights.append(w)
            totals.append(row.home_score + row.away_score)

            if row.home_team_id == home_team_id:
                # Our team was home in this past game
                diff = row.home_score - row.away_score
                if diff > 0:
                    weighted_home_wins += w
            else:
                # Our team was away in this past game
                diff = row.away_score - row.home_score
                if diff > 0:
                    weighted_home_wins += w
            diffs.append(diff)

        if not weights or sum(weights) == 0:
            return {name: 0.0 for name in H2H_FEATURE_NAMES}

        total_weight = sum(weights)

        features = {
            "home_win_count": float(weighted_home_wins),
            "avg_point_diff": float(
                np.average(diffs, weights=weights) if diffs else 0.0
            ),
            "avg_total_pts": float(
                np.average(totals, weights=weights) if totals else 0.0
            ),
        }

        return features
