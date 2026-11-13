"""Roster continuity features — quantifies off-season roster changes.

When a team's roster changes significantly between seasons, historical
data (team stats, H2H, etc.) becomes less predictive. These features
help the model gauge how much to trust historical patterns.

Computed per team by comparing the current season's player set to
the previous season's, weighted by minutes played.

Features (4 per team, ×2 = 8 total):
  - roster_continuity:     Jaccard similarity of player IDs between seasons (0-1)
  - core_retention:        % of top-5 (by minutes) players who stayed (0-1)
  - roster_turnover_rate:  1 - roster_continuity (complement for model)
  - minutes_continuity:    % of this season's total minutes from returning players (0-1)
"""

import logging
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

ROSTER_CONTINUITY_FEATURE_NAMES = [
    "roster_continuity",
    "core_retention",
    "roster_turnover_rate",
    "minutes_continuity",
    "roster_data_missing",
]


def _determine_season(game_date: datetime) -> str:
    """Determine season string from a game date.

    NBA season: Oct–Jun. Example: game on 2026-01-15 → "2025-26".
    """
    year = game_date.year
    if game_date.month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def _previous_season(season: str) -> str:
    """Get the previous season string. e.g. '2025-26' → '2024-25'."""
    start_year = int(season.split("-")[0])
    prev_start = start_year - 1
    return f"{prev_start}-{str(start_year)[-2:]}"


class RosterContinuityFeatures:
    """Computes roster continuity between current and previous season."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        game_date: datetime,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute roster continuity features for a team.

        Args:
            team_id: Internal team ID.
            game_date: Date of the upcoming game (determines the season).
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float.
        """
        season = _determine_season(game_date)
        prev_season = _previous_season(season)

        async def _query(sess: AsyncSession) -> dict[str, float]:
            # Current season roster with minutes — only games already played,
            # otherwise a mid-season game would "know" who played in April.
            current = await self._get_season_roster(
                sess, team_id, season, before_date=game_date
            )
            # Previous season is entirely in the past, no cut-off needed
            previous = await self._get_season_roster(sess, team_id, prev_season)

            return self._compute_continuity(current, previous)

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    async def _get_season_roster(
        sess: AsyncSession,
        team_id: int,
        season: str,
        before_date: datetime | None = None,
    ) -> dict[int, int]:
        """Get {player_id: total_minutes} for a team in a given season.

        Args:
            before_date: When set, only games strictly before this date count
                (prevents using information from games not yet played).

        Minutes is stored as a decimal string (e.g. "29.5"). We cast to
        float and sum, then convert to integer seconds for comparison.

        Returns empty dict if no games found for this team+season.
        """
        from app.models.game import Game
        from app.models.game_player_stats import GamePlayerStats
        from sqlalchemy import Float

        conditions = [
            Game.season == season,
            GamePlayerStats.team_id == team_id,
            Game.status == "Final",
            Game.game_type.in_(("regular", "play_in", "playoffs")),
            Game.player_stats_complete.is_(True),
        ]
        if before_date is not None:
            conditions.append(Game.game_date < before_date)

        query = (
            select(
                GamePlayerStats.player_id,
                func.sum(
                    func.nullif(GamePlayerStats.minutes, "").cast(Float)
                ).label("total_min"),
            )
            .join(Game, GamePlayerStats.game_id == Game.id)
            .where(and_(*conditions))
            .group_by(GamePlayerStats.player_id)
        )
        result = await sess.execute(query)
        rows = result.all()

        roster: dict[int, int] = {}
        for player_id, min_str in rows:
            minutes = _parse_minutes_to_seconds(min_str)
            roster[player_id] = minutes

        return roster

    @staticmethod
    def _compute_continuity(
        current: dict[int, int],
        previous: dict[int, int],
    ) -> dict[str, float]:
        """Compute continuity metrics from two season rosters."""
        if not current:
            # Unknown is represented explicitly.  Neutral numeric values keep
            # old models stable while the mask prevents treating them as fact.
            return {
                "roster_continuity": 0.5,
                "core_retention": 0.5,
                "roster_turnover_rate": 0.5,
                "minutes_continuity": 0.5,
                "roster_data_missing": 1.0,
            }

        if not previous:
            # No previous season data (expansion team?).
            # Can't compute continuity — return neutral values.
            return {
                "roster_continuity": 0.5,
                "core_retention": 0.5,
                "roster_turnover_rate": 0.5,
                "minutes_continuity": 0.5,
                "roster_data_missing": 1.0,
            }

        current_players = set(current.keys())
        previous_players = set(previous.keys())

        # 1. Jaccard similarity: |intersection| / |union|
        intersection = current_players & previous_players
        union = current_players | previous_players
        roster_continuity = len(intersection) / len(union) if union else 0.0

        # 2. Core retention: % of top-5 (by minutes) from previous season
        # who are still on the team this season
        sorted_prev = sorted(previous.items(), key=lambda x: x[1], reverse=True)
        top5_prev = {pid for pid, _ in sorted_prev[:5]}
        if top5_prev:
            core_stayed = top5_prev & current_players
            core_retention = len(core_stayed) / len(top5_prev)
        else:
            core_retention = 0.5

        # 3. Roster turnover: complement of continuity
        roster_turnover_rate = 1.0 - roster_continuity

        # 4. Minutes continuity: what % of this season's minutes come
        # from players who also played for the team last season?
        total_minutes = sum(current.values())
        if total_minutes > 0:
            returning_minutes = sum(
                mins for pid, mins in current.items()
                if pid in previous_players
            )
            minutes_continuity = returning_minutes / total_minutes
        else:
            minutes_continuity = 0.5

        return {
            "roster_continuity": round(roster_continuity, 4),
            "core_retention": round(core_retention, 4),
            "roster_turnover_rate": round(roster_turnover_rate, 4),
            "minutes_continuity": round(minutes_continuity, 4),
            "roster_data_missing": 0.0,
        }


def _parse_minutes_to_seconds(min_str: str | None) -> int:
    """Parse a minutes string to total seconds for comparison.

    The DB stores minutes as a decimal string (e.g., "29.5" = 29:30).
    We convert to integer seconds — precision is not critical since
    we only need relative ordering for top-5 identification.
    """
    if min_str is None or min_str == "":
        return 0
    try:
        return int(float(min_str) * 60)
    except (ValueError, TypeError):
        return 0
