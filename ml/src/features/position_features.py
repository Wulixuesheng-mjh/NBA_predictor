"""Position-based matchup features.

Computes scoring by position group (frontcourt vs backcourt) to capture
matchup advantages at the position level.

Position classification:
  - Backcourt: PG, SG, G (guard)
  - Frontcourt: SF, PF, C, F (forward/center)

Combo positions like "G-F" or "F-C" use the FIRST letter for classification.
"""

import logging
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

POSITION_MATCH_FEATURE_NAMES = [
    "frontcourt_scoring",        # Total PPG from SF+PF+C
    "backcourt_scoring",         # Total PPG from PG+SG
]

POSITION_MATCH_DIFF_NAMES = [
    "frontcourt_scoring_diff",   # home_frontcourt - away_frontcourt
    "backcourt_scoring_diff",    # home_backcourt - away_backcourt
]

# Position → group mapping (primary letter before hyphen)
_BACKCOURT = {"PG", "G", "SG"}
_FRONTCOURT = {"SF", "F", "PF", "C"}


def _classify_position(position: str | None) -> str:
    """Classify a position string into 'frontcourt' or 'backcourt'.

    Returns 'frontcourt' as default for unknown positions (most NBA players
    are forwards/centers; guards are usually clearly labeled).
    """
    if not position:
        return "frontcourt"
    primary = position.split("-")[0].strip().upper()
    if primary in _BACKCOURT:
        return "backcourt"
    return "frontcourt"


class PositionMatchFeatures:
    """Computes position-based scoring features for a team."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute position-based scoring for a team.

        Args:
            team_id: Internal team ID.
            before_date: Reference date.
            n_games: Number of recent games.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name → float.
        """
        rows = await self._query_recent_position_stats(
            team_id, before_date, n_games, session
        )
        return self._compute_from_rows(rows)

    async def _query_recent_position_stats(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[dict]:
        """Query player stats with position info for recent N games."""
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game
        from app.models.player import Player

        async def _query(sess: AsyncSession) -> list[dict]:
            recent_game_subq = (
                select(Game.id)
                .where(
                    and_(
                        Game.status == "Final",
                        Game.game_type.in_(("regular", "play_in", "playoffs")),
                        Game.player_stats_complete.is_(True),
                        Game.game_date < before_date,
                        (Game.home_team_id == team_id)
                        | (Game.away_team_id == team_id),
                    )
                )
                .order_by(desc(Game.game_date))
                .limit(n_games)
            ).scalar_subquery()

            stats_query = (
                select(
                    GamePlayerStats.pts,
                    GamePlayerStats.minutes,
                    GamePlayerStats.game_id,
                    Player.position,
                )
                .join(Player, GamePlayerStats.player_id == Player.id)
                .where(
                    and_(
                        GamePlayerStats.team_id == team_id,
                        GamePlayerStats.game_id.in_(recent_game_subq),
                    )
                )
            )
            result = await sess.execute(stats_query)
            rows = result.all()

            return [
                {
                    "pts": float(r.pts),
                    "minutes": _parse_minutes(r.minutes) or 0.0,
                    "position_group": _classify_position(r.position),
                    "game_id": r.game_id,
                }
                for r in rows
            ]

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_rows(rows: list[dict]) -> dict[str, float]:
        """Compute position scoring from raw rows."""
        defaults = {name: 0.0 for name in POSITION_MATCH_FEATURE_NAMES}
        if not rows:
            return defaults

        frontcourt_pts = sum(r["pts"] for r in rows if r["position_group"] == "frontcourt")
        backcourt_pts = sum(r["pts"] for r in rows if r["position_group"] == "backcourt")

        # Count distinct games for averaging
        total_games = len(set(r.get("game_id", i) for i, r in enumerate(rows)))
        n = max(total_games, 1)

        return {
            "frontcourt_scoring": round(frontcourt_pts / n, 1),
            "backcourt_scoring": round(backcourt_pts / n, 1),
        }

    @staticmethod
    def compute_diff(
        home_f: dict[str, float],
        away_f: dict[str, float],
    ) -> dict[str, float]:
        """Compute position matchup diff features from home/away dicts."""
        home_fc = home_f.get("frontcourt_scoring", 0.0)
        away_fc = away_f.get("frontcourt_scoring", 0.0)
        home_bc = home_f.get("backcourt_scoring", 0.0)
        away_bc = away_f.get("backcourt_scoring", 0.0)

        return {
            "frontcourt_scoring_diff": round(home_fc - away_fc, 1),
            "backcourt_scoring_diff": round(home_bc - away_bc, 1),
        }


def _parse_minutes(minutes_str: str | None) -> float | None:
    """Parse minutes from 'MM:SS' format or raw float."""
    if minutes_str is None:
        return None
    if isinstance(minutes_str, (int, float)):
        return float(minutes_str)
    if ":" in str(minutes_str):
        try:
            parts = str(minutes_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return None
    try:
        return float(minutes_str)
    except ValueError:
        return None
