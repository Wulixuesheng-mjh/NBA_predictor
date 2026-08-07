"""Bench depth and rotation features.

Captures team depth quality beyond what player_form_features covers:
  - Bench player +/- efficiency
  - Starter-to-bench scoring ratio
  - Sixth man scoring punch
  - Rotation depth (how many players get meaningful minutes)

These complement the existing bench_scoring_pct and starter_minutes_share
in player_form_features.py by focusing on efficiency and depth, not just share.
"""

import logging
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

BENCH_FEATURE_NAMES = [
    "bench_plus_minus_per_game",       # Average +/- for bench players
    "starters_vs_bench_scoring_ratio",  # Starters PPG / Bench PPG (higher = top-heavy)
    "sixth_man_scoring",               # PPG of highest-scoring bench player
    "rotation_depth",                  # Number of players averaging ≥15 min/game
]


class BenchFeatures:
    """Computes bench depth and rotation features for a team."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute bench depth features for a team.

        Args:
            team_id: Internal team ID.
            before_date: Reference date.
            n_games: Number of recent games.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name → float.
        """
        rows = await self._query_recent_player_stats(
            team_id, before_date, n_games, session
        )
        return self._compute_from_rows(rows)

    async def _query_recent_player_stats(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[dict]:
        """Query per-player stats for recent N games."""
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[dict]:
            recent_game_subq = (
                select(Game.id)
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
            ).scalar_subquery()

            stats_query = (
                select(GamePlayerStats)
                .where(
                    and_(
                        GamePlayerStats.team_id == team_id,
                        GamePlayerStats.game_id.in_(recent_game_subq),
                    )
                )
            )
            result = await sess.execute(stats_query)
            stats = result.scalars().all()

            rows = []
            for s in stats:
                mins = _parse_minutes(s.minutes) or 0.0
                rows.append({
                    "player_id": s.player_id,
                    "game_id": s.game_id,
                    "minutes": mins,
                    "pts": float(s.pts),
                    "plus_minus": float(s.plus_minus) if s.plus_minus else 0.0,
                })
            return rows

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_rows(rows: list[dict]) -> dict[str, float]:
        """Compute bench features from raw stat rows."""
        defaults = {name: 0.0 for name in BENCH_FEATURE_NAMES}
        if not rows:
            return defaults

        # Group by player
        player_stats: dict[int, dict] = {}
        for r in rows:
            pid = r["player_id"]
            if pid not in player_stats:
                player_stats[pid] = {
                    "total_pts": 0.0,
                    "total_min": 0.0,
                    "total_pm": 0.0,
                    "games": 0,
                }
            ps = player_stats[pid]
            ps["total_pts"] += r["pts"]
            ps["total_min"] += r["minutes"]
            ps["total_pm"] += r["plus_minus"]
            ps["games"] += 1

        # Per-player per-game averages
        player_avgs = {}
        for pid, ps in player_stats.items():
            g = max(ps["games"], 1)
            mpg = ps["total_min"] / g
            player_avgs[pid] = {
                "ppg": ps["total_pts"] / g,
                "mpg": mpg,
                "pm_per_game": ps["total_pm"] / g,
                "is_starter": mpg >= 25.0,
            }

        # Split starters vs bench
        starters = {pid: p for pid, p in player_avgs.items() if p["is_starter"]}
        bench = {pid: p for pid, p in player_avgs.items() if not p["is_starter"]}

        # Bench +/- per game (average across bench players)
        if bench:
            bench_pm = np.mean([p["pm_per_game"] for p in bench.values()])
        else:
            bench_pm = 0.0

        # Starters vs bench scoring ratio
        starter_ppg = np.mean([p["ppg"] for p in starters.values()]) if starters else 0.0
        bench_ppg = np.mean([p["ppg"] for p in bench.values()]) if bench else 0.0
        if bench_ppg > 0:
            starters_vs_bench_ratio = starter_ppg / bench_ppg
        else:
            starters_vs_bench_ratio = 999.0  # All minutes are starter minutes

        # Sixth man scoring (highest PPG among bench)
        sixth_man_ppg = max((p["ppg"] for p in bench.values()), default=0.0)

        # Rotation depth: players averaging ≥ 15 min
        rotation_depth = sum(1 for p in player_avgs.values() if p["mpg"] >= 15.0)

        return {
            "bench_plus_minus_per_game": round(bench_pm, 2),
            "starters_vs_bench_scoring_ratio": round(starters_vs_bench_ratio, 2),
            "sixth_man_scoring": round(sixth_man_ppg, 1),
            "rotation_depth": float(rotation_depth),
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
