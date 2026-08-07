"""Player form and impact features.

Computes team-level metrics derived from individual player performance:
  - Top scorer form (recent PPG, efficiency)
  - Scoring concentration (Herfindahl-like index)
  - Player impact (+/- of key players)
  - Team depth / balance metrics

These complement the injury features in player_features.py by capturing
the *actual* performance of available players.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

PLAYER_FORM_FEATURE_NAMES = [
    "top_scorer_ppg",
    "top_scorer_ppg_trend",
    # NOTE: top3_scorers_ppg removed — redundant with top_scorer_ppg (r=0.994)
    "scoring_concentration",
    "avg_plus_minus",
    "top_scorer_plus_minus",
    "bench_scoring_pct",
    "starter_minutes_share",
]


class PlayerFormFeatures:
    """Computes player form and team composition features."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute player form features for a team.

        Args:
            team_id: Internal team ID.
            before_date: Reference date (only games before this date).
            n_games: Number of recent games to analyze.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float.
        """
        rows = await self._query_recent_player_stats(
            team_id, before_date, n_games, session
        )
        return self._compute_from_rows(rows, n_games)

    async def _query_recent_player_stats(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[dict]:
        """Query player stats for the team's last N games."""
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[dict]:
            # Single combined query: player stats joined with game, filtered by
            # team and date, limited to last N games via a subquery.
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
                minutes_val = _parse_minutes(s.minutes)
                rows.append({
                    "game_id": s.game_id,
                    "player_id": s.player_id,
                    "minutes": minutes_val or 0.0,
                    "pts": float(s.pts),
                    "fga": float(s.fga) if s.fga > 0 else 1.0,
                    "fgm": float(s.fgm),
                    "fg3m": float(s.fg3m),
                    "plus_minus": float(s.plus_minus) if s.plus_minus else 0.0,
                })

            return rows

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_rows(
        rows: list[dict], requested_n: int
    ) -> dict[str, float]:
        """Compute player form features from raw stat rows."""
        defaults = {name: 0.0 for name in PLAYER_FORM_FEATURE_NAMES}
        if not rows:
            return defaults

        # Group by player and compute per-player averages
        player_stats: dict[int, dict] = {}
        game_ids = set()

        for r in rows:
            pid = r["player_id"]
            gid = r["game_id"]
            game_ids.add(gid)
            if pid not in player_stats:
                player_stats[pid] = {
                    "total_pts": 0.0,
                    "total_min": 0.0,
                    "total_plus_minus": 0.0,
                    "total_fga": 0.0,
                    "total_fgm": 0.0,
                    "games": 0,
                    "ppg_by_game": [],
                }
            ps = player_stats[pid]
            ps["total_pts"] += r["pts"]
            ps["total_min"] += r["minutes"]
            ps["total_plus_minus"] += r["plus_minus"]
            ps["total_fga"] += r["fga"]
            ps["total_fgm"] += r["fgm"]
            ps["games"] += 1
            ps["ppg_by_game"].append(r["pts"])

        n_games = len(game_ids)
        if n_games == 0:
            return defaults

        # Per-player PPG
        player_ppg = {}
        for pid, ps in player_stats.items():
            if ps["games"] > 0:
                player_ppg[pid] = ps["total_pts"] / ps["games"]

        # Sort players by PPG
        sorted_players = sorted(player_ppg.items(), key=lambda x: x[1], reverse=True)

        # Top scorer PPG
        top_scorer_ppg = sorted_players[0][1] if sorted_players else 0.0

        # Top scorer PPG trend (last 5 games vs previous)
        top_scorer_trend = 0.0
        if sorted_players:
            top_pid = sorted_players[0][0]
            top_ps = player_stats[top_pid]
            ppg_list = top_ps.get("ppg_by_game", [])
            if len(ppg_list) >= 6:
                recent_5 = np.mean(ppg_list[:5])
                older = np.mean(ppg_list[5:])
                top_scorer_trend = recent_5 - older

        # Top 3 scorers PPG
        top3_ppg = sum(ppg for _, ppg in sorted_players[:3]) if sorted_players else 0.0

        # Scoring concentration: % of team points from top 3 scorers
        total_team_pts = sum(ps["total_pts"] for ps in player_stats.values())
        top3_pts = sum(
            player_stats[pid]["total_pts"]
            for pid, _ in sorted_players[:3]
            if pid in player_stats
        )
        scoring_concentration = top3_pts / total_team_pts if total_team_pts > 0 else 0.0

        # Average plus-minus across all players
        all_pm = [ps["total_plus_minus"] / max(ps["games"], 1) for ps in player_stats.values()]
        avg_plus_minus = float(np.mean(all_pm)) if all_pm else 0.0

        # Top scorer's average plus-minus
        top_scorer_pm = 0.0
        if sorted_players:
            top_pid = sorted_players[0][0]
            top_ps_games = player_stats[top_pid]["games"]
            if top_ps_games > 0:
                top_scorer_pm = player_stats[top_pid]["total_plus_minus"] / top_ps_games

        # Bench scoring %: players with < 25 min/game
        bench_pts = 0.0
        starter_pts = 0.0
        total_starter_min = 0.0
        total_all_min = 0.0
        for pid, ps in player_stats.items():
            mpg = ps["total_min"] / max(ps["games"], 1)
            total_all_min += ps["total_min"]
            if mpg >= 25.0:
                starter_pts += ps["total_pts"]
                total_starter_min += ps["total_min"]
            else:
                bench_pts += ps["total_pts"]

        bench_scoring_pct = bench_pts / total_team_pts if total_team_pts > 0 else 0.0
        starter_minutes_share = total_starter_min / total_all_min if total_all_min > 0 else 0.0

        return {
            "top_scorer_ppg": top_scorer_ppg,
            "top_scorer_ppg_trend": top_scorer_trend,
            # NOTE: top3_scorers_ppg removed — r=0.994 vs top_scorer_ppg
            "scoring_concentration": scoring_concentration,
            "avg_plus_minus": avg_plus_minus,
            "top_scorer_plus_minus": top_scorer_pm,
            "bench_scoring_pct": bench_scoring_pct,
            "starter_minutes_share": starter_minutes_share,
        }


def _parse_minutes(minutes_str: str | None) -> float | None:
    """Parse minutes from 'MM:SS' or 'MM.DD' format."""
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
