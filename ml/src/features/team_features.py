"""Team recent performance features.

For a given team and reference date, queries the last N completed games
and computes ~25 features capturing form, efficiency, and trends.

Adaptive lookback: if fewer than N games are available (early season),
uses all available games with appropriate fallback indicators.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

# ── Feature name constants (home team prefix: "home_", away: "away_") ──

TEAM_FEATURE_NAMES = [
    # Basic stats
    "win_rate",
    "avg_pts",
    "avg_pts_allowed",
    # NOTE: avg_net_pts removed — redundant with avg_net_rtg (r=0.952, pace-adjusted better)
    "avg_reb",
    "avg_ast",
    "avg_stl",
    "avg_blk",
    # NOTE: avg_tov removed — redundant with avg_tov_pct (r=0.966, pace-adjusted better)
    # Shooting
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "avg_ts_pct",
    # NOTE: avg_efg_pct removed — redundant with avg_ts_pct (r=0.964, TS includes FTs)
    # Advanced
    "avg_off_rtg",
    "avg_def_rtg",
    "avg_net_rtg",
    "avg_pace",
    "avg_tov_pct",
    "avg_ftr",
    "avg_ast_pct",
    # Trends
    "trend_pts",
    "trend_net",
    "trend_off_rtg",
    "trend_def_rtg",
    "last5_vs_first5_pts_diff",
    "last5_vs_first5_net_diff",
    "win_rate_last5",
    # Consistency
    "pts_std",
    "net_pts_std",
    # NOTE: consistency_score removed — algebraically derived from pts_std/avg_pts (r=-0.975)
    "games_available",
]


@dataclass
class TeamGameRow:
    """Lightweight row for a single team game result."""

    pts: float
    pts_allowed: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    fgm: float
    fga: float
    fg3m: float
    fg3a: float
    ftm: float
    fta: float
    off_rtg: float | None
    def_rtg: float | None
    pace: float | None
    ts_pct: float | None
    efg_pct: float | None
    tov_pct: float | None
    ast_pct: float | None


class TeamFeatures:
    """Computes team-level features from recent game history."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute all team features for a given team and reference date.

        Args:
            team_id: Internal team ID (database pk).
            before_date: Only games before this date are considered.
            n_games: Number of recent games to look back (5-15).
            session: Optional existing session; one is created if not provided.

        Returns:
            Dict mapping feature_name -> float value.
        """
        rows = await self._query_recent_games(team_id, before_date, n_games, session)
        return self._compute_from_rows(rows, n_games)

    async def _query_recent_games(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int,
        session: AsyncSession | None = None,
    ) -> list[TeamGameRow]:
        """Query the last N completed team games before the reference date.

        Joins through games to get opponent score for win/loss determination.
        """
        # Import models lazily to avoid circular imports at module level
        from app.models.team_game_stats import TeamGameStats
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[TeamGameRow]:
            # Subquery: team stats joined with game for opponent score
            query = (
                select(TeamGameStats, Game)
                .join(Game, TeamGameStats.game_id == Game.id)
                .where(
                    and_(
                        TeamGameStats.team_id == team_id,
                        Game.game_date < before_date,
                        Game.status == "Final",
                    )
                )
                .order_by(desc(Game.game_date))
                .limit(n_games)
            )
            result = await sess.execute(query)
            pairs = result.all()

            rows = []
            for tgs, game in pairs:
                # Determine opponent score
                if tgs.is_home:
                    pts_allowed = game.away_score or 0
                else:
                    pts_allowed = game.home_score or 0

                rows.append(
                    TeamGameRow(
                        pts=float(tgs.pts),
                        pts_allowed=float(pts_allowed),
                        reb=float(tgs.reb),
                        ast=float(tgs.ast),
                        stl=float(tgs.stl),
                        blk=float(tgs.blk),
                        tov=float(tgs.tov),
                        fgm=float(tgs.fgm),
                        fga=float(tgs.fga) if tgs.fga > 0 else 1.0,
                        fg3m=float(tgs.fg3m),
                        fg3a=float(tgs.fg3a) if tgs.fg3a > 0 else 1.0,
                        ftm=float(tgs.ftm),
                        fta=float(tgs.fta) if tgs.fta > 0 else 1.0,
                        off_rtg=tgs.off_rtg,
                        def_rtg=tgs.def_rtg,
                        pace=tgs.pace,
                        ts_pct=tgs.ts_pct,
                        efg_pct=tgs.efg_pct,
                        tov_pct=tgs.tov_pct,
                        ast_pct=tgs.ast_pct,
                    )
                )
            return rows

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_rows(
        rows: list[TeamGameRow], requested_n: int
    ) -> dict[str, float]:
        """Compute all team features from a list of recent game rows."""
        n = len(rows)

        # If no games available, return zeros with games_available=0
        if n == 0:
            return {name: 0.0 for name in TEAM_FEATURE_NAMES}

        # Extract arrays
        pts = np.array([r.pts for r in rows])
        pts_allowed = np.array([r.pts_allowed for r in rows])
        net_pts = pts - pts_allowed
        reb = np.array([r.reb for r in rows])
        ast = np.array([r.ast for r in rows])
        stl = np.array([r.stl for r in rows])
        blk = np.array([r.blk for r in rows])
        # NOTE: tov array no longer extracted — avg_tov removed (redundant with avg_tov_pct)

        # Shooting: aggregate sums then divide (more accurate than per-game avg)
        total_fgm = sum(r.fgm for r in rows)
        total_fga = sum(r.fga for r in rows)
        total_fg3m = sum(r.fg3m for r in rows)
        total_fg3a = sum(r.fg3a for r in rows)
        total_ftm = sum(r.ftm for r in rows)
        total_fta = sum(r.fta for r in rows)

        # Win rate
        wins = sum(1 for r in rows if r.pts > r.pts_allowed)
        win_rate = wins / n

        # Advanced stats averages (handle None)
        off_rtg_vals = [r.off_rtg for r in rows if r.off_rtg is not None]
        def_rtg_vals = [r.def_rtg for r in rows if r.def_rtg is not None]
        pace_vals = [r.pace for r in rows if r.pace is not None]
        ts_pct_vals = [r.ts_pct for r in rows if r.ts_pct is not None]
        # NOTE: efg_pct_vals removed — avg_efg_pct redundant with avg_ts_pct (r=0.964)
        tov_pct_vals = [r.tov_pct for r in rows if r.tov_pct is not None]
        ast_pct_vals = [r.ast_pct for r in rows if r.ast_pct is not None]

        # Computed on-the-fly from raw data:
        # FTR = FTA / FGA per game
        ftr_vals = []
        for r in rows:
            if r.fga > 0:
                ftr_vals.append(r.fta / r.fga)
        # Net RTG per game
        net_rtg_vals = []
        for i in range(len(rows)):
            r = rows[i]
            o = off_rtg_vals[i] if i < len(off_rtg_vals) else None
            d = def_rtg_vals[i] if i < len(def_rtg_vals) else None
            if o is not None and d is not None:
                net_rtg_vals.append(o - d)

        # Trend: linear regression slope over games (chronological order)
        # rows are in DESC order (most recent first), so reverse for trend
        x = np.arange(n)
        trend_pts = _safe_slope(x, pts[::-1])
        trend_net = _safe_slope(x, net_pts[::-1])
        # Advanced stat trends (match x array length to available values)
        trend_off_rtg = _safe_slope(
            np.arange(len(off_rtg_vals)), np.array(off_rtg_vals[::-1])
        ) if len(off_rtg_vals) >= 2 else 0.0
        trend_def_rtg = _safe_slope(
            np.arange(len(def_rtg_vals)), np.array(def_rtg_vals[::-1])
        ) if len(def_rtg_vals) >= 2 else 0.0

        # Last 5 vs first 5 comparison
        if n >= 6:
            last5 = min(5, n)
            first5_start = max(0, n - 10)
            first5_end = min(n - last5, n)

            last5_pts = np.mean(pts[:last5])
            first5_pts = np.mean(pts[-last5:]) if n >= last5 * 2 else np.mean(pts[first5_start:first5_end])

            last5_net = np.mean(net_pts[:last5])
            first5_net = np.mean(net_pts[-last5:]) if n >= last5 * 2 else np.mean(net_pts[first5_start:first5_end])

            last5_vs_first5_pts_diff = last5_pts - first5_pts
            last5_vs_first5_net_diff = last5_net - first5_net
        else:
            last5_vs_first5_pts_diff = 0.0
            last5_vs_first5_net_diff = 0.0

        # Win rate in last 5
        last5_n = min(5, n)
        last5_wins = sum(1 for r in rows[:last5_n] if r.pts > r.pts_allowed)
        win_rate_last5 = last5_wins / last5_n

        # Scoring average & consistency
        avg_pts_val = float(np.mean(pts))
        pts_std_val = float(np.std(pts))
        # NOTE: consistency_score removed — r=-0.975 vs pts_std (algebraically derived from it)

        features = {
            "win_rate": win_rate,
            "avg_pts": avg_pts_val,
            "avg_pts_allowed": float(np.mean(pts_allowed)),
            # avg_net_pts removed (r=0.952 vs avg_net_rtg)
            "avg_reb": float(np.mean(reb)),
            "avg_ast": float(np.mean(ast)),
            "avg_stl": float(np.mean(stl)),
            "avg_blk": float(np.mean(blk)),
            # avg_tov removed (r=0.966 vs avg_tov_pct)
            "fg_pct": total_fgm / total_fga if total_fga > 0 else 0.0,
            "fg3_pct": total_fg3m / total_fg3a if total_fg3a > 0 else 0.0,
            "ft_pct": total_ftm / total_fta if total_fta > 0 else 0.0,
            "avg_ts_pct": float(np.mean(ts_pct_vals)) if ts_pct_vals else 0.0,
            # avg_efg_pct removed (r=0.964 vs avg_ts_pct)
            "avg_off_rtg": float(np.mean(off_rtg_vals)) if off_rtg_vals else 0.0,
            "avg_def_rtg": float(np.mean(def_rtg_vals)) if def_rtg_vals else 0.0,
            "avg_net_rtg": float(np.mean(net_rtg_vals)) if net_rtg_vals else 0.0,
            "avg_pace": float(np.mean(pace_vals)) if pace_vals else 0.0,
            "avg_tov_pct": float(np.mean(tov_pct_vals)) if tov_pct_vals else 0.0,
            "avg_ftr": float(np.mean(ftr_vals)) if ftr_vals else 0.0,
            "avg_ast_pct": float(np.mean(ast_pct_vals)) if ast_pct_vals else 0.0,
            "trend_pts": trend_pts,
            "trend_net": trend_net,
            "trend_off_rtg": trend_off_rtg,
            "trend_def_rtg": trend_def_rtg,
            "last5_vs_first5_pts_diff": last5_vs_first5_pts_diff,
            "last5_vs_first5_net_diff": last5_vs_first5_net_diff,
            "win_rate_last5": win_rate_last5,
            "pts_std": pts_std_val,
            "net_pts_std": float(np.std(net_pts)),
            # consistency_score removed (r=-0.975 vs pts_std, algebraically derived)
            "games_available": float(n),
        }

        return features


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute linear regression slope safely; returns 0.0 if no variance."""
    if len(x) < 2 or np.std(y) == 0:
        return 0.0
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
