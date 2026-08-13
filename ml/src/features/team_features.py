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

# League average total points — used to normalize over/under features
# (matches the league_avg_total constant in predictor.py's consensus logic)
LEAGUE_AVG_TOTAL = 227.5

# Exponential decay factor for rolling averages (Task B):
# recent games are weighted more heavily than older games.
# weight of game i (0 = most recent) is DECAY ** i.
DECAY = 0.85

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
    # Total points (over/under) specific — Task A2
    "total_trend",             # linear slope of (pts + pts_allowed) — momentum toward over/under
    "over_league_avg_ratio",   # fraction of recent games over league average total
    "avg_total_last3",         # mean total points over last 3 games
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

        # ── Exponential decay weights (Task B) ──
        # rows are in DESC order (index 0 = most recent). Weight recent games
        # more heavily: weight of game i is DECAY ** i, normalized to sum 1.
        raw_weights = np.array([DECAY ** i for i in range(n)])
        weights = raw_weights / raw_weights.sum()

        # Extract arrays
        pts = np.array([r.pts for r in rows])
        pts_allowed = np.array([r.pts_allowed for r in rows])
        net_pts = pts - pts_allowed
        reb = np.array([r.reb for r in rows])
        ast = np.array([r.ast for r in rows])
        stl = np.array([r.stl for r in rows])
        blk = np.array([r.blk for r in rows])
        # NOTE: tov array no longer extracted — avg_tov removed (redundant with avg_tov_pct)

        # Shooting: weighted aggregate sums then divide (recent games count more)
        # raw_weights (unnormalized) are fine for ratios — they cancel.
        total_fgm = float(np.sum(np.array([r.fgm for r in rows]) * raw_weights))
        total_fga = float(np.sum(np.array([r.fga for r in rows]) * raw_weights))
        total_fg3m = float(np.sum(np.array([r.fg3m for r in rows]) * raw_weights))
        total_fg3a = float(np.sum(np.array([r.fg3a for r in rows]) * raw_weights))
        total_ftm = float(np.sum(np.array([r.ftm for r in rows]) * raw_weights))
        total_fta = float(np.sum(np.array([r.fta for r in rows]) * raw_weights))

        # Win rate — weighted by recency
        wins = float(np.sum(raw_weights[np.array([r.pts > r.pts_allowed for r in rows])]))
        win_rate = wins / raw_weights.sum()

        # Advanced stats averages (handle None) — with matching decay weights
        off_rtg_vals: list[float] = []
        off_rtg_w: list[float] = []
        def_rtg_vals: list[float] = []
        def_rtg_w: list[float] = []
        pace_vals: list[float] = []
        pace_w: list[float] = []
        ts_pct_vals: list[float] = []
        ts_pct_w: list[float] = []
        # NOTE: efg_pct_vals removed — avg_efg_pct redundant with avg_ts_pct (r=0.964)
        tov_pct_vals: list[float] = []
        tov_pct_w: list[float] = []
        ast_pct_vals: list[float] = []
        ast_pct_w: list[float] = []
        # FTR = FTA / FGA per game (computed on-the-fly)
        ftr_vals: list[float] = []
        ftr_w: list[float] = []
        # Net RTG per game (off - def), only when both present
        net_rtg_vals: list[float] = []
        net_rtg_w: list[float] = []

        for i, r in enumerate(rows):
            w = raw_weights[i]
            if r.off_rtg is not None:
                off_rtg_vals.append(r.off_rtg)
                off_rtg_w.append(w)
            if r.def_rtg is not None:
                def_rtg_vals.append(r.def_rtg)
                def_rtg_w.append(w)
            if r.pace is not None:
                pace_vals.append(r.pace)
                pace_w.append(w)
            if r.ts_pct is not None:
                ts_pct_vals.append(r.ts_pct)
                ts_pct_w.append(w)
            if r.tov_pct is not None:
                tov_pct_vals.append(r.tov_pct)
                tov_pct_w.append(w)
            if r.ast_pct is not None:
                ast_pct_vals.append(r.ast_pct)
                ast_pct_w.append(w)
            if r.fga > 0:
                ftr_vals.append(r.fta / r.fga)
                ftr_w.append(w)
            if r.off_rtg is not None and r.def_rtg is not None:
                net_rtg_vals.append(r.off_rtg - r.def_rtg)
                net_rtg_w.append(w)

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

        # ── Total points (over/under) features — Task A2 ──
        # game totals = own points + opponent points
        total_pts = pts + pts_allowed
        # Momentum: linear slope of game totals over the window (chronological)
        total_trend = _safe_slope(x, total_pts[::-1])
        # How often the team's games exceed the league average total
        over_league_avg_ratio = float(np.mean(total_pts > LEAGUE_AVG_TOTAL))
        # Short-term total (last 3 games) — recent over/under tendency
        last3_n = min(3, n)
        avg_total_last3 = float(np.mean(total_pts[:last3_n]))

        # Scoring average & consistency (weighted by recency)
        avg_pts_val = float(np.average(pts, weights=weights))
        pts_std_val = float(np.std(pts))
        # NOTE: consistency_score removed — r=-0.975 vs pts_std (algebraically derived from it)

        features = {
            "win_rate": win_rate,
            "avg_pts": avg_pts_val,
            "avg_pts_allowed": float(np.average(pts_allowed, weights=weights)),
            # avg_net_pts removed (r=0.952 vs avg_net_rtg)
            "avg_reb": float(np.average(reb, weights=weights)),
            "avg_ast": float(np.average(ast, weights=weights)),
            "avg_stl": float(np.average(stl, weights=weights)),
            "avg_blk": float(np.average(blk, weights=weights)),
            # avg_tov removed (r=0.966 vs avg_tov_pct)
            "fg_pct": total_fgm / total_fga if total_fga > 0 else 0.0,
            "fg3_pct": total_fg3m / total_fg3a if total_fg3a > 0 else 0.0,
            "ft_pct": total_ftm / total_fta if total_fta > 0 else 0.0,
            "avg_ts_pct": float(np.average(ts_pct_vals, weights=ts_pct_w)) if ts_pct_vals else 0.0,
            # avg_efg_pct removed (r=0.964 vs avg_ts_pct)
            "avg_off_rtg": float(np.average(off_rtg_vals, weights=off_rtg_w)) if off_rtg_vals else 0.0,
            "avg_def_rtg": float(np.average(def_rtg_vals, weights=def_rtg_w)) if def_rtg_vals else 0.0,
            "avg_net_rtg": float(np.average(net_rtg_vals, weights=net_rtg_w)) if net_rtg_vals else 0.0,
            "avg_pace": float(np.average(pace_vals, weights=pace_w)) if pace_vals else 0.0,
            "avg_tov_pct": float(np.average(tov_pct_vals, weights=tov_pct_w)) if tov_pct_vals else 0.0,
            "avg_ftr": float(np.average(ftr_vals, weights=ftr_w)) if ftr_vals else 0.0,
            "avg_ast_pct": float(np.average(ast_pct_vals, weights=ast_pct_w)) if ast_pct_vals else 0.0,
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
            "total_trend": total_trend,
            "over_league_avg_ratio": over_league_avg_ratio,
            "avg_total_last3": avg_total_last3,
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
