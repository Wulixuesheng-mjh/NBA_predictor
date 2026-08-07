"""Opponent-adjusted team statistics.

Raw team stats (avg_pts, avg_net_rtg, etc.) don't account for opponent quality.
A team averaging 120 PPG against weak defenses is overrated by raw stats alone.

This module adjusts team metrics by the quality of opponents faced:
  - opp_adj_pts:           Actual PPG / opponent average points allowed
  - opp_adj_pts_allowed:   Actual points allowed / opponent average PPG
  - opp_adj_net_rtg:       Net rating adjusted for opponent strength
  - opp_adj_off_rtg:       Offensive rating vs opponent average defensive rating
  - opp_adj_def_rtg:       Defensive rating vs opponent average offensive rating
  - sos_rating:            Average opponent win rate (strength of schedule)

Performance: During training, use precompute_opponent_cache() to build an in-memory
dict of all (team, season) → stats in one pass, eliminating the N+1 query pattern.
"""

import logging
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

OPPONENT_ADJUSTED_FEATURE_NAMES = [
    "opp_adj_pts",
    "opp_adj_pts_allowed",
    "opp_adj_net_rtg",
    "opp_adj_off_rtg",
    "opp_adj_def_rtg",
    "sos_rating",
]


class OpponentAdjustedFeatures:
    """Compute opponent-adjusted team statistics over a recent window."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    # ── Public API ──────────────────────────────────────────────────────

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
        opponent_cache: dict[tuple[int, str], dict] | None = None,
    ) -> dict[str, float]:
        """Compute opponent-adjusted stats for a team.

        For each of the team's last N games before `before_date`:
        1. Find the opponent
        2. Look up the opponent's season-average stats (from cache if provided)
        3. Compare team's performance to opponent's typical numbers

        Args:
            team_id: Internal team ID.
            before_date: Reference date (only games before this are used).
            n_games: Number of recent games to look back.
            session: Optional existing DB session.
            opponent_cache: Optional precomputed dict[(team_id, season), stats].
                            When provided, skips all per-game DB queries for
                            opponent stats — huge speedup for training.

        Returns:
            Dict with opp_adj_* feature values.
        """
        async def _compute(sess: AsyncSession) -> dict[str, float]:
            from app.models.game import Game

            # ── Get the team's last N games ──
            team_games_query = (
                select(Game)
                .where(
                    and_(
                        Game.game_date < before_date,
                        Game.status == "Final",
                        Game.home_score.isnot(None),
                        Game.away_score.isnot(None),
                        (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                    )
                )
                .order_by(Game.game_date.desc())
                .limit(n_games)
            )
            result = await sess.execute(team_games_query)
            team_games = list(result.scalars().all())

            if not team_games:
                return self._default_result()

            # ── Compute opponent quality per game ──
            opp_adj_pts_list: list[float] = []
            opp_adj_pts_allowed_list: list[float] = []
            opp_adj_net_rtg_list: list[float] = []
            opp_adj_off_rtg_list: list[float] = []
            opp_adj_def_rtg_list: list[float] = []
            opp_win_rates: list[float] = []

            for game in team_games:
                is_home = game.home_team_id == team_id
                opp_id = game.away_team_id if is_home else game.home_team_id

                team_pts = game.home_score if is_home else game.away_score
                opp_pts = game.away_score if is_home else game.home_score

                # ── Opponent season stats (cache or DB fallback) ──
                if opponent_cache is not None:
                    cache_key = (opp_id, game.season or "")
                    opp_stats = opponent_cache.get(cache_key)
                else:
                    opp_stats = await self._get_opponent_season_stats(
                        sess, opp_id, game.game_date
                    )

                if opp_stats is None:
                    continue

                opp_avg_pts_allowed = opp_stats["avg_pts_allowed"]
                opp_avg_pts = opp_stats["avg_pts"]
                opp_avg_off_rtg = opp_stats["avg_off_rtg"]
                opp_avg_def_rtg = opp_stats["avg_def_rtg"]
                opp_win_rate = opp_stats["win_rate"]

                # Opponent-adjusted ratios (>1 = better than opponent typically allows)
                if opp_avg_pts_allowed > 0:
                    opp_adj_pts_list.append(team_pts / opp_avg_pts_allowed)
                if opp_avg_pts > 0:
                    opp_adj_pts_allowed_list.append(opp_pts / opp_avg_pts)

                # Net rating adjustment
                raw_net = team_pts - opp_pts
                opp_avg_net = opp_avg_off_rtg - opp_avg_def_rtg
                opp_adj_net_rtg_list.append(raw_net - opp_avg_net)

                if opp_avg_def_rtg > 0:
                    opp_adj_off_rtg_list.append(team_pts / opp_avg_def_rtg)
                else:
                    opp_adj_off_rtg_list.append(1.0)

                if opp_avg_off_rtg > 0:
                    opp_adj_def_rtg_list.append(opp_pts / opp_avg_off_rtg)
                else:
                    opp_adj_def_rtg_list.append(1.0)

                opp_win_rates.append(opp_win_rate)

            # ── Average across the window ──
            def _mean(lst: list[float]) -> float:
                return sum(lst) / len(lst) if lst else 0.0

            return {
                "opp_adj_pts": round(_mean(opp_adj_pts_list), 4),
                "opp_adj_pts_allowed": round(_mean(opp_adj_pts_allowed_list), 4),
                "opp_adj_net_rtg": round(_mean(opp_adj_net_rtg_list), 2),
                "opp_adj_off_rtg": round(_mean(opp_adj_off_rtg_list), 4),
                "opp_adj_def_rtg": round(_mean(opp_adj_def_rtg_list), 4),
                "sos_rating": round(_mean(opp_win_rates), 4),
            }

        if session is not None:
            return await _compute(session)
        else:
            async with self.session_factory() as sess:
                return await _compute(sess)

    # ── Cache precomputation (bulk, for training) ───────────────────────

    @staticmethod
    async def precompute_opponent_cache(
        session: AsyncSession,
    ) -> dict[tuple[int, str], dict]:
        """Precompute per-team per-season opponent quality stats in ONE pass.

        Scans all completed games once, aggregates pts/pts_allowed/wins per
        (team_id, season), and returns a dict suitable for O(1) lookup during
        training.  This eliminates the N+1 query pattern: without it,
        build_training_dataset() fires ~600K individual opponent-stat queries;
        with it, the entire cache costs 1 query + 1 in-memory pass.

        Returns:
            dict[(team_id, season_str), {
                "avg_pts": float,
                "avg_pts_allowed": float,
                "avg_off_rtg": float,
                "avg_def_rtg": float,
                "win_rate": float,
            }]
        """
        from app.models.game import Game

        query = select(Game).where(
            Game.status == "Final",
            Game.home_score.isnot(None),
            Game.away_score.isnot(None),
        )
        result = await session.execute(query)
        all_games = result.scalars().all()

        # Aggregate: one row per (team, season)
        agg: dict[tuple[int, str], dict[str, float]] = defaultdict(
            lambda: {"pts": 0.0, "pts_allowed": 0.0, "wins": 0, "n": 0}
        )

        for g in all_games:
            season = g.season or "unknown"

            # Home team
            hkey = (g.home_team_id, season)
            agg[hkey]["pts"] += float(g.home_score)
            agg[hkey]["pts_allowed"] += float(g.away_score)
            agg[hkey]["n"] += 1
            if g.home_score > g.away_score:
                agg[hkey]["wins"] += 1

            # Away team
            akey = (g.away_team_id, season)
            agg[akey]["pts"] += float(g.away_score)
            agg[akey]["pts_allowed"] += float(g.home_score)
            agg[akey]["n"] += 1
            if g.away_score > g.home_score:
                agg[akey]["wins"] += 1

        cache: dict[tuple[int, str], dict] = {}
        for (tid, season), s in agg.items():
            n = s["n"]
            if n == 0:
                continue
            cache[(tid, season)] = {
                "avg_pts": round(s["pts"] / n, 2),
                "avg_pts_allowed": round(s["pts_allowed"] / n, 2),
                "avg_off_rtg": round(s["pts"] / n, 2),
                "avg_def_rtg": round(s["pts_allowed"] / n, 2),
                "win_rate": round(s["wins"] / n, 4),
            }

        logger.info(
            "Precomputed opponent cache: %d (team, season) entries from %d games",
            len(cache),
            len(all_games),
        )
        return cache

    # ── Fallback: per-player lookup (used during single-game prediction) ─

    @staticmethod
    async def _get_opponent_season_stats(
        session: AsyncSession,
        team_id: int,
        before_date: datetime,
    ) -> dict | None:
        """Compute a team's season-to-date averages (as of before_date).

        Used as a fallback during single-game prediction when no precomputed
        cache is available.  For training, prefer precompute_opponent_cache().
        """
        from app.models.game import Game

        games_query = (
            select(Game)
            .where(
                and_(
                    Game.game_date < before_date,
                    Game.status == "Final",
                    Game.home_score.isnot(None),
                    Game.away_score.isnot(None),
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                )
            )
        )
        result = await session.execute(games_query)
        games = result.scalars().all()

        if not games:
            return None

        total_pts = 0
        total_pts_allowed = 0
        wins = 0
        n = len(games)

        for g in games:
            is_home = g.home_team_id == team_id
            pts = g.home_score if is_home else g.away_score
            pts_allowed = g.away_score if is_home else g.home_score
            total_pts += pts
            total_pts_allowed += pts_allowed
            if pts > pts_allowed:
                wins += 1

        avg_pts = total_pts / n
        avg_pts_allowed = total_pts_allowed / n
        avg_off_rtg = avg_pts
        avg_def_rtg = avg_pts_allowed
        win_rate = wins / n

        return {
            "avg_pts": avg_pts,
            "avg_pts_allowed": avg_pts_allowed,
            "avg_off_rtg": avg_off_rtg,
            "avg_def_rtg": avg_def_rtg,
            "win_rate": win_rate,
        }

    @staticmethod
    def _default_result() -> dict[str, float]:
        return {
            "opp_adj_pts": 1.0,
            "opp_adj_pts_allowed": 1.0,
            "opp_adj_net_rtg": 0.0,
            "opp_adj_off_rtg": 0.0,
            "opp_adj_def_rtg": 0.0,
            "sos_rating": 0.5,
        }
