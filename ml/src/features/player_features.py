"""Player availability and injury impact features.

Computes firepower loss due to injured/absent players using the
minutes-weighted formula:

    firepower_loss = Σ(absent_player_avg_pts × avg_minutes / 48)

This properly weights player importance: a 20-ppg starter playing 36 min
counts more than a 20-ppg sixth man playing 20 min.

Injury priority for predictions (from plan):
    1. source='manual' (user manually added)
    2. confirmation='modified' (user reviewed and changed)
    3. confirmation='confirmed' (user confirmed)
    4. confirmation='pending' (auto-fetched, unreviewed)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select, and_, func, case, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

PLAYER_FEATURE_NAMES = [
    "injury_firepower_loss",
    "total_minutes_share_lost",
    "key_player_out",
    "defensive_anchor_out",
    "players_out_count",
    "questionable_count",
    # Task B3: separate offensive/defensive loss and probable-but-playing
    "injury_defense_lost",
    "injury_probable_count",
]

# How much of a player's production is expected to be missing for each status.
# Previously Out and Doubtful were treated identically and Questionable/Probable
# were dropped entirely, which understated the impact of a game-time decision.
INJURY_STATUS_WEIGHT: dict[str, float] = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.4,
    "probable": 0.05,
    "available": 0.0,
}

# Player production is estimated from games inside this window (projected
# minutes / usage), falling back to season-to-date when there are none.
RECENT_WINDOW_DAYS = 45


@dataclass
class InjuryInfo:
    """Aggregated injury information for a player."""

    player_id: int
    injury_status: str  # Out / Doubtful / Questionable / Probable
    priority: int  # 1=manual, 2=modified, 3=confirmed, 4=pending
    season_ppg: float
    season_mpg: float  # minutes per game
    season_spg: float  # steals per game
    season_bpg: float  # blocks per game


class PlayerFeatures:
    """Computes player availability and injury impact features."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute_injury_impact(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
        use_historical: bool = False,
        target_game_id: int | None = None,
    ) -> dict[str, float]:
        """Compute injury firepower loss for a team.

        Args:
            team_id: Internal team ID.
            before_date: Reference date for injury status.
            session: Optional existing session.
            use_historical: If True, read from ``historical_player_injuries``
                (daily snapshots, used during training); if False (default),
                read from the live ``player_injuries`` table (prediction).

        Returns:
            Dict mapping feature_name -> float.
        """
        injuries = await self._query_active_injuries(
            team_id, before_date, session, use_historical=use_historical,
            target_game_id=target_game_id,
        )
        return self._compute_from_injuries(injuries)

    async def _query_active_injuries(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
        use_historical: bool = False,
        target_game_id: int | None = None,
    ) -> list[InjuryInfo]:
        """Query active injuries for a team, with player season averages.

        Dispatches to the live (``player_injuries``) or historical
        (``historical_player_injuries``) source depending on ``use_historical``.
        """
        if target_game_id is not None:
            from app.services.injury_snapshot_reader import latest_injury_snapshot, snapshot_statuses

            async def _snapshot_query(sess):
                snapshot = await latest_injury_snapshot(
                    sess, target_game_id, team_id, before_date, reconstructed=use_historical,
                )
                statuses = snapshot_statuses(snapshot)
                # Empty payload with a complete report means confirmed healthy;
                # absent/partial reports remain distinguishable via net masks.
                if not statuses:
                    return []
                stats, minutes = await self._compute_season_stats(sess, list(statuses), before_date)
                return [InjuryInfo(
                    player_id=pid, injury_status=status, priority=4,
                    season_ppg=stats.get(pid, {}).get('ppg', 0.0),
                    season_mpg=float(np.mean(minutes[pid])) if minutes.get(pid) else 0.0,
                    season_spg=stats.get(pid, {}).get('spg', 0.0),
                    season_bpg=stats.get(pid, {}).get('bpg', 0.0),
                ) for pid, status in statuses.items()]

            if session is not None:
                return await _snapshot_query(session)
            async with self.session_factory() as sess:
                return await _snapshot_query(sess)
        if use_historical:
            return await self._query_historical(
                team_id, before_date, session, target_game_id=target_game_id
            )
        return await self._query_live(team_id, before_date, session)

    async def _query_live(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
    ) -> list[InjuryInfo]:
        """Live injury path (prediction): ``player_injuries`` table."""
        from app.models.player_injuries import PlayerInjury
        from app.models.player import Player

        async def _query(sess: AsyncSession) -> list[InjuryInfo]:
            # Priority ordering: manual(1) > modified(2) > confirmed(3) > pending(4)
            priority_expr = case(
                (PlayerInjury.source == "manual", 1),
                (PlayerInjury.confirmation == "modified", 2),
                (PlayerInjury.confirmation == "confirmed", 3),
                else_=4,
            ).label("priority")

            # Get active injuries for players on this team
            query = (
                select(
                    PlayerInjury,
                    Player.nba_player_id,
                    priority_expr,
                )
                .join(Player, PlayerInjury.player_id == Player.id)
                .where(
                    and_(
                        or_(
                            PlayerInjury.team_id == team_id,
                            and_(PlayerInjury.team_id.is_(None), Player.team_id == team_id),
                        ),
                        PlayerInjury.is_active == True,
                        PlayerInjury.game_id.is_(None),
                        PlayerInjury.reported_date <= before_date,
                    )
                )
                .order_by(priority_expr.asc(), PlayerInjury.reported_date.desc())
            )
            result = await sess.execute(query)
            injury_rows = result.all()

            if not injury_rows:
                return []

            player_ids = [row.PlayerInjury.player_id for row in injury_rows]
            stats_map, minutes_data = await self._compute_season_stats(
                sess, player_ids, before_date
            )

            injuries = []
            for inj_row in injury_rows:
                inj = inj_row.PlayerInjury
                pid = inj.player_id
                stats = stats_map.get(pid, {"ppg": 0.0, "spg": 0.0, "bpg": 0.0, "games": 0})
                mpg = np.mean(minutes_data.get(pid, [0.0])) if minutes_data.get(pid) else 0.0
                injuries.append(
                    InjuryInfo(
                        player_id=pid,
                        injury_status=inj.injury_status or "Out",
                        priority=int(inj_row.priority),
                        season_ppg=stats["ppg"],
                        season_mpg=mpg,
                        season_spg=stats["spg"],
                        season_bpg=stats["bpg"],
                    )
                )

            return injuries

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    async def _query_historical(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
        target_game_id: int | None = None,
    ) -> list[InjuryInfo]:
        """Historical injury path (training): ``historical_player_injuries``.

        Matches the daily snapshot whose ``game_date`` equals the reference
        date. ``team_id`` comes from the snapshot itself (not ``Player.team_id``,
        because players change teams across seasons).
        """
        from app.models.historical_player_injuries import HistoricalPlayerInjury

        async def _query(sess: AsyncSession) -> list[InjuryInfo]:
            target_date = before_date.date() if hasattr(before_date, "date") else before_date

            filters = [HistoricalPlayerInjury.team_id == team_id]
            # Unknown legacy publication times are not evidence available at
            # this cutoff. Preserve those rows, but do not turn a later report
            # for the same target game into a historical pregame feature.
            filters.append(or_(
                HistoricalPlayerInjury.published_at <= before_date,
                and_(
                    HistoricalPlayerInjury.published_at.is_(None),
                    HistoricalPlayerInjury.observed_at <= before_date,
                ),
            ))
            if target_game_id is not None:
                filters.append(HistoricalPlayerInjury.game_id == target_game_id)
            else:
                filters.append(HistoricalPlayerInjury.game_date == target_date)
            query = (
                select(
                    HistoricalPlayerInjury.player_id,
                    HistoricalPlayerInjury.injury_status,
                )
                .where(and_(*filters))
            )
            result = await sess.execute(query)
            injury_rows = result.all()

            if not injury_rows:
                return []

            player_ids = [row.player_id for row in injury_rows]
            stats_map, minutes_data = await self._compute_season_stats(
                sess, player_ids, before_date
            )

            injuries = []
            for row in injury_rows:
                pid = row.player_id
                stats = stats_map.get(pid, {"ppg": 0.0, "spg": 0.0, "bpg": 0.0, "games": 0})
                mpg = np.mean(minutes_data.get(pid, [0.0])) if minutes_data.get(pid) else 0.0
                injuries.append(
                    InjuryInfo(
                        player_id=pid,
                        injury_status=row.injury_status or "Out",
                        priority=4,  # no confirmation workflow for historical snapshots
                        season_ppg=stats["ppg"],
                        season_mpg=mpg,
                        season_spg=stats["spg"],
                        season_bpg=stats["bpg"],
                    )
                )

            return injuries

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    async def _compute_season_stats(
        self,
        sess: AsyncSession,
        player_ids: list[int],
        before_date: datetime,
    ) -> tuple[dict[int, dict], dict[int, list[float]]]:
        """Compute per-player season PPG/SPG/BPG and minutes from played games.

        Excludes DNP (did-not-play) rows — a player who was Out/DNP has minutes
        NULL or "0", which would otherwise dilute their averages and understate
        the injury's firepower loss.
        """
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game

        played_filter = and_(
            GamePlayerStats.minutes.isnot(None),
            GamePlayerStats.minutes.notin_(["", "0", "0.0", "0:00", "00:00"]),
        )

        # Recent form first (projected minutes / usage), with a season-to-date
        # fallback for players who have not appeared in the recent window
        # (long-term injuries would otherwise register zero impact).
        recent_start = before_date - timedelta(days=RECENT_WINDOW_DAYS)

        async def _fetch(start: datetime | None, ids: list[int]):
            conditions = [
                GamePlayerStats.player_id.in_(ids),
                Game.game_date < before_date,
                Game.status == "Final",
                Game.game_type.in_(("regular", "play_in", "playoffs")),
                Game.player_stats_complete.is_(True),
                played_filter,
            ]
            if start is not None:
                conditions.append(Game.game_date >= start)

            stats_result = await sess.execute(
                select(
                    GamePlayerStats.player_id,
                    GamePlayerStats.pts, GamePlayerStats.stl, GamePlayerStats.blk,
                    GamePlayerStats.minutes,
                )
                .join(Game, GamePlayerStats.game_id == Game.id)
                .where(and_(*conditions))
            )
            return _aggregate_played_rows(stats_result.all())

        stats_map, minutes_data = await _fetch(recent_start, player_ids)

        missing = [pid for pid in player_ids if pid not in stats_map]
        if missing:
            season_start_year = before_date.year if before_date.month >= 10 else before_date.year - 1
            season_start = before_date.replace(
                year=season_start_year, month=10, day=1,
                hour=0, minute=0, second=0, microsecond=0,
            )
            fallback_stats, fallback_minutes = await _fetch(season_start, missing)
            stats_map.update(fallback_stats)
            minutes_data.update(fallback_minutes)

        return stats_map, minutes_data

    @staticmethod
    def _compute_from_injuries(injuries: list[InjuryInfo]) -> dict[str, float]:
        """Compute injury impact features from a list of injury records.

        Uses confirmation-priority dedup: for the same player, only the
        highest-priority (lowest number) injury record is used.

        Each player's production is weighted by the probability that he
        actually misses the game (Out 1.0, Doubtful 0.75, Questionable 0.4,
        Probable 0.05) instead of lumping Out and Doubtful together.
        """
        # Deduplicate: keep highest-priority record per player
        best_per_player: dict[int, InjuryInfo] = {}
        for inj in injuries:
            if inj.player_id not in best_per_player or inj.priority < best_per_player[inj.player_id].priority:
                best_per_player[inj.player_id] = inj

        firepower_loss = 0.0
        defense_loss = 0.0
        total_minutes_share_lost = 0.0
        key_player_out = 0.0
        defensive_anchor_out = 0.0
        players_out = 0
        questionable = 0
        probable = 0

        for inj in best_per_player.values():
            status = (inj.injury_status or "out").lower()
            weight = INJURY_STATUS_WEIGHT.get(status, 1.0)

            if status in ("out", "doubtful"):
                players_out += 1
            elif status == "questionable":
                questionable += 1
            elif status == "probable":
                probable += 1

            if weight <= 0:
                continue

            minutes_share = (inj.season_mpg / 48.0) if inj.season_mpg > 0 else 0.0
            weighted_pts = weight * inj.season_ppg * minutes_share
            firepower_loss += weighted_pts
            defense_loss += weight * (inj.season_spg + inj.season_bpg) * minutes_share
            total_minutes_share_lost += weight * minutes_share

            # Key player: expected scoring loss > 10 points per 48 minutes
            if weighted_pts > 10.0:
                key_player_out = 1.0

            # Defensive anchor: steals + blocks > 2.5 per game
            if weight >= 0.4 and inj.season_spg + inj.season_bpg > 2.5:
                defensive_anchor_out = 1.0

        return {
            "injury_firepower_loss": firepower_loss,
            "total_minutes_share_lost": total_minutes_share_lost,
            "key_player_out": key_player_out,
            "defensive_anchor_out": defensive_anchor_out,
            "players_out_count": float(players_out),
            "questionable_count": float(questionable),
            "injury_defense_lost": defense_loss,
            "injury_probable_count": float(probable),
        }


def _parse_minutes(minutes_str: str | None) -> float | None:
    """Parse minutes from 'MM:SS' format or raw float.

    Returns float minutes or None if unparseable.
    """
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


def _aggregate_played_rows(rows):
    """Exclude every numeric zero/DNP representation, including 0.000000."""
    totals,minutes={},{}
    for row in rows:
        mins=_parse_minutes(row.minutes)
        if mins is None or not np.isfinite(mins) or mins<=0:
            continue
        pid=row.player_id
        bucket=totals.setdefault(pid,[0.0,0.0,0.0,0])
        bucket[0]+=float(row.pts or 0);bucket[1]+=float(row.stl or 0);bucket[2]+=float(row.blk or 0);bucket[3]+=1
        minutes.setdefault(pid,[]).append(mins)
    stats={pid:dict(ppg=v[0]/v[3],spg=v[1]/v[3],bpg=v[2]/v[3],games=v[3]) for pid,v in totals.items()}
    return stats,minutes
