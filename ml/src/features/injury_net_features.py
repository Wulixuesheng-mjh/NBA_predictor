"""Rotation-based net injury and replacement features.

This module compares projected minutes with the minutes a player actually held
in the team's recent complete rotation.  It never subtracts points from the
final prediction; the deltas are ordinary model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory
from ml.src.features.player_features import INJURY_STATUS_WEIGHT, PlayerFeatures, _parse_minutes

INJURY_NET_FEATURE_NAMES = [
    "injury_net_offense_delta",
    "injury_net_defense_events_delta",
    "injury_replacement_minutes",
    "injury_newly_lost_minutes",
    "injury_position_gap_minutes",
    "injury_guard_gap_minutes",
    "injury_forward_gap_minutes",
    "injury_center_gap_minutes",
    "injury_transferred_role_minutes_share",
    "injury_estimated_minutes_share",
    "injury_net_available",
    "injury_report_complete",
]


@dataclass(frozen=True)
class RotationPlayer:
    player_id: int
    position: str
    reference_minutes: float
    role_minutes: float
    availability: float
    points_per_minute: float | None
    defense_events_per_minute: float | None
    transferred_role: bool = False
    production_observed: bool = True


def _position_group(position: str | None) -> str:
    value = (position or "").upper()
    if "G" in value:
        return "G"
    if "F" in value:
        return "F"
    if "C" in value:
        return "C"
    return "UNK"


def project_rotation(players: list[RotationPlayer], total_minutes: float = 240.0) -> dict[int, float] | None:
    """Water-fill available roles to exactly ``total_minutes``.

    Returns ``None`` when listed players do not have enough availability
    capacity, which prevents silently inventing replacement players.
    """
    if not players:
        return None
    capacity = sum(48.0 * max(0.0, min(1.0, p.availability)) for p in players)
    if capacity + 1e-6 < total_minutes:
        return None

    def allocated(scale: float) -> float:
        return sum(
            min(48.0 * p.availability, scale * p.availability * max(p.role_minutes, 0.0))
            for p in players
        )

    low, high = 0.0, 1.0
    while allocated(high) < total_minutes:
        high *= 2.0
        if high > 1e6:
            return None
    for _ in range(80):
        mid = (low + high) / 2.0
        if allocated(mid) < total_minutes:
            low = mid
        else:
            high = mid
    scale = high
    projected = {
        p.player_id: min(48.0 * p.availability, scale * p.availability * max(p.role_minutes, 0.0))
        for p in players
    }
    # Correct floating-point drift on players with remaining capacity.
    remainder = total_minutes - sum(projected.values())
    if abs(remainder) > 1e-7:
        for p in sorted(players, key=lambda x: x.role_minutes, reverse=True):
            room = 48.0 * p.availability - projected[p.player_id]
            change = min(max(remainder, 0.0), max(room, 0.0))
            projected[p.player_id] += change
            remainder -= change
            if abs(remainder) <= 1e-7:
                break
    return projected


def summarize_rotation(players: list[RotationPlayer], projected: dict[int, float]) -> dict[str, float]:
    net_offense = 0.0
    net_defense = 0.0
    replacement = 0.0
    newly_lost = 0.0
    missing_by_position: dict[str, float] = {}
    estimated_minutes = 0.0
    transferred_minutes = 0.0
    for player in players:
        new_minutes = projected.get(player.player_id, 0.0)
        delta = new_minutes - player.reference_minutes
        if player.points_per_minute is not None:
            net_offense += delta * player.points_per_minute
            if player.production_observed:
                estimated_minutes += new_minutes
        if player.defense_events_per_minute is not None:
            net_defense += delta * player.defense_events_per_minute
        if delta > 0:
            replacement += delta
        elif delta < 0:
            newly_lost += -delta
            position_group = _position_group(player.position)
            missing_by_position[position_group] = (
                missing_by_position.get(position_group, 0.0) - delta
            )
        if player.transferred_role:
            transferred_minutes += new_minutes
    position_gap = max(missing_by_position.values(), default=0.0)
    return {
        "injury_net_offense_delta": net_offense,
        "injury_net_defense_events_delta": net_defense,
        "injury_replacement_minutes": replacement,
        "injury_newly_lost_minutes": newly_lost,
        "injury_position_gap_minutes": position_gap,
        "injury_guard_gap_minutes": missing_by_position.get("G", 0.0),
        "injury_forward_gap_minutes": missing_by_position.get("F", 0.0),
        "injury_center_gap_minutes": missing_by_position.get("C", 0.0),
        "injury_transferred_role_minutes_share": transferred_minutes / 240.0,
        "injury_estimated_minutes_share": estimated_minutes / 240.0,
        "injury_net_available": 1.0,
        "injury_report_complete": 1.0,
    }


class InjuryNetFeatures:
    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory
        self.player_features = PlayerFeatures(session_factory)
        self._position_prior_cache: dict[str, dict[str, tuple[float, float]]] = {}

    async def _position_priors(
        self, sess: AsyncSession, before_date: datetime
    ) -> dict[str, tuple[float, float]]:
        """Previous-season points/defensive-events per minute by position."""
        from app.models.game import Game
        from app.models.game_player_stats import GamePlayerStats
        from app.models.player import Player

        start_year = before_date.year if before_date.month >= 10 else before_date.year - 1
        previous_season = f"{start_year - 1}-{str(start_year)[-2:]}"
        cached = self._position_prior_cache.get(previous_season)
        if cached is not None:
            return cached
        rows = (await sess.execute(
            select(GamePlayerStats, Player.position)
            .join(Game, GamePlayerStats.game_id == Game.id)
            .join(Player, GamePlayerStats.player_id == Player.id)
            .where(
                Game.season == previous_season,
                Game.game_date < before_date,
                Game.status == "Final",
                Game.game_type.in_(("regular", "play_in", "playoffs")),
                Game.player_stats_complete.is_(True),
            )
        )).all()
        totals: dict[str, list[float]] = {}
        for stat, position in rows:
            minutes = _parse_minutes(stat.minutes) or 0.0
            if minutes <= 0:
                continue
            for group in (_position_group(position), "ALL"):
                bucket = totals.setdefault(group, [0.0, 0.0, 0.0])
                bucket[0] += float(stat.pts or 0)
                bucket[1] += float((stat.stl or 0) + (stat.blk or 0))
                bucket[2] += minutes
        priors = {
            group: (values[0] / values[2], values[1] / values[2])
            for group, values in totals.items() if values[2] > 0
        }
        self._position_prior_cache[previous_season] = priors
        return priors

    @staticmethod
    def unavailable(report_complete: bool = False) -> dict[str, float]:
        values = {name: 0.0 for name in INJURY_NET_FEATURE_NAMES}
        values["injury_report_complete"] = 1.0 if report_complete else 0.0
        return values

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        *,
        session: AsyncSession | None = None,
        use_historical: bool = False,
        target_game_id: int | None = None,
        diagnostics: dict | None = None,
    ) -> dict[str, float]:
        from app.models.game import Game
        from app.models.game_player_stats import GamePlayerStats
        from app.services.injury_snapshot_reader import latest_injury_snapshot, snapshot_statuses
        from app.models.player import Player
        from app.models.roster_snapshot import RosterSnapshot

        async def _query(sess: AsyncSession) -> dict[str, float]:
            def unavailable(reason, complete=False):
                if diagnostics is not None:
                    diagnostics['unavailable_reason'] = reason
                return self.unavailable(complete)

            report = await latest_injury_snapshot(
                sess, target_game_id, team_id, before_date, reconstructed=use_historical,
            )
            report_complete = bool(report and report.is_complete)
            if diagnostics is not None:
                diagnostics.update(report_id=report.id if report else None,
                    reconstructed=use_historical, report_complete=report_complete)
            if not report_complete:
                return unavailable('partial_report' if report else 'no_eligible_report')

            complete_games = list((await sess.execute(
                select(Game)
                .where(
                    Game.status == "Final",
                    Game.game_type.in_(("regular", "play_in", "playoffs")),
                    Game.player_stats_complete.is_(True),
                    Game.game_date < before_date,
                    or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                )
                .order_by(desc(Game.game_date))
                .limit(10)
            )).scalars())
            if len(complete_games) < 5:
                return unavailable('fewer_than_five_reference_games', True)
            incomplete_after_reference = await sess.scalar(
                select(Game.id)
                .where(
                    Game.status == "Final",
                    Game.game_type.in_(("regular", "play_in", "playoffs")),
                    Game.player_stats_complete.is_(False),
                    Game.game_date < before_date,
                    Game.game_date > complete_games[-1].game_date,
                    or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                )
                .limit(1)
            )
            if incomplete_after_reference is not None:
                return unavailable('intervening_incomplete_stats', True)

            game_ids = [game.id for game in complete_games]
            stat_rows = list((await sess.execute(
                select(GamePlayerStats, Game.game_date)
                .join(Game, GamePlayerStats.game_id == Game.id)
                .where(GamePlayerStats.game_id.in_(game_ids), GamePlayerStats.team_id == team_id)
            )).all())

            decay = {game.id: 0.85 ** index for index, game in enumerate(complete_games)}
            ref_sum: dict[int, float] = {}
            ref_weight = sum(decay.values())
            by_game: dict[int, list] = {}
            for row, _ in stat_rows:
                by_game.setdefault(row.game_id, []).append(row)
            for game in complete_games:
                rows = by_game.get(game.id, [])
                total = sum(_parse_minutes(row.minutes) or 0.0 for row in rows)
                if total <= 0:
                    return unavailable('invalid_reference_minutes', True)
                scale = 240.0 / total
                for row in rows:
                    ref_sum[row.player_id] = ref_sum.get(row.player_id, 0.0) + (
                        (_parse_minutes(row.minutes) or 0.0) * scale * decay[game.id]
                    )

            roster_ids = set(ref_sum)
            snapshot_ids = set((await sess.execute(
                select(RosterSnapshot.player_id).where(
                    RosterSnapshot.team_id == team_id,
                    RosterSnapshot.effective_from <= before_date,
                    or_(RosterSnapshot.effective_until.is_(None), RosterSnapshot.effective_until > before_date),
                )
            )).scalars())
            if snapshot_ids:
                roster_ids |= snapshot_ids
            else:
                # Current-team fallback is allowed only for live prediction;
                # reconstructed history keeps the causally safe roster formed
                # by the prior ten complete box scores plus the report itself.
                # It must never use today's Player.team_id for an old game.
                if not use_historical:
                    roster_ids |= set((await sess.execute(
                        select(Player.id).where(Player.team_id == team_id)
                    )).scalars())

            availability = {pid: 1.0 - INJURY_STATUS_WEIGHT[status.lower()]
                            for pid, status in snapshot_statuses(report).items()}
            roster_ids |= set(availability)

            positions = dict((await sess.execute(
                select(Player.id, Player.position).where(Player.id.in_(roster_ids))
            )).all())

            history_start = before_date - timedelta(days=365)
            production_rows = list((await sess.execute(
                select(GamePlayerStats, Game.game_date)
                .join(Game, GamePlayerStats.game_id == Game.id)
                .where(
                    GamePlayerStats.player_id.in_(roster_ids),
                    Game.game_date < before_date,
                    Game.game_date >= history_start,
                    Game.status == "Final",
                    Game.game_type.in_(("regular", "play_in", "playoffs")),
                    Game.player_stats_complete.is_(True),
                )
                .order_by(desc(Game.game_date))
            )).all())
            per_player: dict[int, list[tuple[object, datetime]]] = {}
            role_history: dict[int, list] = {}
            for row, played_at in production_rows:
                if (_parse_minutes(row.minutes) or 0.0) <= 0:
                    continue
                if row.team_id == team_id and played_at >= before_date-timedelta(days=180):
                    if len(role_history.setdefault(row.player_id, [])) < 10:
                        role_history[row.player_id].append(row)
                if len(per_player.setdefault(row.player_id, [])) < 20:
                    per_player[row.player_id].append((row, played_at))

            start_year = before_date.year if before_date.month >= 10 else before_date.year-1
            previous_season = f'{start_year-1}-{str(start_year)[-2:]}'
            previous_rows = (await sess.execute(select(GamePlayerStats).join(
                Game, GamePlayerStats.game_id == Game.id,
            ).where(GamePlayerStats.player_id.in_(roster_ids), Game.season == previous_season,
                Game.game_date < before_date, Game.status == 'Final', Game.player_stats_complete.is_(True),
                Game.game_type.in_(('regular','play_in','playoffs')),
            ).order_by(desc(Game.game_date)))).scalars().all()
            previous_roles: dict[int, list] = {}
            for row in previous_rows:
                if (_parse_minutes(row.minutes) or 0.0) > 0 and len(previous_roles.setdefault(row.player_id, [])) < 20:
                    previous_roles[row.player_id].append(row)

            position_priors = await self._position_priors(sess, before_date)
            players: list[RotationPlayer] = []
            for player_id in roster_ids:
                appearances = per_player.get(player_id, [])
                role_cutoff = before_date - timedelta(days=180)
                current_team = role_history.get(player_id, [])
                fallback_rows = previous_roles.get(player_id, [])
                if not fallback_rows:
                    # New transfers without previous-season games may use
                    # recent old-team appearances, never full-career means.
                    fallback_rows = [row for row, played_at in appearances
                        if row.team_id != team_id and played_at >= role_cutoff][:10]
                if len(current_team) >= 5:
                    role_minutes = float(np.mean([
                        _parse_minutes(row.minutes) or 0.0 for row in current_team
                    ]))
                elif fallback_rows:
                    fallback_mean = float(np.mean([
                        _parse_minutes(row.minutes) or 0.0 for row in fallback_rows
                    ]))
                    prior_games = 5 - len(current_team)
                    role_minutes = (
                        sum(_parse_minutes(row.minutes) or 0.0 for row in current_team)
                        + prior_games * fallback_mean
                    ) / 5.0
                else:
                    role_minutes = 0.0
                transferred_role = bool(
                    len(current_team) < 5
                    and any(row.team_id != team_id for row in fallback_rows)
                )
                production = [row for row, _ in appearances]
                played_minutes = sum(_parse_minutes(row.minutes) or 0.0 for row in production)
                group = _position_group(positions.get(player_id))
                prior = position_priors.get(group) or position_priors.get("ALL")
                if prior is not None:
                    ppm = (sum(row.pts for row in production) + 200.0 * prior[0]) / (played_minutes + 200.0)
                    dpm = (
                        sum(row.stl + row.blk for row in production) + 200.0 * prior[1]
                    ) / (played_minutes + 200.0)
                else:
                    ppm = sum(row.pts for row in production) / played_minutes if played_minutes > 0 else None
                    dpm = sum(row.stl + row.blk for row in production) / played_minutes if played_minutes > 0 else None
                players.append(RotationPlayer(
                    player_id=player_id,
                    position=positions.get(player_id) or "UNK",
                    reference_minutes=ref_sum.get(player_id, 0.0) / ref_weight,
                    role_minutes=role_minutes,
                    availability=availability.get(player_id, 1.0),
                    points_per_minute=ppm,
                    defense_events_per_minute=dpm,
                    transferred_role=transferred_role,
                    production_observed=played_minutes > 0,
                ))

            projected = project_rotation(players)
            if projected is None:
                return unavailable('insufficient_rotation_capacity', True)
            summary = summarize_rotation(players, projected)
            if diagnostics is not None:
                diagnostics.update(projected_minutes=sum(projected.values()),
                    historical_minutes_coverage=summary['injury_estimated_minutes_share'])
            if summary["injury_estimated_minutes_share"] < 0.8:
                return unavailable('historical_minutes_coverage_below_80pct', True)
            return summary

        if session is not None:
            return await _query(session)
        async with self.session_factory() as sess:
            return await _query(sess)
