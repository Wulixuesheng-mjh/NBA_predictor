"""Validation and backfill queue for completed-game data."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession


def parse_minutes(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        text = str(value)
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return float(minutes) + float(seconds) / 60.0
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_player_minutes(row) -> float | None:
    """Treat a zero-stat DNP row as zero minutes; reject other missing values."""
    parsed = parse_minutes(row.minutes)
    if parsed is not None:
        return parsed
    zero_fields = ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta")
    if all((getattr(row, name, 0) or 0) == 0 for name in zero_fields):
        return 0.0
    return None


async def bump_data_revision(session: AsyncSession, reason: str) -> int:
    """Increment the explicit cache revision inside the caller's transaction."""
    from app.models.data_revision import DataRevision

    revision = await session.scalar(update(DataRevision).where(DataRevision.id==1).values(
        revision=DataRevision.revision+1,reason=reason[:200],updated_at=datetime.now(timezone.utc),
    ).returning(DataRevision.revision))
    if revision is not None:
        return revision
    row = DataRevision(id=1,revision=1,reason=reason[:200])
    session.add(row)
    await session.flush()
    return row.revision


async def assess_game_data(session: AsyncSession, game_id: int) -> dict:
    """Return strict, explainable completeness state for one game."""
    from app.models.game import Game
    from app.models.team_game_stats import TeamGameStats
    from app.models.game_player_stats import GamePlayerStats

    game = await session.get(Game, game_id)
    if game is None:
        raise ValueError(f"Game {game_id} not found")

    team_rows = list((await session.execute(
        select(TeamGameStats).where(TeamGameStats.game_id == game_id).execution_options(populate_existing=True)
    )).scalars())
    player_rows = list((await session.execute(
        select(GamePlayerStats).where(GamePlayerStats.game_id == game_id).execution_options(populate_existing=True)
    )).scalars())

    score_complete = bool(
        game.status == "Final"
        and game.home_score is not None and game.home_score > 0
        and game.away_score is not None and game.away_score > 0
    )
    team_by_id = {row.team_id: row for row in team_rows}
    team_stats_complete = bool(
        score_complete
        and len(team_rows) == 2
        and set(team_by_id) == {game.home_team_id, game.away_team_id}
        and team_by_id[game.home_team_id].pts == game.home_score
        and team_by_id[game.away_team_id].pts == game.away_score
        and all(row.fga > 0 for row in team_rows)
    )

    issues: list[str] = []
    if not score_complete:
        issues.append("score_incomplete")
    if not team_stats_complete:
        issues.append("team_stats_incomplete")

    player_stats_complete = False
    if team_stats_complete and player_rows:
        player_stats_complete = True
        expected_game_minutes = max(
            [float(row.game_minutes or 48.0) for row in team_rows] or [48.0]
        )
        for team_id, expected_points in (
            (game.home_team_id, game.home_score),
            (game.away_team_id, game.away_score),
        ):
            rows = [row for row in player_rows if row.team_id == team_id]
            parsed = [parse_player_minutes(row) for row in rows]
            if not rows or any(value is None for value in parsed):
                player_stats_complete = False
                issues.append(f"player_minutes_invalid:{team_id}")
                continue
            if sum(row.pts for row in rows) != expected_points:
                player_stats_complete = False
                issues.append(f"player_points_mismatch:{team_id}")
            expected_total_minutes = 5.0 * expected_game_minutes
            if abs(sum(value or 0.0 for value in parsed) - expected_total_minutes) > 2.0:
                player_stats_complete = False
                issues.append(f"player_minutes_mismatch:{team_id}")
    else:
        issues.append("player_stats_incomplete")

    return {
        "score_complete": score_complete,
        "team_stats_complete": team_stats_complete,
        "player_stats_complete": player_stats_complete,
        "issues": sorted(set(issues)),
    }


async def refresh_game_quality(session: AsyncSession, game_id: int) -> dict:
    """Persist derived flags and invalidate caches when the state changes."""
    from app.models.game import Game

    game = await session.get(Game, game_id)
    if game is None:
        raise ValueError(f"Game {game_id} not found")
    assessment = await assess_game_data(session, game_id)
    before = (game.score_complete, game.team_stats_complete, game.player_stats_complete)
    after = (
        assessment["score_complete"],
        assessment["team_stats_complete"],
        assessment["player_stats_complete"],
    )
    game.score_complete, game.team_stats_complete, game.player_stats_complete = after
    game.stats_updated_at = datetime.now(timezone.utc)
    if before != after:
        await bump_data_revision(session, f"game-quality:{game_id}:{before}->{after}")
    return assessment


async def games_needing_stat_backfill(session: AsyncSession, limit: int = 100) -> list:
    """Return Final games whose score exists but box-score state is incomplete."""
    from app.models.game import Game

    result = await session.execute(
        select(Game)
        .where(
            and_(
                Game.status == "Final",
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
                or_(
                    Game.team_stats_complete.is_(False),
                    Game.player_stats_complete.is_(False),
                ),
            )
        )
        .order_by(Game.game_date.asc())
        .limit(limit)
    )
    return list(result.scalars())
