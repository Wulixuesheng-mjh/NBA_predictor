"""Compute advanced stats from raw box score data and update team_game_stats.

NBA advanced metrics computed per game:
  - Pace: possessions per 48 minutes (both teams)
  - ORtg: points scored per 100 possessions
  - DRtg: points allowed per 100 possessions
  - TS%: true shooting percentage
  - eFG%: effective field goal percentage
  - TOV%: turnover percentage
  - FTR: free throw rate
  - AST%: assist percentage (estimated)

Formulas:
  Team Possessions = FGA + 0.44*FTA - OReb + TOV
  (OReb estimated as 0.25 * total rebounds — league-average offensive rebounding rate)
  Pace = 48 * (Poss_A + Poss_B) / (2 * minutes_played)
  ORtg = 100 * PTS / Poss
  DRtg = 100 * Opp_PTS / Opp_Poss
  TS%  = PTS / (2 * (FGA + 0.44 * FTA))
  eFG% = (FGM + 0.5 * FG3M) / FGA
  TOV% = TOV / (FGA + 0.44 * FTA + TOV)

Usage:
    docker compose exec backend python scripts/compute_advanced_stats.py
"""

import asyncio
import sys
from datetime import datetime

sys.path.insert(0, "/app")

from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.team_game_stats import TeamGameStats
from app.models.game import Game
from app.models.game_player_stats import GamePlayerStats


async def compute_possessions(
    fga: float, fta: float, tov: float, reb: float,
) -> float:
    """Estimate possessions from box score data.

    Full formula: Poss = FGA + 0.44*FTA - OReb + TOV
    Since we don't have offensive rebounds separately, estimate OReb as
    25% of total rebounds (league average over the past decade).
    """
    oreb_estimate = 0.25 * reb
    return fga + 0.44 * fta - oreb_estimate + tov


async def compute_ts_pct(pts: float, fga: float, fta: float) -> float:
    """True Shooting Percentage."""
    denom = 2.0 * (fga + 0.44 * fta)
    return pts / denom if denom > 0 else 0.0


async def compute_efg_pct(fgm: float, fg3m: float, fga: float) -> float:
    """Effective Field Goal Percentage."""
    return (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0


async def compute_tov_pct(tov: float, fga: float, fta: float) -> float:
    """Turnover Percentage (per-possession)."""
    denom = fga + 0.44 * fta + tov
    return tov / denom if denom > 0 else 0.0


async def compute_ftr(fta: float, fga: float) -> float:
    """Free Throw Rate."""
    return fta / fga if fga > 0 else 0.0


async def compute_ast_pct(ast: float, fgm: float) -> float:
    """Assist Percentage — % of field goals that were assisted."""
    return ast / fgm if fgm > 0 else 0.0


async def main() -> None:
    """Compute advanced stats for all Final games and update team_game_stats."""
    print("=" * 60)
    print("NBA Advanced Stats Computer")
    print("=" * 60)

    async with async_session_factory() as session:
        # Get all completed games with team stats
        query = (
            select(Game)
            .where(Game.status == "Final")
            .order_by(Game.game_date.asc())
        )
        result = await session.execute(query)
        games = result.scalars().all()

        print(f"Found {len(games)} completed games")

        updated_count = 0
        for game in games:
            # Get both teams' stats for this game
            stats_query = select(TeamGameStats).where(
                TeamGameStats.game_id == game.id
            )
            stats_result = await session.execute(stats_query)
            team_stats = stats_result.scalars().all()

            if len(team_stats) != 2:
                continue

            home_stats = None
            away_stats = None
            for ts in team_stats:
                if ts.is_home:
                    home_stats = ts
                else:
                    away_stats = ts

            if not home_stats or not away_stats:
                continue

            # Game minutes: estimate from player stats or use 240 (48min * 5 players)
            # NBA games are 48 minutes; OT can make them longer
            # Use player minutes to estimate
            minutes_query = select(GamePlayerStats.minutes).where(
                GamePlayerStats.game_id == game.id
            )
            minutes_result = await session.execute(minutes_query)
            all_minutes = [m for m in minutes_result.scalars().all() if m]

            total_player_minutes = 0.0
            for m_str in all_minutes:
                if m_str is None:
                    continue
                if isinstance(m_str, str):
                    if ":" in m_str:
                        parts = m_str.split(":")
                        total_player_minutes += float(parts[0]) + float(parts[1]) / 60.0
                    else:
                        try:
                            total_player_minutes += float(m_str)
                        except ValueError:
                            pass
                else:
                    total_player_minutes += float(m_str)

            # 5 players * 48 min = 240 regulation minutes per team
            # With OT: linearly scale
            game_minutes = 48.0
            if total_player_minutes > 480:  # More than 480 player-minutes = OT
                game_minutes = total_player_minutes / 10.0  # 10 players total on court

            # ── Home team computations ──
            h_poss = await compute_possessions(
                home_stats.fga, home_stats.fta, home_stats.tov, home_stats.reb
            )
            a_poss = await compute_possessions(
                away_stats.fga, away_stats.fta, away_stats.tov, away_stats.reb
            )

            # Pace (possessions per 48 minutes)
            avg_poss = (h_poss + a_poss) / 2.0
            pace = 48.0 * avg_poss / game_minutes if game_minutes > 0 else 100.0

            # Offensive / Defensive Rating
            home_ortg = 100.0 * home_stats.pts / h_poss if h_poss > 0 else 0.0
            home_drtg = 100.0 * away_stats.pts / a_poss if a_poss > 0 else 0.0
            away_ortg = 100.0 * away_stats.pts / a_poss if a_poss > 0 else 0.0
            away_drtg = 100.0 * home_stats.pts / h_poss if h_poss > 0 else 0.0

            # Advanced shooting
            home_ts = await compute_ts_pct(home_stats.pts, home_stats.fga, home_stats.fta)
            home_efg = await compute_efg_pct(home_stats.fgm, home_stats.fg3m, home_stats.fga)
            home_tovp = await compute_tov_pct(home_stats.tov, home_stats.fga, home_stats.fta)
            home_ftr = await compute_ftr(home_stats.fta, home_stats.fga)
            home_astp = await compute_ast_pct(home_stats.ast, home_stats.fgm)

            away_ts = await compute_ts_pct(away_stats.pts, away_stats.fga, away_stats.fta)
            away_efg = await compute_efg_pct(away_stats.fgm, away_stats.fg3m, away_stats.fga)
            away_tovp = await compute_tov_pct(away_stats.tov, away_stats.fga, away_stats.fta)
            away_ftr = await compute_ftr(away_stats.fta, away_stats.fga)
            away_astp = await compute_ast_pct(away_stats.ast, away_stats.fgm)

            # Update home team stats
            await session.execute(
                update(TeamGameStats)
                .where(TeamGameStats.id == home_stats.id)
                .values(
                    pace=round(pace, 2),
                    off_rtg=round(home_ortg, 2),
                    def_rtg=round(home_drtg, 2),
                    ts_pct=round(home_ts, 4),
                    efg_pct=round(home_efg, 4),
                    tov_pct=round(home_tovp, 4),
                    ast_pct=round(home_astp, 4),
                )
            )

            # Update away team stats
            await session.execute(
                update(TeamGameStats)
                .where(TeamGameStats.id == away_stats.id)
                .values(
                    pace=round(pace, 2),
                    off_rtg=round(away_ortg, 2),
                    def_rtg=round(away_drtg, 2),
                    ts_pct=round(away_ts, 4),
                    efg_pct=round(away_efg, 4),
                    tov_pct=round(away_tovp, 4),
                    ast_pct=round(away_astp, 4),
                )
            )

            updated_count += 2

            if updated_count % 500 == 0:
                await session.commit()
                print(f"  ... {updated_count} team stat rows updated")

        await session.commit()

    print(f"\nUpdated {updated_count} team game stat rows with advanced metrics")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
