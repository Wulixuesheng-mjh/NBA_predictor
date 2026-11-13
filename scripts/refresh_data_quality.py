"""Bulk-refresh completed-game quality flags and the cache revision.

The online collector validates a single changed game with the ORM service. A
historical migration must avoid tens of thousands of per-game round trips, so
this command performs the same checks with set-based PostgreSQL aggregates.
"""

import asyncio

from sqlalchemy import text

from ml.src.database import async_session_factory


QUALITY_SQL = text(r"""
WITH team_quality AS (
    SELECT t.game_id,
        count(*) = 2 AND count(DISTINCT t.team_id) = 2
        AND bool_and(t.fga > 0)
        AND bool_and(
            (t.team_id = g.home_team_id AND t.pts = g.home_score)
            OR (t.team_id = g.away_team_id AND t.pts = g.away_score)
        ) AS valid
    FROM team_game_stats t JOIN games g ON g.id = t.game_id
    GROUP BY t.game_id
), parsed_player AS (
    SELECT p.game_id, p.team_id, p.pts,
        CASE
            WHEN (p.minutes IS NULL OR trim(p.minutes) = '')
                AND p.pts = 0 AND p.reb = 0 AND p.ast = 0
                AND p.stl = 0 AND p.blk = 0 AND p.tov = 0
                AND p.fgm = 0 AND p.fga = 0 AND p.ftm = 0 AND p.fta = 0
                THEN 0.0
            WHEN trim(p.minutes) ~ '^[0-9]+:[0-9]+(\.[0-9]+)?$' THEN
                split_part(trim(p.minutes), ':', 1)::double precision
                + split_part(trim(p.minutes), ':', 2)::double precision / 60.0
            WHEN trim(p.minutes) ~ '^[0-9]+(\.[0-9]+)?$' THEN trim(p.minutes)::double precision
            ELSE NULL
        END AS parsed_minutes
    FROM game_player_stats p
), player_team_quality AS (
    SELECT p.game_id, p.team_id,
        count(*) > 0 AND count(p.parsed_minutes) = count(*)
        AND sum(p.pts) = CASE
            WHEN p.team_id = g.home_team_id THEN g.home_score
            WHEN p.team_id = g.away_team_id THEN g.away_score
            ELSE -1
        END
        AND abs(sum(p.parsed_minutes) - 5.0 * coalesce(max(t.game_minutes), 48.0)) <= 2.0 AS valid
    FROM parsed_player p
    JOIN games g ON g.id = p.game_id
    LEFT JOIN team_game_stats t ON t.game_id = p.game_id AND t.team_id = p.team_id
    GROUP BY p.game_id, p.team_id, g.home_team_id, g.away_team_id,
             g.home_score, g.away_score
), player_quality AS (
    SELECT game_id, count(*) = 2 AND bool_and(valid) AS valid
    FROM player_team_quality GROUP BY game_id
)
UPDATE games g
SET score_complete = coalesce(
        g.status = 'Final' AND g.home_score > 0 AND g.away_score > 0,
        false
    ),
    team_stats_complete = (
        coalesce(g.status = 'Final' AND g.home_score > 0 AND g.away_score > 0, false)
        AND coalesce((SELECT valid FROM team_quality q WHERE q.game_id = g.id), false)
    ),
    player_stats_complete = (
        coalesce(g.status = 'Final' AND g.home_score > 0 AND g.away_score > 0, false)
        AND coalesce((SELECT valid FROM team_quality q WHERE q.game_id = g.id), false)
        AND coalesce((SELECT valid FROM player_quality q WHERE q.game_id = g.id), false)
    ),
    source_game_date = coalesce(
        g.source_game_date, (g.game_date AT TIME ZONE 'America/New_York')::date
    ),
    game_type = CASE substring(lpad(g.nba_game_id, 10, '0') from 1 for 3)
        WHEN '001' THEN 'preseason'
        WHEN '002' THEN 'regular'
        WHEN '003' THEN 'exhibition'
        WHEN '004' THEN 'playoffs'
        WHEN '005' THEN 'play_in'
        WHEN '006' THEN 'cup_final'
        ELSE g.game_type
    END,
    stats_updated_at = now()
WHERE g.status = 'Final'
""")


async def main() -> None:
    async with async_session_factory() as session:
        result = await session.execute(QUALITY_SQL)
        await session.execute(text("""
            UPDATE data_revisions
            SET revision = revision + 1,
                reason = 'bulk-game-quality-refresh',
                updated_at = now()
            WHERE id = 1
        """))
        await session.commit()
        counts = (await session.execute(text("""
            SELECT count(*) FILTER (WHERE status = 'Final') AS final_games,
                count(*) FILTER (WHERE status = 'Final' AND score_complete) AS score_complete,
                count(*) FILTER (WHERE status = 'Final' AND team_stats_complete) AS team_complete,
                count(*) FILTER (WHERE status = 'Final' AND player_stats_complete) AS player_complete
            FROM games
        """))).one()
        print({
            "rows_updated": result.rowcount,
            "final_games": counts.final_games,
            "score_complete": counts.score_complete,
            "team_complete": counts.team_complete,
            "player_complete": counts.player_complete,
        })


if __name__ == "__main__":
    asyncio.run(main())
