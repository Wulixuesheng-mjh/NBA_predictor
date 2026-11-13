"""Idempotent, complete-column box-score repairs from the configured CSVs."""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert

from app.collector.advanced_stats import derive_game_advanced_stats
from app.models.game import Game
from app.models.game_player_stats import GamePlayerStats
from app.models.player import Player
from app.models.team import Team
from app.models.team_game_stats import TeamGameStats
from app.models.box_score_snapshot import BoxScoreSnapshot
from app.services.data_quality_service import bump_data_revision, refresh_game_quality

LOG = logging.getLogger(__name__)


def source_id(value):
    value = str(value or '').strip()
    return value.lstrip('0') if value.isdigit() else value


def number(value, default=0):
    try:
        return float(value) if value not in ('', None) else default
    except (ValueError, TypeError):
        return default


def source_start(value):
    """The source explicitly labels these timestamps Eastern; preserve DST."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=ZoneInfo('America/New_York')) if parsed.tzinfo is None else parsed


TEAM_FIELDS = {
    'pts': 'teamScore', 'fgm': 'fieldGoalsMade', 'fga': 'fieldGoalsAttempted',
    'fg3m': 'threePointersMade', 'fg3a': 'threePointersAttempted',
    'ftm': 'freeThrowsMade', 'fta': 'freeThrowsAttempted', 'reb': 'reboundsTotal',
    'ast': 'assists', 'stl': 'steals', 'blk': 'blocks', 'tov': 'turnovers', 'pf': 'foulsPersonal',
}
PLAYER_FIELDS = {**TEAM_FIELDS, 'pts': 'points', 'plus_minus': 'plusMinusPoints'}


def source_player_team(row, game, team_map, teams_by_id):
    """Use explicit source identities; never use today's player affiliation.

    Some historical CSVs omit numeric team IDs but retain the game's home flag
    and both historical team names. Require all three to agree before using
    that alternative source identity. A supplied conflicting ID is not replaced.
    """
    supplied = str(row.get('playerteamId') or '').strip()
    if supplied:
        tid = team_map.get(int(number(supplied)))
        return (tid, 'source_team_id') if tid in (game.home_team_id, game.away_team_id) else (None, 'conflicting_team_id')
    side = str(row.get('home') or '').strip()
    if side not in ('0', '1', '0.0', '1.0'):
        return None, 'missing_source_home_flag'
    home = float(side) == 1.
    tid, oid = (game.home_team_id, game.away_team_id) if home else (game.away_team_id, game.home_team_id)
    team, opponent = teams_by_id[tid], teams_by_id[oid]
    clean = lambda value: str(value or '').strip().casefold()
    if clean(row.get('playerteamName')) != clean(team.name) or clean(row.get('opponentteamName')) != clean(opponent.name):
        return None, 'source_team_name_disagreement'
    return tid, 'source_home_and_team_names'


async def backfill_csv_stats(session_factory, data_dir, start_date='2023-10-01', limit=None,
                             end_date=None, seasons=None):
    folder = Path(data_dir)
    report = {'source': str(folder), 'team_rows': 0, 'player_rows': 0, 'unmapped_players': 0,
              'ambiguous_games': 0, 'source_score_conflicts': 0, 'scheduled_times': 0,
              'player_team_from_home_name_rows': 0, 'unresolved_player_team_rows': 0}
    async with session_factory() as session:
        teams = list((await session.scalars(select(Team))).all())
        team_map = {team.nba_team_id: team.id for team in teams}
        teams_by_id = {team.id: team for team in teams}
        player_map = dict((await session.execute(select(Player.nba_player_id, Player.id))).all())
        query = select(Game).where(
            Game.status == 'Final', Game.game_date >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
            or_(Game.team_stats_complete.is_(False), Game.player_stats_complete.is_(False)),
        ).order_by(Game.game_date.desc())
        if end_date:
            query=query.where(Game.game_date < datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc))
        if seasons:
            query=query.where(Game.season.in_(seasons))
        if limit:
            query = query.limit(limit)
        games = list((await session.execute(query)).scalars())
        # Never downgrade verified second-precision rotations to a coarse CSV.
        protected=set()
        snapshots=(await session.execute(select(BoxScoreSnapshot).where(
            BoxScoreSnapshot.game_id.in_([g.id for g in games]),
        ))).scalars().all() if games else []
        game_map={g.id:g for g in games}
        for snapshot in snapshots:
            game=game_map[snapshot.game_id]
            values={bool(t['is_home']):t['totals']['pts'] for t in snapshot.payload['teams']}
            if values=={True:game.home_score,False:game.away_score}:
                protected.add(game.id)
        report['protected_gamebooks']=len(protected)
        games=[g for g in games if g.id not in protected]
        by_id = {source_id(g.nba_game_id): g for g in games}
        by_match = defaultdict(list)
        for game in games:
            day = game.source_game_date or game.game_date.date()
            by_match[(day, game.home_team_id, game.away_team_id)].append(game)
        matched = {}
        with (folder/'Games.csv').open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                sid = source_id(row['gameId'])
                start = source_start(row['gameDateTimeEst'])
                home = team_map.get(int(row.get('hometeamId') or row.get('homeTeamId') or 0))
                away = team_map.get(int(row.get('awayteamId') or row.get('awayTeamId') or 0))
                game = by_id.get(sid)
                if game is None:
                    candidates = by_match.get((start.date(), home, away), [])
                    if len(candidates) > 1:
                        report['ambiguous_games'] += 1
                        continue
                    game = candidates[0] if candidates else None
                if game is None:
                    continue
                if (game.home_team_id, game.away_team_id) != (home, away):
                    report['ambiguous_games'] += 1
                    continue
                if (game.home_score, game.away_score) != (int(number(row.get('homeScore'))), int(number(row.get('awayScore')))):
                    report['source_score_conflicts'] += 1
                    continue
                game.source_game_date = start.date()
                game.scheduled_start_utc = start.astimezone(timezone.utc)
                game.time_precision = 'exact'
                game.game_date = game.scheduled_start_utc
                report['scheduled_times'] += 1
                matched[sid] = game
        team_rows = defaultdict(dict)
        with (folder/'TeamStatistics.csv').open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                game = matched.get(source_id(row['gameId']))
                if game is None:
                    continue
                tid = team_map.get(int(row.get('teamId') or 0))
                if tid not in (game.home_team_id, game.away_team_id):
                    continue
                values = {key: int(number(row.get(source))) for key, source in TEAM_FIELDS.items()}
                values.update(
                    game_id=game.id, team_id=tid, is_home=tid == game.home_team_id,
                    oreb=number(row.get('reboundsOffensive'), None),
                    dreb=number(row.get('reboundsDefensive'), None),
                    game_minutes=number(row.get('numMinutes'), 240.0)/5,
                    possessions_estimated=row.get('reboundsOffensive') in ('', None),
                )
                team_rows[game.id][tid] = values
        valid_games = set()
        pending = []

        async def save_batch(model, rows, keys):
            if not rows:
                return
            statement = insert(model).values(rows)
            changed = {name: getattr(statement.excluded, name) for name in rows[0] if name not in keys}
            changed['updated_at'] = datetime.now(timezone.utc)
            await session.execute(statement.on_conflict_do_update(index_elements=keys, set_=changed))

        for game in matched.values():
            rows = team_rows.get(game.id, {})
            if set(rows) != {game.home_team_id, game.away_team_id}:
                continue
            home, away = rows[game.home_team_id], rows[game.away_team_id]
            if (home['pts'], away['pts']) != (game.home_score, game.away_score):
                continue
            if home['fga'] <= 0 or away['fga'] <= 0:
                continue
            h_adv, a_adv = derive_game_advanced_stats(SimpleNamespace(**home), SimpleNamespace(**away), home['game_minutes'])
            pending.extend([{**home, **h_adv}, {**away, **a_adv}])
            valid_games.add(game.id)
            if len(pending) >= 500:
                await save_batch(TeamGameStats, pending, ['game_id', 'team_id'])
                report['team_rows'] += len(pending)
                pending = []
        await save_batch(TeamGameStats, pending, ['game_id', 'team_id'])
        report['team_rows'] += len(pending)
        pending = []
        with (folder/'PlayerStatistics.csv').open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                game = matched.get(source_id(row['gameId']))
                if game is None or game.id not in valid_games:
                    continue
                player_id = player_map.get(int(row.get('personId') or 0))
                if player_id is None:
                    report['unmapped_players'] += 1
                    continue
                tid, provenance = source_player_team(row, game, team_map, teams_by_id)
                if tid is None:
                    report['unresolved_player_team_rows'] += 1
                    continue
                if provenance == 'source_home_and_team_names':
                    report['player_team_from_home_name_rows'] += 1
                values = {key: int(number(row.get(source))) for key, source in PLAYER_FIELDS.items()}
                minutes = row.get('numMinutes', '').strip()
                # Keep minute precision. Missing nonzero-stat minutes stay
                # missing; the quality gate will reject incomplete box scores.
                values.update(game_id=game.id, player_id=player_id, team_id=tid,
                              minutes=f'{number(minutes):.6f}' if minutes else None)
                pending.append(values)
                if len(pending) >= 500:
                    await save_batch(GamePlayerStats, pending, ['game_id', 'player_id'])
                    report['player_rows'] += len(pending)
                    pending = []
        await save_batch(GamePlayerStats, pending, ['game_id', 'player_id'])
        report['player_rows'] += len(pending)
        await session.flush()
        for gid in valid_games:
            await refresh_game_quality(session, gid)
        if valid_games:
            await bump_data_revision(session, 'verified-csv-stat-backfill')
        await session.commit()
        report['pending_games'] = len(games)
        report['matched_games'] = len(matched)
        report['refreshed_games'] = len(valid_games)
    LOG.info('Box score backfill: %s', report)
    return report
