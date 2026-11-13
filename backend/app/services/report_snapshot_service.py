"""Ingest immutable verified official-report payloads; retain unknowns."""

import unicodedata
import re
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from app.models.game import Game
from app.models.team import Team
from app.models.player import Player
from app.models.injury_report import InjuryReport
from app.services.data_quality_service import bump_data_revision
from app.collector.player_aliases import PLAYER_ALIASES

STATUSES = {'Out', 'Doubtful', 'Questionable', 'Probable', 'Available'}


def normalize(value):
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(c for c in value if not unicodedata.combining(c)).replace('.', '').lower()
    value = re.sub(r'\s+', ' ', value).strip()
    return 'los angeles ' + value[3:] if value.startswith('la ') else value


def player_key(value):
    if ',' in value:
        last, first = value.split(',', 1)
        value = first.strip() + ' ' + last.strip()
    return normalize(value)


def report_complete(rows, entries, issues):
    # Being named in a PDF is not submission. Explicit placeholders and any
    # unresolved relevant row invalidate complete-report eligibility.
    if issues:
        return False
    return bool(entries) or any(
        'no injuries' in normalize(row.get('Player Name'))
        or normalize(row.get('Player Name')) in ('none', 'no injuries to report') for row in rows
    )


async def ingest_report(session, rows, *, published_at, observed_at, source_url, sha256):
    if published_at.tzinfo is None or observed_at.tzinfo is None or published_at > observed_at:
        raise ValueError('Report timestamps must be verified, timezone-aware and nonfuture')
    teams = list((await session.execute(select(Team))).scalars())
    team_map = {}
    for team in teams:
        for name in (team.full_name, team.name, team.abbreviation, f'{team.city} {team.name}'):
            team_map[normalize(name)] = team.id
    players = (await session.execute(select(Player.id, Player.full_name, Player.first_name, Player.last_name, Player.nba_player_id))).all()
    player_map = defaultdict(set)
    for pid, full, first, last, nba_pid in players:
        for name in (full, f'{first or ""} {last or ""}', *PLAYER_ALIASES.get(nba_pid,())):
            player_map[normalize(name)].add(pid)
    grouped = defaultdict(list)
    for row in rows:
        try:
            day = datetime.strptime(str(row.get('Game Date') or ''), '%m/%d/%Y').date()
        except ValueError:
            continue
        tid = team_map.get(normalize(row.get('Team')))
        if tid:
            grouped[(day, tid)].append(row)
    counts = {'snapshots': 0, 'complete': 0, 'partial': 0, 'unbound': 0}
    for (day, tid), values in grouped.items():
        candidates = list((await session.execute(select(Game).where(
            Game.source_game_date == day,
            (Game.home_team_id == tid) | (Game.away_team_id == tid),
            Game.game_type.in_(('regular', 'play_in', 'playoffs')),
        ))).scalars())
        if len(candidates) != 1:
            counts['unbound'] += 1
            continue
        game = candidates[0]
        entries, issues = [], []
        for row in values:
            raw_name = str(row.get('Player Name') or '')
            status = str(row.get('Current Status') or '').strip()
            if 'not yet submitted' in normalize(raw_name + ' ' + status):
                issues.append('not_yet_submitted')
                continue
            if status not in STATUSES:
                if not any(text in normalize(raw_name) for text in ('no injuries', 'none')):
                    issues.append('unrecognized_status:' + status)
                continue
            ids = player_map.get(player_key(raw_name), set())
            if len(ids) != 1:
                issues.append('unmatched_player:' + raw_name)
            entries.append({'player_id': next(iter(ids)) if len(ids) == 1 else None,
                            'player_name': raw_name, 'status': status,
                            'reason': str(row.get('Reason') or '')})
        complete = report_complete(values, entries, issues)
        # Reimporting an identical PDF is idempotent; observed_at remains the
        # actual first collection time, rather than the publication time.
        existing = await session.scalar(select(InjuryReport.id).where(
            InjuryReport.game_id == game.id, InjuryReport.team_id == tid,
            InjuryReport.source_sha256 == sha256,
        ).limit(1))
        if existing:
            continue
        statement = insert(InjuryReport).values(
            game_id=game.id, team_id=tid, source='nba_official_pdf',
            published_at=published_at, observed_at=observed_at,
            source_url=source_url, source_sha256=sha256, publication_verified=True,
            entries_json=entries, parse_issues=issues, is_complete=complete,
        ).on_conflict_do_nothing(index_elements=['game_id','team_id','source_sha256'])
        added = await session.scalar(statement.returning(InjuryReport.id))
        if added is None:
            continue
        counts['snapshots'] += 1
        counts['complete' if complete else 'partial'] += 1
    if counts['snapshots']:
        await bump_data_revision(session, 'verified-official-injury-snapshot')
    return counts
