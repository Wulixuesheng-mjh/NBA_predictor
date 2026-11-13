"""Select one immutable payload under an explicit information cutoff."""
from types import SimpleNamespace
from sqlalchemy import text


async def resolve_known_aliases(session, snapshot):
    """Resolve verified identities in a copy, preserving archived raw payloads."""
    from app.models.player import Player
    from app.collector.player_aliases import PLAYER_ALIASES
    from app.services.report_snapshot_service import player_key
    from sqlalchemy import select
    if snapshot is None or not isinstance(snapshot.entries_json,list):
        return snapshot
    aliases=session.info.get('official_player_aliases')
    if aliases is None:
        rows=(await session.execute(select(Player.nba_player_id,Player.id).where(
            Player.nba_player_id.in_(PLAYER_ALIASES)))).all()
        aliases={player_key(name):pid for nba_pid,pid in rows for name in PLAYER_ALIASES[nba_pid]}
        session.info['official_player_aliases']=aliases
    resolved=[];fixed_names=set()
    for entry in snapshot.entries_json:
        entry=dict(entry)
        if entry.get('player_id') is None:
            pid=aliases.get(player_key(entry.get('player_name','')))
            if pid is not None:
                entry['player_id']=pid
                fixed_names.add(entry.get('player_name',''))
        resolved.append(entry)
    if fixed_names:
        snapshot.raw_is_complete=snapshot.is_complete
        snapshot.raw_entries_json=snapshot.entries_json
        snapshot.entries_json=resolved
        issues=[issue for issue in snapshot.parse_issues or []
                if not (issue.startswith('unmatched_player:') and issue.split(':',1)[1] in fixed_names)]
        snapshot.identity_resolved_names=sorted(fixed_names)
        snapshot.resolved_parse_issues=issues
        if not issues and resolved and all(e.get('player_id') is not None for e in resolved):
            snapshot.is_complete=True
    return snapshot


async def latest_injury_snapshot(session, game_id, team_id, as_of, *, reconstructed=False):
    """Reconstruction permits later collection, never a later publication.

    Live requests require both clocks. Do not filter is_complete here: a newer
    partial submission must mask an older complete report. Unknown legacy
    publication times cannot establish historical availability.
    """
    if game_id is None:
        return None
    # Select stored columns directly: an already loaded application may still
    # hold the previous ORM class while a compatible additive migration lands.
    extra = 'publication_verified = true' if reconstructed else 'observed_at <= :as_of'
    row = (await session.execute(text(f'''
        SELECT * FROM injury_reports WHERE game_id=:gid AND team_id=:tid
        AND published_at <= :as_of AND entries_json IS NOT NULL AND {extra}
        ORDER BY published_at DESC, observed_at DESC, id DESC LIMIT 1
    '''), dict(gid=game_id, tid=team_id, as_of=as_of))).mappings().one_or_none()
    return await resolve_known_aliases(session,SimpleNamespace(**row)) if row else None


def snapshot_statuses(snapshot):
    """Available explicitly clears prior Out; never merge different PDFs."""
    if snapshot is None:
        return {}
    return {int(row['player_id']): row['status'] for row in snapshot.entries_json or []
            if row.get('player_id') is not None and row.get('status') in (
                'Out', 'Doubtful', 'Questionable', 'Probable', 'Available')}
