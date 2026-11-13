"""Read-only identity probe for rejected scorer rows; never fuzzy assign IDs."""
import asyncio
import json
from pathlib import Path
from sqlalchemy import or_, select
from app.database import async_session_factory
from app.models.player import Player


async def main():
    async with async_session_factory() as session:
        players = (await session.scalars(select(Player).where(or_(
            Player.full_name.ilike('%karim%'), Player.first_name.ilike('%karim%'))))).all()
        report = dict(query='Karim', matches=[dict(id=p.id, nba_player_id=p.nba_player_id,
            full_name=p.full_name, first_name=p.first_name, last_name=p.last_name,
            team_id=p.team_id) for p in players], writes=0)
    Path('/tmp/scorer_identity_probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
