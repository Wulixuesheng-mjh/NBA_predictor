"""Acceptance checks for scorer tables, DNP semantics and persisted seconds."""
import asyncio
from types import SimpleNamespace
from sqlalchemy import select
from app.database import async_session_factory
from app.models.box_score_snapshot import BoxScoreSnapshot
from app.models.game_player_stats import GamePlayerStats
from app.services.data_quality_service import assess_game_data,bump_data_revision
from app.collector.gamebook_parser import parse_final_box
from ml.src.features.player_features import _aggregate_played_rows,_parse_minutes


async def main():
    stats,minutes=_aggregate_played_rows([
        SimpleNamespace(player_id=1,minutes='36.000000',pts=20,stl=2,blk=1),
        SimpleNamespace(player_id=1,minutes='0.000000',pts=0,stl=0,blk=0)])
    assert stats[1]['ppg']==20 and stats[1]['games']==1 and minutes[1]==[36]
    rows='\n'.join(f'{i} Tester {i} F 48:00 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 2' for i in range(1,6))
    table=rows+'\n240:00 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 10\n'
    text="NATIONAL BASKETBALL ASSOCIATION OFFICIAL SCORER'S REPORT\nFINAL BOX\nMonday, June 17, 2024\nVISITOR: Dallas Mavericks\n"+table+'HOME: Boston Celtics\n'+table
    assert parse_final_box(text)['teams'][0]['total_seconds']==14400
    try:
        parse_final_box(text.replace('48:00','47:50',1))
    except ValueError:
        pass
    else:
        raise AssertionError('Minute mismatch accepted')
    async with async_session_factory() as session:
        snapshots=(await session.execute(select(BoxScoreSnapshot).limit(5))).scalars().all()
        assert len(snapshots)==5
        for snapshot in snapshots:
            state=await assess_game_data(session,snapshot.game_id)
            assert state['player_stats_complete'],state
            values=(await session.execute(select(GamePlayerStats).where(
                GamePlayerStats.game_id==snapshot.game_id))).scalars().all()
            for side in snapshot.payload['teams']:
                actual=sum(_parse_minutes(r.minutes) or 0 for r in values if r.team_id==side['team_id'])
                assert abs(actual*60-side['total_seconds'])<0.001
        try:
            first=await bump_data_revision(session,'test-atomic-counter')
            second=await bump_data_revision(session,'test-atomic-counter')
            assert second==first+1
        finally:
            await session.rollback()
    print('PASS: scorer columns, exact minutes, mismatch rejection, DNP exclusion, DB completeness, atomic revision')


if __name__=='__main__':
    asyncio.run(main())
