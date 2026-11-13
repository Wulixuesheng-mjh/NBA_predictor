"""Attribute a pinned-cache difference to a real earlier-game data correction."""
import asyncio
import json
from pathlib import Path
import numpy as np
from sqlalchemy import select, update
from app.database import async_session_factory
from app.models.game import Game
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder


async def main():
    data=DatasetCache().load('3cf61fa525ab4784',len(FEATURE_NAMES))
    async with async_session_factory() as session:
        target=await session.get(Game,31788)
        correction=await session.get(Game,31783)
        assert correction.game_date<target.game_date and correction.player_stats_complete
        index=next(i for i,r in enumerate(data.sample_info) if r['nba_game_id']==target.nba_game_id)
        home,away,day,gid=target.home_team_id,target.away_team_id,target.game_date,target.id
        async def build():
            return (await FeatureBuilder().build_features(home,away,
                day,session=session,use_historical=True,target_game_id=gid)).features
        current=await build()
        try:
            await session.execute(update(Game).where(Game.id==correction.id).values(player_stats_complete=False))
            restored=await build()
            np.testing.assert_allclose(restored,data.X[index],rtol=1e-6,atol=2e-5)
        finally:
            await session.rollback()
        after=await build()
        np.testing.assert_allclose(current,after,rtol=0,atol=1e-8)
        changed=[FEATURE_NAMES[i] for i in np.flatnonzero(abs(current-data.X[index])>1e-4)]
    report=dict(status='PASS',target_game=31788,corrected_earlier_game=31783,
        changed_features=changed,old_vector_reproduced_by_restoring_old_completeness=True,
        rollback_verified=True,
        conclusion='Earlier-game completeness correction changed former-team player production; pinned cache must be rebuilt')
    output=Path('/app/ml/saved_models/data_quality/injury_cache_correction_20261029.json')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    asyncio.run(main())
