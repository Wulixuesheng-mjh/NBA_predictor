"""Compare cached vectors with direct queries on real nonzero/missing rows."""
import asyncio
import argparse
import json
import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder


async def main(key, through_date=None, mask_injuries=False):
    data = DatasetCache().load(key,len(FEATURE_NAMES))
    if data is None:
        raise ValueError('Missing cache or incompatible feature semantics')
    if through_date:
        from ml.src.training_policy import subset_arrays
        from ml.src.evaluation import nba_day
        data=subset_arrays(data,np.array([nba_day(r)<=through_date for r in data.sample_info]))
        if not len(data.X):
            raise ValueError('No samples before requested cutoff')
    net = np.maximum(data.X[:,FEATURE_NAMES.index('home_injury_net_available')],
                     data.X[:,FEATURE_NAMES.index('away_injury_net_available')])
    candidates = [np.flatnonzero(net)[:5], np.flatnonzero(net)[-5:],
                  np.flatnonzero(net == 0)[:3], np.array([len(data.X)-1])]
    # Include middle seasons as well as the chronological endpoints, so an
    # audit-season backfill cannot escape verification through sample choice.
    for season in sorted({row['season'] for row in data.sample_info}):
        season_mask = np.array([row['season'] == season for row in data.sample_info])
        for available in (True, False):
            selected = np.flatnonzero(season_mask & ((net > 0) == available))
            if len(selected):
                candidates.append(selected[[0, len(selected)//2, -1]])
    indexes = np.unique(np.concatenate(candidates))
    ignored=set()
    if mask_injuries:
        from ml.src.features.player_features import PLAYER_FEATURE_NAMES
        from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
        ignored={FEATURE_NAMES.index(f'{side}_{name}') for side in ('home','away')
                 for name in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES}
    builder, maximum = FeatureBuilder(), 0.0
    raw_changes=[]
    async with async_session_factory() as session:
        for index in indexes:
            info = data.sample_info[index]
            game = await session.scalar(select(Game).where(Game.nba_game_id == info['nba_game_id']))
            value = await builder.build_features(game.home_team_id,game.away_team_id,game.game_date,
                session=session,use_historical=True,target_game_id=game.id)
            difference = np.abs(data.X[index]-value.features)
            changed=np.flatnonzero(difference>1e-4)
            if len(changed):
                raw_changes.append(dict(game_id=game.id,day=str(game.source_game_date),
                    features={FEATURE_NAMES[i]:[float(data.X[index,i]),float(value.features[i])]
                              for i in changed}))
            cached, direct = data.X[index].copy(),value.features.copy()
            if ignored:
                cached[list(ignored)]=0
                direct[list(ignored)]=0
            difference=np.abs(cached-direct)
            maximum = max(maximum,float(difference.max()))
            assert np.allclose(cached,direct,rtol=1e-6,atol=1e-5), [
                (FEATURE_NAMES[i],float(data.X[index,i]),float(value.features[i]))
                for i in np.flatnonzero(difference>1e-4)]
    print(json.dumps(dict(status='PASS',samples=len(indexes),through_date=through_date,
                          masked_injuries=mask_injuries,raw_changes=raw_changes,
                          max_float32_roundoff=maximum)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-cache-key',required=True)
    parser.add_argument('--through-date')
    parser.add_argument('--mask-injuries',action='store_true')
    args=parser.parse_args()
    asyncio.run(main(args.feature_cache_key,args.through_date,args.mask_injuries))
