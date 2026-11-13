"""Ensure later-season backfills do not change earlier causal feature vectors."""
import argparse
import json
from pathlib import Path

import numpy as np

from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES


def main(args):
    cache = DatasetCache()
    before, after = [cache.load(key, len(FEATURE_NAMES)) for key in (args.before, args.after)]
    if before is None or after is None:
        raise ValueError('Both pinned caches must exist with compatible semantics')
    indexes = {row['nba_game_id']: i for i, row in enumerate(after.sample_info)}
    count, maximum = 0, 0.0
    other_count, other_changes = 0, []
    for i, row in enumerate(before.sample_info):
        if row['season'] not in args.unchanged_seasons:
            if getattr(args, 'describe_other_seasons', False):
                j = indexes[row['nba_game_id']]
                difference = np.abs(before.X[i] - after.X[j])
                other_count += 1
                changed = np.flatnonzero(difference > 1e-5)
                if len(changed):
                    other_changes.append(dict(game=row['nba_game_id'], season=row['season'],
                        day=str(row.get('source_game_date') or row['game_date'])[:10],
                        max_difference=float(difference.max()),
                        features=[FEATURE_NAMES[k] for k in changed]))
                for target in ('y_win', 'y_spread', 'y_total'):
                    assert getattr(before, target)[i] == getattr(after, target)[j], (row, target)
            continue
        j = indexes[row['nba_game_id']]
        difference = np.abs(before.X[i] - after.X[j])
        maximum = max(maximum, float(difference.max()))
        if not np.allclose(before.X[i], after.X[j], rtol=1e-6, atol=1e-5):
            raise AssertionError(dict(game=row['nba_game_id'], changed=[
                FEATURE_NAMES[k] for k in np.flatnonzero(difference > 1e-5)]))
        for target in ('y_win', 'y_spread', 'y_total'):
            assert getattr(before, target)[i] == getattr(after, target)[j], (row, target)
        count += 1
    if not count:
        raise ValueError('No overlapping samples in the requested unchanged seasons')
    result = dict(status='PASS', before=args.before, after=args.after, samples=count,
                  unchanged_seasons=args.unchanged_seasons, max_difference=maximum)
    if getattr(args, 'describe_other_seasons', False):
        result['other_seasons'] = dict(samples=other_count, changed_samples=len(other_changes),
            changes=other_changes, interpretation='Descriptive only; not certified unchanged. '
            'Review against the actual dates and scope of statistic repairs.')
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--unchanged-seasons', nargs='+', default=['2023-24', '2024-25'])
    parser.add_argument('--output')
    parser.add_argument('--describe-other-seasons', action='store_true')
    main(parser.parse_args())
