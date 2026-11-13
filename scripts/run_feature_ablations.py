"""Controlled feature ablations, reusing a single pinned 213-column cache."""

import argparse
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from run_rolling_evaluation import run


async def main(args):
    root = Path(args.output_dir)
    experiments = [
        ('E3_no_injury_direct', 'full', 'none'),
        ('E3_aggregate_direct', 'full', 'aggregate'),
        ('E3_net_only_direct', 'full', 'net'),
    ]
    if args.include_earlier:
        experiments[:0] = [('E0_core_direct','core','aggregate'), ('E2_prior_direct','prior','aggregate')]
    for phase in (['development','audit'] if args.phase == 'both' else [args.phase]):
        for name, features, injuries in experiments:
            output = root / f'{phase}_{name}.json'
            await run(SimpleNamespace(
                phase=phase, policies=args.policies, features=features,
                injuries=injuries, direct_total=True, include_mlp=False, resume=True,
                feature_cache_key=args.feature_cache_key, output=str(output),
            ))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-cache-key', required=True)
    parser.add_argument('--phase', choices=['development','audit','both'], default='both')
    parser.add_argument('--policies', nargs='+', default=['T0'])
    parser.add_argument('--output-dir', default='/app/ml/saved_models/ablations_v8')
    parser.add_argument('--include-earlier', action='store_true')
    asyncio.run(main(parser.parse_args()))
