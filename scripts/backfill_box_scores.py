"""Repair partial box scores from a pinned, checksummed source snapshot."""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.collector.kaggle_collector import KaggleCollector
from app.collector.stat_backfill import backfill_csv_stats
from ml.src.database import async_session_factory


async def main(args):
    collector = KaggleCollector(async_session_factory)
    collector._data_dir = args.data_dir
    await collector.sync_players()
    report = await backfill_csv_stats(async_session_factory, args.data_dir, args.start, args.limit)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/app/ml/saved_models/data_sources/kaggle/v525')
    parser.add_argument('--start', default='2023-10-01')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', default='/app/ml/saved_models/data_quality/box_score_backfill.json')
    asyncio.run(main(parser.parse_args()))
