"""Refresh a pinned CSV source with the same atomic code as production."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from app.collector.source_snapshot import fetch_snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--output', default='/app/ml/saved_models/data_sources/kaggle')
    args = parser.parse_args()
    result = fetch_snapshot(args.version,args.output)
    print(json.dumps(dict(version=result['version'], source_game_date_max=result['source_game_date_max'],
                         files=len(result['files']), dataset_dir=result['dataset_dir'])),flush=True)
