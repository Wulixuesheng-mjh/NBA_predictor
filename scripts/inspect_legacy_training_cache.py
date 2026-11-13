"""Read original contemporaneous 166-column cache without changing it."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ml.src.features.feature_builder import FEATURE_NAMES


def main(args):
    path=Path('/app/ml/saved_models/feature_cache/train_99ec73146ff0f31e.npz')
    with np.load(path,allow_pickle=False) as f:
        X=f['X'];meta=json.loads(str(f['meta']))
    names=['home_games_available','away_games_available','home_avg_ast_pct','away_avg_ast_pct','same_conference','same_division']
    report=dict(kind='contemporaneous_legacy_cache_constant_channel_audit',cache_meta=meta,
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),labels_used=False,writes=0,
        original_feature_names_stored=False,names_from='archived 166-column compatibility prefix',
        features={name:dict(column=FEATURE_NAMES.index(name),minimum=float(X[:,FEATURE_NAMES.index(name)].min()),
            maximum=float(X[:,FEATURE_NAMES.index(name)].max()),std=float(X[:,FEATURE_NAMES.index(name)].std()),
            unique_values=int(len(np.unique(X[:,FEATURE_NAMES.index(name)])))) for name in names})
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
