"""Prepare reproducible baseline evidence; never rewrite active.json or train."""
import argparse
import json
from pathlib import Path
from ml.src.config import ml_settings
from ml.src.release_provenance import archive_active_manifest


def main(args):
    manifest=archive_active_manifest(ml_settings.models_dir,legacy_semantic_version=args.legacy_semantic_version)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(manifest,indent=2)
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError('Archived baseline cannot be overwritten')
    else:
        with path.open('x',encoding='utf-8') as stream: stream.write(raw)
    print(json.dumps(dict(status='PASS',version=manifest['version'],output=str(path),
        source_active_sha256=manifest['origin_active_sha256'],model_hash_counts={k:len(v)
        for k,v in manifest['file_hashes'].items()},active_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--legacy-semantic-version',type=int,choices=[5])
    p.add_argument('--output',required=True);main(p.parse_args())
