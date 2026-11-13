"""Checksummed, atomic CSV snapshots from the existing configured dataset."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET = 'eoinamoore/historical-nba-data-and-player-box-scores'
FILES = ('Games.csv','Players.csv','TeamStatistics.csv','PlayerStatistics.csv')


def checksum(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def client():
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5,
        status_forcelist=(429,500,502,503,504), allowed_methods=('GET',))))
    return session


def file_url(version, name):
    return f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{name}?datasetVersionNumber={version}'


def fetch_snapshot(version, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Collection workers and manual refreshes must not switch the pointer in
    # different orders. Keep this lock separate from the immutable manifests.
    import fcntl
    with os.fdopen(os.open(root/'.snapshot.lock', os.O_CREAT | os.O_RDONLY, 0o644), 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _fetch_locked(int(version), root)


def _fetch_locked(version, root):
    folder = root/f'v{version}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/'source_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    manifest = dict(dataset=DATASET, version=version, files={})
    with client() as session:
        for name in FILES:
            path, url = folder/name, file_url(version, name)
            record = previous.get('files',{}).get(name)
            if path.exists() and record and checksum(path) == record['sha256']:
                manifest['files'][name] = record
                continue
            partial, sha, total = path.with_suffix('.csv.partial'), hashlib.sha256(), 0
            with session.get(url, stream=True, timeout=(10,60)) as response:
                response.raise_for_status()
                expected = int(response.headers.get('Content-Length') or 0)
                with partial.open('wb') as output:
                    for block in response.iter_content(1024*1024):
                        output.write(block)
                        sha.update(block)
                        total += len(block)
                if expected and total != expected:
                    raise ValueError(f'Truncated CSV: {name}')
            with partial.open(encoding='utf-8-sig') as stream:
                columns = next(csv.reader(stream))
            if not any(c in columns for c in ('gameId','personId')):
                raise ValueError(f'Unexpected source schema: {name}')
            os.replace(partial,path)
            manifest['files'][name] = dict(url=url, bytes=total, sha256=sha.hexdigest(),
                fetched_at=datetime.now(timezone.utc).isoformat(), columns=columns)
            manifest_path.write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    dates = []
    with (folder/'Games.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            value = row.get('gameDateTimeEst','')[:10]
            if value:
                dates.append(value)
    manifest.update(dataset_dir=str(folder.resolve()), source_game_date_min=min(dates),
                    source_game_date_max=max(dates))
    manifest_path.write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    pointer = root/'current.json'
    current = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else {}
    if int(current.get('version',0)) <= version:
        tmp = pointer.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest,indent=2), encoding='utf-8')
        os.replace(tmp,pointer)
    return manifest


def refresh_snapshot(root):
    """Probe sequential versions, bounded to three advances per collection."""
    root = Path(root)
    current = json.loads((root/'current.json').read_text(encoding='utf-8'))
    version = int(current['version'])
    latest = version
    with client() as session:
        for candidate in range(version+1,version+4):
            with session.get(file_url(candidate,'Games.csv'),stream=True,timeout=(5,12)) as response:
                if response.status_code == 404:
                    break
                response.raise_for_status()
                columns = next(response.iter_lines()).decode('utf-8-sig')
                if not columns.startswith('gameId,'):
                    raise ValueError('Source probe returned non-CSV content')
                latest = candidate
    return fetch_snapshot(latest,root) if latest > version else current
