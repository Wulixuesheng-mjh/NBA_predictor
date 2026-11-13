"""Restore selected pre-fix code into staging, never into the workspace source."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root / 'reports/auto_shadow_20261029'
    archive = previous / 'code_snapshot_20261030.zip'
    checks = json.loads((previous / 'code_snapshot_20261030.sha256.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checks['archive_sha256']
    staging = root / 'tmp/feedback_runtime_rollback'
    names = ('ml/src/feedback.py', 'ml/src/trainer.py', 'backend/app/api/admin.py',
             'backend/app/scheduler/jobs.py', 'backend/app/scheduler/scheduler.py')
    with ZipFile(archive) as source:
        for name in names:
            value = source.read(name)
            assert hashlib.sha256(value).hexdigest() == checks['files'][name]
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    print(json.dumps(dict(status='PASS', files_restored_to_staging=len(names),
                         workspace_source_unchanged=True)))


if __name__ == '__main__':
    main()
