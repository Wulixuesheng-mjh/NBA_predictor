"""Archive reviewable code without secrets, datasets, dependencies or Git internals."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
folder = root/'reports/data_quality_20261027'
paths = []
for name in ('backend/app','backend/alembic','backend/tests','ml/src','frontend/src','scripts'):
    paths += [p for p in (root/name).rglob('*') if p.is_file()
              and p.suffix in ('.py','.ts','.tsx','.css','.json')]
for name in ('docker-compose.yml','docker-compose.override.yml','backend/Dockerfile',
             'backend/pyproject.toml','frontend/Dockerfile','frontend/package.json',
             'frontend/package-lock.json','ml/Dockerfile','ml/Dockerfile.injuries',
             'docs/data-optimization-checkpoint-2026-10-28.md'):
    paths.append(root/name)
paths = sorted(set(paths))
archive = folder/'code_snapshot_20261028.zip'
manifest = {}
with ZipFile(archive,'w',ZIP_DEFLATED) as output:
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        manifest[relative] = hashlib.sha256(data).hexdigest()
        output.writestr(relative,data)
(folder/'code_snapshot_20261028.sha256.json').write_text(json.dumps(dict(
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=manifest),indent=2),encoding='utf-8')
print('Saved',len(paths),'code files; excluded .env, model weights and raw data')
