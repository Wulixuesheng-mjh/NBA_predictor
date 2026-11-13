"""Archive the rollout code and verification summary, excluding runtime secrets."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main(output_dir=None, snapshot_date='20261030'):
    root = Path(__file__).resolve().parents[1]
    folder = Path(output_dir) if output_dir else root / 'reports/auto_shadow_20261029'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ('backend/app', 'backend/alembic', 'backend/tests', 'ml/src', 'scripts', 'frontend/src'):
        paths.extend(p for p in (root / name).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.ts', '.tsx', '.css', '.json'))
    paths.extend(p for p in (root / 'docs').glob('*.md')
                 if any(word in p.name for word in ('checkpoint', 'model-', 'total-', 'shadow-', 'history-study')))
    for name in ('README.md', 'docker-compose.yml', 'docker-compose.override.yml', 'docker-compose.release.yml',
                 'docker-compose.win-shadow.yml', 'tmp/win_shadow_worker/Dockerfile',
                 'docker-compose.paired-models.yml', 'tmp/paired_models_worker/Dockerfile',
                 'tmp/mlp_production_audit/mlp_model.py', 'tmp/expanded_smoke_20261105.py',
                 'backend/Dockerfile', 'backend/pyproject.toml', 'backend/alembic.ini',
                 'frontend/Dockerfile', 'frontend/package.json', 'frontend/package-lock.json',
                 'frontend/index.html', 'frontend/nginx.conf', 'frontend/postcss.config.js',
                 'frontend/tailwind.config.js', 'ml/Dockerfile', 'ml/Dockerfile.injuries',
                 'tmp/rollback_auto_shadow/Dockerfile', 'tmp/auto_shadow_gate_overlay/Dockerfile',
                 'tmp/feedback_runtime_overlay/Dockerfile', 'tmp/feedback_runtime_rollback/Dockerfile',
                 'tmp/feedback_runtime_rollback/compose.yml', 'tmp/model_test_environment/Dockerfile',
                 'tmp/inference_provenance_overlay/Dockerfile', 'tmp/current_gamebook_fetch/Dockerfile'):
        if (root / name).is_file():
            paths.append(root / name)
    paths.extend((root / 'frontend').glob('tsconfig*.json'))
    paths.extend((root / 'frontend').glob('vite.config.*'))
    paths.extend((root / 'tmp/feedback_runtime_rollback').rglob('*.py'))
    manifest = {}
    parsed = 0
    from datetime import datetime
    datetime.strptime(snapshot_date, '%Y%m%d')
    archive = folder / f'code_snapshot_{snapshot_date}.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
        for path in sorted(set(paths)):
            data = path.read_bytes()
            if path.suffix == '.py':
                ast.parse(data.decode('utf-8-sig'), filename=str(path))
                parsed += 1
            relative = path.relative_to(root).as_posix()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            output.writestr(relative, data)
    (folder / f'code_snapshot_{snapshot_date}.sha256.json').write_text(json.dumps(dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        python_ast_parsed=parsed, files=manifest), indent=2), encoding='utf-8')
    print(json.dumps(dict(code_files=len(manifest), python_ast_parsed=parsed,
                         archive=str(archive)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    parser.add_argument('--snapshot-date',default='20261030')
    args=parser.parse_args()
    main(args.output_dir,args.snapshot_date)
