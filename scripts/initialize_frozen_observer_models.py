"""Initialize an empty /frozen Docker volume; source model volume must be read-only.

Run only against a new empty volume; does not activate production models.
"""
import shutil,json,hashlib
from pathlib import Path
source=Path('/app/ml/saved_models');target=Path('/frozen');version='20261022_050952'
base=source/'repair_releases/mlp_input_repair_20261112_release_v1/baseline.json'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='9824b2248129c7c9a71348807660883ca2a620416725bf78290a1b626e42ff3a'
assert not (target/'active.json').exists()
for group in ('xgboost','mlp'):
 shutil.copytree(source/group/version,target/group/version)
for group in ('diagnostics','win_shadows','release_baselines'):
 if (source/group).exists():shutil.copytree(source/group,target/group)
for p in source.iterdir():
 if p.name in {'active.json','xgboost','mlp','diagnostics','win_shadows','release_baselines'}:continue
 shutil.copy2(p,target/p.name) if p.is_file() else (target/p.name).symlink_to(p)
(target/'active.json').write_bytes(base.read_bytes());(target/'active.json').chmod(0o444)
print(json.dumps(dict(status='INDEPENDENT_BASELINE_AND_MODELS_READY',sha256=hashlib.sha256((target/'active.json').read_bytes()).hexdigest())))
