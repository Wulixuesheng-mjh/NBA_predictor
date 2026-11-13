"""Validate repository boundaries and immutable artifacts without live services."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    failures=[]
    for name in filter(None,tracked):
        p=Path(name)
        if name in {'compact_docker.ps1','move_docker_to_e.ps1','_diskpart_compact.txt','.env'}:
            failures.append('Local tool or secret at repository root: '+name)
        if name.startswith(('notes/local/','tmp/')) or p.suffix in {'.dump','.tar','.db','.sqlite3','.pyc'}:
            failures.append('Generated/private artifact tracked: '+name)
        if name.endswith('.zip') and not name.startswith('releases/'):
            failures.append('Unscoped archive tracked: '+name)
    for name in ['README.md','docs/README.md','docs/operations.md','scripts/README.md','reports/README.md','tools/windows/README.md']:
        path=ROOT/name
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#',1)[0]
            if target and not (path.parent/target).exists():failures.append('Broken entry link: '+name+' -> '+target)
    evidence=json.loads((ROOT/'reports/published_index.json').read_text(encoding='utf-8'))
    for name,expected in evidence['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append('Published evidence changed: '+name)
    bundle=ROOT/'releases/mlp_input_repair_20261112_release_v1/bundle.json'
    meta=json.loads(bundle.read_text(encoding='utf-8'));archive=bundle.parent/meta['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:
        failures.append('Release archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(meta['members']):failures.append('Release member inventory differs')
        for name,expected in meta['members'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:failures.append('Release member changed: '+name)
    if failures:raise SystemExit('\n'.join(failures))
    print(json.dumps(dict(status='PASS',tracked_files=len(list(filter(None,tracked))),
        published_evidence_files=len(evidence['files']),release_members=len(meta['members']),
        database_access=False,model_execution=False)))


if __name__=='__main__':main()
