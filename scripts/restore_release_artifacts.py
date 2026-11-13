"""Verify an immutable release archive and install without activating it."""
import argparse,hashlib,json,stat,zipfile
from pathlib import Path


def restore(bundle,models_dir):
    bundle=Path(bundle);manifest=json.loads(bundle.read_text(encoding='utf-8'))
    archive=bundle.parent/manifest['archive'];root=Path(models_dir).resolve()
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:
        raise ValueError('Release archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(manifest['members']):
            raise ValueError('Unexpected or duplicate archive members')
        for name,expected in manifest['members'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or stat.S_ISLNK(z.getinfo(name).external_attr>>16):
                raise ValueError('Unsafe archive member')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Artifact checksum mismatch')
            if path.exists() and (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
                raise ValueError('Existing artifact differs; never overwrite')
        # Validate the complete archive before installing any member.
        for name in manifest['members']:
            path=(root/name).resolve()
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as out:out.write(z.read(name))
    return dict(status='RESTORED',release=manifest['release'],members=len(manifest['members']),activated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--models-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(restore(args.bundle,args.models_dir)))
