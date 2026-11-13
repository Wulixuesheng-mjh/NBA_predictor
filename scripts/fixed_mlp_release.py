"""Narrow fixed-1022 repair release path. Never admits arbitrary model upgrades."""
import argparse,fcntl,hashlib,json,os,tempfile
from pathlib import Path
from ml.src.legacy_mlp_compatibility import contract_spec,FILES,VERSION

BASE_SHA='9824b2248129c7c9a71348807660883ca2a620416725bf78290a1b626e42ff3a'
RELEASE='mlp_input_repair_20261112_release_v1'
XGB_FILES={'spread_model.json':'5e55f10a47384d1a9a8778f6f3601f3a6255f2e304cee06dbcf87c74b0fabf89',
    'total_model.json':'63cb20e81a97911dc02492194c6b6d76d841cd19132f4e9e95ddfbe11295c64a',
    'win_model.json':'cb7194b0a24f07c66ee6f6ebe93f5f8345f83bd718a2f9d21605d3042dac2d11',
    'metadata.json':'e737501ef4ac261b347faf064601cd4e972f5c5b7b26d36a91542a5a7b89d5bf'}
EVIDENCE={'season_results.json':'cc753365b4dfa6efef537902e9bcc0f92752ffd61191a6498309fb0e16c5c39d',
    'engineering.json':'f305a5a3f53498b995bbd58cba4dd54fe6da7c988b62c5c5bc7ee73d3265ee80'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(root,package):
    root=Path(root).resolve();package=Path(package).resolve()
    proof=read(package/'evidence.json')
    for name,expected in EVIDENCE.items():
        if sha(package/name)!=expected:raise ValueError('Unrecognized fixed-repair research evidence')
    for name,value in proof['files'].items():
        path=(package/name).resolve()
        if not path.is_relative_to(package) or sha(path)!=value:raise ValueError('Release evidence or manifest changed')
    if sha(package/'baseline.json')!=BASE_SHA:raise ValueError('Only pinned production baseline is accepted')
    base=read(package/'baseline.json');candidate=read(package/'candidate.json')
    expected={**base,'version':RELEASE,'mlp_input_compatibility':contract_spec(),'file_hashes':proof['model_hashes']}
    if candidate!=expected:raise ValueError('Unapproved weights, preprocessing, target or compatibility change')
    if candidate['feature_dim']!=166 or candidate['calibration_enabled'] is not False:raise ValueError('Legacy schema changed')
    if proof['model_hashes']['mlp']!=FILES:raise ValueError('MLP artifact contract changed')
    if proof['model_hashes']['xgboost']!=XGB_FILES:raise ValueError('Unapproved XGB artifacts')
    required={'win_model.json','spread_model.json','total_model.json','metadata.json'}
    if not required<=proof['model_hashes']['xgboost'].keys():raise ValueError('Incomplete XGB bundle')
    for group,hashes in proof['model_hashes'].items():
        if group not in ('mlp','xgboost'):raise ValueError('Unknown model group')
        for name,value in hashes.items():
            if Path(name).name!=name or sha(root/group/VERSION/name)!=value:raise ValueError('Model artifact changed')
    season=read(package/'season_results.json');engine=read(package/'engineering.json');api=read(package/'api_preflight.json')
    if season['quality']['failures'] or season['quality']['samples']!=156:raise ValueError('Unexpected frozen season evidence')
    for target in ('win','spread','total'):
        result=season['repair_three_heads'][target];loss='brier' if target=='win' else 'mae'
        if result['samples']!=156 or result['missing_pairs'] or result['paired_day_95ci'][1]>.01*result['baseline'][loss]:
            raise ValueError('Affected head failed fixed-repair safety')
    if engine['status']!='PASS' or engine['cohort']<100 or engine['original_training_max_difference']!=0 or not engine['xgb_all_outputs_unchanged']:
        raise ValueError('Engineering control evidence failed')
    if engine['pytorch_onnx']['samples']<100 or any(a>b for a,b in zip(engine['pytorch_onnx']['max_difference'],[1e-5,1e-3,1e-3])):
        raise ValueError('Runtime parity failed')
    if not all(api.get(k) is True for k in ('full_version_retained','both_models_persisted','snapshots_replayed','backdated_write_rejected')):
        raise ValueError('Required package API preflight evidence missing')
    if api.get('candidate_manifest_sha256')!=sha(package/'candidate.json'):raise ValueError('API preflight tested another manifest')
    return candidate


def atomic(path,data):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        temporary=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
    try:
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def switch(root,package,action):
    root=Path(root);package=Path(package);validate(root,package)
    with (root/'.fixed_mlp_release.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        active=root/'active.json';candidate=(package/'candidate.json').read_bytes();base=(package/'baseline.json').read_bytes()
        expected,target=(base,candidate) if action=='activate' else (candidate,base)
        if active.read_bytes()!=expected:raise ValueError('Active selector changed; refusing stale or duplicate switch')
        atomic(active,target)
    return dict(action=action,active_sha256=sha(active),status='PASS')


async def require_compatible_database():
    from app.database import async_session_factory
    from sqlalchemy import text
    async with async_session_factory() as db:
        width=await db.scalar(text("SELECT character_maximum_length FROM information_schema.columns WHERE table_name='predictions' AND column_name='model_version'"))
        if width is None or width<128:raise ValueError('Apply 010_mlp_compat_version before activation; full behavior identifiers must fit')
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['validate','activate','rollback'])
    parser.add_argument('--models-dir',type=Path,required=True);parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='validate':validate(args.models_dir,args.package);print(json.dumps(dict(status='PASS',action='validate')))
    else:
        if args.action=='activate':
            import asyncio
            asyncio.run(require_compatible_database())
        print(json.dumps(switch(args.models_dir,args.package,args.action)))
