import importlib.util
from pathlib import Path
import pytest


def test_fixed_cohort_and_bad_important_slice_cannot_pass_development_gate():
    path=Path('/app/scripts/lock_complete_history_study.py')
    if not path.exists(): path=Path(__file__).resolve().parents[2]/'scripts/lock_complete_history_study.py'
    spec=importlib.util.spec_from_file_location('history_gates',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reference=[dict(nba_game_id=str(i),source_game_date=f'2024-04-{i%28+1:02d}',
        actual=[1.,5.,225.],prediction=[.6,0.,215.],early_season=i<100,game_type='regular') for i in range(400)]
    candidate=[r|dict(prediction=[.7,5.,225.]) for r in reference]
    assert m.assess(reference,candidate,'total')['development_gate_pass']
    harmed=[r|dict(prediction=[.7,5.,210.] if i<100 else [.7,5.,225.]) for i,r in enumerate(reference)]
    result=m.assess(reference,harmed,'total')
    assert result['development_improvement']>.02 and not result['development_gate_pass']
    candidate[0]=candidate[0]|dict(actual=[1.,5.,226.])
    with pytest.raises(ValueError,match='cohort'): m.assess(reference,candidate,'total')


def test_audit_refuses_failed_development_and_modified_locked_report(tmp_path,monkeypatch):
    import hashlib,json,sys
    from types import SimpleNamespace
    path=Path('/app/scripts/run_locked_history_audit.py')
    if not path.exists(): path=Path(__file__).resolve().parents[2]/'scripts/run_locked_history_audit.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('locked_history_audit',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    hashes={}
    for mode in ('direct','residual'):
        file=tmp_path/f'development_{mode}.json';file.write_text('{}')
        hashes[mode]=hashlib.sha256(file.read_bytes()).hexdigest()
    lock=dict(source_sha256=hashes,selected={'total':'direct:xgboost:T0'},
              release_gates={'total':{'development_gate_pass':False}})
    (tmp_path/'development_lock.json').write_text(json.dumps(lock))
    args=SimpleNamespace(directory=str(tmp_path),reference=str(tmp_path/'not_read.json'))
    with pytest.raises(ValueError,match='No target'): m.main(args)
    assert not (tmp_path/'audit_protocol.json').exists()
    (tmp_path/'development_direct.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'): m.main(args)
