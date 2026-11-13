import importlib.util
from pathlib import Path
import pytest


def test_pace_efficiency_decomposition_closes_with_and_without_overtime():
    path=Path('/app/scripts/diagnose_pace_efficiency.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/diagnose_pace_efficiency.py'
    spec=importlib.util.spec_from_file_location('pace_diagnosis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for minutes in (48,53,58):
        r=module.decompose_total(100,225,110,260,minutes)
        assert r['baseline']==225
        assert sum(r['contributions'].values())==pytest.approx(35)
        assert r['contributions']['extra_duration']==pytest.approx(260*(1-48/minutes))
    with pytest.raises(ValueError):
        module.decompose_total(0,225,110,260,48)
