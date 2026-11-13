from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_nested_total_selection_never_reads_future_test_labels(monkeypatch):
    path=Path('/app/scripts/run_nested_total_research.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/run_nested_total_research.py'
    spec=importlib.util.spec_from_file_location('nested_totals',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'PARAMETERS',module.PARAMETERS|{'n_estimators':20})
    rng=np.random.default_rng(12)
    X=rng.normal(size=(140,len(module.FEATURE_NAMES))).astype(np.float32)
    X[:,module.FEATURE_NAMES.index('shrunk_expected_total')]=230
    labels=230+5*X[:,0]+rng.normal(size=140)
    start=datetime(2024,1,1,tzinfo=timezone.utc)
    info=[dict(game_date=(start+timedelta(days=i)).isoformat(),
               source_game_date=(start+timedelta(days=i)).date().isoformat(),season='2023-24') for i in range(140)]
    def data(indices):
        return SimpleNamespace(X=X[indices].copy(),y_total=labels[indices].copy(),
            y_win=np.zeros(len(X[indices])),y_spread=np.zeros(len(X[indices])),
            sample_info=info[indices],cache_key='test')
    past,test=data(slice(0,120)),data(slice(120,None))
    prediction,details=module.fit_fold(past,test)
    test.y_total[:]=1e9
    poisoned,other=module.fit_fold(past,test)
    np.testing.assert_allclose(prediction,poisoned,rtol=0,atol=0)
    assert details==other
    assert details['inner_fit_until']<details['validation_from']<=details['full_fit_until']<info[120]['source_game_date']
    assert 0<=details['chosen']['rounds']<=20
