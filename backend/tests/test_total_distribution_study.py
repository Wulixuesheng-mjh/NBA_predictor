"""Research safety and distribution semantics, independent of live predictors."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def study():
    path=Path('/app/scripts/run_total_distribution_study.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/run_total_distribution_study.py'
    spec=importlib.util.spec_from_file_location('distribution_study',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_tail_projection_is_monotonic_and_preserves_point_prediction(study):
    raw=np.array([[240.,230.,220.,210.,200.],[180.,200.,225.,240.,260.]])
    before=raw.copy();result=study.ordered_quantiles(raw)
    assert np.all(np.diff(result,axis=1)>=0)
    assert np.array_equal(result[:,2],raw[:,2])
    assert np.array_equal(raw,before), 'Input must remain unchanged'
    assert np.array_equal(result[1],raw[1])


@pytest.mark.parametrize('raw', [np.ones((2,4)),np.ones(5),np.array([[1,2,np.nan,4,5]])])
def test_invalid_quantile_shape_or_values_fail_closed(study,raw):
    with pytest.raises(ValueError): study.ordered_quantiles(raw)


def test_interval_scores_penalize_tail_misses_and_do_not_claim_calibration(study):
    raw=np.array([[190.,200.,220.,240.,250.]])
    center=study.interval_metrics([220.],raw)
    outside=study.interval_metrics([270.],raw)
    assert center['calibrated'] is False
    assert center['intervals']['0.8']['empirical_coverage']==1
    assert outside['intervals']['0.8']['empirical_coverage']==0
    assert outside['intervals']['0.8']['interval_score']>center['intervals']['0.8']['interval_score']
    assert center['pinball']['0.5']==0


def test_distribution_research_publication_cannot_overwrite_protocol(study,tmp_path):
    path=tmp_path/'protocol.json'
    study.publish_once(path,{'frozen':True})
    before=path.read_bytes()
    with pytest.raises(FileExistsError): study.publish_once(path,{'frozen':False})
    assert path.read_bytes()==before and json.loads(before)['frozen']
    assert sorted(p.name for p in tmp_path.iterdir())==['protocol.json']


def test_quantile_fitting_never_uses_outer_test_labels(study):
    from ml.src.features.feature_builder import TrainingDataset,FEATURE_NAMES
    from datetime import date,timedelta
    def dataset(count,offset):
        X=np.zeros((count,len(FEATURE_NAMES)),dtype=np.float32)
        X[:,FEATURE_NAMES.index('shrunk_expected_total')]=225.
        X[:,0]=np.arange(offset,offset+count)
        days=[str(date(2023,1,1)+timedelta(days=i)) for i in range(offset,offset+count)]
        return TrainingDataset(X=X,y_win=np.zeros(count),y_spread=np.zeros(count),
            y_total=225.+np.arange(offset,offset+count)%11,
            sample_info=[dict(season='2022-23',game_date=d,source_game_date=d) for d in days])
    past,test=dataset(50,0),dataset(10,50)
    p={'candidate':dict(parameters=dict(n_estimators=8,max_depth=2,n_jobs=1,
        tree_method='hist',random_state=42),early_stopping_rounds=3,
        objective='reg:quantileerror',eval_metric='quantile')}
    a,detail=study.fit_one(past,test,p,.5)
    test.y_total[:]=9999.
    b,other=study.fit_one(past,test,p,.5)
    assert np.array_equal(a,b)
    assert detail==other and detail['rounds']>=1
    assert detail['full_fit_until']<test.sample_info[0]['source_game_date']


def test_concurrent_research_writer_is_rejected(study,tmp_path):
    with study.exclusive_study(tmp_path):
        with pytest.raises(BlockingIOError):
            with study.exclusive_study(tmp_path): pass
