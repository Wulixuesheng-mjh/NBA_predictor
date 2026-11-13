import importlib.util,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pytest


@pytest.fixture
def worker():
    path=Path('/app/scripts/paired_models_worker.py')
    if not path.exists():path=Path(__file__).resolve().parents[2]/'scripts/paired_models_worker.py'
    spec=importlib.util.spec_from_file_location('paired_worker_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_append_does_not_overwrite_observation(worker,tmp_path):
    file=tmp_path/'first.json'
    assert worker.publish(file,{'first':1})
    before=file.read_bytes()
    assert not worker.publish(file,{'second':2}) and file.read_bytes()==before
    assert len(list(tmp_path.iterdir()))==1


def test_nonfinite_outputs_remain_unknown_and_are_accounted_for(worker):
    invalid=[];value=worker.safe_values({'models':{'mlp':[float('nan'),float('inf'),210.]}},invalid)
    assert value['models']['mlp']==[None,None,210.]
    assert invalid==['.models.mlp[0]','.models.mlp[1]']
    json.dumps(value,allow_nan=False)


def test_only_real_time_exact_scheduled_window_is_accepted(worker):
    now=datetime.now(timezone.utc)
    g=SimpleNamespace(status='Scheduled',time_precision='exact',scheduled_start_utc=now+timedelta(minutes=55))
    assert worker.in_window(g,now)
    for minutes in (-1,49,61):
        g.scheduled_start_utc=now+timedelta(minutes=minutes);assert not worker.in_window(g,now)
    g.scheduled_start_utc=now+timedelta(minutes=55);g.time_precision='date'
    assert not worker.in_window(g,now)


def test_failed_mlp_is_preserved_without_fabricating_scaler_or_final_output(worker):
    class Dummy:
        mlp_trainer=None
        def predict(self,fv,apply_calibration):
            assert apply_calibration is False
            return SimpleNamespace(raw_outputs={'mlp':{'predicted_total':100.}},
                xgboost=None,mlp=None,data_quality={'warnings':['rejected']},calibration_applied=False)
        def _prepare_release_features(self,X):return X
        def _adapt_features(self,X,n):return X[:,:n]
    result=worker.paired_outputs(Dummy(),SimpleNamespace(features=np.zeros(213)))
    assert result['models']['mlp']['raw']['predicted_total']==100
    assert result['models']['mlp']['guard_rejected']
    assert result['models']['mlp']['final'] is None
    assert result['mlp_scaled_clipped_input'] is None
    assert result['models']['xgboost']['inference_failed']
