"""Temporal decay must not accidentally change regularization through its scale."""
import numpy as np
import pytest
from ml.src.training_policy import normalize_fit_weights
from ml.src.models.xgboost_model import XGBoostTrainer


def test_fit_boundary_normalizes_only_positive_rows_and_does_not_mutate_input():
    original=np.array([0.,.2,.4,.6])
    normalized=normalize_fit_weights(original,4)
    assert normalized[0]==0 and normalized[1:].mean()==pytest.approx(1.)
    np.testing.assert_array_equal(original,[0.,.2,.4,.6])
    np.testing.assert_allclose(normalize_fit_weights(original*1000,4),normalized)
    for invalid in (np.zeros(4),[1.,-1.,2.,3.],[1.,np.nan,2.,3.],[1.], [1.,np.inf,2.,3.]):
        with pytest.raises(ValueError): normalize_fit_weights(invalid,4)


def test_scaled_weight_equivalent_selects_and_refits_same_xgb_models(monkeypatch):
    from ml.src.config import ml_settings
    monkeypatch.setattr(ml_settings,'xgboost_early_stopping_rounds',3)
    rng=np.random.default_rng(42);X=rng.normal(size=(100,4)).astype(np.float32)
    win=(X[:,0]+.8*rng.normal(size=len(X))>0).astype(float);spread=X[:,0]*5;total=225+X[:,1]*4
    weights=np.arange(1,81,dtype=float)
    results=[]
    for scale in (1., .03):
        model=XGBoostTrainer()
        model._common_params=lambda:dict(n_estimators=8,max_depth=2,learning_rate=.1,
            n_jobs=1,random_state=42,min_child_weight=1)
        model.train(X[:80],win[:80],spread[:80],total[:80],X[80:],win[80:],spread[80:],total[80:],
                    sample_weight=weights*scale,total_strategy='direct')
        rounds=[m.best_iteration for m in (model.model_win,model.model_spread,model.model_total)]
        model.refit_full(X,win,spread,total,sample_weight=np.arange(1,101)*scale)
        predictions=model.predict(X[80:])
        results.append((rounds,np.array([[p.win_prob_home,p.predicted_spread,p.predicted_total] for p in predictions])))
    assert results[0][0]==results[1][0]
    np.testing.assert_allclose(results[0][1],results[1][1],rtol=0,atol=0)
