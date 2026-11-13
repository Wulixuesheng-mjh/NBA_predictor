"""Exercise explicit residual/direct training, refit and missing-baseline fallback."""
import numpy as np
from ml.src.config import ml_settings
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.training_policy import policy_mask_and_weights


def main():
    ml_settings.xgboost_n_estimators = 12
    ml_settings.xgboost_early_stopping_rounds = 3
    rng = np.random.default_rng(42)
    X = rng.normal(size=(240, 4)).astype(np.float32)
    X[:, 0] = rng.uniform(200, 240, len(X))
    y = X[:, 0] + 7
    win = np.arange(len(X)) % 2
    spread = X[:, 1] * 10
    for strategy in ('direct', 'residual'):
        model = XGBoostTrainer(version='strategy-test')
        model.train(X[:180], win[:180], spread[:180], y[:180],
                    X[180:], win[180:], spread[180:], y[180:],
                    total_baseline_train=X[:180, 0], total_baseline_val=X[180:, 0],
                    total_baseline_feature_index=0, total_strategy=strategy)
        assert model._total_mode == strategy
        expected_trees = model.model_total.best_iteration + 1
        model.refit_full(X, win, spread, y, total_baseline=X[:, 0])
        assert model.model_total.n_estimators == expected_trees
        if strategy == 'residual':
            assert np.max(np.abs(model._predict_totals(X) - y)) < 1e-3
            missing = X[:1].copy()
            missing[:, 0] = 0
            assert np.allclose(model._predict_totals(missing), model._model_total_direct.predict(missing))
    rows = [dict(source_game_date=f'{year}-06-01', game_date=f'{year}-06-02')
            for year in range(2017, 2027)]
    for name in ('T0', 'T1', 'T2', 'T3', 'T4', 'T5'):
        mask, weights, report = policy_mask_and_weights(rows, name, cutoff='2026-06-01')
        assert len(weights) == mask.sum() and np.isclose(weights.mean(), 1)
        assert report['fit_until'] == '2026-06-01'
        if name in ('T3', 'T4', 'T5'):
            half_life = report['half_life_days']
            ratio = weights[-2] / weights[-1]
            assert np.isclose(ratio, 2 ** (-365 / half_life))
    print('PASS: forced strategies, exact refit rounds, residual add-back, fallback, temporal weighting')


if __name__ == '__main__':
    main()
