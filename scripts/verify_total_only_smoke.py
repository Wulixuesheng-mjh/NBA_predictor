"""Synthetic gradient and full-refit checks for the offline target ablation."""
import json,tempfile
from datetime import date,timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts import run_total_only_mlp_study as study
from ml.src.models import mlp_model
from ml.src.features.feature_builder import TrainingDataset,FEATURE_NAMES


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    net=study.TotalOnlyNetwork(input_dim=8,input_normalization='none');net.eval()
    X=torch.randn(20,8);target=torch.randn(20,1)
    win,spread,total=net(X)
    assert not win.requires_grad and not spread.requires_grad and total.requires_grad
    loss=.3*torch.nn.functional.binary_cross_entropy(win,torch.zeros_like(win))+.35*(spread**2).mean()+.35*((total-target)**2).mean()
    loss.backward();gradient=net.layer1[0].weight.grad.clone()
    assert net.win_head[0].weight.grad is None and net.spread_head[0].weight.grad is None
    net.zero_grad();(.35*((net(X)[2]-target)**2).mean()).backward()
    assert torch.allclose(gradient,net.layer1[0].weight.grad,rtol=0,atol=1e-7)
    trainer=study.TotalOnlyTrainer(device='cpu');trainer.model=net
    crit=torch.nn.HuberLoss(reduction='none')
    val=(X,torch.zeros_like(target),torch.zeros_like(target),target)
    a=trainer._compute_val_loss(val,crit,crit,crit)
    b=trainer._compute_val_loss((X,torch.ones_like(target),torch.ones_like(target)*100,target),crit,crit,crit)
    assert a==b
    trainer.sanity_baseline=np.zeros(2)
    trainer.predict=lambda X:[SimpleNamespace(predicted_total=-500.) for _ in range(len(X))]
    try:trainer._run_sanity_check(np.ones((2,8)))
    except ValueError:pass
    else:raise AssertionError('Trained total safety check was bypassed')
    rng=np.random.default_rng(42)
    def dataset(n,offset):
        X=rng.normal(size=(n,len(FEATURE_NAMES))).astype('float32');X[:,FEATURE_NAMES.index('shrunk_expected_total')]=225
        return TrainingDataset(X=X,y_win=rng.integers(0,2,n).astype(float),y_spread=rng.normal(0,8,n),
            y_total=225+rng.normal(0,6,n),sample_info=[dict(nba_game_id=str(i),season='2021-22',
                source_game_date=str(date(2022,1,1)+timedelta(days=i)),game_date=str(date(2022,1,1)+timedelta(days=i)))
                for i in range(offset,offset+n)])
    study.ml_settings.mlp_epochs=2;study.ml_settings.mlp_patience=2
    past,test=dataset(200,0),dataset(30,200);original=mlp_model.NBA_MLP
    with tempfile.TemporaryDirectory() as temporary:
        for name in study.CHOICES:
            values,detail=study.fit(past,test,name,Path(temporary),np.array([0,29]))
            assert np.isfinite(values).all() and len(values)==30 and detail['single_batch_samples']==2
            assert mlp_model.NBA_MLP is original
            metadata=json.loads((Path(temporary)/name/'metadata.json').read_text())
            assert metadata['targets']==['total'] and metadata['auxiliary_heads_untrained']
    print(json.dumps(dict(status='PASS',kind='synthetic_total_only_gradient_refit_smoke',real_labels_used=False,
        unused_head_gradients_absent=True,total_gradient_matches_explicit_loss=True,
        stopping_ignores_auxiliary_labels=True,trained_target_safety_retained=True,factory_restored=True)))


if __name__=='__main__':main()
