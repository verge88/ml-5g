from pathlib import Path
import numpy as np
import pytest
import torch
from paper5g.prepare import P,build_dataset
from paper5g.benchmark import load,seed_all,svdd_fit,confusion
from paper5g.engines import Existing,PublishedTGN,causal_batch,fit_links,predict_links,negative_batch
from paper5g import references

@pytest.fixture
def data(tmp_path):
    rows=[dict(source='a' if i%3 else 'c',destination='b' if i%2 else 'a',time=i*.7,
        method='POST' if i%4 else 'GET',path='/service/v1/resource',wire='HTTP2',label=0,kind='normal') for i in range(160)]
    build_dataset('fixture',rows,{'mixed':[dict(r,label=int(i%3==0)) for i,r in enumerate(rows[:40])]},tmp_path)
    return load(tmp_path/'fixture','context')

def test_causal_windows_do_not_include_future_and_keep_current_last():
    x=np.arange(60,dtype=np.float32).reshape(20,3)
    batch,l=causal_batch(x,0,8,4)
    for i in range(8):np.testing.assert_array_equal(batch[i,l[i]-1],x[i])
    changed=x.copy();changed[8:]=-123
    np.testing.assert_array_equal(batch,causal_batch(changed,0,8,4)[0])

@pytest.mark.parametrize('name',['lstm','transformer','gnn','tgn','granomaly_tgn'])
def test_models_really_train_without_eval_label_access(data,monkeypatch,name,tmp_path):
    monkeypatch.setitem(P,'epochs',2);monkeypatch.setitem(P,'batch_size',16)
    torch.set_num_threads(1);seed_all(11);info,streams=data
    root=Path(__file__).resolve().parents[1]/'.references'
    if name=='granomaly_tgn':
        if not (root/'granomaly/tgn.py').exists():pytest.skip('reference modules fetched in CI controls')
        engine=PublishedTGN(streams['train']['x'].shape[1],info,root)
    else:engine=Existing(name,streams['train']['x'].shape[1],info)
    trained=fit_links(engine,streams['train'],info,11)
    assert trained['changed_parameter_tensors']>0 and len(trained['epoch_loss'])==2
    score=predict_links(engine,streams['train'],streams['mixed'])
    changed={**streams['mixed'],'y':1-streams['mixed']['y']}
    np.testing.assert_allclose(score,predict_links(engine,streams['train'],changed),rtol=0,atol=0)
    assert len(score)==len(streams['mixed']['y']) and np.isfinite(score).all()

def test_svdd_constraints_and_real_weight_update(data,monkeypatch):
    monkeypatch.setitem(P,'epochs',2);info,streams=data
    _,center,score,detail=svdd_fit(streams['train']['x'],11)
    assert detail['changed_parameter_tensors']>0 and not detail['bias_terms']
    assert torch.all(center.abs()>=.1) and np.isfinite(score(streams['mixed'])).all()

def test_unknown_labels_do_not_become_normal():
    result=confusion(np.array([0,1,-1]),np.array([.1,.9,.9]),.5)
    assert result['tp']==1 and result['tn']==1 and result['fp']==0

def test_original_kitnet_executes_without_fitting_test_rows(data):
    root=Path(__file__).resolve().parents[1]/'.references'
    if not (root/'kitnet/KitNET.py').exists():pytest.skip('reference modules fetched in CI controls')
    _,streams=data;cls=references.kitnet(root);x=streams['train']['x']
    model=cls(x.shape[1],FM_grace_period=8,AD_grace_period=len(x)-9)
    for row in x:model.train(row.astype(np.float64))
    before=model.n_trained
    scores=np.array([model.execute(row.astype(np.float64)) for row in streams['mixed']['x']])
    assert model.n_trained==before and np.isfinite(scores).all()
