import numpy as np
import pytest
import torch
from paper5g.mechanism import Ablation,objective,train
from paper5g.engines import predict_links
from paper5g.benchmark import seed_all
from paper5g.prepare import P

@pytest.mark.parametrize('current',[False,True])
@pytest.mark.parametrize('history',[False,True])
def test_only_intended_path_is_removed(current,history,monkeypatch):
    for k,v in dict(epochs=2,batch_size=8,hidden=16,temporal_neighbors=2).items():monkeypatch.setitem(P,k,v)
    torch.set_num_threads(1);seed_all(11);rng=np.random.default_rng(4)
    x=rng.normal(size=(32,10)).astype('float32');src=np.arange(32,dtype='int64')%3+1;dst=(src%3)+1
    stream=dict(x=x,source=src,destination=dst,y=np.zeros(32,dtype='int8'))
    info=dict(n_nodes=4,center=[0.]*10,scale=[1.]*10)
    eng=Ablation(10,info,current,history);d=train(eng,stream,info,11,True)
    assert d['changed_tensors']>0
    if not history:assert torch.count_nonzero(eng.model.memory)==0 and not eng.model.pending_mask.any()
    eng.train(False);eng.reset(40)
    first=eng.score(stream,0,8).detach().clone()
    changed={**stream,'x':stream['x']+7}
    second=eng.score(changed,0,8).detach().clone()
    if not current:torch.testing.assert_close(first,second,rtol=0,atol=0)
    else:assert not torch.equal(first,second)
    a=predict_links(eng,stream,stream)
    b=predict_links(eng,stream,{**stream,'y':np.ones(32,dtype='int8')})
    np.testing.assert_array_equal(a,b)

def test_collisions_contribute_no_negative_gradient():
    p=torch.tensor([.2,.3],requires_grad=True);n=torch.tensor([.7,.9],requires_grad=True)
    objective(p,n,np.array([True,False]),True).backward()
    assert n.grad[0]==0 and n.grad[1]!=0 and torch.all(p.grad!=0)

def test_all_collision_batch_still_trains_normal_class():
    p=torch.tensor([.2],requires_grad=True);n=torch.tensor([.7],requires_grad=True)
    loss=objective(p,n,np.array([True]),True);loss.backward()
    assert torch.isfinite(loss) and p.grad is not None and n.grad is None
