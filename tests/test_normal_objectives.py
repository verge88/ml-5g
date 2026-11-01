import numpy as np
import pytest
import torch
from paper5g.normal_objectives import Backbone,batch,fit,score,loss_or_score,metrics
from paper5g.benchmark import seed_all

@pytest.fixture
def fixture():
    rng=np.random.default_rng(3);n=32;width=65
    x=rng.normal(size=(n,width)).astype('float32');x[:,:8]=0
    s=np.arange(n,dtype='int64')%3+1;d=s%3+1
    x[np.arange(n),s]=1;x[np.arange(n),4+d]=1
    for a,b in [(8,15),(15,23),(23,55)]:
        x[:,a:b]=0;x[np.arange(n),a+np.arange(n)%(b-a)]=1
    return dict(x=x,source=s,destination=d,y=np.zeros(n,dtype='int8'),time=np.arange(n,dtype=float)),dict(identity_dim=8,n_nodes=4,base_dim=width,center=[0.]*width,scale=[1.]*width)

@pytest.mark.parametrize('goal',['permutation','next','masked'])
@pytest.mark.parametrize('mode',['current','history','joint'])
def test_real_fitting_and_label_blindness(fixture,goal,mode):
    stream,info=fixture;torch.set_num_threads(1);seed_all(11)
    model=Backbone(65,mode);details=fit(model,stream,info,goal,11)
    assert details['changed_tensors']>0
    a=score(model,stream,info,goal)
    b=score(model,{**stream,'y':np.ones(32,dtype='int8')},info,goal)
    np.testing.assert_array_equal(a,b);assert len(a)==32 and np.isfinite(a).all()

def test_history_excludes_current_and_future(fixture):
    s,info=fixture;x,h,l=batch(s,8,9)
    changed={**s,'x':s['x'].copy()};changed['x'][8:]+=123
    xx,hh,ll=batch(changed,8,9)
    torch.testing.assert_close(h,hh,rtol=0,atol=0);assert not torch.equal(x,xx)

def test_next_logits_cannot_copy_target_operation(fixture):
    s,info=fixture;model=Backbone(65,'joint');model.eval()
    x,h,l=batch(s,8,9);other=x.clone();other[:,8:55]+=19
    x[:,8:55]=0;other[:,8:55]=0
    torch.testing.assert_close(model.forecast(model(x,h,l)),model.forecast(model(other,h,l)),rtol=0,atol=0)

def test_known_subset_preserves_unknown_exclusion(fixture):
    s,_=fixture;s={**s,'y':np.array([1,1,0,-1],dtype='int8')}
    result=metrics(s,np.array([.9,.9,.9,.9]),.5,np.array([True,False,True,True]))
    assert result['tp']==1 and result['positive']==1 and result['fp']==1 and result['labelled']==2
