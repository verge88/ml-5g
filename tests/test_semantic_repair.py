import numpy as np
import pytest,torch
from paper5g.semantic_repair import Repair,fit,predict,batch,codes,vocabularies
from paper5g.benchmark import seed_all

@pytest.fixture
def sample():
    rng=np.random.default_rng(7);cards=[4,4,8,5,9,2];cat=np.stack([rng.integers(n,size=32) for n in cards],1);num=rng.normal(size=(32,8)).astype('float32')
    x=np.c_[*[np.eye(n,dtype='float32')[cat[:,j]] for j,n in enumerate(cards)],num,np.zeros((32,4),dtype='float32')].astype('float32')
    return dict(cat=cat,num=num,x=x,y=np.zeros(32,dtype='int8')),cards

@pytest.mark.parametrize('objective',['ce','mse'])
@pytest.mark.parametrize('mode',['current','global'])
def test_fit_changes_weights_and_scores_are_label_blind(sample,objective,mode):
    s,cards=sample;torch.set_num_threads(1);seed_all(11);model=Repair(cards,mode);detail=fit(model,s,objective)
    assert detail['changed_tensors']>0 and len(detail['losses'])==12
    a=predict(model,s,objective);b=predict(model,{**s,'y':np.ones(32,dtype='int8')},objective)
    np.testing.assert_array_equal(a,b);assert np.isfinite(a).all()

def test_current_and_future_excluded_from_history(sample):
    s,_=sample;_,a,n=batch(s,8,9);changed={**s,'x':s['x'].copy()};changed['x'][8:]+=123
    _,b,nn=batch(changed,8,9);torch.testing.assert_close(a,b,rtol=0,atol=0);assert n.item()==nn.item()==8

@pytest.mark.parametrize('group',[0,1,2,3])
def test_hidden_fields_cannot_be_copied(sample,group):
    s,cards=sample;model=Repair(cards,'global');x,h,l=batch(s,8,9);other=x.clone();start=sum(cards[:group+2]) if group<3 else sum(cards);end=sum(cards[:group+3]) if group<3 else sum(cards)+8
    other[:,start:end]+=7
    torch.testing.assert_close(model.masked_logits(x,h,l,group),model.masked_logits(other,h,l,group),rtol=0,atol=0)

def test_exact_vocabulary_never_learns_test_categories():
    rows=[dict(source='a',destination='b',service='one',path='/one/a',method='GET')];voc=vocabularies(rows);nodes={'a':1,'b':2}
    test=[dict(source='a',destination='b',service='new',path='/new/x',method='POST')]
    cat,cards=codes(test,voc,'exact',nodes);assert cat[0,3]==cat[0,4]==0 and cards[3:5]==[2,2]
    assert list(voc[2])==['one'] and list(voc[3])==['/one/a']
