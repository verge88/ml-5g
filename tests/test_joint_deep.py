import numpy as np,pytest,torch
from joint5g.deep import windows,Net,score

def test_windows_do_not_cross_future():
 x=np.arange(60,dtype='float32').reshape(20,3);a,c=windows(x,16);changed=x.copy();changed[9:]+=100;b,d=windows(changed,16)
 np.testing.assert_array_equal(a[:9],b[:9]);np.testing.assert_array_equal(a[8,:9],x[:9]);assert c[8]==9

@pytest.mark.parametrize('arch',['mlp','lstm','transformer'])
def test_real_gradient_fit_and_masked_padding(arch):
 torch.set_num_threads(1);torch.manual_seed(11);model=Net(3,arch,16);x,c=windows(np.random.default_rng(3).normal(size=(20,3)).astype('float32'),16);old=score(model,x,c)
 opt=torch.optim.AdamW(model.parameters(),lr=.001);model.train();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x),torch.from_numpy(c)),torch.arange(20).remainder(2).float());loss.backward();opt.step();new=score(model,x,c)
 assert np.isfinite(new).all() and not np.array_equal(old,new)
 changed=x.copy()
 for i,n in enumerate(c):changed[i,n:]=999
 np.testing.assert_allclose(score(model,changed,c),new,rtol=1e-5,atol=1e-6)
