import copy
import numpy as np
import pytest
import torch
from neural5g.data import P, Stream, api_vector, windows
from neural5g.models import make_model
from neural5g.benchmark import fit_neural, predict, seed_all

CONFIG = dict(P, hidden=16, attention_heads=4, epochs=2, batch_size=8, sequence_length=6,
    temporal_neighbors=3, torch_threads=1, dropout=0.)
torch.set_num_threads(1)


def stream(n=40):
    rng=np.random.default_rng(19)
    x=rng.normal(size=(n,6)).astype(np.float32)
    return Stream(x,np.arange(n,dtype=np.int64)%3+1,(np.arange(n,dtype=np.int64)+1)%3+1,
        np.arange(n,dtype=np.float64)+1,(x[:,0]>0).astype(np.float32),[{} for _ in range(n)],'fixture')


def model(name):
    seed_all(7)
    return make_model(name,6,4,CONFIG)


def test_windows_are_causal_and_right_padded():
    x=np.arange(12,dtype=np.float32).reshape(6,2)
    history,lengths=windows(x,3)
    assert lengths.tolist()==[1,2,3,3,3,3]
    assert np.array_equal(history[3],x[1:4])
    assert np.all(history[0,1:]==0)
    changed=x.copy();changed[4:]+=1000
    assert np.array_equal(windows(changed,3)[0][:4],history[:4])


def test_api_encoding_excludes_annotation_and_response_information():
    r=dict(method='PUT',path='/namf-callback/v1/{subscriber}/dereg-notify',label='normal',status=200,time=1,frame=1)
    assert np.array_equal(api_vector(r),api_vector(dict(r,label='attack',status=404,time=99999,frame=900)))
    assert not np.array_equal(api_vector(r),api_vector(dict(r,method='POST')))


@pytest.mark.parametrize('name',['tgn','gnn','lstm','transformer'])
def test_future_events_do_not_change_past_predictions(name):
    m=model(name);s=stream();original=predict(m,s,name,CONFIG)
    changed=copy.deepcopy(s);changed.x[25:]+=20;changed.destination[25:]=1
    result=predict(m,changed,name,CONFIG)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(original[:25],result[:25],atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('name',['lstm','transformer'])
def test_sequence_padding_cannot_change_valid_output(name):
    m=model(name).eval()
    x=torch.randn(3,6,6);lengths=torch.tensor([1,3,6])
    altered=x.clone();altered[0,1:]+=100;altered[1,3:]-=100
    with torch.no_grad():
        torch.testing.assert_close(m(x,lengths),m(altered,lengths),atol=1e-6,rtol=1e-6)


def test_transformer_layers_have_independent_initial_weights():
    m=model('transformer')
    assert not torch.equal(m.layers[0].self_attn.in_proj_weight,m.layers[1].self_attn.in_proj_weight)


def test_gnn_observes_only_after_scoring_and_learns_neighbors():
    m=model('gnn');s=stream(8)
    x,src,dst=torch.from_numpy(s.x),torch.from_numpy(s.source),torch.from_numpy(s.destination)
    before=m(x,src,dst)
    assert m.adjacency.sum()==0
    m.observe(x,src,dst)
    assert m.adjacency.sum()>0
    after=m(x,src,dst)
    assert not torch.allclose(before,after)
    after.sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in m.neighbor_layers.parameters() if p.grad is not None)>0


def test_tgn_memory_is_persistent_deferred_and_trainable():
    m=model('tgn');s=stream(8)
    args=[torch.from_numpy(a) for a in (s.x,s.source,s.destination,s.time)]
    m(*args)
    assert m.memory.abs().sum()==0 and not m.pending_mask.any()
    m.observe(*args)
    assert m.pending_mask.any() and m.memory.abs().sum()==0
    next_args=[a.clone() for a in args];next_args[-1]+=8
    m(*next_args)
    assert m.active_memory.abs().sum()>0
    m.reset()
    record=fit_neural(m,[stream()], 'tgn',7,CONFIG)
    assert all(record['max_gradient_l1_by_module'][k]>0 for k in ['updater','message','attention','time_encoder'])
    assert record['changed_parameter_tensors']>0


def test_tgn_uses_elapsed_time_not_absolute_epoch():
    m=model('tgn');s=stream()
    original=predict(m,s,'tgn',CONFIG)
    shifted=copy.deepcopy(s);shifted.time+=1700000000
    np.testing.assert_allclose(original,predict(m,shifted,'tgn',CONFIG),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('name',['tgn','gnn','lstm','transformer'])
def test_weight_checkpoint_round_trip_resets_capture_state(name,tmp_path):
    m=model(name);s=stream()
    original=predict(m,s,name,CONFIG)
    path=tmp_path/'weights.pt';torch.save(m.state_dict(),path)
    other=model(name);other.load_state_dict(torch.load(path,weights_only=True))
    np.testing.assert_allclose(original,predict(other,s,name,CONFIG),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('name',['tgn','gnn','lstm','transformer'])
def test_training_changes_weights_with_finite_loss(name):
    m=model(name)
    record=fit_neural(m,[stream()],name,7,dict(CONFIG,epochs=1))
    assert np.isfinite(record['epoch_loss']).all()
    assert record['changed_parameter_tensors']>0
