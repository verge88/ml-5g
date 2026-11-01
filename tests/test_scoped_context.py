import base64,csv,json
import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from paper5g.context_prepare import history_indices
from paper5g.normal_objectives import batch
from paper5g.bound_observations import read_bound,canonical
from paper5g.scoped_context import load_data
from semantic5g.analyze import read_csv

def signed(observations,capture='abc',deployment='test'):
    key=ec.generate_private_key(ec.SECP256R1());payload=dict(schema='paper5g-bound-observations-v1',capture_sha256=capture,deployment_id=deployment,observations=observations)
    pem=key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
    signature=key.sign(canonical(payload),ec.ECDSA(hashes.SHA256()))
    return dict(payload=payload,signature=base64.b64encode(signature).decode()),pem

def test_scope_isolation_and_strict_past():
    h=history_indices(['a','b','a','b','a',None]);assert h[0].max()==-1
    np.testing.assert_array_equal(h[4,-2:],[0,2]);np.testing.assert_array_equal(h[3,-1:],[1]);assert h[5].max()==-1
    assert np.all((h<0)|(h<np.arange(6)[:,None]))
    s=dict(x=np.arange(6*3,dtype='float32').reshape(6,3),history_index=h)
    _,past,length=batch(s,4,5);assert length.item()==2
    np.testing.assert_array_equal(past[0,:2],s['x'][[0,2]])
    s['x'][3]+=1000;_,past2,_=batch(s,4,5);torch.testing.assert_close(past,past2,rtol=0,atol=0)

def test_future_history_rejected():
    with pytest.raises(AssertionError,match='future'):batch(dict(x=np.zeros((2,3),dtype='float32'),history_index=np.ones((2,16),dtype='int64')),0,1)

def test_correlation_does_not_change_old_parser(tmp_path):
    p=tmp_path/'sample.csv';fields=['_ws.col.info','ip.src','srcport','ip.dst','dstport','frame.number','frame.time_epoch','Label']
    infos=['[SYN]','HEADERS[1]: GET /nnrf-nfm/v1/nf-instances','HEADERS[3]: GET /nudm-sdm/v2/imsi-123/am-data','[SYN]','HEADERS[1]: GET /nnrf-nfm/v1/nf-instances']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,text in enumerate(infos):w.writerow(dict(zip(fields,[text,'a','10','b','80',i+1,float(i),'normal'])))
    old,_=read_csv(p);new,_=read_csv(p,retain_correlation=True)
    assert old==[{k:v for k,v in r.items() if k!='transport_key'} for r in new]
    assert new[0]['transport_key']==new[1]['transport_key'] and new[0]['transport_key']!=new[2]['transport_key']

def test_bound_values_distinguish_failure_from_missing():
    b,pem=signed([dict(request_index=0,observed_at=9.,procedure_id='p',checks=dict(integrity=False,authorization=True))])
    v,p,m=read_bound(b,pem,'abc','test',[10.,11.])
    np.testing.assert_array_equal(v[0],[0,1,0,1,1,0]);np.testing.assert_array_equal(v[1],np.zeros(6));np.testing.assert_array_equal(p,[0,-1]);assert m['matched_requests']==1

@pytest.mark.parametrize('capture,deployment',[('other','test'),('abc','other')])
def test_cross_capture_or_deployment_rejected(capture,deployment):
    b,pem=signed([])
    with pytest.raises(ValueError,match='binding'):read_bound(b,pem,capture,deployment,[10.])

@pytest.mark.parametrize('issued',[4.,11.,float('inf')])
def test_stale_or_future_observations_rejected(issued):
    if not np.isfinite(issued):
        with pytest.raises(ValueError):signed([dict(request_index=0,observed_at=issued)])
        return
    b,pem=signed([dict(request_index=0,observed_at=issued)])
    with pytest.raises(ValueError,match='stale'):read_bound(b,pem,'abc','test',[10.])

def test_tampered_observation_rejected():
    b,pem=signed([dict(request_index=0,observed_at=9.,checks=dict(integrity=True))]);b['payload']['observations'][0]['checks']['integrity']=False
    with pytest.raises(InvalidSignature):read_bound(b,pem,'abc','test',[10.])

def test_wrong_collector_key_rejected():
    b,pem=signed([]);_,other=signed([])
    with pytest.raises(InvalidSignature):read_bound(b,other,'abc','test',[10.])

@pytest.mark.parametrize('observations',[
    [dict(request_index=0,observed_at=9.),dict(request_index=0,observed_at=9.)],
    [dict(request_index=2,observed_at=9.)],
    [dict(request_index=0,observed_at=9.,checks=dict(integrity=1))],
    [dict(request_index=0,observed_at=9.,checks=dict(label=True))],
    [dict(request_index=0,observed_at=9.,label='attack')]])
def test_invalid_or_label_fields_rejected(observations):
    b,pem=signed(observations)
    with pytest.raises(ValueError):read_bound(b,pem,'abc','test',[10.])

@pytest.mark.parametrize('arm',['procedure','procedure_witness'])
def test_missing_real_observations_block_full_comparison(tmp_path,arm):
    (tmp_path/'alignment.json').write_text(json.dumps(dict(eligible_procedure=False,eligible_procedure_witness=False)))
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='No complete'):load_data(tmp_path,arm)
