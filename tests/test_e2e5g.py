import base64,copy,json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from e2e5g.schedule import Scheduler,predict,raw_features
from e2e5g.evidence import receipt
from e2e5g.controls import cases
import pytest

def observation(missing=False):
    return dict(direct=dict(ok=not missing,elapsed_ms=400 if missing else 1),nrf=dict(ok=True,status=200,elapsed_ms=1,body=dict(nfInstances=[dict(nfStatus='REGISTERED')])))

def test_budget_and_starvation_bound():
    models={s:dict(mean=[0]*6,scale=[1]*6,coef=[8,0,0,0,0,0],intercept=-4) for s in ['ausf','nssf']}
    for mode in ['fixed','risk','ml_risk']:
        scheduler=Scheduler(mode,models);picked=[]
        for tick in range(100):
            s,d=scheduler.choose(tick,dict(ausf=observation(True),nssf=observation()))
            picked.append(s);assert max(d['ages'].values())<=4
        assert len(picked)==100 and set(picked)=={'ausf','nssf'}
        if mode!='fixed':assert picked.count('ausf')>picked.count('nssf')

def test_public_receipt_binding_tamper_and_age():
    key=Ed25519PrivateKey.generate();public=dict(public_key=base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode(),boot_id='guest')
    request=dict(nonce='a'*32,body_sha256='b'*64);p=dict(schema='e2e-application-receipt-v1',**request,boot_id='guest')
    row=dict(returncode=0,interface='uesimtun0',unix_ns=1_000_000_000,request=request,response=dict(payload=p,signature=base64.b64encode(key.sign(json.dumps(p,sort_keys=True,separators=(',',':')).encode())).decode()))
    assert receipt(row,public,now_ns=1_000_000_000)
    assert not receipt(row,public,now_ns=6_000_000_000)
    bad=copy.deepcopy(row);bad['request']['nonce']='c'*32;assert not receipt(bad,public,now_ns=1_000_000_000)
    bad=copy.deepcopy(row);bad['response']['payload']['other']='tamper'
    with pytest.raises(Exception):receipt(bad,public,now_ns=1_000_000_000)

def test_authorization_contract_never_promotes_missing_context():
    report=cases();assert len(report['cases'])==14 and report['passed']
