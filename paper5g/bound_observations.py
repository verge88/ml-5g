"""Import independently collected request-bound observations; never infer checks from labels.

A signature authenticates the configured collector's report, not the correctness
or independence of its underlying physical measurements. No collector is run here.
"""
import base64,hashlib,json,math
import numpy as np
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec

CHECKS=('integrity','authorization','lifecycle')
def canonical(payload):return json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()

def read_bound(bundle,public_pem,capture_sha256,deployment_id,request_times,max_age=5.):
    key=serialization.load_pem_public_key(public_pem)
    if not isinstance(key,ec.EllipticCurvePublicKey) or not isinstance(key.curve,ec.SECP256R1):raise ValueError('expected pinned P-256 collector key')
    payload=bundle['payload'];key.verify(base64.b64decode(bundle['signature'],validate=True),canonical(payload),ec.ECDSA(hashes.SHA256()))
    if payload.get('schema')!='paper5g-bound-observations-v1':raise ValueError('unsupported schema')
    if payload.get('capture_sha256')!=capture_sha256 or payload.get('deployment_id')!=deployment_id:raise ValueError('capture/deployment binding mismatch')
    n=len(request_times);values=np.zeros((n,6),dtype=np.float32);procedure=np.full(n,-1,dtype=np.int64);ids={};seen=set()
    for o in payload['observations']:
        i=o['request_index']
        if type(i) is not int or not 0<=i<n or i in seen:raise ValueError('duplicate/out-of-range request')
        seen.add(i);now=float(request_times[i]);issued=float(o['observed_at'])
        if not math.isfinite(issued) or not math.isfinite(now) or not 0<=now-issued<=max_age:raise ValueError('future/stale observation')
        if set(o)-{'request_index','observed_at','procedure_id','checks'}:raise ValueError('unsupported observation fields')
        checks=o.get('checks',{})
        if set(checks)-set(CHECKS):raise ValueError('unsupported check')
        for j,name in enumerate(CHECKS):
            value=checks.get(name)
            if value is not None:
                if type(value) is not bool:raise ValueError('check must be boolean or missing')
                values[i,j]=int(value);values[i,j+3]=1
        pid=o.get('procedure_id')
        if pid is not None:
            if not isinstance(pid,str) or not pid:raise ValueError('invalid procedure identity')
            if pid not in ids:ids[pid]=len(ids)
            procedure[i]=ids[pid]
    return values,procedure,dict(matched_requests=len(seen),procedure_requests=int((procedure>=0).sum()),
        check_requests=int(values[:,3:].any(1).sum()),collector_public_key_sha256=hashlib.sha256(public_pem).hexdigest(),
        trust='Pinned collector assertions; signature alone does not prove independent measurement')
