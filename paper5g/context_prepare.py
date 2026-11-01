"""Align original public requests to existing matrices; recover causal scope only."""
import argparse,hashlib,json
from collections import defaultdict,deque,Counter
from pathlib import Path
import numpy as np
from hybrid.fetch_data import fetch
from semantic5g.analyze import read_csv
from .prepare import partition,features
from .benchmark import load
from .bound_observations import read_bound

def history_indices(keys):
    past=defaultdict(lambda:deque(maxlen=16));out=np.full((len(keys),16),-1,dtype=np.int64)
    for i,key in enumerate(keys):
        if key is None:continue
        if past[key]:out[i,-len(past[key]):]=past[key]
        past[key].append(i)
    return out

def prepare(raw,original,out,observations=None,public_key=None,deployment_id=None):
    if any(v is not None for v in [observations,public_key,deployment_id]) and not all(v is not None for v in [observations,public_key,deployment_id]):raise ValueError('observations, pinned public key and deployment id must be provided together')
    fetch(raw);info,old=load(original,'context');rows={};capture_hashes={};bindings={}
    for filename,name in [('bridge_normal.csv','normal'),('bridge_attack_nrf.csv','nrf'),('bridge_attack_smf.csv','smf')]:
        path=raw/filename;sha=hashlib.sha256(path.read_bytes()).hexdigest();capture_hashes[name]=sha
        rr,_=read_csv(path,retain_correlation=True)
        if observations is None:
            checks=np.zeros((len(rr),6),dtype=np.float32);proc=np.full(len(rr),-1,dtype=np.int64);bindings[name]=dict(matched_requests=0,procedure_requests=0,check_requests=0)
        else:
            bundle=json.loads((observations/(path.stem+'.observations.json')).read_text())
            checks,proc,bindings[name]=read_bound(bundle,public_key.read_bytes(),sha,deployment_id,[r['time'] for r in rr])
        for i,r in enumerate(rr):
            r.update(original_annotation=r['label'],label=int(r['label']!='normal'),checks=checks[i],procedure=None if proc[i]<0 else int(proc[i]))
        rows[name]=sorted(rr,key=lambda r:r['time'])
    streams={**partition(rows['normal']),**{k:rows[k] for k in ['nrf','smf']}}
    nodes={v:i+1 for i,v in enumerate(sorted({r[k] for r in streams['train'] for k in ('source','destination')}))}
    out.mkdir(parents=True,exist_ok=True);manifest=dict(original_manifest_sha256=hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest(),
        captures=capture_hashes,bindings=bindings,streams={},input_dim=info['base_dim']+6,independent_new_deployments=0,
        notes=['Connection = directional transport endpoint tuple and observed TCP SYN generation; not a procedure',
               'Edge = actual directional endpoint pair, including unknown endpoints without conflating their histories',
               'Original labels and base features retained; response status is audit-only',
               'All scopes reset at the original partition boundaries; no connection/procedure IDs enter model vectors',
               'No real aligned collector observations supplied unless explicitly provided and verified'])
    for name,rr in streams.items():
        s=old[name];x,c,src,dst=features(rr,nodes);rebuilt=np.clip((np.concatenate([x,c],axis=1)-np.asarray(info['center'],dtype=np.float32))/np.asarray(info['scale'],dtype=np.float32),-20,20).astype(np.float32)
        np.testing.assert_array_equal(rebuilt,s['x']);np.testing.assert_array_equal(src,s['source']);np.testing.assert_array_equal(dst,s['destination'])
        np.testing.assert_array_equal([r['label'] for r in rr],s['y']);np.testing.assert_array_equal([r['time']-rr[0]['time'] for r in rr],s['time'])
        values={**s,'x':s['x'][:,:info['base_dim']].copy(),'witness':np.stack([r['checks'] for r in rr])}
        keys=dict(edge=[(r['source'],r['destination']) for r in rr],connection=[r['transport_key'] for r in rr],procedure=[r['procedure'] for r in rr])
        for scope,kk in keys.items():values['history_'+scope]=history_indices(kk)
        audit={}
        known=(src!=0)&(dst!=0)
        for annotation in sorted(set(r['original_annotation'] for r in rr)):
            ii=[i for i,r in enumerate(rr) if r['original_annotation']==annotation and known[i]]
            audit[annotation]=dict(known_requests=len(ii),responses=dict(Counter('unknown' if rr[i]['status'] is None else str(rr[i]['status']) for i in ii)))
        path=out/(name+'.npz');np.savez_compressed(path,**values)
        manifest['streams'][name]=dict(rows=len(rr),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),procedure_complete=all(r['procedure'] is not None for r in rr),
            witness_complete=bool(values['witness'][:,3:].any(1).all()),known_positive=int(((s['y']==1)&known).sum()),
            scopes={scope:dict(groups=len(set(k for k in kk if k is not None)),requests_with_past=int((values['history_'+scope]>=0).any(1).sum())) for scope,kk in keys.items()},annotation_audit=audit)
    manifest['eligible_procedure']=all(v['procedure_complete'] for v in manifest['streams'].values())
    manifest['eligible_procedure_witness']=manifest['eligible_procedure'] and all(v['witness_complete'] for v in manifest['streams'].values())
    (out/'alignment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');(out/'manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(dict(rows={k:v['rows'] for k,v in manifest['streams'].items()},procedure=manifest['eligible_procedure'],procedure_witness=manifest['eligible_procedure_witness'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observations',type=Path);p.add_argument('--public-key',type=Path);p.add_argument('--deployment-id');a=p.parse_args()
    prepare(a.raw,a.original,a.output,a.observations,a.public_key,a.deployment_id)
