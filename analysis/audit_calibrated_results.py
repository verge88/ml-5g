"""Read-only artifact verification independent of the experiment summarizer."""
import hashlib
import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=Path(__file__).parent/'calibrated-probe-results')
ROOT=parser.parse_args().input
rows=json.loads((ROOT/'per_trace.json').read_text(encoding='utf-8'))
result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
keys=[tuple(r[k] for k in ['seed','load','work','mode','policy','mechanism','burst','detector','calibration']) for r in rows]
assert len(keys)==len(set(keys))==149760
assert {r['seed'] for r in rows}==set(range(12001,12121))
physical=defaultdict(list)
no_interference={}
for r in rows:
    assert 0<=r['queries']<=72
    assert 0<=r['request_fresh_fraction']<=r['completion_fresh_fraction']<=1
    assert 0<=r['unresolved_probes']<=r['queries']
    key=tuple(r[k] for k in ['seed','load','work','mode','policy','mechanism','burst'])
    physical[key].append(r)
    if r['mode']=='isolated' or r['policy']=='passive':
        k=tuple(r[k] for k in ['seed','load','work','mechanism','burst'])
        if k in no_interference:assert r['user_p95']==no_interference[k]
        else:no_interference[k]=r['user_p95']
for rr in physical.values():
    assert len(rr)==4
    for field in ['queries','user_p95','request_fresh_fraction','completion_fresh_fraction']:
        assert len({r[field] for r in rr})==1

def cp_upper(k,n):
    if k==n:return 1.
    lo,hi=0.,1.
    for _ in range(80):
        p=(lo+hi)/2
        cdf=sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
        if cdf>.05:lo=p
        else:hi=p
    return (lo+hi)/2

for r in result['summaries']:
    assert abs(cp_upper(r['false_events'],r['healthy_deployments'])-r['false_event_upper95'])<1e-10

primary=defaultdict(list)
for r in rows:
    if r['load']==.92 and r['work']==.6 and r['mode']=='shared' and not r['burst'] and r['detector']=='full' and r['calibration']=='block':
        primary[(r['seed'],r['policy'])].append(r)
lat=[];rec=[];diff=[];fresh=[]
for seed in range(12001,12121):
    aa=primary[(seed,'guarded_risk')];bb=primary[(seed,'risk')]
    a=sum(r['user_p95'] for r in aa)/4;b=sum(r['user_p95'] for r in bb)/4
    ar=sum(r['detected_within30'] for r in aa if r['mechanism'])/3
    br=sum(r['detected_within30'] for r in bb if r['mechanism'])/3
    lat.append((a-b)/b);rec.append(ar);diff.append(ar-br)
    fresh.append(sum(r['request_fresh_fraction'] for r in aa)/4)
for values,field in [(lat,'latency_relative'),(rec,'guarded_detection'),(diff,'detection_difference')]:
    assert abs(sum(values)/120-result['primary'][field]['mean'])<1e-10
assert abs(sum(fresh)/120-result['primary']['request_fresh_fraction'])<1e-10
audit=dict(unique_diagnostic_rows=len(keys),physical_simulations=len(physical),independent_seeds=120,
    budget_and_request_freshness_verified=True,acquisition_identical_across_detectors_and_thresholds=True,
    primary_point_estimates_recomputed=True,all_clopper_pearson_upper_limits_independently_verified=True,
    test_model_fits=result['test_model_fits'],result_sha256=hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),
    scope='Read-only audit of the same Actions artifact; not an independent replication')
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
