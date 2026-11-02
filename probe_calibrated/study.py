"""Separate fit, healthy calibration and held-out test stages."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .simulator import PLAN,PASSIVE,FEATURES,cells,simulate

ROOT=Path(__file__).parent
SCOPES=['passive','full']
CALIBRATIONS=['pooled_window','block']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest():
    return dict(plan_sha256=digest(ROOT/'protocol.json'),simulator_sha256=digest(ROOT/'simulator.py'),
                study_sha256=digest(ROOT/'study.py'),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__)


def fit(out,smoke=False):
    xs=[];ys=[]
    ranges=[range(*PLAN['driver_train_seeds']),range(*PLAN['model_train_seeds'])]
    if smoke:ranges=[list(v)[:1] for v in ranges]
    for seed in ranges[0]:
        for c in cells():
            if c['policy']!='periodic':continue
            for k in range(4):
                _,x,y,_=simulate(seed,c,k)
                xs.append(x[::4,:len(PASSIVE)]);ys.append((y[::4]!=0).astype(int))
    x=np.concatenate(xs);y=np.concatenate(ys)
    mean=x.mean(0);scale=np.maximum(x.std(0),.01)
    lr=LogisticRegression(C=1.,class_weight='balanced',max_iter=500,random_state=19)
    lr.fit((x-mean)/scale,y)
    driver=dict(mean=mean,scale=scale,coef=lr.coef_[0],intercept=float(lr.intercept_[0]))
    xs=[];ys=[]
    for seed in ranges[1]:
        for c in cells():
            for k in range(4):
                _,x,y,_=simulate(seed,c,k,driver)
                xs.append(x[::4]);ys.append(y[::4])
    x=np.concatenate(xs);y=np.concatenate(ys)
    models={}
    for scope in SCOPES:
        m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,learning_rate=.08,
            l2_regularization=1.,class_weight='balanced',early_stopping=False,random_state=19)
        m.fit(x[:,:len(PASSIVE)] if scope=='passive' else x,y)
        assert m.classes_.tolist()==[0,1,2,3]
        models[scope]=m
    joblib.dump(dict(driver=driver,models=models),out/'models.joblib')
    write(out/'frozen.json',dict(**manifest(),model_sha256=digest(out/'models.joblib'),
          driver_seed_range=PLAN['driver_train_seeds'],model_seed_range=PLAN['model_train_seeds'],
          feature_names=FEATURES,training_rows=len(y),smoke=smoke,hyperparameter_search=False))
    print(json.dumps(dict(stage='fit',training_rows=len(y),smoke=smoke)),flush=True)


def load_models(root,smoke=False):
    meta=read(root/'frozen.json')
    for k,v in manifest().items():
        if k.endswith('_sha256'):assert meta[k]==v,(k,meta[k],v)
    assert meta['model_sha256']==digest(root/'models.joblib')
    if not smoke:assert not meta['smoke']
    return joblib.load(root/'models.joblib'),meta


def scores(models,x):
    result={}
    for scope in SCOPES:
        m=models[scope]
        p=m.predict_proba(x[:,:len(PASSIVE)] if scope=='passive' else x)
        result[scope]=(np.asarray(1.-p[:,0],dtype=np.float32),1+p[:,1:].argmax(1))
    return result


def conformal_threshold(values,alpha=.05):
    values=np.asarray(values)
    rank=math.ceil((len(values)+1)*(1-alpha))
    return None if rank>len(values) else float(np.sort(values)[rank-1])


def exceeds(values,threshold):
    return np.zeros(len(values),dtype=bool) if threshold is None else np.asarray(values)>threshold


def calibration_shard(out,frozen,start,count,smoke=False):
    obj,meta=load_models(frozen,smoke)
    cs=cells();seeds=list(range(start,start+count))
    lo,hi=PLAN['healthy_calibration_seeds']
    if not smoke:assert lo<=start<start+count<=hi
    arr=np.empty((count,len(cs),len(SCOPES),PLAN['tick_count']-PLAN['burn_in']),dtype=np.float32)
    for i,seed in enumerate(seeds):
        for j,c in enumerate(cs):
            _,x,y,_=simulate(seed,c,0,obj['driver'])
            assert not y.any()
            ss=scores(obj['models'],x)
            for z,scope in enumerate(SCOPES):arr[i,j,z]=ss[scope][0]
        if (i+1)%10==0:print(f'healthy calibration completed {i+1}/{count}',flush=True)
    np.savez_compressed(out/'healthy_scores.npz',scores=arr,seeds=seeds)
    write(out/'shard.json',dict(**manifest(),model_sha256=meta['model_sha256'],seeds=seeds,
                              cells=cs,smoke=smoke,scores_sha256=digest(out/'healthy_scores.npz')))


def calibration_merge(out,frozen,inputs,smoke=False):
    _,meta=load_models(frozen,smoke)
    chunks=[];seeds=[]
    for path in sorted(inputs.glob('**/healthy_scores.npz')):
        mm=read(path.with_name('shard.json'))
        assert mm['model_sha256']==meta['model_sha256']
        assert mm['plan_sha256']==meta['plan_sha256'] and mm['cells']==cells()
        assert mm['scores_sha256']==digest(path)
        with np.load(path) as a:
            assert a['seeds'].tolist()==mm['seeds']
            seeds+=a['seeds'].tolist();chunks.append(a['scores'])
    assert len(seeds)==len(set(seeds))
    if not smoke:assert sorted(seeds)==list(range(*PLAN['healthy_calibration_seeds']))
    array=np.concatenate(chunks)
    thresholds={};maxima={}
    for j,c in enumerate(cells()):
        thresholds[str(j)]={};maxima[str(j)]={}
        for z,scope in enumerate(SCOPES):
            vv=array[:,j,z]
            mm=vv.max(1)
            thresholds[str(j)][scope]=dict(block=conformal_threshold(mm),
                pooled_window=float(np.quantile(vv,.99,method='higher')))
            maxima[str(j)][scope]=mm.tolist()
    write(out/'thresholds.json',dict(**manifest(),model_sha256=meta['model_sha256'],
        calibration_seeds=seeds,cells=cells(),thresholds=thresholds,calibration_maxima=maxima,
        smoke=smoke,alpha_block=.05,alpha_pooled_window=.01,test_accessed=False))
    print(json.dumps(dict(stage='calibration_merge',healthy_deployments_per_cell=len(seeds))),flush=True)


def diagnostic_metrics(base,y,score,classes,threshold,scope,method):
    alarm=exceeds(score,threshold)
    pred=np.where(alarm,classes,0).astype(int)
    cm=np.zeros((4,4),dtype=int);np.add.at(cm,(y,pred),1)
    onset=PLAN['impairment_interval'][0]-PLAN['burn_in']
    episode=(y!=0)
    first=np.flatnonzero(alarm & episode)
    short=slice(onset,onset+30)
    truth_class=base['mechanism']
    normal=y==0
    return dict(**base,detector=scope,calibration=method,threshold=threshold,confusion=cm.tolist(),
        healthy_event_false_alarm=bool(alarm.any()) if truth_class==0 else None,
        normal_window_false_alarms=int((alarm & normal).sum()),normal_windows=int(normal.sum()),
        detected_within30=bool(alarm[short].any()) if truth_class else None,
        correct_mechanism_within30=bool((pred[short]==truth_class).any()) if truth_class else None,
        detection_delay=int(first[0]-onset) if len(first) else None,
        total_alarm_windows=int(alarm.sum()),scored_windows=len(y))


def test_shard(out,frozen,threshold_root,start,count,smoke=False):
    obj,meta=load_models(frozen,smoke)
    th=read(threshold_root/'thresholds.json')
    assert th['model_sha256']==meta['model_sha256'] and th['plan_sha256']==meta['plan_sha256']
    if not smoke:assert not th['smoke']
    assert set(range(start,start+count)).isdisjoint(th['calibration_seeds'])
    lo,hi=PLAN['test_seeds']
    if not smoke:assert lo<=start<start+count<=hi
    rows=[]
    for i,seed in enumerate(range(start,start+count)):
        for j,c in enumerate(cells()):
            for k in range(4):
                b,x,y,_=simulate(seed,c,k,obj['driver'])
                ss=scores(obj['models'],x)
                for scope,(score,classes) in ss.items():
                    for method in CALIBRATIONS:
                        rows.append(diagnostic_metrics(b,y,score,classes,th['thresholds'][str(j)][scope][method],scope,method))
            if c['load']==.70:
                b,x,y,_=simulate(seed,c,0,obj['driver'],burst=True)
                for scope,(score,classes) in scores(obj['models'],x).items():
                    for method in CALIBRATIONS:
                        rows.append(diagnostic_metrics(b,y,score,classes,th['thresholds'][str(j)][scope][method],scope,method))
        if (i+1)%5==0:print(f'test completed {i+1}/{count}',flush=True)
    write(out/'per_trace.json',rows)
    write(out/'test_manifest.json',dict(**manifest(),model_sha256=meta['model_sha256'],
        thresholds_sha256=digest(threshold_root/'thresholds.json'),seeds=list(range(start,start+count)),
        rows=len(rows),test_model_fits=0,smoke=smoke,per_trace_sha256=digest(out/'per_trace.json')))


def bootstrap(values):
    a=np.array(values);rng=np.random.default_rng(6741)
    rr=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(rr,[.025,.975]).tolist(),n=len(a))


def upper_binomial(k,n):
    return 1. if k==n else float(beta.ppf(.95,k+1,n-k))


def macro_f1(cm):
    a=np.asarray(cm);den=a.sum(0)+a.sum(1)
    return float(np.divide(2*a.diagonal(),den,out=np.zeros(4),where=den>0).mean())


def report(out,inputs,smoke=False):
    rows=[];manifests=[]
    for p in sorted(inputs.glob('**/per_trace.json')):
        mm=read(p.with_name('test_manifest.json'))
        assert mm['per_trace_sha256']==digest(p)
        assert mm['plan_sha256']==manifest()['plan_sha256']
        assert mm['test_model_fits']==0
        rows+=read(p);manifests.append(mm)
    assert len({m['model_sha256'] for m in manifests})==1
    assert len({m['thresholds_sha256'] for m in manifests})==1
    seeds=sum([m['seeds'] for m in manifests],[])
    assert len(seeds)==len(set(seeds))
    if not smoke:assert sorted(seeds)==list(range(*PLAN['test_seeds']))
    keys=[tuple(r[k] for k in ['seed','load','work','mode','policy','mechanism','burst','detector','calibration']) for r in rows]
    assert len(keys)==len(set(keys))==len(seeds)*(len(cells())*4+24)*4
    groups={}
    for r in rows:
        key=tuple(r[k] for k in ['load','work','mode','policy','burst','detector','calibration','seed'])
        groups.setdefault(key,[]).append(r)
    units={}
    for key,rr in groups.items():
        healthy=[r for r in rr if r['mechanism']==0]
        assert len(healthy)==1
        abnormal=[r for r in rr if r['mechanism']!=0]
        units[key]=dict(p95=float(np.mean([r['user_p95'] for r in rr])),
            f1=macro_f1(np.sum([r['confusion'] for r in rr],0)),
            recall=float(np.mean([r['detected_within30'] for r in abnormal])) if abnormal else 0.,
            mechanism_recall=float(np.mean([r['correct_mechanism_within30'] for r in abnormal])) if abnormal else 0.,
            false_event=int(healthy[0]['healthy_event_false_alarm']),
            normal_window_fpr=sum(r['normal_window_false_alarms'] for r in rr)/sum(r['normal_windows'] for r in rr),
            queries=float(np.mean([r['queries'] for r in rr])),fresh=float(np.mean([r['request_fresh_fraction'] for r in rr])),
            receipt_fresh=float(np.mean([r['completion_fresh_fraction'] for r in rr])),
            late=float(np.mean([r['late_receipts'] for r in rr])))
    summaries=[]
    for key in sorted({k[:-1] for k in units}):
        uu=[units[(*key,s)] for s in sorted(seeds)]
        record=dict(zip(['load','work','mode','policy','burst','detector','calibration'],key))
        record.update({k:float(np.mean([u[k] for u in uu])) for k in uu[0]})
        k=sum(u['false_event'] for u in uu)
        record.update(false_events=k,healthy_deployments=len(uu),false_event_upper95=upper_binomial(k,len(uu)))
        summaries.append(record)
    a=PLAN['primary_cell']
    aa=[units[(a['load'],a['work'],a['mode'],'guarded_risk',False,'full','block',s)] for s in sorted(seeds)]
    bb=[units[(a['load'],a['work'],a['mode'],'risk',False,'full','block',s)] for s in sorted(seeds)]
    latency=bootstrap([(x['p95']-y['p95'])/y['p95'] for x,y in zip(aa,bb)])
    detection=bootstrap([x['recall'] for x in aa])
    contrast=bootstrap([x['recall']-y['recall'] for x,y in zip(aa,bb)])
    fa=[upper_binomial(sum(x['false_event'] for x in uu),len(uu)) for uu in [aa,bb]]
    fresh=float(np.mean([x['fresh'] for x in aa]))
    checks=dict(healthy_error=all(v<=.10 for v in fa),detection=detection['ci95'][0]>=.60,
                detection_noninferiority=contrast['ci95'][0]>-.05,
                latency=latency['ci95'][1]<-.10,freshness=fresh>=.80)
    primary=dict(latency_relative=latency,guarded_detection=detection,detection_difference=contrast,
                 healthy_false_alarm_upper95_guarded_risk=fa[0],healthy_false_alarm_upper95_risk=fa[1],
                 request_fresh_fraction=fresh,checks=checks,joint_success=all(checks.values()))
    result=dict(primary=primary,summaries=summaries,provenance=manifest(),independent_test_seeds=len(seeds),
        diagnostic_rows=len(rows),unique_physical_simulations=len(rows)//4,test_model_fits=0,smoke=smoke,
        model_sha256=manifests[0]['model_sha256'],thresholds_sha256=manifests[0]['thresholds_sha256'],
        comparator='pooled_window controls no deployment-level error; block calibration assumes exchangeable healthy deployments')
    write(out/'results.json',result)
    write(out/'per_trace.json',rows)
    lines=['# Calibrated queue intervention study','',
           'Synthetic model, no physical 5G, cryptography or security attacks.',
           f'Independent test seeds: {len(seeds)}; physical simulated traces: {len(rows)//4}.','',
           '## Predeclared primary conjunction','```json',json.dumps(primary,indent=2),'```','',
           '## All cells','',
           '| Load | Work | Mode | Policy | Burst | Detector | Calibration | p95 | Within30 recall | Healthy event FA | FA upper95 | Fresh |',
           '|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|'+ '|'.join(str(r[k]) if k in ['load','work','mode','policy','burst','detector','calibration'] else f'{r[k]:.4f}' for k in
            ['load','work','mode','policy','burst','detector','calibration','p95','recall','false_event','false_event_upper95','fresh'])+'|')
    lines+=['','Calibration is separate from testing; no model was refitted on test data.',
            'Only the primary conjunction is confirmatory. High pointwise accuracy or silence alone cannot satisfy it.',
            'Burst is an explicit distribution-shift stress test; the exchangeability guarantee does not apply.',
            'The probability guarantee is marginal over healthy calibration and future deployments, not conditional on this fixed calibration set.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(primary=primary,independent_test_seeds=len(seeds),physical_traces=len(rows)//4),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','calibrate','merge-calibration','test','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,default=Path('frozen'))
    parser.add_argument('--thresholds',type=Path,default=Path('calibrated'))
    parser.add_argument('--input',type=Path,default=Path('inputs'))
    parser.add_argument('--start',type=int)
    parser.add_argument('--count',type=int)
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if a.stage=='fit':fit(a.output,a.smoke)
        elif a.stage=='calibrate':calibration_shard(a.output,a.models,a.start,a.count,a.smoke)
        elif a.stage=='merge-calibration':calibration_merge(a.output,a.models,a.input,a.smoke)
        elif a.stage=='test':test_shard(a.output,a.models,a.thresholds,a.start,a.count,a.smoke)
        else:report(a.output,a.input,a.smoke)


if __name__=='__main__':main()
