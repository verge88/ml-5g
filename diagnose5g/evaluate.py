import argparse,csv,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from .data import PLAN,CLASSES,eligible,features,load
from .policy import ProbePolicy


def metrics(y,p):
    return dict(rows=len(y),accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
                confusion=confusion_matrix(y,p,labels=range(4)).tolist())


def replay(trial,model,mode,cutoff):
    policy=ProbePolicy(mode,PLAN['optional_probe_budget_per_trial'],PLAN['optional_maximum_age_seconds'])
    first=trial['rows'][0]['monotonic_ns']/1e9;last_value=None;last_time=None;records=[]
    for tick,row in enumerate(trial['rows']):
        now=row['monotonic_ns']/1e9-first
        if not eligible(row):
            records.append(dict(tick=tick,eligible=False,queried=False,prediction=None,committed=False));continue
        end=float(row['measured']['end_ms'])
        age=now-last_time if last_time is not None else None
        retained=last_value if age is not None and age<=PLAN['optional_maximum_age_seconds'] else None
        prior=model.posterior(end,retained,age or 0.)
        benefit=model.expected_risk_reduction(prior)
        query,reason=policy.choose(tick,now,benefit)
        # The bypass value and even its response count are accessed only here.
        if query:
            m=row['measured']
            if m['bypass_signed']>=2 and m['bypass_ms'] is not None and np.isfinite(m['bypass_ms']):
                last_value=float(m['bypass_ms']);last_time=now;policy.observe_valid(now)
        age=now-last_time if last_time is not None else None
        retained=last_value if age is not None and age<=PLAN['optional_maximum_age_seconds'] else None
        posterior=model.posterior(end,retained,age or 0.)
        records.append(dict(tick=tick,eligible=True,queried=query,reason=reason,benefit=benefit,age=age,
            optional_fresh=retained is not None,prediction=int(posterior.argmax()),confidence=float(posterior.max()),committed=bool(posterior.max()>=cutoff)))
    assert policy.used<=PLAN['optional_probe_budget_per_trial']
    return dict(replica=trial['replica'],trial=trial['name'],mode=mode,queries=policy.used,records=records)


def run(root,frozen,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);frozen=Path(frozen)
    meta=json.loads((frozen/'frozen.json').read_text())
    assert meta['plan_sha256']==hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest()
    assert meta['model_sha256']==hashlib.sha256((frozen/'models.joblib').read_bytes()).hexdigest()
    assert {p['replica'] for p in meta['train_source']}==set(PLAN['train'])
    assert {p['replica'] for p in meta['calibration_source']}==set(PLAN['calibration'])
    models=joblib.load(frozen/'models.joblib');test,source=load(root,PLAN['test'])
    names=['gaussian_end','gaussian_both','logistic_end','logistic_both','hgb_end','hgb_both']
    all_y=[];all_predictions={n:[] for n in names};by_deployment=[];effects=[];coverage=[];replays=[]
    for trial in test:
        rows=trial['rows'];steady=rows[PLAN['steady_start']:]
        complete=[r for r in steady if eligible(r,True)]
        coverage.append(dict(replica=trial['replica'],trial=trial['name'],steady_rows=len(steady),
            eligible_end=sum(eligible(r) for r in steady),complete_rows=len(complete)))
        x=np.array([features(r) for r in complete]);y=[trial['label']]*len(x);all_y+=y
        predictions={}
        if len(x):
            predictions['gaussian_end']=[int(models['gaussian'].posterior(a[0]).argmax()) for a in x]
            predictions['gaussian_both']=[int(models['gaussian'].posterior(*a).argmax()) for a in x]
            for name in names[2:]:predictions[name]=models[name].predict(x[:,:1] if name.endswith('_end') else x).astype(int).tolist()
            for name in names:all_predictions[name]+=predictions[name]
        by_deployment.append(dict(replica=trial['replica'],trial=trial['name'],label=trial['label'],truth=y,predictions=predictions))
        base=[r for r in rows[:8] if eligible(r,True)]
        if base and complete:
            b=np.median([features(r) for r in base],axis=0);s=np.median(x,axis=0)
            effects.append(dict(replica=trial['replica'],trial=trial['name'],family=CLASSES[trial['label']],
                baseline_end_ms=float(b[0]),steady_end_ms=float(s[0]),end_delta_ms=float(s[0]-b[0]),
                baseline_bypass_ms=float(b[1]),steady_bypass_ms=float(s[1]),bypass_delta_ms=float(s[1]-b[1])))
        for mode in ['fixed','oldest','expected_risk']:replays.append(replay(trial,models['gaussian'],mode,meta['abstention_cutoff']))
    summary={n:metrics(all_y,all_predictions[n]) for n in names}
    grouped=[]
    for replica in PLAN['test']:
        units=[r for r in by_deployment if r['replica']==replica];y=sum([r['truth'] for r in units],[])
        grouped.append(dict(replica=replica,models={n:metrics(y,sum([r['predictions'].get(n,[]) for r in units],[])) for n in names}))
    replay_summary={}
    for mode in ['fixed','oldest','expected_risk']:
        rr=[r for r in replays if r['mode']==mode];pairs=[]
        for trial,r in zip(test,[r for r in replays if r['mode']==mode]):
            pairs.extend((trial['label'],v) for v in r['records'][PLAN['steady_start']:])
        committed=[(y,r) for y,r in pairs if r['eligible'] and r['committed']]
        eligible_pairs=[(y,r) for y,r in pairs if r['eligible']]
        replay_summary[mode]=dict(queries=sum(r['queries'] for r in rr),maximum_queries=10*len(test),
            steady_rows=len(pairs),eligible_rows=len(eligible_pairs),committed=len(committed),
            correct_committed=sum(y==r['prediction'] for y,r in committed),wrong_committed=sum(y!=r['prediction'] for y,r in committed),
            abstained_or_unavailable=len(pairs)-len(committed),fresh_optional=sum(r.get('optional_fresh',False) for y,r in eligible_pairs),
            raw_accuracy=float(np.mean([y==r['prediction'] for y,r in eligible_pairs])) if eligible_pairs else None,
            physical_savings_measured=False)
    result=dict(source_commit=meta['source_commit'],frozen_sha256=hashlib.sha256((frozen/'frozen.json').read_bytes()).hexdigest(),test_source=source,
        classes=CLASSES,models=summary,by_deployment=grouped,coverage=coverage,effects=effects,replay=replay_summary,
        test_model_fits=0,independent_test_deployments=len(PLAN['test']),scientific_priority_established=False,
        scope=PLAN['limits'])
    (out/'results.json').write_text(json.dumps(result,indent=2));(out/'predictions.json').write_text(json.dumps(by_deployment))
    (out/'replay.json').write_text(json.dumps(replays))
    with (out/'effects.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(effects[0]));writer.writeheader();writer.writerows(effects)
    lines=['# Diagnostic observability pilot','',f'Test deployments: {len(PLAN["test"])}. Complete-case steady rows: {len(all_y)}. No significance claim.','',
        '| Method | Accuracy | Macro F1 |','|---|---:|---:|']
    lines += [f'|{n}|{v["accuracy"]:.4f}|{v["macro_f1"]:.4f}|' for n,v in summary.items()]
    lines += ['','Primary question: contribution of a measured additional observation, not ML superiority.',
        'Optional probe savings are OFFLINE counterfactual replay; all probes were physically collected.',
        'Nominal mechanism labels are evaluator-only. Measured effects and all failed/no-effect cases are retained.',
        'Application delay is a controlled bounded wait, not an endogenous fault or semantic attack.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.models,a.output)
