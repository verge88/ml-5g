"""Fit on separate deployments and combine integrity verdicts with frozen ML."""
import argparse,hashlib,json,warnings,os
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.asymmetric import ec
from live5g.evaluate import features
from cu5g.common import validate_witness,POLICY
P=json.loads(Path(__file__).with_name('protocol.json').read_text())
STATES=['AUTHENTICATED_CURRENT_CU','PENDING_CURRENT_CU','UNKNOWN_STALE_CU','UNKNOWN_BINDING_AFTER_CHANGE','UNKNOWN_CU_OR_PROFILE_MISSING','AUTHENTICATION_ALARM']
NORMAL={'fit_normal','calibration_normal','test_normal','train_baseline','test_baseline'}

def combine(hard,ml):return np.asarray(hard,dtype=bool)|np.asarray(ml,dtype=bool)

def bound_index(records,bound):
    # Readiness and recovery probes also invoke the sampler; they are not scored.
    assert len(bound)>=len(records)
    index={r['observation_id']:r for r in bound};assert len(index)==len(bound)
    ids=[r['observations']['ausf']['integrity_monitor']['observation_id'] for r in records]
    assert len(set(ids))==len(ids) and all(i in index for i in ids)
    return index

def vector(observations):
    raw=[];context=[]
    for slot in ['ausf','nssf']:
        obs=observations[slot];raw.extend(features(slot,obs)[:6]);m=obs['integrity_monitor']
        context.extend([float(m['states'].get(slot)==state) for state in STATES])
    m=observations['ausf']['integrity_monitor']
    context.extend([float(m['witness_available']),float(m['confirmed']),float(m['hard_alarm']),float(m['decision']['verdict']=='UNKNOWN')])
    return raw,raw+context

def read_run(folder,replica):
    manifest=json.loads((folder/'native/capture-manifest.json').read_text());assert manifest['replica']==replica
    records=[json.loads(v) for v in (folder/'native/observations.jsonl').read_text().splitlines()]
    events=json.loads((folder/'native/events.json').read_text());bound=json.loads((folder/'sampler-bound-observations.json').read_text())
    index=bound_index(records,bound);raw=[];context=[];normal=[];truth=[];hard=[];unknown=[];train=[];rows=[];recovery=[]
    signature_checks=0
    for r in records:
        obs=r['observations'];m=obs['ausf']['integrity_monitor'];assert m==obs['nssf']['integrity_monitor']
        b=index[m['observation_id']];assert b['states']==m['states'] and b['decision']==m['decision'] and b['hard_alarm']==m['hard_alarm']
        if m['witness_available']:
            assert b['witness'] is not None
            validate_witness(b['witness'],folder/'witness.pub',b['challenge'],now_ns=b['observed_unix_ns']);signature_checks+=1
        if m['confirmed']:assert m['decision']['verdict']=='CONFIRMED_MATCH_IN_DECLARED_SCOPE' and m['witness_available']
        if not m['witness_available']:assert not m['confirmed']
        a,c=vector(obs);raw.append(a);context.append(c);normal.append(r['stage'] in NORMAL);recovery.append(r['stage'].endswith('_recovery'))
        active=any(e['start_tick']<=r['tick']<e['end_tick'] for e in events);truth.append(active)
        train.append(r['stage'] in {'fit_normal','calibration_normal','train_baseline','train_fault'})
        hard.append(m['hard_alarm']);unknown.append(m['decision']['verdict']=='UNKNOWN')
        rows.append(dict(observation_id=m['observation_id'],tick=r['tick'],ns=r['monotonic_ns'],stage=r['stage']))
    native=json.loads((folder/'evaluation.json').read_text());assert all(native['controls'].values()) and len(native['real_restarts'])==2
    return dict(raw=np.asarray(raw),context=np.asarray(context),normal=np.asarray(normal),recovery=np.asarray(recovery),truth=np.asarray(truth),train=np.asarray(train),hard=np.asarray(hard),unknown=np.asarray(unknown),rows=rows,events=events,
        signature_checks=signature_checks,native=native,source_sha256=hashlib.sha256((folder/'native/observations.jsonl').read_bytes()).hexdigest())

def metrics(run,alarms):
    normal=run['normal'];fault=run['truth'];recovery=run['recovery'];outcomes=[]
    for e in run['events']:
        if e['split']!='test':continue
        ix=np.array([e['start_tick']<=r['tick']<e['end_tick'] for r in run['rows']]);hits=np.flatnonzero(ix&alarms)
        outcomes.append(dict(episode=e['episode'],kind=e['kind'],slot=e['slot'],detected=bool(len(hits)),delay_seconds=(run['rows'][hits[0]]['ns']-e['start_ns'])/1e9 if len(hits) else None))
    return dict(normal_points=int(normal.sum()),fp=int(alarms[normal].sum()),fpr=float(alarms[normal].mean()),fault_points=int(fault.sum()),tp=int(alarms[fault].sum()),recall=float(alarms[fault].mean()),
        unknown_normal=int((run['unknown']&normal).sum()),unknown_fault=int((run['unknown']&fault).sum()),recovery_alarm_fraction=float(alarms[recovery].mean()),episodes=outcomes,
        detected_episodes=sum(e['detected'] for e in outcomes),test_episodes=len(outcomes))

def run(inputs,out):
    out.mkdir(parents=True,exist_ok=True)
    # Test runs are not opened until after all fitted models and thresholds are saved.
    training=[read_run(inputs/f'joint-native-{n}',n) for n in P['train_deployments']]
    calibration=[read_run(inputs/f'joint-native-{n}',n) for n in P['calibration_deployments']]
    fitted=[];provenance=[]
    for model_name in ['isolation_forest','mlp']:
        for composition in ['raw','context']:
            x=np.concatenate([r[composition][r['train'] & (r['normal'] if model_name=='isolation_forest' else np.ones(len(r['train']),dtype=bool))] for r in training])
            y=np.concatenate([r['truth'][r['train'] & (r['normal'] if model_name=='isolation_forest' else np.ones(len(r['train']),dtype=bool))] for r in training])
            normal_train=np.concatenate([r[composition][r['train']&r['normal']] for r in training]);scaler=StandardScaler().fit(normal_train)
            cal=np.concatenate([r[composition][r['normal']] for r in calibration])
            for seed in P['model_seeds']:
                if model_name=='isolation_forest':model=IsolationForest(n_estimators=200,max_samples=256,random_state=seed,n_jobs=1)
                else:model=MLPClassifier(hidden_layer_sizes=(48,24),max_iter=300,batch_size=64,alpha=.001,random_state=seed,early_stopping=False)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);model.fit(scaler.transform(x),y) if model_name=='mlp' else model.fit(scaler.transform(x))
                z=scaler.transform(cal);scores=model.predict_proba(z)[:,1] if model_name=='mlp' else -model.score_samples(z);tau=float(scores.max())
                name=f'{model_name}-{composition}-{seed}';joblib.dump(dict(model=model,scaler=scaler),out/(name+'.joblib'));np.save(out/(name+'-calibration.npy'),scores)
                detail=dict(model=model_name,composition=composition,seed=seed,threshold=tau,training_rows=len(x),training_positive=int(y.sum()),calibration_rows=len(cal),
                    warning=[str(w.message) for w in caught],weights_sha256=hashlib.sha256((out/(name+'.joblib')).read_bytes()).hexdigest())
                fitted.append((name,model,scaler,detail));provenance.append(detail)
    (out/'frozen-models.json').write_text(json.dumps(dict(protocol=P,code_commit=os.environ.get('GITHUB_SHA','local'),models=provenance),indent=2))
    results=[];tests=[]
    for n in P['test_deployments']:
        r=read_run(inputs/f'joint-native-{n}',n);tests.append(dict(replica=n,signature_checks=r['signature_checks'],source_sha256=r['source_sha256'],native_controls=r['native']['controls']))
        results.append(dict(replica=n,model='integrity_only',seed=None,method='integrity_only',**metrics(r,r['hard'])))
        arrays=dict(truth=r['truth'],normal=r['normal'],hard=r['hard'],unknown=r['unknown'],recovery=r['recovery'],tick=np.array([v['tick'] for v in r['rows']]),ns=np.array([v['ns'] for v in r['rows']]))
        for name,model,scaler,d in fitted:
            z=scaler.transform(r[d['composition']]);scores=model.predict_proba(z)[:,1] if d['model']=='mlp' else -model.score_samples(z);ml=scores>d['threshold'];full=combine(r['hard'],ml)
            assert np.all(full[r['hard']]);arrays['score_'+name]=scores;arrays['alarm_'+name]=ml;arrays['full_'+name]=full
            for method,alarm in [('ml_'+d['composition'],ml),('full_'+d['composition'],full)]:results.append(dict(replica=n,model=d['model'],seed=d['seed'],method=method,**metrics(r,alarm)))
        np.savez_compressed(out/f'test-{n}.npz',**arrays)
    (out/'results.json').write_text(json.dumps(dict(models=provenance,results=results,test_evidence=tests),indent=2))
    lines=['# Совместный опыт контроля целостности и машинного обучения','', 'Новые развёртывания: 701/702 обучение,703 калибровка,704/705/706 проверка. Реальные локальные отказы stop/pause — не семантические кибератаки. Офлайн-контроли целостности не смешаны с отказами.','',
        '| Модель | Состав | Обнаружено эпизодов (сумма по тестовым развёртываниям и инициализациям) | Проверено эпизодов | Средний FPR, % |','|---|---|---:|---:|---:|']
    for model_name in ['integrity_only','isolation_forest','mlp']:
        for method in (['integrity_only'] if model_name=='integrity_only' else ['ml_raw','ml_context','full_raw','full_context']):
            v=[x for x in results if x['model']==model_name and x['method']==method]
            lines.append(f"| {model_name} | {method} | {sum(x['detected_episodes'] for x in v)} | {sum(x['test_episodes'] for x in v)} | {100*np.mean([x['fpr'] for x in v]):.4f} |")
    lines+=['','Сумма по инициализациям повторяет одни и те же физические эпизоды, а не увеличивает число независимых событий. Физических проверочных эпизодов —12, новых тестовых развёртываний —3. Integrity UNKNOWN сохранён отдельно; ML не подтверждает целостность и не отменяет нарушения.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.input,a.output)
