"""Controlled offline comparison of causal history scopes and bound observations."""
import argparse,csv,hashlib,itertools,json,os
from pathlib import Path
import numpy as np
import torch
from . import normal_objectives as obj
from .benchmark import seed_all,confusion
from sklearn.metrics import average_precision_score

ARMS=['current','global','edge','connection']
GOALS=['permutation','next','masked']
SEEDS=[11,29,47]
def load_data(folder,arm):
    meta=json.loads((folder/'alignment.json').read_text());info=json.loads((folder/'manifest.json').read_text())
    if arm=='procedure' and not meta['eligible_procedure']:raise ValueError('No complete independently bound procedure identities')
    if arm=='procedure_witness' and not meta['eligible_procedure_witness']:raise ValueError('No complete request-bound procedure/witness observations')
    streams={}
    for name,spec in meta['streams'].items():
        path=folder/(name+'.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        with np.load(path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        assert len(s['y'])==spec['rows']
        extra=s['witness'] if arm=='procedure_witness' else np.zeros_like(s['witness'])
        s['x']=np.concatenate([s['x'],extra],axis=1)
        if arm not in ['current','global']:s['history_index']=s['history_'+('procedure' if arm=='procedure_witness' else arm)]
        streams[name]=s
    return info,meta,streams

def run(data,out,goal,arm):
    info,meta,streams=load_data(data,arm);torch.set_num_threads(2)
    assert np.all(streams['train']['y']==0) and np.all(streams['calibration']['y']==0)
    for seed in SEEDS:
        seed_all(seed);model=obj.Backbone(meta['input_dim'],'current' if arm=='current' else 'joint')
        detail=obj.fit(model,streams['train'],info,goal,seed);folder=out/str(seed);folder.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),folder/'weights.pt');cal=obj.score(model,streams['calibration'],info,goal);np.save(folder/'calibration.npy',cal)
        thresholds={str(a):float(np.quantile(cal,1-a,method='higher')) for a in [.001,.01]}
        r=dict(goal=goal,arm=arm,seed=seed,training=detail,thresholds=thresholds,commit=os.environ.get('GITHUB_SHA','local'),
            alignment_sha256=hashlib.sha256((data/'alignment.json').read_bytes()).hexdigest(),input_dim=meta['input_dim'],weights_sha256=hashlib.sha256((folder/'weights.pt').read_bytes()).hexdigest())
        (folder/'frozen.json').write_text(json.dumps(r,indent=2));results={}
        for t,s in streams.items():
            if t in ['train','calibration']:continue
            scores=obj.score(model,s,info,goal);assert np.isfinite(scores).all()
            known=(s['source']!=0)&(s['destination']!=0);hours=float(np.ptp(s['time']))/3600
            np.savez_compressed(folder/(t+'.npz'),score=scores,y=s['y'],known=known,time=s['time'])
            results[t]={}
            for a in [.001,.01]:
                all_m=obj.metrics(s,scores,thresholds[str(a)]);known_m=obj.metrics(s,scores,thresholds[str(a)],known)
                for m in [all_m,known_m]:m['false_alarms_per_capture_hour']=m['fp']/hours if hours>0 else None
                results[t][str(a)]=dict(all=all_m,known=known_m,capture_hours=hours)
        (folder/'result.json').write_text(json.dumps({**r,'results':results},indent=2))

def report(inputs,data,out):
    meta=json.loads((data/'alignment.json').read_text());rows=[];keys=set();arrays=0
    for p in inputs.rglob('result.json'):
        r=json.loads(p.read_text());key=(r['goal'],r['arm'],r['seed']);assert key not in keys;keys.add(key)
        assert r['training']['changed_tensors']>0 and len(r['training']['losses'])==12
        assert hashlib.sha256((p.parent/'weights.pt').read_bytes()).hexdigest()==r['weights_sha256']
        frozen=json.loads((p.parent/'frozen.json').read_text());assert all(r[k]==v for k,v in frozen.items())
        cal=np.load(p.parent/'calibration.npy',allow_pickle=False)
        for a in [.001,.01]:assert r['thresholds'][str(a)]==np.quantile(cal,1-a,method='higher')
        for t,rr in r['results'].items():
            with np.load(p.parent/(t+'.npz'),allow_pickle=False) as z:y=z['y'];scores=z['score'];known=z['known'];arrays+=1
            with np.load(data/(t+'.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(y,z['y']);np.testing.assert_array_equal(known,(z['source']!=0)&(z['destination']!=0))
            if t=='smf':assert int(((y==1)&known).sum())==706
            for a in [.001,.01]:
                for subset in ['all','known']:
                    c=rr[str(a)][subset];yy=y.copy()
                    if subset=='known':yy[~known]=-1
                    assert all(c[k]==v for k,v in confusion(yy,scores,r['thresholds'][str(a)]).items())
                    if c['ap'] is not None:assert abs(c['ap']-average_precision_score(yy[yy>=0],scores[yy>=0]))<1e-12
                    clean=r['results']['clean_holdout'][str(a)][subset]['fpr']
                    rows.append(dict(goal=r['goal'],arm=r['arm'],seed=r['seed'],target=t,subset=subset,alpha=a,ap=c['ap'],recall=c['recall'],fpr=c['fpr'],
                        tp=c['tp'],fp=c['fp'],positive=c['positive'],clean_fpr=clean,budget_met=c['fpr'] is not None and c['fpr']<=a and clean<=a,
                        false_alarms_per_capture_hour=c['false_alarms_per_capture_hour'],capture_hours=rr[str(a)]['capture_hours'],
                        recall_ci=json.dumps(c['conditional_intervals']['intervals']['recall']),fpr_ci=json.dumps(c['conditional_intervals']['intervals']['fpr'])))
    assert keys==set(itertools.product(GOALS,ARMS,SEEDS));assert arrays==108
    out.mkdir(parents=True,exist_ok=True)
    with (out/'operating.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    names={'current':'Вектор запроса','global':'Общая история','edge':'История пары функций','connection':'История соединения'}
    titles=dict(permutation='Перестановка',next='Предсказание операции',masked='Восстановление')
    lines=['# Проверка группировки истории','', '36 обучений на Granomaly. Первичный результат: 706 положительных и 2509 штатных SMF с известными участниками. Средние трёх инициализаций, номинальный калибровочный уровень 0,1%.','',
        '| Цель | Состав | AP | Полнота, % | FPR SMF, % | FPR штатного остатка, % | Ограничение при всех инициализациях |','|---|---|---:|---:|---:|---:|---|']
    for goal,arm in itertools.product(GOALS,ARMS):
        v=[r for r in rows if (r['goal'],r['arm'],r['target'],r['subset'],r['alpha'])==(goal,arm,'smf','known',.001)]
        mean=lambda k:np.mean([r[k] for r in v])
        lines.append(f"| {titles[goal]} | {names[arm]} | {mean('ap'):.4f} | {100*mean('recall'):.3f} | {100*mean('fpr'):.4f} | {100*mean('clean_fpr'):.4f} | {'Да' if all(r['budget_met'] for r in v) else 'Нет'} |")
    lines+=['','Идентификаторы сквозных процедур отсутствуют. Транспортное соединение не считается процедурой. Связанных с этими запросами свидетельств нет. Полное сравнение трёх исходно предложенных составов не выполнено: готов механизм приёма подписанных наблюдений, но эмпирическая проверка процедур и свидетельств требует новых связанных данных. Независимых новых развёртываний — 0. Метки исходных сценариев сохранены и не доказывают вредоносность каждого запроса.']
    status=dict(fits=len(keys),thresholds=2*len(keys),prediction_arrays=arrays,unique_test_requests=20350,known_smf_positives=706,
        procedure_eligible=meta['eligible_procedure'],procedure_witness_eligible=meta['eligible_procedure_witness'],independent_new_deployments=0,
        original_requested_compositions_complete=False)
    (out/'verification.json').write_text(json.dumps(status,indent=2));(out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--goal',choices=GOALS);p.add_argument('--arm',choices=ARMS+['procedure','procedure_witness']);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.report:report(a.report,a.data,a.output)
    else:run(a.data,a.output,a.goal,a.arm)
