"""Registered offline 2x2x2x2 TGN mechanism experiment; no traffic generation."""
import argparse,csv,hashlib,itertools,json,os,time
from pathlib import Path
import numpy as np
import torch
from .benchmark import load,seed_all,confusion
from .prepare import P
from .engines import Existing,negative_batch,predict_links
from sklearn.metrics import average_precision_score

class Ablation(Existing):
    def __init__(self,width,info,current=True,history=True):
        super().__init__('tgn',width,info);self.current=current;self.history=history
    def score(self,stream,start,end,negative=None):
        x,s,d=(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end]) if negative is None else negative
        return self.model(torch.from_numpy(x if self.current else np.zeros_like(x)),torch.from_numpy(s),torch.from_numpy(d),torch.arange(start,end,dtype=torch.float64)+stream.get('offset',0))
    def observe(self,stream,start,end):
        if self.history:super().observe(stream,start,end)

def objective(positive,negative,collisions,drop):
    a=torch.nn.functional.binary_cross_entropy_with_logits(positive,torch.zeros_like(positive))
    mask=~torch.from_numpy(collisions) if drop else torch.ones(len(negative),dtype=torch.bool)
    if mask.any():
        b=torch.nn.functional.binary_cross_entropy_with_logits(negative[mask],torch.ones_like(negative[mask]))
        return (a+b)/2
    return a

def train(engine,stream,info,seed,drop):
    engine.train(True);initial={n:p.detach().clone() for n,p in engine.parameters()}
    rng=np.random.default_rng(seed);losses=[];collisions=total=known=0;grad={}
    edges=set(zip(stream['source'].tolist(),stream['destination'].tolist()))
    for epoch in range(P['epochs']):
        engine.reset(len(stream['x'])+512);loss_sum=0.;n=0
        for start in range(0,len(stream['x']),P['batch_size']):
            end=min(start+P['batch_size'],len(stream['x']))
            neg=negative_batch(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end],info,rng)
            same=(neg[1]==stream['source'][start:end])&(neg[2]==stream['destination'][start:end])
            collisions+=int(same.sum());total+=len(same)
            known+=sum((int(s),int(d)) in edges for s,d in zip(neg[1],neg[2]))
            engine.optimizer.zero_grad(set_to_none=True)
            loss=objective(engine.score(stream,start,end),engine.score(stream,start,end,neg),same,drop)
            loss.backward()
            for name,p in engine.parameters():
                if p.grad is not None:grad[name.split('.')[0]]=max(grad.get(name.split('.')[0],0),float(p.grad.abs().sum()))
            torch.nn.utils.clip_grad_norm_([p for _,p in engine.parameters()],P['gradient_clip'])
            engine.optimizer.step();engine.observe(stream,start,end)
            loss_sum+=float(loss.detach())*len(same);n+=len(same)
        losses.append(loss_sum/n);print('epoch',epoch+1,'loss',losses[-1],flush=True)
    changed=sum(bool(torch.any(p.detach()!=initial[n])) for n,p in engine.parameters())
    assert changed and np.isfinite(losses).all()
    if engine.history:assert all(grad.get(k,0)>0 for k in ['message','updater','attention','time_encoder'])
    return dict(losses=losses,changed_tensors=changed,gradient_max=grad,negative_examples=total,
        exact_endpoint_collisions=collisions,negative_pairs_seen_anywhere_in_normal_train=known,
        dropped_negative_examples=collisions if drop else 0)

def run(data,out,composition,current,history,drop):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=True)
    for dataset in P['datasets']:
        info,streams=load(data/dataset,composition)
        for seed in P['seeds']:
            seed_all(seed);engine=Ablation(streams['train']['x'].shape[1],info,bool(current),bool(history))
            folder=out/dataset/str(seed);folder.mkdir(parents=True,exist_ok=True)
            detail=train(engine,streams['train'],info,seed,bool(drop));engine.save(folder/'weights.pt')
            score=lambda stream:predict_links(engine,streams['train'],stream)
            cal=score(streams['calibration']);np.save(folder/'calibration.npy',cal)
            thresholds={str(a):float(np.quantile(cal,1-a,method='higher')) for a in P['calibration_fprs']}
            frozen=dict(dataset=dataset,seed=seed,input=composition,current=current,history=history,drop=drop,
                training=detail,thresholds=thresholds,protocol=P,commit=os.environ.get('GITHUB_SHA','local'),
                train_rows=len(streams['train']['x']),calibration_rows=len(cal),manifest=info)
            (folder/'frozen.json').write_text(json.dumps(frozen,indent=2))
            results={}
            for target,s in streams.items():
                if target in ('train','calibration'):continue
                ss=score(s);assert len(ss)==len(s['y']) and np.isfinite(ss).all()
                np.savez_compressed(folder/(target+'.npz'),score=ss,y=s['y'])
                mask=s['y']>=0;y=s['y'][mask]
                results[target]=dict(rows=len(ss),unknown=int((~mask).sum()),ap=float(average_precision_score(y,ss[mask])) if len(set(y))==2 else None,
                    operating={str(a):confusion(s['y'],ss,thresholds[str(a)]) for a in P['calibration_fprs']})
            (folder/'result.json').write_text(json.dumps({**frozen,'results':results},indent=2))

def report(inputs,out):
    out.mkdir(parents=True,exist_ok=True);records={};rows=[]
    for p in inputs.rglob('result.json'):
        r=json.loads(p.read_text());key=(r['dataset'],r['seed'],r['input'],r['current'],r['history'],r['drop'])
        assert key not in records;records[key]=r
        cal=np.load(p.parent/'calibration.npy',allow_pickle=False)
        for a in P['calibration_fprs']:assert float(np.quantile(cal,1-a,method='higher'))==r['thresholds'][str(a)]
        for target,metrics in r['results'].items():
            with np.load(p.parent/(target+'.npz'),allow_pickle=False) as f:y=f['y'];s=f['score']
            mask=y>=0
            if len(set(y[mask]))==2:assert abs(average_precision_score(y[mask],s[mask])-metrics['ap'])<1e-12
            for a in P['calibration_fprs']:
                c=confusion(y,s,r['thresholds'][str(a)]);assert c==metrics['operating'][str(a)]
                rows.append(dict(dataset=r['dataset'],seed=r['seed'],input=r['input'],current=r['current'],history=r['history'],drop=r['drop'],target=target,alpha=a,ap=metrics['ap'],**c,
                    budget_met=c['fpr']<=a and r['results']['clean_holdout']['operating'][str(a)]['fpr']<=a))
    assert len(records)==96,len(records)
    expected=set(itertools.product(P['datasets'],P['seeds'],['base','context'],[0,1],[0,1],[0,1]));assert set(records)==expected
    with (out/'operating.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    effects=[]
    for key,r in records.items():
        for index,factor in [(2,'context'),(3,'current'),(4,'history'),(5,'drop')]:
            if key[index] not in ('base',0):continue
            alternate=list(key);alternate[index]='context' if index==2 else 1;other=records[tuple(alternate)]
            for t in r['results']:
                if t=='clean_holdout':continue
                ap_delta=other['results'][t]['ap']-r['results'][t]['ap']
                lo=r['results'][t]['operating']['0.001'];hi=other['results'][t]['operating']['0.001']
                effects.append(dict(dataset=key[0],seed=key[1],factor=factor,target=t,condition=list(key[2:]),delta_ap=ap_delta,
                    delta_recall=hi['recall']-lo['recall'],delta_fpr=hi['fpr']-lo['fpr']))
    (out/'paired-effects.json').write_text(json.dumps(effects,indent=2))
    lines=['# Контролируемая проверка механизмов TGN','','96 обучений. Все исходные тестовые запросы. Разности при попарно совпадающих остальных условиях. Это причинное вмешательство в алгоритм на фиксированных данных, а не доказательство причин реальной атаки.','',
        '| Данные | Фактор включён | Среднее ΔAP | Минимум ΔAP | Максимум ΔAP |','|---|---|---:|---:|---:|']
    for d,t in [('ctd5g','mixed'),('granomaly','nrf'),('granomaly','smf')]:
        for fac in ['context','current','history','drop']:
            vals=[e['delta_ap'] for e in effects if (e['dataset'],e['target'],e['factor'])==(d,t,fac)]
            lines.append(f'| {d}/{t} | {fac} | {np.mean(vals):.6f} | {min(vals):.6f} | {max(vals):.6f} |')
    lines+=['','Средние по условиям и инициализациям не являются доверительными интервалами по независимым развёртываниям. Drop удаляет только отрицательные примеры с точно той же парой участников; другие известные штатные связи и несогласованность контекста сохраняются. Тестовые пороги не подстраиваются.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--input',choices=['base','context']);p.add_argument('--current',type=int,choices=[0,1]);p.add_argument('--history',type=int,choices=[0,1]);p.add_argument('--drop',type=int,choices=[0,1]);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.data,a.output,a.input,a.current,a.history,a.drop)
