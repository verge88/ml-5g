"""Offline normal-only objectives under a shared causal backbone."""
import argparse,csv,itertools,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from .prepare import P
from .benchmark import load,seed_all,confusion,block_intervals
from .engines import negative_batch

class Backbone(nn.Module):
    def __init__(self,width,mode):
        super().__init__();self.mode=mode
        self.project=nn.Sequential(nn.Linear(width,48),nn.Tanh())
        self.past=nn.GRU(48,48,batch_first=True)
        self.binary=nn.Sequential(nn.Linear(96,48),nn.ReLU(),nn.Linear(48,1))
        self.forecast=nn.Linear(96,47)
        self.restore=nn.Linear(96,width)
    def forward(self,current,past,lengths):
        if self.mode=='history':a=torch.zeros(len(current),48)
        else:a=self.project(current)
        if self.mode=='current':b=torch.zeros_like(a)
        else:
            h,_=self.past(self.project(past));b=h[torch.arange(len(h)),(lengths-1).clamp_min(0)]
            b=b*(lengths>0)[:,None]
        return torch.cat([a,b],1)

def batch(stream,start,end):
    x=stream['x'];ii=np.arange(start,end)[:,None]-np.arange(16,0,-1)[None,:];valid=ii>=0
    v=x[np.maximum(ii,0)].copy();v[~valid]=0;length=valid.sum(1)
    for j,n in enumerate(length):
        if n:v[j,:n]=v[j,-n:].copy()
        v[j,n:]=0
    return torch.from_numpy(x[start:end].copy()),torch.from_numpy(v),torch.from_numpy(length.astype('int64'))

def groups(info):
    k=info['identity_dim'];return [list(range(k,k+7)),list(range(k+7,k+15)),list(range(k+15,k+47)),list(range(k+47,k+57))]

def loss_or_score(model,stream,start,end,info,goal,rng=None):
    x,h,l=batch(stream,start,end);k=info['identity_dim'];gg=groups(info)
    if goal=='permutation':
        logits=model.binary(model(x,h,l)).flatten()
        if rng is None:return logits.sigmoid()
        negative,_,_=negative_batch(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end],info,rng)
        other=model.binary(model(torch.from_numpy(negative),h,l)).flatten()
        return (nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))+nn.functional.binary_cross_entropy_with_logits(other,torch.ones_like(other)))/2
    if goal=='next':
        # Predict the CURRENT operation from strictly PAST operations and only
        # non-target current metadata. The target operation never enters readout.
        query=x.clone();query[:,k:k+47]=0
        pred=model.forecast(model(query,h,l));losses=[]
        center=torch.tensor(info['center'][k:k+47],dtype=torch.float32);scale=torch.tensor(info['scale'][k:k+47],dtype=torch.float32)
        raw=x[:,k:k+47]*scale+center
        for a,b in [(0,7),(7,15),(15,47)]:
            valid=raw[:,a:b].sum(1)>.5
            ce=nn.functional.cross_entropy(pred[:,a:b],raw[:,a:b].argmax(1),reduction='none')
            losses.append(ce*valid)
        scores=torch.stack(losses).sum(0)/3
    else:
        # Four complete field-block masks, equal weighting, no identity targets.
        errors=[]
        for g in gg:
            query=x.clone();query[:,g]=0
            pred=model.restore(model(query,h,l))
            errors.append(((pred[:,g]-x[:,g])**2).mean(1))
        scores=torch.stack(errors).mean(0)
    return scores.mean() if rng is not None else scores

def fit(model,train,info,goal,seed):
    model.train();initial={n:p.detach().clone() for n,p in model.named_parameters()};rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);losses=[]
    for epoch in range(12):
        total=0
        for start in range(0,len(train['x']),512):
            end=min(start+512,len(train['x']));opt.zero_grad(set_to_none=True)
            loss=loss_or_score(model,train,start,end,info,goal,rng);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*(end-start)
        losses.append(total/len(train['x']));print(epoch+1,losses[-1],flush=True)
    changed=sum(bool(torch.any(p.detach()!=initial[n])) for n,p in model.named_parameters());assert changed
    return dict(losses=losses,changed_tensors=changed)

@torch.no_grad()
def score(model,stream,info,goal):
    model.eval();return np.concatenate([loss_or_score(model,stream,i,min(i+512,len(stream['x'])),info,goal).numpy() for i in range(0,len(stream['x']),512)])

def metrics(stream,scores,tau,mask=None):
    y=stream['y'].copy()
    if mask is not None:y[~mask]=-1
    valid=y>=0;positive=int((y==1).sum());c=confusion(y,scores,tau)
    c.update(labelled=int(valid.sum()),positive=positive,ap=float(average_precision_score(y[valid],scores[valid])) if positive and (y==0).any() else None,
        conditional_intervals=block_intervals(y,scores,tau))
    return c

def run(data,out,goal,mode):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=True)
    for d in P['datasets']:
        info,streams=load(data/d,'base')
        for seed in P['seeds']:
            seed_all(seed);model=Backbone(info['base_dim'],mode);detail=fit(model,streams['train'],info,goal,seed)
            folder=out/d/str(seed);folder.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),folder/'weights.pt')
            cal=score(model,streams['calibration'],info,goal);np.save(folder/'calibration.npy',cal)
            thresholds={str(a):float(np.quantile(cal,1-a,method='higher')) for a in [.001,.01]}
            r=dict(dataset=d,seed=seed,goal=goal,mode=mode,training=detail,thresholds=thresholds,manifest=info,commit=os.environ.get('GITHUB_SHA','local'))
            (folder/'frozen.json').write_text(json.dumps(r,indent=2));results={}
            for t,s in streams.items():
                if t in ['train','calibration']:continue
                ss=score(model,s,info,goal);assert len(ss)==len(s['y']) and np.isfinite(ss).all()
                known=(s['source']!=0)&(s['destination']!=0)
                np.savez_compressed(folder/(t+'.npz'),score=ss,y=s['y'],known=known,time=s['time'])
                duration=float(np.ptp(s['time']))/3600
                results[t]={}
                for a in [.001,.01]:
                    all_m=metrics(s,ss,thresholds[str(a)]);known_m=metrics(s,ss,thresholds[str(a)],known)
                    for m in [all_m,known_m]:
                        m['false_alarms_per_capture_hour']=m['fp']/duration if duration>0 else None
                    results[t][str(a)]=dict(all=all_m,known=known_m,capture_hours=duration,
                        normal_unknown_participants=int(((s['y']==0)&~known).sum()))
            (folder/'result.json').write_text(json.dumps({**r,'results':results},indent=2))

def report(inputs,out):
    out.mkdir(parents=True,exist_ok=True);rr=[];keys=set();arrays=0
    for p in inputs.rglob('result.json'):
        r=json.loads(p.read_text());key=tuple(r[k] for k in ['dataset','seed','goal','mode']);assert key not in keys;keys.add(key)
        assert r['training']['changed_tensors']>0
        cal=np.load(p.parent/'calibration.npy',allow_pickle=False)
        for a in [.001,.01]:assert np.quantile(cal,1-a,method='higher')==r['thresholds'][str(a)]
        for t,target in r['results'].items():
            with np.load(p.parent/(t+'.npz'),allow_pickle=False) as z:y=z['y'];s=z['score'];known=z['known'];arrays+=1
            if r['dataset']=='granomaly' and t=='smf':assert int(((y==1)&known).sum())==706
            for a in [.001,.01]:
                for subset in ['all','known']:
                    c=target[str(a)][subset];yy=y.copy()
                    if subset=='known':yy[~known]=-1
                    for k,v in confusion(yy,s,r['thresholds'][str(a)]).items():assert c[k]==v
                    m=yy>=0
                    if c['ap'] is not None:assert abs(c['ap']-average_precision_score(yy[m],s[m]))<1e-12
                    rr.append(dict(dataset=r['dataset'],target=t,seed=r['seed'],goal=r['goal'],mode=r['mode'],subset=subset,alpha=a,
                        ap=c['ap'],recall=c['recall'],fpr=c['fpr'],tp=c['tp'],positive=c['positive'],fp=c['fp'],
                        capture_hours=target[str(a)]['capture_hours'],false_alarms_per_capture_hour=c['false_alarms_per_capture_hour'],
                        recall_ci=json.dumps(c['conditional_intervals']['intervals']['recall']),fpr_ci=json.dumps(c['conditional_intervals']['intervals']['fpr']),
                        clean_fpr=r['results']['clean_holdout'][str(a)][subset]['fpr'],
                        budget_met=c['fpr'] is not None and c['fpr']<=a and r['results']['clean_holdout'][str(a)][subset]['fpr']<=a))
    assert keys==set(itertools.product(P['datasets'],P['seeds'],['permutation','next','masked'],['current','history','joint']))
    with (out/'operating.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    lines=['# Проверка обучающих целей на известных участниках','','54 обучения, полная проверка исходных матриц. Основной результат — SMF, 706 положительных запросов с известными участниками.','',
        '| Цель | Вход | AP | Полнота % | FPR % | Лимит во всех состояниях |','|---|---|---:|---:|---:|---|']
    for goal,mode in itertools.product(['permutation','next','masked'],['current','history','joint']):
        v=[x for x in rr if (x['dataset'],x['target'],x['goal'],x['mode'],x['subset'],x['alpha'])==('granomaly','smf',goal,mode,'known',.001)]
        lines.append(f"| {goal} | {mode} | {np.mean([x['ap'] for x in v]):.6f} | {100*np.mean([x['recall'] for x in v]):.3f} | {100*np.mean([x['fpr'] for x in v]):.4f} | {all(x['budget_met'] for x in v)} |")
    lines+=['','Это ретроспективное сравнение целей на общем GRU/MLP каркасе; не прямое воспроизведение прежней TGN. Независимых новых развёртываний и подтверждённой записи легитимной смены экземпляров нет. Три штатных запроса NRF с неизвестным участником не считаются такой записью. Доверительные интервалы условны внутри записи. Частота тревог в час относится к длительности записи, не эксплуатационной сети.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(fits=len(keys),thresholds=len(keys)*2,prediction_arrays=arrays,independent_new_deployments=0),indent=2));print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--goal',choices=['permutation','next','masked']);p.add_argument('--mode',choices=['current','history','joint']);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.data,a.output,a.goal,a.mode)
