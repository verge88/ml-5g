"""Registered offline repair of categorical likelihood and operation representation."""
import argparse,csv,hashlib,itertools,json,os
from pathlib import Path
import joblib,numpy as np,torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from semantic5g.analyze import read_csv
from hybrid.fetch_data import fetch
from .prepare import partition,features,bin_id,METHODS
from .benchmark import load,seed_all,svdd_fit,confusion
from .normal_objectives import metrics

SEEDS=[11,29,47]
MODELS=['mse_current','mse_global','ce_current','ce_global','isolation_forest','deep_svdd']
ENCODINGS=['hash','exact']

def vocabularies(train):
    return [{v:i+1 for i,v in enumerate(sorted(set(r[key] for r in train)))} for key in ['source','destination','service','path']]

def codes(rows,voc,encoding,nodes):
    service,path=voc[2:];cards=[len(nodes)+1,len(nodes)+1,8,(9 if encoding=='hash' else len(service)+1),(33 if encoding=='hash' else len(path)+1),2]
    values=[]
    for r in rows:
        values.append([nodes.get(r['source'],0),nodes.get(r['destination'],0),METHODS.index(r['method'])+1 if r['method'] in METHODS else 0,
            bin_id(r['service'],8)+1 if encoding=='hash' else service.get(r['service'],0),bin_id(r['path'],32)+1 if encoding=='hash' else path.get(r['path'],0),1])
    return np.asarray(values,dtype=np.int64),cards

def prepare(raw,original,out):
    fetch(raw);info,old=load(original,'context');rr={};hashes={}
    for filename,name in [('bridge_normal.csv','normal'),('bridge_attack_nrf.csv','nrf'),('bridge_attack_smf.csv','smf')]:
        p=raw/filename;rows,_=read_csv(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in rows:r.update(label=int(r['label']!='normal'),service=r['path'].split('/')[1] if '/' in r['path'] else '')
        rr[name]=sorted(rows,key=lambda r:r['time'])
    streams={**partition(rr['normal']),**{k:rr[k] for k in ['nrf','smf']}}
    nodes={v:i+1 for i,v in enumerate(sorted({r[key] for r in streams['train'] for key in ['source','destination']}))};voc=vocabularies(streams['train'])
    raw_features={}
    for t,rows in streams.items():
        x,c,src,dst=features(rows,nodes);rebuilt=np.clip((np.c_[x,c]-np.asarray(info['center'],dtype='float32'))/np.asarray(info['scale'],dtype='float32'),-20,20)
        np.testing.assert_array_equal(rebuilt,old[t]['x']);np.testing.assert_array_equal(src,old[t]['source']);np.testing.assert_array_equal(dst,old[t]['destination'])
        np.testing.assert_array_equal([r['label'] for r in rows],old[t]['y']);np.testing.assert_array_equal([r['time']-rows[0]['time'] for r in rows],old[t]['time']);raw_features[t]=x[:,info['identity_dim']+49:info['identity_dim']+57]
    center=np.median(raw_features['train'],axis=0);scale=np.maximum(1.4826*np.median(abs(raw_features['train']-center),axis=0),.1)
    k=info['identity_dim'];s=old['train'];train_signatures={tuple(np.r_[s['source'][i],s['destination'][i],s['x'][i,k:k+47]].tolist()) for i in range(len(s['y']))}
    manifest=dict(captures=hashes,original_manifest_sha256=hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest(),streams={},encodings={},numeric_center=center.tolist(),numeric_scale=scale.tolist(),
        vocab_sha256=hashlib.sha256(json.dumps(voc,sort_keys=True).encode()).hexdigest(),independent_new_deployments=0)
    out.mkdir(parents=True,exist_ok=True)
    for encoding in ENCODINGS:
        folder=out/encoding;folder.mkdir(exist_ok=True);spec={}
        for t,rows in streams.items():
            cat,cards=codes(rows,voc,encoding,nodes);num=np.arcsinh((raw_features[t]-center)/scale).astype('float32');s=old[t]
            seen=np.array([tuple(np.r_[s['source'][i],s['destination'][i],s['x'][i,k:k+47]].tolist()) in train_signatures for i in range(len(s['y']))])
            np.savez_compressed(folder/(t+'.npz'),cat=cat,num=num,y=s['y'],source=s['source'],destination=s['destination'],time=s['time'],seen_signature=seen,kind=s['kind'])
            spec[t]=hashlib.sha256((folder/(t+'.npz')).read_bytes()).hexdigest()
            manifest['streams'][t]=dict(rows=len(rows),positive=int((s['y']==1).sum()),known_seen_positive=int(((s['y']==1)&seen&(s['source']!=0)&(s['destination']!=0)).sum()))
        manifest['encodings'][encoding]=dict(cards=cards,sha256=spec,input_dim=sum(cards)+12)
    assert manifest['streams']['smf']['known_seen_positive']==646
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(streams=manifest['streams'],cards={k:v['cards'] for k,v in manifest['encodings'].items()})))

def load_data(data,encoding):
    meta=json.loads((data/'manifest.json').read_text());spec=meta['encodings'][encoding];streams={}
    for t,h in spec['sha256'].items():
        p=data/encoding/(t+'.npz');assert hashlib.sha256(p.read_bytes()).hexdigest()==h
        with np.load(p,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        onehot=[np.eye(n,dtype='float32')[s['cat'][:,j]] for j,n in enumerate(spec['cards'])]
        s['x']=np.c_[*onehot,s['num'],np.zeros((len(s['y']),4),dtype='float32')].astype('float32');streams[t]=s
    return meta,spec,streams

class Repair(nn.Module):
    def __init__(self,cards,mode):
        super().__init__();self.mode=mode;self.cards=cards;self.offset=np.cumsum([0]+cards).tolist();self.width=sum(cards)+12
        self.project=nn.Sequential(nn.Linear(self.width,48),nn.Tanh());self.past=nn.GRU(48,48,batch_first=True)
        self.heads=nn.ModuleList([nn.Linear(96,cards[j]) for j in [2,3,4]]);self.numeric=nn.Linear(96,8)
    def embedding(self,x,h,length):
        a=self.project(x)
        if self.mode=='current':b=torch.zeros_like(a)
        else:
            v,_=self.past(self.project(h));b=v[torch.arange(len(v)),(length-1).clamp_min(0)]*(length>0)[:,None]
        return torch.cat([a,b],1)
    def masked_logits(self,x,h,length,group):
        query=x.clone();n=sum(self.cards)
        if group<3:query[:,self.offset[group+2]:self.offset[group+3]]=0
        else:query[:,n:n+8]=0
        query[:,n+8+group]=1;v=self.embedding(query,h,length)
        return self.heads[group](v) if group<3 else self.numeric(v)

def batch(s,start,end):
    ii=np.arange(start,end)[:,None]-np.arange(16,0,-1)[None,:];valid=ii>=0;past=s['x'][np.maximum(ii,0)].copy();past[~valid]=0;length=valid.sum(1)
    for j,n in enumerate(length):
        if n:past[j,:n]=past[j,-n:].copy()
        past[j,n:]=0
    return torch.from_numpy(s['x'][start:end].copy()),torch.from_numpy(past),torch.from_numpy(length.astype('int64'))

def errors(model,s,start,end,objective):
    x,h,length=batch(s,start,end);cat=torch.from_numpy(s['cat'][start:end]);num=torch.from_numpy(s['num'][start:end]);loss=[]
    for g in range(4):
        pred=model.masked_logits(x,h,length,g)
        if g<3:
            target=cat[:,g+2]
            e=nn.functional.cross_entropy(pred,target,reduction='none') if objective=='ce' else ((pred.softmax(1)-nn.functional.one_hot(target,model.cards[g+2]))**2).mean(1)
        else:e=nn.functional.smooth_l1_loss(pred,num,reduction='none').mean(1) if objective=='ce' else ((pred-num)**2).mean(1)
        loss.append(e)
    return torch.stack(loss,1)

@torch.no_grad()
def predict(model,s,objective):
    model.eval();return np.concatenate([errors(model,s,i,min(i+512,len(s['y'])),objective).mean(1).numpy() for i in range(0,len(s['y']),512)])

def fit(model,s,objective):
    model.train();before={k:v.detach().clone() for k,v in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);losses=[]
    for epoch in range(12):
        total=0
        for i in range(0,len(s['y']),512):
            end=min(i+512,len(s['y']));opt.zero_grad(set_to_none=True);loss=errors(model,s,i,end,objective).mean();assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*(end-i)
        losses.append(total/len(s['y']));print(epoch+1,losses[-1],flush=True)
    changed=sum(bool(torch.any(v.detach()!=before[k])) for k,v in model.named_parameters());assert changed
    return dict(losses=losses,changed_tensors=changed,parameters=sum(p.numel() for p in model.parameters()))

def run(data,out,encoding,method):
    torch.set_num_threads(2);meta,spec,streams=load_data(data,encoding)
    assert np.all(streams['train']['y']==0) and np.all(streams['calibration']['y']==0)
    for seed in SEEDS:
        seed_all(seed);folder=out/str(seed);folder.mkdir(parents=True,exist_ok=True)
        if method=='isolation_forest':
            model=IsolationForest(n_estimators=200,max_samples=256,random_state=seed,n_jobs=2).fit(streams['train']['x']);joblib.dump(model,folder/'weights.joblib')
            score=lambda s:-model.score_samples(s['x']);detail=dict(trees=200,training_rows=len(streams['train']['y']))
        elif method=='deep_svdd':
            model,center,score,detail=svdd_fit(streams['train']['x'],seed);torch.save(dict(weights=model.state_dict(),center=center),folder/'weights.pt')
        else:
            objective,mode=method.split('_');model=Repair(spec['cards'],mode);detail=fit(model,streams['train'],objective);torch.save(model.state_dict(),folder/'weights.pt');score=lambda s:predict(model,s,objective)
        cal=score(streams['calibration']);assert np.isfinite(cal).all();np.save(folder/'calibration.npy',cal)
        thresholds=dict(max_calibration=float(np.max(cal)),nominal_001=float(np.quantile(cal,.999,method='higher')))
        r=dict(encoding=encoding,method=method,seed=seed,thresholds=thresholds,training=detail,commit=os.environ.get('GITHUB_SHA','local'),data_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
        (folder/'frozen.json').write_text(json.dumps(r,indent=2));results={}
        for t,s in streams.items():
            if t in ['train','calibration']:continue
            scores=score(s);assert np.isfinite(scores).all();known=(s['source']!=0)&(s['destination']!=0);seen=known&s['seen_signature'];hours=float(np.ptp(s['time']))/3600
            np.savez_compressed(folder/(t+'.npz'),score=scores,y=s['y'],known=known,seen=seen,time=s['time']);results[t]={}
            for point,tau in thresholds.items():
                results[t][point]={}
                for subset,mask in [('all',None),('known',known),('seen',seen)]:
                    c=metrics(s,scores,tau,mask);c['false_alarms_per_capture_hour']=c['fp']/hours;results[t][point][subset]=c
        (folder/'result.json').write_text(json.dumps({**r,'results':results},indent=2))

def report(inputs,out):
    rows=[];keys=set();arrays=0
    for p in inputs.rglob('result.json'):
        r=json.loads(p.read_text());key=(r['encoding'],r['method'],r['seed']);assert key not in keys;keys.add(key)
        assert all(r[k]==v for k,v in json.loads((p.parent/'frozen.json').read_text()).items());cal=np.load(p.parent/'calibration.npy',allow_pickle=False)
        assert r['thresholds']['max_calibration']==cal.max() and r['thresholds']['nominal_001']==np.quantile(cal,.999,method='higher')
        for t,result in r['results'].items():
            with np.load(p.parent/(t+'.npz'),allow_pickle=False) as z:y=z['y'];scores=z['score'];known=z['known'];seen=z['seen'];arrays+=1
            if t=='smf':assert ((y==1)&known).sum()==706 and ((y==1)&seen).sum()==646
            for point,tau in r['thresholds'].items():
                for subset,mask in [('all',None),('known',known),('seen',seen)]:
                    yy=y.copy()
                    if mask is not None:yy[~mask]=-1
                    c=result[point][subset];assert all(c[k]==v for k,v in confusion(yy,scores,tau).items())
                    if c['ap'] is not None:assert abs(c['ap']-average_precision_score(yy[yy>=0],scores[yy>=0]))<1e-12
                    clean=r['results']['clean_holdout'][point][subset]['fpr']
                    rows.append(dict(encoding=r['encoding'],method=r['method'],seed=r['seed'],target=t,point=point,subset=subset,ap=c['ap'],recall=c['recall'],fpr=c['fpr'],
                        tp=c['tp'],fp=c['fp'],positive=c['positive'],clean_fpr=clean,budget_met=c['fpr'] is not None and c['fpr']<=.001 and clean<=.001,
                        recall_ci=json.dumps(c['conditional_intervals']['intervals']['recall']),fpr_ci=json.dumps(c['conditional_intervals']['intervals']['fpr']),false_alarms_per_capture_hour=c['false_alarms_per_capture_hour']))
    assert keys==set(itertools.product(ENCODINGS,MODELS,SEEDS)) and arrays==108
    primary=[r for r in rows if (r['encoding'],r['method'],r['target'],r['point'],r['subset'])==('exact','ce_global','smf','max_calibration','known')]
    hard=[r for r in rows if (r['encoding'],r['method'],r['target'],r['point'],r['subset'])==('exact','ce_global','smf','max_calibration','seen')]
    feasible=all(r['budget_met'] for r in primary);useful=all(r['recall']>=.05 for r in hard);success=feasible and useful
    out.mkdir(parents=True,exist_ok=True)
    with (out/'operating.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Исправление представления операций и функции ошибки','','Основная модель заранее фиксирована: exact + ce_global, порог max_calibration. Успех: FPR SMF и штатного остатка ≤0,1% при всех трёх инициализациях и полнота не ниже 5% на трудной группе из 646 при всех инициализациях.',f'Основной критерий выполнен: {success}. Ограничение ложных тревог: {feasible}; полнота трудной группы: {useful}.','',
        '| Представление | Метод | AP | Полнота 706, % | Полнота 646, % | FPR SMF, % | Ограничение во всех инициализациях |','|---|---|---:|---:|---:|---:|---|']
    for encoding,method in itertools.product(ENCODINGS,MODELS):
        v=[r for r in rows if (r['encoding'],r['method'],r['target'],r['point'],r['subset'])==(encoding,method,'smf','max_calibration','known')]
        h=[r for r in rows if (r['encoding'],r['method'],r['target'],r['point'],r['subset'])==(encoding,method,'smf','max_calibration','seen')]
        lines.append(f"| {encoding} | {method} | {np.mean([r['ap'] for r in v]):.4f} | {100*np.mean([r['recall'] for r in v]):.3f} | {100*np.mean([r['recall'] for r in h]):.3f} | {100*np.mean([r['fpr'] for r in v]):.4f} | {all(r['budget_met'] for r in v)} |")
    lines+=['','36 обучений; 20 350 уникальных тестовых запросов. Исходные метки сохранены. Ретроспективный поиск улучшения на уже изученных данных: новой независимой проверки нет. Максимум калибровки не гарантирует FPR при смене распределения. Случайный успех одной инициализации или другого варианта не считается выполнением основного критерия. Свидетельств и сквозных процедур нет.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');(out/'verification.json').write_text(json.dumps(dict(fits=len(keys),thresholds=72,prediction_arrays=arrays,primary_success=success,primary_fpr_feasible=feasible,primary_hard_recall_useful=useful,independent_new_deployments=0),indent=2));print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-raw',type=Path);p.add_argument('--original',type=Path)
    p.add_argument('--encoding',choices=ENCODINGS);p.add_argument('--method',choices=MODELS);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.prepare_raw:prepare(a.prepare_raw,a.original,a.output)
    elif a.report:report(a.report,a.output)
    else:run(a.data,a.output,a.encoding,a.method)
