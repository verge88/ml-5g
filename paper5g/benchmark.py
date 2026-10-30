"""Train actual ML models and score every held-out request, without rules."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from .prepare import P
from . import references
from .engines import Existing,PublishedTGN,SVDD,fit_links,predict_links

def load(folder,composition):
    info=json.loads((folder/'manifest.json').read_text());streams={}
    if info['protocol']!=P:raise ValueError('prepared-data protocol differs from registered model protocol')
    for name,spec in info['streams'].items():
        path=folder/(name+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('prepared matrix hash mismatch')
        with np.load(path,allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
        if composition=='base':values['x']=values['x'][:,:info['base_dim']].copy()
        assert len(values['x'])==spec['rows'] and np.isfinite(values['x']).all()
        streams[name]=values
    assert np.all(streams['train']['y']==0) and np.all(streams['calibration']['y']==0)
    return info,streams

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def confusion(y,scores,threshold):
    selected=y>=0;y=y[selected];flags=scores[selected]>threshold
    tp=int((flags&(y==1)).sum());fp=int((flags&(y==0)).sum())
    fn=int((~flags&(y==1)).sum());tn=int((~flags&(y==0)).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
        fpr=fp/(fp+tn) if fp+tn else None,precision=tp/(tp+fp) if tp+fp else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def block_intervals(y,scores,threshold,repeats=200):
    blocks=[]
    for start in range(0,len(y),512):
        c=confusion(y[start:start+512],scores[start:start+512],threshold)
        blocks.append([c[k] for k in ('tp','fp','fn','tn')])
    values=np.array(blocks);rng=np.random.default_rng(20261030);estimates=[]
    for _ in range(repeats):
        tp,fp,fn,tn=values[rng.integers(len(values),size=len(values))].sum(0)
        estimates.append([tp/(tp+fn) if tp+fn else np.nan,fp/(fp+tn) if fp+tn else np.nan,
                          2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.])
    result={}
    for i,key in enumerate(('recall','fpr','f1')):
        finite=np.asarray(estimates)[:,i];finite=finite[np.isfinite(finite)]
        result[key]=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    return dict(block_requests=512,replicates=repeats,scope='conditional within-record dependence; not independent deployments',intervals=result)

def evaluate(stream,scores,threshold,info):
    y=stream['y'];selected=y>=0;ys=y[selected];ss=scores[selected]
    result=confusion(y,scores,threshold)
    result.update(rows=len(y),unknown_excluded=int((y<0).sum()),labelled=int(selected.sum()),
        average_precision=float(average_precision_score(ys,ss)) if len(set(ys))==2 else None,
        roc_auc=float(roc_auc_score(ys,ss)) if len(set(ys))==2 else None)
    result['per_family']={}
    for name,k in info['categories'].items():
        mask=(stream['kind']==k)&(y==1)
        if mask.any():result['per_family'][name]=dict(requests=int(mask.sum()),detected=int((scores[mask]>threshold).sum()),
            recall=float((scores[mask]>threshold).mean()))
    result['conditional_block_bootstrap']=block_intervals(y,scores,threshold)
    return result

def svdd_fit(x,seed):
    model=SVDD(x.shape[1]);tx=torch.from_numpy(x)
    with torch.no_grad():c=torch.cat([model(tx[i:i+2048]) for i in range(0,len(x),2048)]).mean(0)
    c=torch.where(c.abs()<.1,torch.where(c<0,-.1,.1),c)
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);losses=[]
    for _ in range(P['epochs']):
        total=0.
        for i in range(0,len(x),P['batch_size']):
            optimizer.zero_grad(set_to_none=True);loss=((model(tx[i:i+P['batch_size']])-c)**2).sum(1).mean()
            loss.backward();optimizer.step();total+=float(loss.detach())*len(tx[i:i+P['batch_size']])
        losses.append(total/len(x))
    changed=sum(bool(torch.any(p!=initial[k])) for k,p in model.named_parameters())
    assert changed>0 and all(m.bias is None for m in model.modules() if isinstance(m,torch.nn.Linear))
    @torch.no_grad()
    def score(stream):
        return np.concatenate([((model(torch.from_numpy(stream['x'][i:i+2048]))-c)**2).sum(1).numpy() for i in range(0,len(stream['x']),2048)])
    return model,c,score,dict(epoch_loss=losses,changed_parameter_tensors=changed,center=c.tolist(),bias_terms=False)

def run(data,output,method,composition,root):
    torch.set_num_threads(P['torch_threads']);output.mkdir(parents=True,exist_ok=True)
    ref_manifest=references.fetch(root)
    records=[]
    for dataset in P['datasets']:
        info,streams=load(data/dataset,composition);training=streams['train']
        for seed in P['kitnet_seeds'] if method=='kitnet' else P['seeds']:
            seed_all(seed);folder=output/dataset/str(seed);folder.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter();engine=None
            if method=='isolation_forest':
                model=IsolationForest(n_estimators=200,max_samples=256,random_state=seed,n_jobs=2)
                model.fit(training['x']);score=lambda stream:-model.score_samples(stream['x'])
                detail=dict(trees=200,max_samples=256,training_rows_supplied=len(training['x']))
                joblib.dump(model,folder/'model.joblib')
            elif method=='kitnet':
                cls=references.kitnet(root);fm=max(1,int(len(training['x'])*.2))
                model=cls(training['x'].shape[1],max_autoencoder_size=10,FM_grace_period=fm,
                    AD_grace_period=len(training['x'])-fm-1,learning_rate=.1,hidden_ratio=.75)
                for row in training['x']:model.train(row.astype(np.float64))
                score=lambda stream:np.asarray([model.execute(row.astype(np.float64)) for row in stream['x']])
                detail=dict(training_rows=len(training['x']),feature_mapping_rows=fm+1,fixed_author_seed=1234,
                    ensemble_autoencoders=len(model.ensembleLayer),independent_random_repeats=1)
                weights={}
                for i,ae in enumerate([*model.ensembleLayer,model.outputLayer]):
                    for name in ('W','hbias','vbias','norm_min','norm_max'):weights[f'{i}_{name}']=getattr(ae,name)
                np.savez_compressed(folder/'autoencoder-parameters.npz',**weights)
            elif method=='deep_svdd':
                model,center,score,detail=svdd_fit(training['x'],seed)
                torch.save(dict(weights=model.state_dict(),center=center),folder/'weights.pt')
            else:
                engine=PublishedTGN(training['x'].shape[1],info,root) if method=='granomaly_tgn' else Existing(method,training['x'].shape[1],info)
                detail=fit_links(engine,training,info,seed);score=lambda stream:predict_links(engine,training,stream)
                engine.save(folder/'weights.pt')
            seconds=time.perf_counter()-started
            cal=score(streams['calibration'])
            if not np.isfinite(cal).all():raise ValueError('nonfinite calibration')
            points={str(alpha):float(np.quantile(cal,1-alpha,method='higher')) for alpha in P['calibration_fprs']}
            frozen=dict(dataset=dataset,method=method,input=composition,seed=seed,protocol=P,training=detail,
                training_rows=len(training['x']),training_seconds=seconds,input_dim=training['x'].shape[1],
                thresholds=points,calibration_rows=len(cal),prepared_manifest=info,
                code_commit=os.environ.get('GITHUB_SHA','local'),references=ref_manifest)
            # This file exists before the first test-set score is requested.
            (folder/'frozen-before-test.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
            np.save(folder/'calibration.npy',cal)
            results={};latencies={}
            for name,stream in streams.items():
                if name in ('train','calibration'):continue
                before=time.perf_counter();scores=np.asarray(score(stream));latencies[name]=time.perf_counter()-before
                if len(scores)!=len(stream['y']) or not np.isfinite(scores).all():raise ValueError('test scores incomplete/nonfinite')
                np.savez_compressed(folder/(name+'-scores.npz'),score=scores,y=stream['y'],kind=stream['kind'],episode=stream['episode'])
                results[name]={str(alpha):evaluate(stream,scores,points[str(alpha)],info) for alpha in P['calibration_fprs']}
            record={**frozen,'test_results':results,'test_seconds':latencies,
                'latency_scope':'batch throughput including replay of normal training history for graph models; not live network alert latency'}
            (folder/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
            print(json.dumps(dict(dataset=dataset,method=method,input=composition,seed=seed,finished=True)),flush=True)
    (output/'summary.json').write_text(json.dumps(dict(method=method,input=composition,records=records),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=P['models'],required=True);p.add_argument('--input',choices=P['inputs'],required=True)
    p.add_argument('--references',type=Path,default=Path('.references'));a=p.parse_args()
    with threadpool_limits(limits=P['torch_threads']):run(a.data,a.output,a.method,a.input,a.references)
