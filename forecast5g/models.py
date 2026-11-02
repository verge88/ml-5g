"""Fixed model families, common causal inputs and no test-time fitting."""
import json,hashlib
from pathlib import Path
import joblib,numpy as np,torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import PLAN,evaluate,threshold,load
DROP_CONTEXT=[4,5,6,7,15,19,23]
KEEP_NO_CONTEXT=[i for i in range(24) if i not in DROP_CONTEXT]
REG_CONTEXT=[5,6,7,8,9]

class Net(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='mlp':self.net=nn.Sequential(nn.Linear(24,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        else:self.net=nn.LSTM(12,32,batch_first=True);self.head=nn.Linear(32,1)
    def forward(self,x):
        if self.kind=='mlp':return self.net(x).flatten()
        z,_=self.net(x);return self.head(z[:,-1]).flatten()

def score_stats(t,normalizer):
    x=t['point'];f=t['features'];expected=normalizer['regressor'].predict(x[:,REG_CONTEXT]);res=x[:,0]-expected
    ewma=[];cusum=[];a=0.;b=0.
    for r in res:
        a=.3*r+.7*a;b=max(0,b+r/normalizer['scale']-.5);ewma.append(a);cusum.append(b)
    return dict(current_latency=x[:,0],conditional_residual=res,ewma=np.array(ewma),cusum=np.array(cusum),linear_forecast=x[:,0]+PLAN['forecast_horizon_seconds']*f[:,18])

def fit(train,calibration,output,provenance,commit):
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    train_ids={t['replica'] for t in train};cal_ids={t['replica'] for t in calibration}
    assert train_ids==set(PLAN['train']) and cal_ids==set(PLAN['calibration']) and not (train_ids|cal_ids)&set(PLAN['test'])
    x=np.concatenate([t['features'][t['eligible']] for t in train]);p=np.concatenate([t['point'][t['eligible']] for t in train]);w=np.concatenate([t['windows'][t['eligible']] for t in train]);y=np.concatenate([t['y'][t['eligible']] for t in train]).astype(int)
    assert y.sum()>20 and (1-y).sum()>100
    normal=np.concatenate([t['point'][t['eligible']] for t in train if t['onset'] is None]);assert len(normal)>100
    reg=HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_iter=100,max_leaf_nodes=15,min_samples_leaf=15,random_state=11,early_stopping=False).fit(normal[:,REG_CONTEXT],normal[:,0])
    residual=normal[:,0]-reg.predict(normal[:,REG_CONTEXT]);stats=dict(regressor=reg,scale=max(1.,float(np.median(np.abs(residual-np.median(residual)))*1.4826)))
    joblib.dump(stats,output/'statistics.joblib')
    descriptors=[];cal_scores={}
    for name in PLAN['statistical_comparators']:
        values=[score_stats(t,stats)[name] for t in calibration];cut=threshold(calibration,values)
        descriptors.append(dict(name=name,kind='statistical',threshold=cut,calibration=evaluate(calibration,values,cut)));cal_scores[name]=np.stack(values)
    scaler=StandardScaler().fit(x);pscaler=StandardScaler().fit(p)
    joblib.dump(dict(features=scaler,point=pscaler),output/'scalers.joblib')
    learners=[('logistic_history','history',LogisticRegression(class_weight='balanced',max_iter=1000,C=1.,random_state=11)),
        ('hgb_history','history',HistGradientBoostingClassifier(class_weight='balanced',max_iter=150,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=15,early_stopping=False,random_state=11)),
        ('hgb_instant','instant',HistGradientBoostingClassifier(class_weight='balanced',max_iter=150,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=15,early_stopping=False,random_state=11)),
        ('hgb_without_context','without_context',HistGradientBoostingClassifier(class_weight='balanced',max_iter=150,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=15,early_stopping=False,random_state=11))]
    for name,composition,model in learners:
        xx=p if composition=='instant' else x[:,KEEP_NO_CONTEXT] if composition=='without_context' else scaler.transform(x)
        model.fit(xx,y);joblib.dump(model,output/(name+'.joblib'))
        values=[]
        for t in calibration:
            xx=t['point'] if composition=='instant' else t['features'][:,KEEP_NO_CONTEXT] if composition=='without_context' else scaler.transform(t['features'])
            values.append(model.predict_proba(xx)[:,1])
        cut=threshold(calibration,values);descriptors.append(dict(name=name,kind='learned',composition=composition,threshold=cut,calibration=evaluate(calibration,values,cut)));cal_scores[name]=np.stack(values)
    for kind in ['mlp','lstm']:
        xx=scaler.transform(x).astype('float32') if kind=='mlp' else ((w-pscaler.mean_)/pscaler.scale_).astype('float32')
        for seed in PLAN['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Net(kind);optimizer=torch.optim.Adam(net.parameters(),lr=PLAN['neural_learning_rate'])
            loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y)-y.sum())/y.sum(),dtype=torch.float32));tx=torch.from_numpy(xx);ty=torch.from_numpy(y.astype('float32'));losses=[]
            for epoch in range(PLAN['neural_epochs']):
                order=rng.permutation(len(y));total=0.
                for start in range(0,len(y),PLAN['neural_batch']):
                    idx=order[start:start+PLAN['neural_batch']];optimizer.zero_grad();value=loss(net(tx[idx]),ty[idx]);value.backward();optimizer.step();total+=float(value.detach())*len(idx)
                losses.append(total/len(y))
            name=f'{kind}_history_{seed}';torch.save(net.state_dict(),output/(name+'.pt'));net.eval();values=[]
            with torch.no_grad():
                for t in calibration:
                    v=scaler.transform(t['features']).astype('float32') if kind=='mlp' else ((t['windows']-pscaler.mean_)/pscaler.scale_).astype('float32')
                    values.append(torch.sigmoid(net(torch.from_numpy(v))).numpy())
            cut=threshold(calibration,values);descriptors.append(dict(name=name,kind='neural',architecture=kind,seed=seed,threshold=cut,losses=losses,calibration=evaluate(calibration,values,cut)));cal_scores[name]=np.stack(values)
    stat=[d for d in descriptors if d['kind']=='statistical']
    best=max(stat,key=lambda d:(d['calibration']['mean_early_lead_including_misses'] or 0,d['calibration']['early_detected'],-d['calibration']['benign_false_alerts'],-d['calibration']['premature_false_alerts']))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*') if p.suffix in ['.pt','.joblib']}
    frozen=dict(commit=commit,plan=PLAN,models=descriptors,selected_statistical=best['name'],primary_model=PLAN['primary_model'],training_eligible_rows=len(y),training_positives=int(y.sum()),training_deployments=sorted(train_ids),calibration_deployments=sorted(cal_ids),test_access_during_fit=False,inputs=provenance,sha256=hashes)
    (output/'frozen.json').write_text(json.dumps(frozen,indent=2));np.savez_compressed(output/'calibration-scores.npz',**cal_scores)
    (output/'calibration-order.json').write_text(json.dumps([dict(replica=t['replica'],trial=t['name']) for t in calibration]))
    return frozen

def predict(trials,folder):
    frozen=json.loads((folder/'frozen.json').read_text());assert frozen['plan']==PLAN
    for name,digest in frozen['sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    stats=joblib.load(folder/'statistics.joblib');scalers=joblib.load(folder/'scalers.joblib');result={};torch.set_num_threads(1)
    for d in frozen['models']:
        name=d['name'];values=[]
        if d['kind']=='statistical':values=[score_stats(t,stats)[name] for t in trials]
        elif d['kind']=='learned':
            model=joblib.load(folder/(name+'.joblib'))
            for t in trials:
                xx=t['point'] if d['composition']=='instant' else t['features'][:,KEEP_NO_CONTEXT] if d['composition']=='without_context' else scalers['features'].transform(t['features'])
                values.append(model.predict_proba(xx)[:,1])
        else:
            kind=d['architecture'];net=Net(kind);net.load_state_dict(torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True));net.eval()
            with torch.no_grad():
                for t in trials:
                    xx=scalers['features'].transform(t['features']).astype('float32') if kind=='mlp' else ((t['windows']-scalers['point'].mean_)/scalers['point'].scale_).astype('float32')
                    values.append(torch.sigmoid(net(torch.from_numpy(xx))).numpy())
        result[name]=values
    return result,frozen
