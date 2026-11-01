"""Learn per-NF scheduling scores on the eight prior deployments only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joint5g.evaluate import read_run
from .schedule import SLOTS

def train(inputs,out):
    plan=json.loads(Path(__file__).with_name('plan.json').read_text()); runs=[]
    for n in plan['training_deployments']:
        r=read_run(inputs/f'joint-native-{n}',n); runs.append(r)
    models={}
    for j,s in enumerate(SLOTS):
        xx=[];yy=[]
        for r in runs:
            mask=r['train']; y=np.array([any(e['slot']==s and e['start_tick']<=v['tick']<e['end_tick'] for e in r['events']) for v in r['rows']])
            xx.append(r['raw'][mask,j*6:(j+1)*6]); yy.append(y[mask])
        x=np.concatenate(xx); y=np.concatenate(yy); scaler=StandardScaler().fit(x[~y])
        m=LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=11).fit(scaler.transform(x),y)
        models[s]=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=m.coef_[0].tolist(),intercept=float(m.intercept_[0]),training_rows=len(y),positive=int(y.sum()),iterations=m.n_iter_.tolist())
    out.mkdir(parents=True,exist_ok=True)
    (out/'risk-models.json').write_text(json.dumps(dict(models=models,training_deployments=plan['training_deployments'],training_sources=[r['source_sha256'] for r in runs],model='logistic_regression',seed=11),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();train(a.input,a.output)
