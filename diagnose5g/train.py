import argparse,hashlib,json,os
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import PLAN,load,steady
from .policy import GaussianDiagnostic


def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records,source=load(root,PLAN['train']);calibration,cal_source=load(root,PLAN['calibration'])
    x,y=steady(records);cx,cy=steady(calibration)
    assert len(x)>0 and set(y)==set(range(4))
    gaussian=GaussianDiagnostic().fit(x,y)
    models={'gaussian':gaussian}
    for inputs in ['end','both']:
        a=x[:,:1] if inputs=='end' else x
        models[f'logistic_{inputs}']=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=11)).fit(a,y)
        models[f'hgb_{inputs}']=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,random_state=11).fit(a,y)
    scores=np.array([gaussian.posterior(*a) for a in cx]);confidence=scores.max(1);pred=scores.argmax(1)
    cutoff=1.01
    for cut in sorted(set([.25]+confidence.tolist())):
        use=confidence>=cut
        if use.sum()>=8 and np.mean(pred[use]==cy[use])>=PLAN['calibration_minimum_precision']:cutoff=float(cut);break
    joblib.dump(models,out/'models.joblib')
    meta=dict(source_commit=os.environ.get('SOURCE_COMMIT','local'),plan_sha256=hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest(),
        train_source=source,calibration_source=cal_source,train_rows=len(y),calibration_rows=len(cy),train_class_counts=np.bincount(y,minlength=4).tolist(),
        abstention_cutoff=cutoff,model_sha256=hashlib.sha256((out/'models.joblib').read_bytes()).hexdigest(),test_files_downloaded=False,
        features=['measured_end_ms','measured_bypass_ms'],scope='pilot observed mechanism diagnosis, not attack classification')
    (out/'frozen.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
