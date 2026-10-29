"""All applicable detector families on fixed external capture transfer folds."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from hybrid import external as old
from .common import P,balanced_indices,cutoff,fit_classifier,provenance,robust,temporal,write_csv,write_json


def run(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    normal=old.read(Path(data)/"bridge_normal.csv");assert not old.labels(normal).any()
    te,ce=int(.6*len(normal)),int(.8*len(normal))
    reference=old.memory(normal[:te]);nx=old.features(normal,reference)
    train,cal=nx[:te],nx[te+32:ce]
    center,scale=robust(train,.1)
    z=(train-center)/scale
    precision=np.linalg.inv(np.cov(z.T)+.01*np.eye(z.shape[1]))
    packet=IsolationForest(n_estimators=100,max_samples=512,random_state=3401,n_jobs=1).fit(train[:,:12])
    graph=IsolationForest(n_estimators=100,max_samples=512,random_state=3401,n_jobs=1).fit(train)
    gc,gs=robust((-graph.score_samples(train))[:,None],.001)
    def scores(x):
        zz=(x-center)/scale
        raw=np.max(abs(zz),axis=1)
        s=-graph.score_samples(x)
        return {"edgebank":x[:,-1],"max_mad":raw,
                "mahalanobis":np.sqrt(np.maximum(0,np.einsum("bi,ij,bj->b",zz,precision,zz))),
                "ewma":temporal(raw,"ewma"),"cusum":temporal(raw,"cusum"),
                "if_packet":-packet.score_samples(x[:,:12]),"if_packet_graph":s,
                "if_structure_guard":np.maximum(((s[:,None]-gc)/gs).ravel(),raw)}
    limits={k:cutoff(v) for k,v in scores(cal).items()}
    captures={name:old.read(Path(data)/name) for name in ("bridge_attack_nrf.csv","bridge_attack_smf.csv")}
    cx={k:old.features(v,reference) for k,v in captures.items()}
    evaluations={"normal_holdout":(normal[ce+32:],nx[ce+32:]),**{k:(v,cx[k]) for k,v in captures.items()}}
    unsupervised={}
    for capture,(rows,x) in evaluations.items():
        sc=scores(x);y=old.labels(rows)
        unsupervised[capture]={k:old.metrics(y,s,limits[k],rows) for k,s in sc.items()}
        write_csv(out/(capture.replace(".csv","")+"-predictions.csv"),[{"row":i,"label":row["Label"],**{k:float(v[i]) for k,v in sc.items()}} for i,row in enumerate(rows)])
    supervised={};warnings={}
    for fold,source,target in (("train_nrf_test_smf","bridge_attack_nrf.csv","bridge_attack_smf.csv"),("train_smf_test_nrf","bridge_attack_smf.csv","bridge_attack_nrf.csv")):
        lx=np.concatenate([train,cx[source]]);y=np.r_[np.zeros(len(train),dtype=bool),old.labels(captures[source])]
        take=balanced_indices(y,3402)
        xx=np.clip((lx-center)/scale,-20,20)
        supervised[fold]={};warnings[fold]={}
        for name,model in (("hgb_cross_capture",HistGradientBoostingClassifier(max_iter=150,max_depth=4,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=3402)),
                           ("mlp_cross_capture",MLPClassifier(hidden_layer_sizes=tuple(P["mlp_hidden"]),max_iter=P["mlp_iterations"],batch_size=256,alpha=.001,early_stopping=False,random_state=3402))):
            warnings[fold][name]=fit_classifier(model,xx[take],y[take])
            threshold=cutoff(model.predict_proba(np.clip((cal-center)/scale,-20,20))[:,1])
            results={}
            for capture in (target,"normal_holdout"):
                rows,x=evaluations[capture]
                score=model.predict_proba(np.clip((x-center)/scale,-20,20))[:,1]
                results[capture]=old.metrics(old.labels(rows),score,threshold,rows)
                np.savez_compressed(out/(fold+"-"+name+"-"+capture.replace(".csv","")+".npz"),score=score,label=old.labels(rows))
            supervised[fold][name]={"threshold":threshold,"results":results,"fit_points":len(take)}
    summary={**provenance("external"),"train_rows":te,"calibration_rows":len(cal),"normal_holdout_rows":len(nx[ce+32:]),
             "features":old.FEATURES,"thresholds":limits,"unsupervised":unsupervised,"supervised":supervised,"mlp_warnings":warnings,
             "source_manifest":json.loads((Path(data)/"manifest.json").read_text()),
             "integrity_status":"NOT_APPLICABLE: no NFProfile, CU or witness inputs", "population_repeats":0}
    write_json(out/"summary.json",summary)
    print(json.dumps({"domain":"external","unsupervised_methods":len(limits),"supervised_folds":list(supervised)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.data,a.output)
