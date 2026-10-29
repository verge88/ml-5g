"""Identical live rows for every applicable model; causal CU state is separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from live5g.evaluate import features,FEATURES
from .common import P,balanced_indices,cutoff,fit_classifier,provenance,robust,temporal,write_json,write_csv


def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    observations=[json.loads(s) for s in (root/"observations.jsonl").read_text().splitlines()]
    events=json.loads((root/"events.json").read_text());manifest=json.loads((root/"capture-manifest.json").read_text())
    rows=[];data=[];hard_cu=[];policy=[];statuses=[]
    for r in observations:
        for slot,obs in r["observations"].items():
            rows.append({"tick":r["tick"],"ns":r["monotonic_ns"],"stage":r["stage"],"slot":slot})
            data.append(features(slot,obs));hard_cu.append(obs["integrity_monitor"]["hard_alarm"])
            policy.append(obs["deployment_policy_alert"]);statuses.append(obs["integrity_monitor"]["decision"]["verdict"])
    x=np.asarray(data);fit=np.array([r["stage"]=="fit_normal" for r in rows]);cal=np.array([r["stage"]=="calibration_normal" for r in rows])
    train=fit|np.array([r["stage"].startswith("train_") for r in rows]);normal=np.array([r["stage"] in ("test_normal","test_baseline") for r in rows])
    recovery=np.array([r["stage"]=="test_recovery" for r in rows]);truth=np.zeros(len(rows),dtype=bool)
    for e in events:truth|=np.array([e["start_tick"]<=r["tick"]<e["end_tick"] and r["slot"]==e["slot"] for r in rows])
    raw_center,raw_scale=robust(x[fit,:6],.1)
    regression=Ridge(alpha=1).fit(x[fit,6:],x[fit,:6]);res=x[:,:6]-regression.predict(x[:,6:])
    rc,rs=robust(res[fit],.1);context=(res-rc)/rs;raw=(x[:,:6]-raw_center)/raw_scale
    inputs={"raw":np.c_[raw,x[:,6:]],"context":np.c_[context,x[:,6:]]}
    rawmax=np.max(abs(raw),axis=1);contextmax=np.max(abs(context),axis=1)
    precision=np.linalg.inv(np.cov(context[fit].T)+.01*np.eye(6))
    scores={"availability_rule":x[:,0],"profile_policy":np.asarray(policy,dtype=float),"cu_witness_alarm":np.asarray(hard_cu,dtype=float),
            "robust_raw":rawmax,"robust_context":contextmax,
            "mahalanobis":np.sqrt(np.maximum(0,np.einsum("bi,ij,bj->b",context,precision,context)))}
    for name in ("ewma","cusum"):
        scores[name]=temporal(contextmax.reshape(-1,2),name).ravel()
    take0=np.flatnonzero(train);take=take0[balanced_indices(truth[train],manifest["replica"]+100)]
    warnings={};times={}
    for representation,values in inputs.items():
        forest=IsolationForest(n_estimators=150,max_samples=min(256,int(fit.sum())),random_state=manifest["replica"],n_jobs=1).fit(values[fit])
        scores["if_"+representation]=-forest.score_samples(values)
        vv=np.clip(values,-20,20)
        for name,model in (("hgb",HistGradientBoostingClassifier(max_iter=150,max_depth=4,min_samples_leaf=5,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=manifest["replica"])),
                           ("mlp",MLPClassifier(hidden_layer_sizes=tuple(P["mlp_hidden"]),max_iter=P["mlp_iterations"],batch_size=64,alpha=.001,early_stopping=False,random_state=manifest["replica"]))):
            key=name+"_"+representation;warnings[key]=fit_classifier(model,vv[take],truth[take]);scores[key]=model.predict_proba(vv)[:,1]
    fc,fs=robust(scores["if_context"][fit,None],.001)
    scores["if_structure_guard"]=np.maximum(((scores["if_context"][:,None]-fc)/fs).ravel(),contextmax)
    assert set(scores)==set(P["native_methods"])
    hard=(scores["availability_rule"]>.5)|(scores["profile_policy"]>.5)|(scores["cu_witness_alarm"]>.5)
    models={};event_rows=[];arrays={"x":x,"fit":fit,"cal":cal,"train":train,"truth":truth,"hard":hard}
    for name,score in scores.items():
        threshold=.5 if name in ("availability_rule","profile_policy","cu_witness_alarm") else cutoff(score[cal])
        pred=score>threshold;arrays["score_"+name]=score
        for hybrid in (False,True):
            key=("hybrid_" if hybrid else "")+name
            alarms=pred|hard if hybrid else pred
            if hybrid:assert np.all(alarms[hard])
            outcomes=[]
            for e in events:
                if e["split"]!="test":continue
                ix=np.array([e["start_tick"]<=r["tick"]<e["end_tick"] and r["slot"]==e["slot"] for r in rows]);hits=np.flatnonzero(ix&alarms)
                outcomes.append({**e,"detected":bool(len(hits)),"delay_seconds":(rows[hits[0]]["ns"]-e["start_ns"])/1e9 if len(hits) else None,
                                 "restricted_delay_seconds":(rows[hits[0]]["ns"]-e["start_ns"])/1e9 if len(hits) else (e["end_ns"]-e["start_ns"])/1e9})
            models[key]={"threshold":threshold,"episodes":outcomes,"detected_episodes":sum(e["detected"] for e in outcomes),
                         "normal_points":int(normal.sum()),"false_points":int(alarms[normal].sum()),"normal_point_fpr":float(alarms[normal].mean()),
                         "recovery_alarm_fraction":float(alarms[recovery].mean())}
            event_rows.extend({"model":key,**e} for e in outcomes)
    np.savez_compressed(out/"scores.npz",**arrays);write_csv(out/"events.csv",event_rows)
    summary={**provenance("native"),"replica":manifest["replica"],"models":models,"features":FEATURES,"supervised_fit_points":len(take),
             "supervised_positive_points":int(truth[take].sum()),"training_warnings":warnings,"monitor_states":dict(Counter(statuses)),
             "normal_monitor_states":dict(Counter(s for s,b in zip(statuses,normal) if b)),
             "unknown_is_not_clear":True,"full_profile_verdict_scope":"selected profile fields, runtime identity and source trust assumptions", "capture_manifest":manifest}
    write_json(out/"summary.json",summary)
    print(json.dumps({"replica":manifest["replica"],"models":{k:{"detected":v["detected_episodes"],"false":v["false_points"]} for k,v in models.items() if not k.startswith("hybrid_")},"monitor_states":summary["monitor_states"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.input,a.output)
