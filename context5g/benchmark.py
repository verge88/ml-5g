"""Compare fixed context rules and five existing models on recorded SBI requests."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits
from neural5g.benchmark import fit_neural,predict,seed_all
from neural5g.models import make_model
from .data import P,NUMERIC,FEATURES,prepare,scopes


def evaluate(stream,scores,threshold,banks):
    scores=np.asarray(scores)
    if len(scores)!=len(stream) or not np.isfinite(scores).all():
        raise ValueError("invalid prediction array")
    flags=scores>threshold
    normal=np.array([r["label"]=="normal" for r in stream.rows])
    n=int(normal.sum());fp=int(flags[normal].sum())
    groups={}
    for name,mask in scopes(stream.rows,banks).items():
        positive=int(mask.sum());tp=int(flags[mask].sum())
        groups[name]=dict(requests=positive,detected=tp,recall=tp/positive if positive else None,
            status="OBSERVED" if positive else "NOT_IDENTIFIABLE_NO_EXAMPLES")
    span=float(stream.time[-1]-stream.time[0]) if len(stream)>1 else 0.
    return dict(requests=len(stream),normal_requests=n,false_alerts=fp,
        fpr=fp/n if n else None,scopes=groups,request_span_seconds=span,
        false_marks_per_request_span_hour=fp*3600/span if n==len(stream) and span>0 else None,
        exposure_scope="clean holdout request time span only; mixed-capture normal exposure UNKNOWN")


def cutoff(scores,alpha):
    return float(np.quantile(scores,1-alpha,method="higher"))


def operating_points(cal_scores,policy=False):
    return {str(alpha):dict(threshold=.5 if policy else cutoff(cal_scores,alpha),
        calibration_fpr=float((cal_scores>(.5 if policy else cutoff(cal_scores,alpha))).mean()))
        for alpha in P["calibration_fprs"]}


def record_points(cal_scores,test_scores,streams,banks,points):
    output=[]
    for alpha in P["calibration_fprs"]:
        frozen=points[str(alpha)]
        metrics={name:evaluate(streams[name],scores,frozen["threshold"],banks)
            for name,scores in test_scores.items()}
        target=next(name for name in metrics if name!="normal_holdout")
        budget=all(metrics[name]["fpr"] is not None and metrics[name]["fpr"]<=alpha
            for name in (target,"normal_holdout"))
        output.append(dict(alpha=alpha,threshold=frozen["threshold"],calibration_fpr=frozen["calibration_fpr"],
            calibration_budget_met=frozen["calibration_fpr"]<=alpha,test_budget_met=budget,results=metrics))
    return output


def rule_scores(raw,streams,center,scale):
    result={}
    for name,values in raw.items():
        z=(values-center)/scale
        maximum=np.maximum(np.max(abs(z[:,:len(NUMERIC)]),axis=1),np.max(abs(streams[name].x[:,:17]),axis=1))
        state={};cumulative=[]
        for row,value in zip(streams[name].rows,z[:,5]):
            src=row["source"]
            state[src]=max(0.,state.get(src,0.)+float(value)-P["cusum_reference"])
            cumulative.append(state[src])
        result[name]={"edge_policy":values[:,0],"edge_operation_policy":values[:,1],
            "caller_operation_policy":values[:,2],"transition_policy":np.max(values[:,[2,6,7]],axis=1),
            "markov_order1":values[:,4],"markov_order2":values[:,5],
            "context_max_mad":maximum,"context_cusum":np.array(cumulative)}
    return result


def save_scores(folder,streams,test_scores,cal_scores):
    np.save(folder/"calibration.npy",cal_scores)
    for name,scores in test_scores.items():
        np.savez_compressed(folder/(name.replace(".csv","")+".npz"),score=scores,
            label=streams[name].y,raw_row=np.array([r["packet_row"] for r in streams[name].rows]))


def model_run(data,output,architecture):
    torch.set_num_threads(P["torch_threads"])
    records=[]
    for fold,source,target in P["folds"]:
        views,raw,info,banks=prepare(data,source,target)
        for composition in P["inputs"]:
            streams=views[composition]
            training=[streams["train_normal"],streams["train_source"]]
            for seed in P["seeds"]:
                seed_all(seed)
                folder=output/fold/composition/str(seed);folder.mkdir(parents=True,exist_ok=True)
                started=time.perf_counter()
                if architecture=="hgb":
                    xx=np.concatenate([s.x for s in training]);yy=np.concatenate([s.y for s in training])
                    positive_weight=float((len(yy)-yy.sum())/yy.sum())
                    model=HistGradientBoostingClassifier(max_iter=P["hgb_iterations"],max_depth=P["hgb_max_depth"],
                        learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=seed)
                    model.fit(xx,yy,sample_weight=np.where(yy>0,positive_weight,1.))
                    score=lambda s:model.predict_proba(s.x)[:,1]
                    training_info=dict(points=len(yy),positive_rows=int(yy.sum()))
                    joblib.dump(model,folder/"model.joblib")
                else:
                    model=make_model(architecture,training[0].x.shape[1],info["n_nodes"],P)
                    training_info=fit_neural(model,training,architecture,seed,P)
                    score=lambda s:predict(model,s,architecture,P)
                    torch.save(model.state_dict(),folder/"weights.pt")
                seconds=time.perf_counter()-started
                cal=score(streams["calibration"]);points=operating_points(cal)
                frozen=dict(model=architecture,fold=fold,source=source,target=target,input=composition,seed=seed,
                    protocol=P,features=info,input_dim=training[0].x.shape[1],points=points,
                    training=training_info,training_seconds=seconds)
                (folder/"frozen-before-evaluation.json").write_text(json.dumps(frozen,indent=2))
                tests={name:score(streams[name]) for name in (target,"normal_holdout")}
                save_scores(folder,streams,tests,cal)
                frozen["operating_points"]=record_points(cal,tests,streams,banks,points)
                records.append(frozen)
                print(json.dumps(dict(model=architecture,fold=fold,input=composition,seed=seed)),flush=True)
    return records


def rules_run(data,output):
    records=[]
    for fold,source,target in P["folds"]:
        views,raw,info,banks=prepare(data,source,target)
        streams=views["metadata_api_context"]
        scored=rule_scores(raw,streams,np.array(info["context_center"]),np.array(info["context_scale"]))
        for method in P["rules"]:
            folder=output/fold/method;folder.mkdir(parents=True,exist_ok=True)
            cal=scored["calibration"][method]
            points=operating_points(cal,policy=method.endswith("policy"))
            frozen=dict(model=method,fold=fold,source=source,target=target,input="same_observable_context",seed=None,
                protocol=P,features=info,points=points)
            (folder/"frozen-before-evaluation.json").write_text(json.dumps(frozen,indent=2))
            tests={name:scored[name][method] for name in (target,"normal_holdout")}
            save_scores(folder,streams,tests,cal)
            frozen["operating_points"]=record_points(cal,tests,streams,banks,points)
            records.append(frozen)
        # A small learned-rule control receives the same labelled source as ML.
        for composition in P["inputs"]:
            streams=views[composition]
            training=[streams["train_normal"],streams["train_source"]]
            xx=np.concatenate([s.x for s in training]);yy=np.concatenate([s.y for s in training])
            model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,class_weight="balanced",random_state=P["seeds"][0])
            model.fit(xx,yy)
            folder=output/fold/("shallow_tree_"+composition);folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,folder/"model.joblib")
            (folder/"learned-rules.txt").write_text(export_text(model))
            cal=model.predict_proba(streams["calibration"].x)[:,1];points=operating_points(cal)
            frozen=dict(model="shallow_tree",fold=fold,source=source,target=target,input=composition,
                seed=P["seeds"][0],protocol=P,features=info,points=points)
            (folder/"frozen-before-evaluation.json").write_text(json.dumps(frozen,indent=2))
            tests={name:model.predict_proba(streams[name].x)[:,1] for name in (target,"normal_holdout")}
            save_scores(folder,streams,tests,cal)
            frozen["operating_points"]=record_points(cal,tests,streams,banks,points)
            records.append(frozen)
        print(json.dumps(dict(fold=fold,rules_complete=True)),flush=True)
    return records


def main(data,output,model):
    output.mkdir(parents=True,exist_ok=True)
    records=rules_run(data,output) if model=="rules" else model_run(data,output,model)
    manifest=json.loads((data/"manifest.json").read_text())
    summary=dict(model=model,records=records,protocol=P,source_manifest=manifest,
        code_commit=__import__("os").environ.get("GITHUB_SHA","local"),
        protocol_sha256=hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest())
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=P["models"]+["rules"],required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=P["torch_threads"]):
        main(args.data,args.output,args.model)
