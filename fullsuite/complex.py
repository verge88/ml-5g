"""Rich shared features, instantaneous statistical rival and budgeted schedules."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from complex5g import benchmark as old
from complex5g.features import LearnedFeatures
from complex5g.generator import datasets
from hybrid.benchmark import schedule
from .common import P, cutoff, provenance, robust, temporal, write_csv


def rich_statistics(representation,trace,center,scale):
    rich,components=representation.transform(trace)
    # Unlike the old averaged rival, this also sees instantaneous transition
    # and cross-NF surprise already present in the classifiers' rich features.
    all_components=np.concatenate([components,rich[...,-10,None],rich[...,-8,None]],axis=2)
    signal=np.max((all_components-center)/scale,axis=2)
    return {"instant_structure":signal,"ewma_structure":temporal(signal,"ewma"),"cusum_structure":temporal(signal,"cusum")}


def run(seed,out):
    record=old.experiment(seed,out)
    folder=Path(out)/f"seed-{seed}"
    normal,train,calibration,test=datasets(seed)
    representation=LearnedFeatures(normal)
    rows=[]
    for r in normal:
        rich,components=representation.transform(r)
        rows.append(np.concatenate([components,rich[...,-10,None],rich[...,-8,None]],axis=2).reshape(-1,9))
    center,scale=robust(np.concatenate(rows),.1)
    cs=[rich_statistics(representation,r,center,scale) for r in calibration]
    ss={r.run_id:rich_statistics(representation,r,center,scale) for r in test}
    with np.load(folder/"test-observations.npz") as stored:
        original={m:stored["score_"+m].copy() for m in old.PROTOCOL["models"]}
    scores={r.run_id:{m:original[m][i] for m in old.PROTOCOL["models"]} for i,r in enumerate(test)}
    events=[]
    for name in ("instant_structure","ewma_structure","cusum_structure"):
        limit=cutoff(np.concatenate([s[name].ravel() for s in cs]))
        block=old.PROTOCOL["event_length"]
        ep_limit=cutoff(np.concatenate([s[name].reshape(-1,block,s[name].shape[1]).max(axis=1).ravel() for s in cs]),P["episode_alpha"])
        record["thresholds"][name]=limit;record["episode_thresholds"][name]=ep_limit
        for r in test:scores[r.run_id][name]=ss[r.run_id][name]
        for suffix,threshold in (("",limit),("@episode",ep_limit)):
            predictions={r.run_id:ss[r.run_id][name]>threshold for r in test}
            for hybrid in (False,True):
                key=("hybrid_" if hybrid else "")+name+suffix
                pp={r.run_id:predictions[r.run_id]|r.hard if hybrid else predictions[r.run_id] for r in test}
                if hybrid:
                    assert all(np.all(pp[r.run_id][r.hard]) for r in test)
                record["models"][key]=old.metrics(test,pp,key)
                events.extend(old.event_rows(test,pp,key))
                assert record["models"][key]["per_kind"]["invisible"]["paired_detection_gain"]==0
    with (folder/"events.csv").open(newline="") as f:previous=list(csv.DictReader(f))
    write_csv(folder/"events.csv",previous+events)
    scheduler={};scheduler_events=[];selected_arrays={}
    mapping={"ml_hgb":"hgb_rich","ml_mlp":"mlp_rich","ml_if":"if_rich","statistical":"instant_structure"}
    for mode in P["scheduler_modes"]:
        case_rows=[];budget=0
        selections=[]
        for r in test:
            priority=scores[r.run_id][mapping.get(mode,"hgb_rich")]
            selected=schedule(priority,mode if mode in ("uniform","oldest") else "ml")
            selections.append(selected);budget+=int(selected.sum())
            if r.event and r.event["has_integrity"]:
                e=r.event;hit=np.flatnonzero((selected&r.hard)[e["start"]:e["end"],e["nf"]])
                case_rows.append({"mode":mode,**e,"detected":bool(len(hit)),"restricted_delay":int(hit[0]) if len(hit) else e["end"]-e["start"]})
        selected_arrays[mode]=np.stack(selections)
        scheduler[mode]={"checks":budget,"max_gap_ticks":13,"per_kind":{k:{"events":sum(e["kind"]==k for e in case_rows),
            "recall":float(np.mean([e["detected"] for e in case_rows if e["kind"]==k])),
            "restricted_delay":float(np.mean([e["restricted_delay"] for e in case_rows if e["kind"]==k]))} for k in ("integrity_only","combined")}}
        scheduler_events.extend(case_rows)
    assert len({s["checks"] for s in scheduler.values()})==1
    np.savez_compressed(folder/"additional-scores-and-schedules.npz",**selected_arrays,
                        **{name:np.stack([ss[r.run_id][name] for r in test]) for name in ("instant_structure","ewma_structure","cusum_structure")})
    write_csv(folder/"scheduler-events.csv",scheduler_events)
    record.update(provenance("complex"),scheduler=scheduler,stronger_comparator="instantaneous and averaged normal-learned relations")
    (folder/"summary.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({"seed":seed,"instant_structure_episode":record["models"]["instant_structure@episode"]["behavior_event_recall"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            if seed not in P["complex_seeds"]:raise ValueError("unregistered test seed")
            run(seed,a.output)
