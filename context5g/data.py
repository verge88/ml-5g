"""Frozen normal-only context banks, causally updated histories, no label input."""
from collections import Counter, defaultdict, deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from neural5g.data import prepare as base_prepare
from semantic5g.analyze import candidate

P = json.loads(Path(__file__).with_name("protocol.json").read_text())
NUMERIC = ["edge_unseen","edge_operation_unseen","caller_operation_unseen","operation_surprise",
    "transition1_surprise","transition2_surprise","transition1_unseen","transition2_unseen",
    "caller_gap_log","edge_gap_log","recent_caller_count_log","recent_edge_count_log",
    "caller_same_operation_fraction","caller_destination_diversity","caller_operation_diversity"]
FEATURES = NUMERIC+[f"past_{lag}_hash_{i}" for lag in (1,2) for i in range(P["history_hash_bins"])]


def operation(row):
    return row["destination"],row["method"],row["path"]


class ContextBank:
    def __init__(self, rows):
        self.edge = set();self.edge_op=set();self.caller_op=set()
        self.counts=[Counter(),Counter(),Counter()]
        self.totals=[Counter(),Counter(),Counter()]
        vocabulary=set();history=defaultdict(lambda:deque(maxlen=2))
        for row in rows:
            src=row["source"];op=operation(row);past=tuple(history[src])
            self.edge.add((src,row["destination"]))
            self.edge_op.add((src,*op));self.caller_op.add((src,row["method"],row["path"]))
            vocabulary.add(op)
            for order in (0,1,2):
                ctx=(src,tuple(past[-order:]) if order else ())
                self.counts[order][(ctx,op)]+=1;self.totals[order][ctx]+=1
            history[src].append(op)
        self.vocabulary_size=len(vocabulary)+1

    def score(self, rows):
        history=defaultdict(lambda:deque(maxlen=P["history_window"]))
        global_recent=deque(maxlen=32);last_caller={};last_edge={};result=[]
        for row in rows:
            src=row["source"];dst=row["destination"];op=operation(row);past=tuple(history[src])
            surprises=[];unseen=[]
            for order in (0,1,2):
                ctx=(src,tuple(past[-order:]) if order else ())
                n=self.counts[order][(ctx,op)]
                surprises.append(-np.log((n+1)/(self.totals[order][ctx]+self.vocabulary_size)))
                unseen.append(float(n==0))
            edge=(src,dst);t=row["time"]
            caller_gap=max(0.,t-last_caller.get(src,t));edge_gap=max(0.,t-last_edge.get(edge,t))
            values=[float(edge not in self.edge),float((src,*op) not in self.edge_op),
                float((src,row["method"],row["path"]) not in self.caller_op),*surprises,
                unseen[1],unseen[2],np.log1p(caller_gap),np.log1p(edge_gap),
                np.log1p(sum(s==src for s,d in global_recent)),
                np.log1p(sum((s,d)==edge for s,d in global_recent)),
                sum(p==op for p in past)/max(1,len(past)),
                len({p[0] for p in past})/P["history_window"],len(set(past))/P["history_window"]]
            for lag in (1,2):
                vector=np.zeros(P["history_hash_bins"],dtype=np.float32)
                if len(past)>=lag:
                    index=int.from_bytes(hashlib.sha256(repr(past[-lag]).encode()).digest()[:8],"big")%len(vector)
                    vector[index]=1
                values.extend(vector.tolist())
            result.append(values)
            # Counts are frozen; only past observable events update local history.
            history[src].append(op);global_recent.append(edge);last_caller[src]=t;last_edge[edge]=t
        return np.asarray(result,dtype=np.float32)


def prepare(root, source, target):
    training,calibration,tests,info,banks=base_prepare(root,source,target,True)
    bank=ContextBank(training[0].rows)
    named={"train_normal":training[0],"train_source":training[1],"calibration":calibration,**tests}
    raw={name:bank.score(stream.rows) for name,stream in named.items()}
    center=np.median(raw["train_normal"],axis=0)
    scale=np.maximum(1.4826*np.median(abs(raw["train_normal"]-center),axis=0),.1)
    views={name:{} for name in P["inputs"]}
    for name,stream in named.items():
        stamps=stream.time-stream.time[0] if len(stream) else stream.time
        views["metadata_api"][name]=replace(stream,time=stamps)
        extended=np.concatenate([stream.x,np.clip((raw[name]-center)/scale,-20,20)],axis=1).astype(np.float32)
        views["metadata_api_context"][name]=replace(stream,x=extended,time=stamps)
    info.update(context_feature_names=FEATURES,context_center=center.tolist(),context_scale=scale.tolist(),
        caller_roles="UNKNOWN; observed network addresses only",authorization="UNKNOWN",time="relative per stream")
    return views,raw,info,banks


def scopes(rows, banks):
    candidates=np.array([candidate(r) for r in rows],dtype=bool)
    known=np.array([(r["source"],r["destination"]) in banks[0] for r in rows],dtype=bool)
    access=np.array([r["family"] in ("authentication_data","subscription_data") for r in rows],dtype=bool)
    return {"all_producer_candidates":candidates,"known_edge_candidates":candidates&known,
        "data_access_known_edge":candidates&known&access,"data_access_all_edges":candidates&access,
        "subscription_data":candidates&np.array([r["family"]=="subscription_data" for r in rows]),
        "deregistration_notification":candidates&np.array([r["family"]=="deregistration_notification" for r in rows])}
