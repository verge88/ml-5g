"""All basic detectors with corrected behavioral labels and identical fit rows."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from hybrid import benchmark as old
from .common import P, balanced_indices, fit_classifier, provenance, robust


class Models:
    def __init__(self, seed, normal, labeled):
        x, c = old.flatten(normal,"x"), old.flatten(normal,"context")
        self.expected = Ridge(alpha=1).fit(c,x)
        residual = x-self.expected.predict(c)
        self.center,self.scale = robust(residual,1e-4)
        self.raw_center,self.raw_scale = robust(x,.001)
        z = self.residual(x,c)
        self.precision = np.linalg.inv(np.cov(z.T)+.01*np.eye(z.shape[1]))
        self.models = {"if_raw": IsolationForest(n_estimators=100,max_samples=512,random_state=seed,n_jobs=1).fit(x),
                       "if_context": IsolationForest(n_estimators=100,max_samples=512,random_state=seed,n_jobs=1).fit(z)}
        lx,lc = old.flatten(labeled,"x"),old.flatten(labeled,"context")
        ys=[]
        for r in labeled:
            y=np.zeros_like(r.y)
            for e in r.events:
                if e["kind"] in {"fanout","burst","sequence","mixed"}:
                    y[e["start"]:e["end"],e["nf"]]=True
            ys.append(y.ravel())
        ly=np.concatenate(ys)
        take=balanced_indices(ly,seed+90000)
        self.warnings={}
        for representation,values in (("raw",(lx-self.raw_center)/self.raw_scale),("context",self.residual(lx,lc))):
            values=np.clip(values,-P["clip_standardized"],P["clip_standardized"])
            for name,model in (("hgb",HistGradientBoostingClassifier(max_iter=100,max_depth=3,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=seed)),
                               ("mlp",MLPClassifier(hidden_layer_sizes=tuple(P["mlp_hidden"]),max_iter=P["mlp_iterations"],batch_size=256,alpha=.001,early_stopping=False,random_state=seed))):
                key=name+"_"+representation
                self.warnings[key]=fit_classifier(model,values[take],ly[take])
                self.models[key]=model
        self.fit_points=len(take)
        Models.latest=self

    def residual(self,x,c):
        return (x-self.expected.predict(c)-self.center)/self.scale

    def scores(self,run):
        shape=run.y.shape
        x,c=run.x.reshape(-1,6),run.context.reshape(-1,run.context.shape[-1])
        z=self.residual(x,c)
        zz=z.reshape(*shape,6)
        result={"max_residual":z.max(axis=1).reshape(shape),
                "mahalanobis":np.sqrt(np.maximum(0,np.einsum("bi,ij,bj->b",z,self.precision,z))).reshape(shape),
                "ewma":old.sequential(zz,"ewma"),"cusum":old.sequential(zz,"cusum"),
                "if_raw":-self.models["if_raw"].score_samples(x).reshape(shape),
                "if_context":-self.models["if_context"].score_samples(z).reshape(shape)}
        for representation,values in (("raw",(x-self.raw_center)/self.raw_scale),("context",z)):
            values=np.clip(values,-P["clip_standardized"],P["clip_standardized"])
            for name in ("hgb","mlp"):
                key=name+"_"+representation
                result[key]=self.models[key].predict_proba(values)[:,1].reshape(shape)
        return result


def run(seed,out):
    old.Models=Models
    old.PROTOCOL["models"]=[m for m in P["basic_methods"] if m!="integrity"]
    record=old.experiment(seed,out)
    record.update(provenance("basic"),behavior_training_labels="exclude integrity-only and invisible",equal_supervised_fit_rows=True,
                  training_warnings=Models.latest.warnings,supervised_fit_points=Models.latest.fit_points)
    path=Path(out)/f"seed-{seed}"/"summary.json"
    path.write_text(json.dumps(record,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            if seed not in P["basic_seeds"]:raise ValueError("unregistered test seed")
            run(seed,a.output)
