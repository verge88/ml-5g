"""Frozen likelihood model and causal optional-probe policy for OFFLINE replay."""
import numpy as np


def normalize(logp):
    v=np.exp(logp-np.max(logp));return v/v.sum()


class GaussianDiagnostic:
    def fit(self,x,y):
        self.mean=np.array([x[y==k].mean(0) for k in range(4)])
        self.var=np.array([np.maximum(x[y==k].var(0),100.) for k in range(4)])
        self.prior=np.full(4,.25)
        assert np.isfinite(self.mean).all() and np.isfinite(self.var).all()
        return self

    def likelihood(self,value,index,age=0.):
        # Descriptive ageing model, not a validity guarantee under drift.
        var=self.var[:,index]+(16*max(0.,age))**2
        return -.5*(np.log(2*np.pi*var)+(value-self.mean[:,index])**2/var)

    def posterior(self,end,bypass=None,age=0.):
        logp=np.log(self.prior)+self.likelihood(end,0)
        if bypass is not None:logp+=self.likelihood(bypass,1,age)
        return normalize(logp)

    def expected_risk_reduction(self,prior):
        # Integrate possible answers; actual unselected observation is never read.
        nodes,weights=np.polynomial.hermite.hermgauss(15)
        expected_accuracy=0.
        for k in range(4):
            for node,weight in zip(nodes,weights):
                value=self.mean[k,1]+np.sqrt(2*self.var[k,1])*node
                posterior=normalize(np.log(np.maximum(prior,1e-300))+self.likelihood(value,1))
                expected_accuracy+=prior[k]*weight/np.sqrt(np.pi)*float(posterior.max())
        return max(0.,float(expected_accuracy-prior.max()))


class ProbePolicy:
    def __init__(self,mode,budget=10,max_age=6.):
        assert mode in ['fixed','oldest','expected_risk']
        self.mode=mode;self.budget=budget;self.max_age=max_age;self.used=0;self.last=None;self.requests=[]

    def choose(self,tick,now,benefit):
        assert np.isfinite(benefit) and 0<=benefit<=1
        if self.used>=self.budget:return False,'budget_exhausted'
        due=self.last is None or now-self.last>=self.max_age-.1
        if self.mode=='fixed':choose=tick%4==0;reason='fixed'
        elif due:choose=True;reason='freshness_due'
        elif self.mode=='oldest':choose=False;reason='wait'
        else:
            # At most one new credit per4ticks; up to2unused credits may accumulate.
            credits=min(self.budget,1+tick//4)-self.used
            choose=benefit>.02 and credits>0;reason='estimated_risk' if choose else 'wait'
        if choose:self.used+=1;self.requests.append(tick)
        return choose,reason

    def observe_valid(self,now):self.last=now
