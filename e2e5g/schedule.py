"""Causal bounded-age allocation; no event, stage, label or future input."""
import math
SLOTS=('ausf','nssf')

def raw_features(obs):
    direct,nrf=obs['direct'],obs['nrf']; body=nrf['body']
    rows=body.get('nfInstances',[]) if isinstance(body,dict) else []
    registered=sum(r.get('nfStatus')=='REGISTERED' for r in rows)/max(1,len(rows))
    return [float(not direct['ok']),math.log1p(direct['elapsed_ms']),float(not nrf['ok'] or nrf['status']!=200),math.log1p(nrf['elapsed_ms']),abs(len(rows)-1),1-registered]

def predict(model,x):
    z=[(a-b)/c for a,b,c in zip(x,model['mean'],model['scale'])]
    value=model['intercept']+sum(a*b for a,b in zip(z,model['coef']))
    return 1/(1+math.exp(-max(-40,min(40,value))))

class Scheduler:
    def __init__(self,mode,models,max_age=4):
        if mode not in ('fixed','risk','ml_risk'): raise ValueError('mode')
        self.mode,self.models,self.max_age=mode,models,max_age
        self.last={s:-1 for s in SLOTS}; self.ewma={s:0. for s in SLOTS}
    def choose(self,tick,observations):
        ages={s:tick-self.last[s] for s in SLOTS}; risk={}
        for s in SLOTS:
            x=raw_features(observations[s]); value=max(x[0],x[2],min(1,x[1]/7))
            self.ewma[s]=.6*value+.4*self.ewma[s]
            risk[s]=predict(self.models[s],x) if self.mode=='ml_risk' else self.ewma[s]
        if self.mode=='fixed': chosen=SLOTS[tick%len(SLOTS)]
        else:
            overdue=[s for s in SLOTS if ages[s]>=self.max_age]
            chosen=max(overdue,key=lambda s:(ages[s],-SLOTS.index(s))) if overdue else max(SLOTS,key=lambda s:(risk[s]+.08*ages[s],ages[s],-SLOTS.index(s)))
        self.last[chosen]=tick
        return chosen,dict(ages=ages,risk=risk)
