"""Queue simulator retaining only causal public observations at decision time."""
import heapq
import json
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

PLAN = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
PASSIVE = ['log_user_mean','log_user_median','log_user_q75','log_user_std',
           'past_arrival_rate','past_completion_rate','user_completion_age',
           'own_past_query_rate','probe_work','shared_queue']
FEATURES = PASSIVE + ['log_probe_rtt','probe_present','request_age','probe_minus_user']


def cells():
    return [dict(load=a,work=b,mode=c,policy=d) for a,b,c,d in
            product(PLAN['loads'],PLAN['probe_work'],PLAN['modes'],PLAN['policies'])]


def workload(seed,load,burst=False):
    rng=np.random.default_rng(seed)
    arrivals=[]
    start,end=PLAN['impairment_interval']
    for t in range(PLAN['tick_count']):
        actual=.97 if burst and start<=t<end else load
        n=rng.poisson(actual)
        tt=np.sort(rng.uniform(t,t+1,n))
        arrivals.append(list(zip(tt.tolist(),rng.exponential(1,n).tolist())))
    return arrivals,rng.normal(0,.08,PLAN['tick_count']),rng.random(PLAN['tick_count'])


def make_features(users,arrivals,completions,query_history,last_user,last_probe,t,work,mode):
    vals=list(users) or [1.]
    mean=float(np.mean(vals))
    fresh=last_probe is not None and t-last_probe[0]<=PLAN['maximum_request_age']
    x=[np.log1p(mean),np.log1p(np.median(vals)),np.log1p(np.quantile(vals,.75)),
       np.log1p(np.std(vals)),sum(arrivals)/len(arrivals),sum(completions)/len(completions),
       min(30.,t-last_user)/30.,sum(query_history)/len(query_history),work,float(mode=='shared')]
    if fresh:
        requested,completed,value=last_probe
        x += [np.log1p(value),1.,(t-requested)/PLAN['maximum_request_age'],
              np.log1p(value)-np.log1p(mean)]
    else:x += [0.,0.,1.,0.]
    return np.array(x,dtype=np.float64),fresh


def driver_probability(x,driver):
    if driver is None:return .5
    z=float(np.dot((x[:len(PASSIVE)]-driver['mean'])/driver['scale'],driver['coef'])+driver['intercept'])
    return 1./(1.+np.exp(-np.clip(z,-40,40)))


def acquire(policy,t,prob,fresh,arrival_rate,work,credits,used,coin):
    if policy=='passive' or used>=PLAN['budget'] or credits<1:return False
    if policy.startswith('guarded_') and arrival_rate+PLAN['token_rate']*work>=.95:return False
    p=policy.removeprefix('guarded_')
    if p=='periodic':return t%5==0
    if p=='random':return coin<.2
    return not fresh or prob>.35


def simulate(seed,cell,mechanism,driver=None,burst=False,ledger=False):
    load,work,mode,policy=(cell[k] for k in ('load','work','mode','policy'))
    arrivals,noise,coins=workload(seed,load,burst)
    queue=[];server=probe_server=0.
    users=deque(maxlen=12);counts=deque([0],maxlen=20);completed_counts=deque([0],maxlen=20)
    query_history=deque([0],maxlen=20)
    last_user=0.;last_probe=None;credits=1.;used=0
    user_lat=[];xx=[];yy=[];fresh_count=completion_fresh=late=returned=0
    events=[]
    start,end=PLAN['impairment_interval']
    for t in range(PLAN['tick_count']):
        completions=0
        while queue and queue[0][0]<=t:
            done,kind,requested,value=heapq.heappop(queue)
            if kind==0:
                users.append(value);last_user=done;completions+=1
            else:
                returned+=1
                late+=int(done-requested>PLAN['maximum_request_age'])
                # An older, slower response cannot overwrite a newer request's evidence.
                if last_probe is None or requested>last_probe[0]:last_probe=(requested,done,value)
                if ledger:events.append(dict(kind='returned',requested=requested,completed=done,observed_at=t))
        counts.append(len(arrivals[t-1]) if t else 0)
        completed_counts.append(completions)
        x,fresh=make_features(users,counts,completed_counts,query_history,last_user,last_probe,t,work,mode)
        probability=driver_probability(x,driver)
        credits=min(2.,credits+PLAN['token_rate'])
        query=acquire(policy,t,probability,fresh,x[4],work,credits,used,coins[t])
        label=mechanism if start<=t<end else 0
        path=PLAN['impairment_wait'] if label in (1,3) else 0.
        app=PLAN['impairment_wait'] if label in (2,3) else 0.
        if query:
            credits-=1;used+=1
            if mode=='shared':
                server=max(server,float(t))+work;done=server+app+.1+abs(noise[t])
            else:
                probe_server=max(probe_server,float(t))+work;done=probe_server+app+.1+abs(noise[t])
            heapq.heappush(queue,(done,1,t,done-t))
            if ledger:events.append(dict(kind='requested',requested=t,will_complete=done))
        query_history.append(int(query))
        for arrival,service in arrivals[t]:
            server=max(server,arrival)+service
            done=server+path+app
            heapq.heappush(queue,(done,0,arrival,done-arrival))
            if t>=PLAN['burn_in']:user_lat.append(done-arrival)
        if t>=PLAN['burn_in']:
            xx.append(x);yy.append(label);fresh_count+=int(fresh)
            completion_fresh+=int(last_probe is not None and t-last_probe[1]<=PLAN['maximum_request_age'])
    metric=dict(seed=seed,**cell,mechanism=mechanism,burst=burst,queries=used,
                probe_service_work=used*work,user_p95=float(np.quantile(user_lat,.95)),
                user_mean=float(np.mean(user_lat)),request_fresh_fraction=fresh_count/len(yy),
                completion_fresh_fraction=completion_fresh/len(yy),late_receipts=late,
                unresolved_probes=used-returned,user_requests=len(user_lat))
    return metric,np.array(xx),np.array(yy),events
