"""Strict causal features and measured future outcomes; no actuator input."""
import hashlib,json
from pathlib import Path
import numpy as np
PLAN=json.loads(Path(__file__).with_name('plan.json').read_text())
POINT_NAMES=['median_ms','p90_ms','std_ms','success_fraction','signed','requested','session_established','seconds_since_session','direct_rtt_mean','nrf_rtt_mean','cu_age_ausf','cu_age_nssf']
CONTEXT=[3,4,5,6,7,8,9,10,11]

def point(row):
    p=row['probe_summary'];o=row['observations']
    return np.array([p[k] for k in POINT_NAMES[:7]]+[min(300,p['seconds_since_session']),np.mean([o[s]['direct']['elapsed_ms'] for s in ['ausf','nssf']]),np.mean([o[s]['nrf']['elapsed_ms'] for s in ['ausf','nssf']]),min(10,row['cu']['ages']['ausf']),min(10,row['cu']['ages']['nssf'])],dtype=float)

def causal_features(rows):
    x=np.array([point(r) for r in rows]);windows=[];features=[]
    for i in range(len(rows)):
        past=x[max(0,i-PLAN['history_rows']+1):i+1];padded=np.vstack([np.repeat(past[:1],PLAN['history_rows']-len(past),axis=0),past]);windows.append(padded)
        extra=[]
        for n in [4,8,12]:
            start=max(0,i-n+1);v=x[start:i+1,0];tt=np.array([(r['monotonic_ns']-rows[start]['monotonic_ns'])/1e9 for r in rows[start:i+1]])
            slope=np.polyfit(tt,v,1)[0] if len(v)>1 and tt[-1]>0 else 0.
            extra.extend([float(v.mean()),float(v.std()),float(slope),float(x[start:i+1,5].mean())])
        features.append(np.r_[x[i],extra])
    return x,np.array(windows,dtype='float32'),np.array(features)

def measured_outcome(rows):
    bad=[r['probe_summary']['signed']>=PLAN['minimum_signed_probes'] and r['probe_summary']['median_ms']>PLAN['latency_slo_ms'] for r in rows]
    n=PLAN['slo_consecutive_batches']
    for i in range(len(rows)-n+1):
        if all(bad[i:i+n]):return i
    return None

def make_trial(rows,replica,name):
    x,w,f=causal_features(rows);onset=measured_outcome(rows);ts=np.array([r['monotonic_ns']/1e9 for r in rows]);H=PLAN['forecast_horizon_seconds']
    eligible=[]
    for i,r in enumerate(rows):
        p=r['probe_summary'];cu=r['cu']
        ok=(i+1>=PLAN['minimum_history'] and p['signed']>=PLAN['minimum_signed_probes'] and p['median_ms']<=PLAN['latency_slo_ms'] and not cu['hard_alarm'] and cu['witness_available'] and max(cu['ages'].values())<=PLAN['maximum_cu_age_seconds'])
        # Prefix ends at first measured outcome; non-events need full follow-up.
        ok=ok and (i<onset if onset is not None else ts[-1]-ts[i]>=H+PLAN['slo_consecutive_batches'])
        eligible.append(ok)
    y=np.array([onset is not None and 0<ts[onset]-t<=H for t in ts])
    return dict(replica=replica,name=name,rows=rows,point=x,windows=w,features=f,times=ts,eligible=np.array(eligible),y=y,onset=onset)

def load(root,ids):
    trials=[];provenance=[]
    for replica in ids:
        folder=root/f'forecast-native-{replica}';capture=json.loads((folder/'capture.json').read_text());assert not capture['smoke'] and capture['replica']==replica
        assert capture['plan_sha256']==hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest()
        for entry in capture['trials']:
            p=folder/'trials'/entry['name']/'observations.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()];assert len(rows)==PLAN['case_ticks']
            trials.append(make_trial(rows,replica,entry['name']));provenance.append(dict(replica=replica,trial=entry['name'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return trials,provenance

def threshold(trials,scores,alpha=None):
    # Maxima on non-target eligible prefixes, grouped by complete trial.
    maxima=[]
    for t,s in zip(trials,scores):
        mask=t['eligible']&~t['y']
        if mask.any():maxima.append(float(np.max(s[mask])))
    assert maxima
    alpha=PLAN['calibration_false_episode_fraction'] if alpha is None else alpha
    k=max(0,len(maxima)-1-int(alpha*len(maxima)));return float(np.sort(maxima)[k])

def evaluate(trials,scores,cutoff):
    records=[]
    for t,s in zip(trials,scores):
        hits=np.flatnonzero(t['eligible']&(np.asarray(s)>cutoff));first=int(hits[0]) if len(hits) else None;on=t['onset']
        lead=float(t['times'][on]-t['times'][first]) if first is not None and on is not None else None
        early=lead is not None and 0<lead<=PLAN['forecast_horizon_seconds']
        records.append(dict(replica=t['replica'],trial=t['name'],outcome=on is not None,onset=on,first_alert=first,
            early_detected=bool(early),premature_false_alert=bool(lead is not None and lead>PLAN['forecast_horizon_seconds']),
            benign_false_alert=bool(first is not None and on is None),lead_seconds=lead,eligible=int(t['eligible'].sum()),rows=len(s)))
    positive=[r for r in records if r['outcome']];negative=[r for r in records if not r['outcome']]
    return dict(positive_trials=len(positive),early_detected=sum(r['early_detected'] for r in positive),benign_trials=len(negative),
        benign_false_alerts=sum(r['benign_false_alert'] for r in negative),premature_false_alerts=sum(r['premature_false_alert'] for r in positive),
        eligible_rows=sum(r['eligible'] for r in records),rows=sum(r['rows'] for r in records),
        mean_early_lead_including_misses=float(np.mean([r['lead_seconds'] if r['early_detected'] else 0 for r in positive])) if positive else None,
        median_correct_lead_seconds=float(np.median([r['lead_seconds'] for r in positive if r['early_detected']])) if any(r['early_detected'] for r in positive) else None,records=records)
