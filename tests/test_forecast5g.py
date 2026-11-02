import copy
import numpy as np
from forecast5g.data import PLAN,causal_features,make_trial,evaluate,threshold
from forecast5g.profiles import configuration,command

def rows(n=40):
    return [dict(monotonic_ns=i*1_000_000_000,trial='arbitrary',tick=i,
        probe_summary=dict(median_ms=40.,p90_ms=45.,std_ms=2.,success_fraction=1.,signed=3,requested=3,session_established=True,seconds_since_session=i),
        observations={s:dict(direct=dict(elapsed_ms=1.),nrf=dict(elapsed_ms=1.)) for s in ['ausf','nssf']},
        cu=dict(ages=dict(ausf=1.,nssf=1.),hard_alarm=False,witness_available=True)) for i in range(n)]

def test_no_future_or_actuator_inputs():
    a=rows();b=copy.deepcopy(a)
    for r in b[25:]:r['probe_summary']['median_ms']=10000
    for r in b:r.update(trial='other-class',tick=9999,injected_delay_ms=180,future_outcome=True)
    for aa,bb in zip(causal_features(a),causal_features(b)):assert np.array_equal(aa[:25],bb[:25])
    assert causal_features(a)[0].shape==(40,12) and causal_features(a)[2].shape==(40,24)

def test_future_onset_and_healthy_current_gate():
    a=rows()
    for r in a[25:28]:r['probe_summary']['median_ms']=250
    t=make_trial(a,1201,'unobserved-case');assert t['onset']==25
    assert np.array_equal(np.flatnonzero(t['y']),np.arange(17,25))
    assert not t['eligible'][25:].any()
    a[20]['cu']['hard_alarm']=True;a[21]['cu']['ages']['ausf']=100
    a[22]['probe_summary']['signed']=0
    t=make_trial(a,1201,'x');assert not t['eligible'][20:23].any()

def test_one_early_alert_prevents_later_credit():
    a=rows()
    for r in a[25:28]:r['probe_summary']['median_ms']=250
    t=make_trial(a,1201,'x');s=np.zeros(40);s[10]=1;s[20]=1
    r=evaluate([t],[s],.5);assert r['premature_false_alerts']==1 and r['early_detected']==0 and r['mean_early_lead_including_misses']==0
    s[10]=0;r=evaluate([t],[s],.5);assert r['early_detected']==1 and r['mean_early_lead_including_misses']==5

def test_calibration_false_trial_budget_and_strict_threshold():
    trials=[make_trial(rows(),1207,str(i)) for i in range(40)];scores=[np.full(40,i,dtype=float) for i in range(40)]
    cut=threshold(trials,scores);assert sum(np.any(t['eligible']&(s>cut)) for t,s in zip(trials,scores))<=2

def test_normal_reconnection_and_transient_not_sustained_outcome():
    a=rows();a[12]['probe_summary']['signed']=0;a[16]['probe_summary']['median_ms']=250;a[17]['probe_summary']['median_ms']=250
    t=make_trial(a,1210,'normal');assert t['onset'] is None and not t['eligible'][12] and not t['eligible'][16:18].any()
    assert not t['eligible'][-11:].any()

def test_disjoint_deployments_and_bounded_internal_workload():
    assert not set(PLAN['train'])&set(PLAN['calibration']) and not set(PLAN['train']+PLAN['calibration'])&set(PLAN['test'])
    assert not set(PLAN['smoke'])&set(PLAN['train']+PLAN['calibration']+PLAN['test'])
    for replica in PLAN['train']+PLAN['calibration']+PLAN['test']:
        for name in PLAN['cases']:
            c=configuration(replica,name)
            for i in range(c['ticks']):
                q=command(c,i);assert 0<=q['delay_ms']<=180 and q['count'] in [3,6]
