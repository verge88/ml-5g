import numpy as np
from joint5g.evaluate import combine,vector,metrics

def observation():
    monitor=dict(states=dict(ausf='AUTHENTICATED_CURRENT_CU',nssf='PENDING_CURRENT_CU'),witness_available=False,confirmed=False,hard_alarm=False,decision=dict(verdict='UNKNOWN'),observation_id='not-a-feature')
    return {s:dict(direct=dict(ok=True,elapsed_ms=2.),nrf=dict(ok=True,status=200,elapsed_ms=3.,body=dict(nfInstances=[dict(nfStatus='REGISTERED',nfInstanceId='not-a-feature')])),integrity_monitor=monitor) for s in ['ausf','nssf']}

def test_fusion_never_suppresses_integrity_alarm():
    hard=np.array([True,False,True,False]);ml=np.array([False,True,True,False]);np.testing.assert_array_equal(combine(hard,ml),[True,True,True,False])

def test_features_exclude_ids_truth_and_scheduling():
    a=observation();raw,context=vector(a);a['ausf']['integrity_monitor']['observation_id']='test-fault-704'
    a['ausf']['nrf']['body']['nfInstances'][0]['nfInstanceId']='other-instance';a['ausf']['label']='attack';a['ausf']['stage']='train_fault'
    assert vector(a)==(raw,context) and len(raw)==12 and len(context)==28

def test_unknown_evidence_remains_explicit():
    a=observation();_,v=vector(a);assert v[-4:]==[0.,0.,0.,1.]

def test_detection_requires_hit_within_active_episode():
    r=dict(normal=np.array([True,False,False,False]),truth=np.array([False,True,True,False]),recovery=np.array([False,False,False,True]),unknown=np.zeros(4,dtype=bool),
        rows=[dict(tick=i,ns=i*1000000000) for i in range(4)],events=[dict(split='test',episode='e',kind='stop',slot='ausf',start_tick=1,end_tick=3,start_ns=1000000000)])
    m=metrics(r,np.array([False,False,False,True]));assert m['detected_episodes']==0 and m['tp']==0
    m=metrics(r,np.array([False,False,True,False]));assert m['detected_episodes']==1 and m['episodes'][0]['delay_seconds']==1.
