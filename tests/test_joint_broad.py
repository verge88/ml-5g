import pytest
from joint5g.broad_collect import settings

def test_registered_training_variety_and_separation():
 train=[settings(n) for n in range(801,807)]
 assert {p['period_seconds'] for p in train}=={.3,.45,.6}
 assert {p['fault_ticks'] for p in train}=={8,12,20}
 assert all(set(tuple(x) for x in p['training_faults'])=={('ausf','stop'),('nssf','stop'),('ausf','pause'),('nssf','pause')} for p in train)
 assert all(settings(n)['fault_ticks']==12 for n in [809,810,811,812,813])

def test_data_expansion_counts_from_registered_plan():
 plans=[settings(n) for n in range(801,807)]
 normal=448+sum(p['fit_ticks']+p['calibration_ticks']+len(p['training_faults'])*p['baseline_ticks'] for p in plans)
 positive=48+sum(len(p['training_faults'])*p['fault_ticks'] for p in plans)
 assert (normal,positive)==(5332,368)
 assert sum(settings(n)['fit_ticks']+settings(n)['calibration_ticks']+settings(n)['normal_test_ticks']+8*settings(n)['baseline_ticks'] for n in [809,810])==2356

def test_unregistered_collection_refused():
 with pytest.raises(ValueError):settings(899)
