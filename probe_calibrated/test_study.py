import unittest
from unittest.mock import patch

import numpy as np

from .simulator import cells,make_features,simulate,workload
from .study import conformal_threshold,diagnostic_metrics,exceeds,upper_binomial


class CalibrationTests(unittest.TestCase):
    def test_exact_order_statistic_and_unattainable_tail(self):
        self.assertEqual(conformal_threshold(range(19),.05),18.)
        self.assertIsNone(conformal_threshold(range(3),.05))
        np.testing.assert_array_equal(exceeds(np.array([.1,.2]),.2),[False,False])
        self.assertFalse(exceeds(np.ones(3),None).any())

    def test_zero_false_events_has_nonzero_uncertainty(self):
        self.assertAlmostEqual(upper_binomial(0,120),1-.05**(1/120))
        self.assertEqual(upper_binomial(120,120),1.)

    def test_silent_detector_cannot_satisfy_detection(self):
        b=dict(mechanism=1)
        y=np.zeros(280,dtype=int);y[40:160]=1
        r=diagnostic_metrics(b,y,np.zeros(280),np.ones(280),.5,'full','block')
        self.assertFalse(r['detected_within30'])
        self.assertIsNone(r['detection_delay'])


class CausalTests(unittest.TestCase):
    def cell(self,mode='shared',policy='periodic'):
        return dict(load=.92,work=.6,mode=mode,policy=policy)

    def test_late_receipt_does_not_reset_request_age(self):
        x,fresh=make_features([1.],[1],[1],[0],29.,(10.,29.,19.),30,.6,'shared')
        self.assertFalse(fresh)
        self.assertEqual(x[11],0.)

    def test_common_inputs_and_probe_isolation(self):
        a=simulate(88001,self.cell(policy='passive'),2)[0]
        b=simulate(88001,self.cell(mode='isolated'),2)[0]
        c=simulate(88001,self.cell(),2)[0]
        self.assertEqual(a['user_p95'],b['user_p95'])
        self.assertGreater(c['user_p95'],b['user_p95'])
        self.assertLessEqual(c['queries'],72)

    def test_future_workload_cannot_change_prefix(self):
        original=workload(88002,.92)
        changed=[list(v) for v in original[0]]
        for t in range(200,len(changed)):changed[t]=[(t+.5,100.)]
        with patch('probe_calibrated.simulator.workload',return_value=original):
            _,x,_,_=simulate(88002,self.cell(),1)
        with patch('probe_calibrated.simulator.workload',return_value=(changed,original[1],original[2])):
            _,z,_,_=simulate(88002,self.cell(),1)
        np.testing.assert_array_equal(x[:120],z[:120])

    def test_receipts_observed_only_after_completion(self):
        _,_,_,events=simulate(88003,self.cell(),3,ledger=True)
        self.assertTrue(all(v['observed_at']>=v['completed'] for v in events if v['kind']=='returned'))
        self.assertEqual(len(cells()),72)


if __name__=='__main__':unittest.main()
