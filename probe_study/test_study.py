import unittest
import numpy as np
from probe_study.study import Gaussian, choose, simulate, PLAN


class CausalTests(unittest.TestCase):
    def test_no_probe_changes_no_user_latencies(self):
        rows=[simulate(9001,.92,.6,3,'passive',mode)[0] for mode in PLAN['modes']]
        self.assertEqual(len({r['user_p95'] for r in rows}),1)
        self.assertTrue(all(r['queries']==0 for r in rows))

    def test_isolated_probe_work_does_not_slow_users(self):
        passive=simulate(9001,.92,.6,3,'passive','shared')[0]
        isolated=simulate(9001,.92,.6,3,'periodic','isolated')[0]
        shared=simulate(9001,.92,.6,3,'periodic','shared')[0]
        self.assertEqual(passive['user_p95'],isolated['user_p95'])
        self.assertGreater(shared['user_p95'],passive['user_p95'])

    def test_guard_and_budget_use_only_observed_arguments(self):
        posterior=np.full(4,.25)
        self.assertFalse(choose('guarded_risk',5,posterior,False,.9,.6,2,0,.1))
        self.assertTrue(choose('risk',5,posterior,False,.9,.6,2,0,.1))
        self.assertFalse(choose('risk',5,posterior,False,.1,.6,2,72,.1))
        self.assertFalse(choose('risk',5,posterior,False,.1,.6,.5,0,.1))

    def test_reproducibility_budget_and_missing_optional(self):
        model=Gaussian().fit([[1.,1.],[2.,1.],[1.,2.],[2.,2.]],[0,1,2,3])
        aa=simulate(9002,.7,.6,2,'risk','shared',model)[0]
        bb=simulate(9002,.7,.6,2,'risk','shared',model)[0]
        self.assertEqual(aa,bb)
        self.assertLessEqual(aa['queries'],72)
        self.assertTrue(np.isfinite(model.posterior(1.)).all())
        self.assertAlmostEqual(model.posterior(1.).sum(),1.)


if __name__=='__main__':unittest.main()
