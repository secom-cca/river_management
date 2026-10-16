"""Guards against changed scaling, discarded ties and unpaired resampling."""
import copy
import unittest

import numpy as np
import pandas as pd

from analyze_optuna_to7_weather_selection_stability import fixed_regrets, selection_masks, resample_selection
from evaluate_optuna_to7_cross_scenario import OBJECTIVES, SCENARIO_ORDER


class SelectionStabilityTests(unittest.TestCase):
    def test_fixed_reference_and_regret_above_one_survive_subsetting(self):
        settings = {c:{'ideal_losses':[0,0,0,-2,-2],'scales':[.5,.5,.5,1,1]} for c in SCENARIO_ORDER}
        before = copy.deepcopy(settings)
        frame = pd.DataFrame([{'target_scenario':'present',**dict(zip(OBJECTIVES,[.8,.2,.4,1,1]))},
                              {'target_scenario':'present',**dict(zip(OBJECTIVES,[.1,.1,.1,3,3]))}])
        whole = fixed_regrets(frame,settings)
        subset = fixed_regrets(frame.iloc[:1],settings)
        np.testing.assert_array_equal(whole[:1],subset)
        self.assertEqual(whole[0,0],1.6)
        self.assertEqual(whole[1,3],0)
        self.assertEqual(settings,before)

    def test_ties_at_first_and_third_are_included(self):
        best,top,ranks = selection_masks(np.array([1.,1.,2.,2.,3.]),np.ones(5,dtype=bool))
        np.testing.assert_array_equal(best,[True,True,False,False,False])
        np.testing.assert_array_equal(top,[True,True,True,True,False])
        np.testing.assert_array_equal(ranks,[1.,1.,3.,3.,5.])

    def test_common_draws_reproducible_and_policy_order_invariant(self):
        scores = np.array([[0.,2.,0.,2.],[2.,0.,2.,0.],[3.,3.,3.,3.]])
        before=scores.copy()
        f,draws = resample_selection(scores,np.ones(3,dtype=bool),[2,4],20,42)
        g,other = resample_selection(scores[[1,0,2]],np.ones(3,dtype=bool),[2,4],20,42)
        self.assertEqual(draws,other)
        for p,q in [(0,1),(1,0),(2,2)]:
            np.testing.assert_array_equal(f.loc[f.policy_index.eq(p),'best_frequency'],g.loc[g.policy_index.eq(q),'best_frequency'])
        full=f.loc[f.sequence_count.eq(4)]
        np.testing.assert_array_equal(full.best_frequency,[1.,1.,0.])
        self.assertTrue(full.draw_count.eq(1).all())
        self.assertTrue(all(len(d['sequence_indices'])==len(set(d['sequence_indices'])) for d in draws))
        np.testing.assert_array_equal(before,scores)


if __name__ == '__main__':
    unittest.main()
