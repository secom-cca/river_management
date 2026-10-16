"""Scientific direction, strictness and tradeoff guards for five-objective tests."""
import unittest
from compare_optuna_to7_gray_infrastructure_pareto import losses, dominates, METRICS


class DominanceTests(unittest.TestCase):
    def test_losses_minimize_damage_and_maximize_revenue_biodiversity(self):
        baseline=dict(zip(METRICS,[100,200,2,1000,1]))
        good=dict(zip(METRICS,[90,180,1.8,1100,1.1]))
        self.assertTrue(dominates(losses(good,baseline),losses(baseline,baseline)))
        self.assertFalse(dominates(losses(baseline,baseline),losses(good,baseline)))

    def test_equality_is_not_strict_dominance_and_tradeoff_is_neither(self):
        a=(1,1,1,-1,-1);b=(.9,1.1,1,-1,-1)
        self.assertFalse(dominates(a,a))
        self.assertFalse(dominates(a,b));self.assertFalse(dominates(b,a))
        self.assertTrue(dominates((.9,1,1,-1,-1),a))

    def test_serialization_noise_is_equal_and_cost_is_not_an_objective(self):
        baseline={m:1 for m in METRICS}
        a={**baseline,'policy_additional_cost':500};b={**baseline,'policy_additional_cost':0}
        self.assertFalse(dominates(losses(a,baseline),losses(b,baseline)))
        self.assertFalse(dominates((1-1e-12,1,1,-1,-1),(1,1,1,-1,-1)))


if __name__=='__main__':unittest.main()
