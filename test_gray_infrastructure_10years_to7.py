"""Budget/calendar guards for the dedicated recurring infrastructure policy."""
import unittest
import numpy as np
from compare_optuna_to7_gray_infrastructure_10years import ACTIONS, scheduled_cost


class RecurringInfrastructureTests(unittest.TestCase):
    def test_exact_annual_budget_and_stop_after_ten_years(self):
        for action in ACTIONS:
            costs=[scheduled_cost(action,y) for y in range(2009,2024)]
            expected=[5e9]*10+[0.]*5 if action!='no_policy' else [0.]*15
            self.assertEqual(costs,expected)
            self.assertEqual(sum(costs),0. if action=='no_policy' else 50e9)

    def test_leap_and_regular_years_have_equal_integrated_spending(self):
        for length in [365,366]:
            daily=np.full(length,5e9/length)
            self.assertAlmostEqual(daily.sum()/5e9,1.,places=12)


if __name__=='__main__':unittest.main()
