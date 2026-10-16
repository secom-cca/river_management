"""Experiment design guards for paired infrastructure sensitivity tests."""
import copy
import unittest

from diagnose_optuna_to7_infrastructure import POLICIES, build_tasks, isolated_decision
from run_vensim_with_pysd_to7_optuna import _zero_decision


class InfrastructureDesignTests(unittest.TestCase):
    def setUp(self):
        self.zero = _zero_decision()
        self.decisions = {sid: self.zero.copy() for sid in POLICIES}
        self.decisions['2C_trial_00008685']['migration_per_year'] = 100.

    def test_each_case_has_compatible_no_policy_denominator(self):
        tasks = build_tasks(self.decisions)
        index = {t['case_id']: t for t in tasks}
        self.assertEqual(len(index), len(tasks))
        for task in tasks:
            base = index[task['baseline_case']]
            self.assertEqual(base['scenario'], task['scenario'])
            self.assertEqual(base['policy_id'], 'no_policy')
            self.assertEqual(base['parameters'], task['parameters'])
            self.assertIn(task['comparison_case'], index)

    def test_experiment_building_does_not_change_source_decisions(self):
        before = copy.deepcopy(self.decisions)
        build_tasks(self.decisions)
        self.assertEqual(before, self.decisions)

    def test_single_investment_and_full_paddy_schedule_are_distinct(self):
        one = isolated_decision(self.zero, 'paddy_dam', 0)
        full = isolated_decision(self.zero, 'paddy_dam', 0, end=5)
        self.assertEqual(one['paddy_dam_end_year'], 0)
        self.assertEqual(full['paddy_dam_end_year'], 5)
        self.assertEqual(one['annual_paddy_dam_investment'], 5e9)
        self.assertEqual(one['migration_per_year'], 0)
        self.assertEqual(self.zero['annual_paddy_dam_investment'], 0)


if __name__ == '__main__':
    unittest.main()
