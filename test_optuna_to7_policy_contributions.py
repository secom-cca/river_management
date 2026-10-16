"""Checks for paired outcome signs, isolation and exact calendar cost accounting."""
import unittest

from analyze_optuna_to7_policy_contributions import (
    COMPONENTS, METRICS, REMOVE_KEYS, costs_by_year, effect_values, remove_action,
)


class ContributionTests(unittest.TestCase):
    def test_positive_benefit_for_damage_and_revenue(self):
        baseline = {name: 100. for name in METRICS}
        original = dict(zip(METRICS, [60., 60., 60., 120., 120.]))
        removed = dict(zip(METRICS, [70., 70., 70., 110., 110.]))
        row = effect_values(original, removed, baseline)
        for name in METRICS:
            self.assertEqual(row[f"contribution_{name}_baseline_pp"], 10.)
            self.assertTrue(row[f"changed_{name}"])

    def test_adverse_and_zero_baseline_are_not_hidden(self):
        original = {name: 100. for name in METRICS}
        removed = original.copy()
        removed[METRICS[0]] = 90.
        baseline = original.copy()
        baseline[METRICS[1]] = 0.
        row = effect_values(original, removed, baseline)
        self.assertEqual(row[f"contribution_{METRICS[0]}_baseline_pp"], -10.)
        self.assertNotEqual(row[f"contribution_{METRICS[1]}_baseline_pp"], row[f"contribution_{METRICS[1]}_baseline_pp"])
        self.assertFalse(row[f"changed_{METRICS[2]}"])

    def test_removal_preserves_all_other_decisions_and_input(self):
        decision = {key: 123. for key in REMOVE_KEYS.values()}
        decision.update({"breeding_enabled": 1, "breeding_start_year": 2,
                         "breeding_end_year": 11, "migration_start_year": 0,
                         "migration_end_year": 9, "baseline_capacity": 5200.})
        for action in COMPONENTS:
            changed = remove_action(decision, action)
            self.assertEqual([k for k in decision if decision[k] != changed[k]], [REMOVE_KEYS[action]])
            self.assertEqual(changed["baseline_capacity"], 5200.)
            self.assertEqual(decision["breeding_enabled"], 1)

    def test_calendar_breeding_cost_uses_budget_not_integrated_flow(self):
        record = {"yearly": [{"year_index": i, "breeding_actual_cost": 49_986_000.} for i in range(15)]}
        decision = {"breeding_enabled": 1, "breeding_start_year": 1, "breeding_end_year": 10}
        values = costs_by_year(record, decision, "breeding")
        self.assertEqual(sum(values), 500_000_000.)
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[10], 50_000_000.)
        self.assertEqual(values[11], 0.)


if __name__ == "__main__":
    unittest.main()
