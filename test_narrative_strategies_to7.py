"""Budget/calendar invariants of the fixed narrative policies."""
import unittest

import numpy as np

from compare_optuna_to7_narrative_strategies import (
    ANNUAL, TOTAL, STRATEGIES, SECTORS, build_design,
)


class NarrativeStrategyTests(unittest.TestCase):
    def test_all_designs_feasible_equal_disaster_and_exact_envelopes(self):
        for spec in STRATEGIES.values():
            d = build_design(spec["shares"])
            p = d["decision"]
            np.testing.assert_allclose([d["sector_envelope_yen"][s] for s in SECTORS], np.array(spec["shares"]) * TOTAL)
            self.assertLessEqual(max(d["annual_spending_upper_yen"]), ANNUAL)
            self.assertLessEqual(sum(d["annual_spending_upper_yen"]), TOTAL)
            amounts = [p[f"{a}_investment_amount"] for a in ["drainage", "dam", "levee"]]
            amounts += [p[f"{a}_per_year"] * 3e6 * 10 for a in ["house_elevation", "migration"]]
            np.testing.assert_allclose(amounts, amounts[0])
            for sector in SECTORS:
                self.assertLessEqual(d["sector_reserved_upper_yen"][sector], d["sector_envelope_yen"][sector] + .001)

    def test_agreed_calendar_and_fixed_breeding(self):
        for spec in STRATEGIES.values():
            d = build_design(spec["shares"])["decision"]
            self.assertEqual([d[f"{a}_start_year"] for a in ["drainage", "dam", "levee"]], [0, 1, 2])
            for a in ["house_elevation", "migration", "paddy_dam", "forest_management", "breeding"]:
                self.assertEqual((d[f"{a}_start_year"], d[f"{a}_end_year"]), (0, 9))
            self.assertEqual(d["breeding_enabled"], 1)

    def test_forest_maintenance_reserved_after_new_conversion_stops(self):
        d = build_design([.2, .6, .2])
        costs = np.array(d["annual_forest_cost_upper_yen"])
        np.testing.assert_allclose(costs[9:], costs[9])
        self.assertGreater(costs[10:].sum(), 0)
        self.assertEqual(d["decision"]["annual_forest_management_conversion_area"], 3000.)
        self.assertGreater(d["sector_envelope_yen"]["ecosystem"], d["sector_reserved_upper_yen"]["ecosystem"])

    def test_invalid_shares_rejected(self):
        for shares in [[.6, .2, .3], [1, -1, 1], [.5, .5], [np.nan, .5, .5], [1, 0, 0]]:
            with self.assertRaises(ValueError):
                build_design(shares)


if __name__ == "__main__":
    unittest.main()
