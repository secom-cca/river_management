"""Focused tests for regret normalization and robustness aggregation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import analyze_optuna_to7_robustness as target


class RegretNormalizationTests(unittest.TestCase):
    def setUp(self):
        ideal = [0.5, 0.5, 0.5, 1.2, 1.2]
        baseline = [1.0, 1.0, 1.0, 1.0, 1.0]
        policy_rows = []
        for scenario in target.SCENARIO_ORDER:
            policy_rows.extend(
                [
                    {"scenario": scenario, **dict(zip(target.OBJECTIVES, ideal))},
                    {"scenario": scenario, **dict(zip(target.OBJECTIVES, baseline))},
                ]
            )
        self.policies = pd.DataFrame(policy_rows)
        result_rows = []
        for solution_id, values in (("ideal", ideal), ("baseline", baseline)):
            for scenario in target.SCENARIO_ORDER:
                result_rows.append(
                    {
                        "source_solution_id": solution_id,
                        "source_scenario": "2C",
                        "target_scenario": scenario,
                        **dict(zip(target.OBJECTIVES, values)),
                        "feasible": True,
                        "total_budget_utilisation": 0.8,
                        "annual_budget_utilisation": 0.9,
                    }
                )
        self.results = pd.DataFrame(result_rows)

    def test_pareto_range_maps_ideal_to_zero_and_nadir_to_one(self):
        regrets = target.add_regrets(self.results, self.policies, "pareto_range")
        ideal = regrets[regrets["source_solution_id"] == "ideal"]
        baseline = regrets[regrets["source_solution_id"] == "baseline"]
        np.testing.assert_allclose(ideal.loc[:, target.REGRET_COLUMNS], 0.0)
        np.testing.assert_allclose(baseline.loc[:, target.REGRET_COLUMNS], 1.0)

    def test_baseline_to_ideal_has_same_anchors(self):
        regrets = target.add_regrets(self.results, self.policies, "baseline_to_ideal")
        scores = target.score_policies(regrets, np.ones(5), np.ones(3))
        indexed = scores.set_index("source_solution_id")
        self.assertAlmostEqual(indexed.loc["ideal", "mean_regret"], 0.0)
        self.assertAlmostEqual(indexed.loc["baseline", "mean_regret"], 1.0)
        self.assertAlmostEqual(indexed.loc["baseline", "worst_regret"], 1.0)


class WeightedScoreTests(unittest.TestCase):
    def test_objective_weights_change_mean_and_worst_scores(self):
        rows = []
        for solution_id, regrets in (
            ("uniform", [0.2] * 5),
            ("tradeoff", [0.1, 0.1, 0.1, 0.1, 1.0]),
        ):
            for scenario in target.SCENARIO_ORDER:
                rows.append(
                    {
                        "source_solution_id": solution_id,
                        "source_scenario": "2C",
                        "target_scenario": scenario,
                        **dict(zip(target.REGRET_COLUMNS, regrets)),
                        "feasible": True,
                        "total_budget_utilisation": 0.8,
                        "annual_budget_utilisation": 0.9,
                    }
                )
        frame = pd.DataFrame(rows)
        equal = target.score_policies(frame, np.ones(5), np.ones(3)).set_index(
            "source_solution_id"
        )
        weighted = target.score_policies(
            frame, [1, 1, 1, 1, 2], np.ones(3)
        ).set_index("source_solution_id")
        self.assertAlmostEqual(equal.loc["uniform", "mean_regret"], 0.2)
        self.assertAlmostEqual(equal.loc["tradeoff", "mean_regret"], 0.28)
        self.assertAlmostEqual(weighted.loc["tradeoff", "mean_regret"], 0.4)
        self.assertGreater(
            weighted.loc["tradeoff", "worst_regret"],
            equal.loc["tradeoff", "worst_regret"],
        )


if __name__ == "__main__":
    unittest.main()
