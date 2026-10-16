"""Regression checks for calendar integration and optional breeding response."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import breeding_response_to7 as response
import run_vensim_with_pysd_to7_opt as core
import run_vensim_with_pysd_to7_optuna as optuna_runner


class ResponseTests(unittest.TestCase):
    def test_progress_is_zero_anchored_bounded_and_monotone(self):
        values = [response.logistic_effect_ratio(x) for x in [0, 1e6, 100e6, 300e6, 500e6, 1e12]]
        self.assertEqual(values[0], 0)
        self.assertTrue(all(0 <= x <= 1 for x in values))
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 1)
        self.assertGreater(values[-2], 0.94)
        self.assertLess(values[-2], 0.95)

    def test_invalid_assumptions_are_rejected(self):
        for kwargs in ({"scale": 0}, {"midpoint": -1}, {"max_reduction": 1.1}, {"scale": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                response.model_parameters(**kwargs)

    def test_default_uses_adopted_threshold_model(self):
        self.assertEqual(response.DEFAULT_MODE, "threshold")
        self.assertEqual(response.model_parameters()["breeding_effect_mode"], 0)


class ResumeTests(unittest.TestCase):
    def test_resume_accepts_same_model_and_rejects_old_trials(self):
        args = SimpleNamespace(scenarios=["present"], population_size=10, dividing_parameter=3, seed=42)
        config = optuna_runner._new_config(args, Path("dummy"), None)
        optuna_runner._validate_resume_config(config, args)
        old = copy.deepcopy(config)
        old.pop("model_configuration")
        with self.assertRaisesRegex(ValueError, "model_configuration"):
            optuna_runner._validate_resume_config(old, args)
        changed = copy.deepcopy(config)
        changed["model_configuration"]["parameters"]["breeding_effect_mode"] = 1
        with self.assertRaisesRegex(ValueError, "model_configuration"):
            optuna_runner._validate_resume_config(changed, args)


class CalendarIntegrationTests(unittest.TestCase):
    def test_yearly_baseline_matches_one_continuous_calendar_run(self):
        decision = optuna_runner._zero_decision()
        yearly_runs = core.run_model_yearly("present", decision)
        actual = pd.concat(yearly_runs)
        total_days = sum(core.YEAR_LENGTHS)
        full = core.get_model().run(
            params=core.fixed_params_for_scenario("present"),
            initial_condition="original", final_time=total_days,
            return_timestamps=list(range(total_days)), return_columns=core.RETURN_COLS,
        )
        self.assertEqual(actual.index.tolist(), list(range(total_days)))
        np.testing.assert_allclose(actual.to_numpy(), full.to_numpy(), rtol=1e-11, atol=1e-6)

    def test_response_override_does_not_change_default_or_funding(self):
        decision = optuna_runner._zero_decision()
        decision.update(breeding_enabled=1, breeding_start_year=0, breeding_end_year=9)
        _, logistic = core.evaluate_policy("present", decision, model_params=response.model_parameters("logistic"))
        _, threshold = core.evaluate_policy("present", decision)
        np.testing.assert_allclose(logistic.accumulated_breeding_investment, threshold.accumulated_breeding_investment, rtol=0, atol=0.01)
        self.assertEqual(threshold.breeding_effect_ratio.iloc[8], 0)
        self.assertEqual(threshold.breeding_effect_ratio.iloc[9], 1)
        self.assertGreater(logistic.breeding_effect_ratio.iloc[8], 0)
        self.assertLess(logistic.breeding_effect_ratio.iloc[9], 1)
        self.assertEqual(threshold.breeding_effect_ratio_at_harvest.iloc[9], 0)
        self.assertEqual(threshold.breeding_effect_ratio_at_harvest.iloc[10], 1)
        np.testing.assert_allclose(logistic.breeding_effect_ratio.iloc[9:], logistic.breeding_effect_ratio.iloc[9], rtol=0, atol=1e-12)
        self.assertLessEqual((threshold.accumulated_breeding_investment - np.minimum(np.arange(1, 16), 10) * 50e6).abs().max(), 0.01)


if __name__ == "__main__":
    unittest.main()
