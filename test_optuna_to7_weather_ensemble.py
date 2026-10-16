"""Checks for paired annual permutations, calendar mapping and ensemble scores."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import weather_ensemble_to7 as weather
from analyze_optuna_to7_weather_ensemble import sensitivity_scores
from evaluate_optuna_to7_cross_scenario import OBJECTIVES, SCENARIO_ORDER


def artificial_weather():
    # Shared year/day markers reveal any loss of month-day or station pairing.
    marker = np.array([d.year - 2000 + d.dayofyear / 1000 for d in weather.DATES])
    columns = {}
    for name in weather.WEATHER_COMPONENTS:
        columns[name] = marker + (10 if "max_temp" in name else -10 if "min_temp" in name else 0)
    return pd.DataFrame(columns, index=weather.DATES)


class WeatherPermutationTests(unittest.TestCase):
    def test_reproducible_unique_full_year_permutations(self):
        first, second = weather.year_permutations(50, 42), weather.year_permutations(50, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 51)
        self.assertEqual(len({tuple(row["source_years"]) for row in first}), 51)
        for row in first:
            self.assertEqual(set(row["source_years"]), set(weather.YEARS))

    def test_original_order_is_exact(self):
        source = artificial_weather()
        result, audit = weather.permute_weather(source, list(weather.YEARS))
        np.testing.assert_array_equal(source, result)
        self.assertEqual(audit["leap_days_inserted"], 0)
        self.assertEqual(audit["leap_days_removed"], 0)

    def test_joint_month_day_mapping_and_leap_day_audit(self):
        source = artificial_weather()
        order = list(weather.YEARS)
        i, j = order.index(2009), order.index(2012)
        order[i], order[j] = order[j], order[i]
        result, audit = weather.permute_weather(source, order)
        np.testing.assert_array_equal(result.loc["2009-08-01"], source.loc["2012-08-01"])
        np.testing.assert_array_equal(result.loc["2012-08-01"], source.loc["2009-08-01"])
        np.testing.assert_array_equal(result.loc["2012-02-29"],
                                      (source.loc["2009-02-28"] + source.loc["2009-03-01"]) / 2)
        self.assertEqual(audit["leap_days_inserted"], 1)
        self.assertEqual(audit["leap_days_removed"], 1)
        self.assertEqual(len(result), len(source))

    def test_invalid_repeated_year_is_rejected(self):
        with self.assertRaises(ValueError):
            weather.permute_weather(artificial_weather(), [2009] * 15)

    def test_external_arrays_restore_on_failure(self):
        externals = {name: SimpleNamespace(data=object()) for name in weather.WEATHER_COMPONENTS}
        original = {name: value.data for name, value in externals.items()}
        components = SimpleNamespace(**{f"_ext_data_{name}": value for name, value in externals.items()})
        model = SimpleNamespace(components=components, cache=SimpleNamespace(clean=lambda: None))
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with weather.weather_override(model, artificial_weather()):
                self.assertEqual(len(externals["daily_precip_up"].data), len(weather.DATES) + 1)
                raise RuntimeError("injected failure")
        self.assertTrue(all(value.data is original[name] for name, value in externals.items()))


class EnsembleScoringTests(unittest.TestCase):
    def test_weather_and_climate_weights_are_aggregated_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Source Pareto ideal=.8/-1.2, nadir=1/-1 for each objective.
            for scenario in SCENARIO_ORDER:
                pd.DataFrame([[.8, .8, .8, 1.2, 1.2], [1., 1., 1., 1., 1.]],
                             columns=OBJECTIVES).to_csv(root / f"pareto_summary_{scenario}.csv", index=False)
            rows = []
            for sequence, regret in [("perm_0001", .25), ("perm_0002", .75)]:
                for scenario in SCENARIO_ORDER:
                    ratios = [.8 + .2 * regret] * 3 + [1.2 - .2 * regret] * 2
                    rows.append({"source_solution_id": "p", "sequence_id": sequence,
                                 "target_scenario": scenario, "feasible": True,
                                 **dict(zip(OBJECTIVES, ratios))})
            policies = pd.DataFrame({"solution_id": ["p"], "scenario": ["present"]})
            scores, _ = sensitivity_scores(pd.DataFrame(rows), policies, ["perm_0001", "perm_0002"], root)
            base = scores.loc[scores["variant_id"].eq("pareto_range__equal__equal")].iloc[0]
            self.assertAlmostEqual(base["mean_regret"], .5)
            self.assertAlmostEqual(base["worst_regret"], .75)
            self.assertAlmostEqual(base["regret_q90"], .7)
            self.assertEqual(len(scores), 72)
            for _key, group in scores.groupby(["normalization", "objective_weight_scheme"]):
                self.assertAlmostEqual(group["worst_regret"].max(), group["worst_regret"].min())
            # Distinct climates expose axis/order mistakes that identical climates cannot.
            varied = pd.DataFrame(rows)
            for j, scenario in enumerate(SCENARIO_ORDER):
                mask = varied["target_scenario"].eq(scenario)
                varied.loc[mask, list(OBJECTIVES[:3])] += .02 * j
                varied.loc[mask, list(OBJECTIVES[3:])] -= .02 * j
            changed, _ = sensitivity_scores(varied, policies, ["perm_0001", "perm_0002"], root)
            equal = changed.loc[changed["variant_id"].eq("pareto_range__equal__equal")].iloc[0]
            heavy = changed.loc[changed["variant_id"].eq("pareto_range__equal__4C_heavy")].iloc[0]
            self.assertAlmostEqual(equal["mean_regret"], .6)
            self.assertAlmostEqual(heavy["mean_regret"], .645)
            self.assertAlmostEqual(equal["worst_regret"], .95)
            self.assertAlmostEqual(heavy["worst_regret"], .95)


if __name__ == "__main__":
    unittest.main()
