"""Focused tests for mixed policies, constraints, Pareto filtering and storage."""

from __future__ import annotations

import csv
import importlib.util
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_vensim_with_pysd_to7_optuna as target


def _example_params(breeding: bool = True):
    params = {
        "drainage_investment_amount": 1.0,
        "dam_investment_amount": 2.0,
        "levee_investment_amount": 3.0,
        "house_elevation_per_year": 4.0,
        "migration_per_year": 5.0,
        "annual_paddy_dam_investment": 6.0,
        "annual_forest_management_conversion_area": 7.0,
        "drainage_start_year": 1,
        "dam_start_year": 2,
        "levee_start_year": 3,
        "house_elevation_start_year": 4,
        "house_elevation_duration_years": 3,
        "migration_start_year": 5,
        "migration_duration_years": 2,
        "paddy_dam_start_year": 6,
        "paddy_dam_duration_years": 4,
        "forest_management_start_year": 7,
        "forest_management_duration_years": 5,
        "breeding_enabled": breeding,
    }
    if breeding:
        params["breeding_start_year"] = 5
    return params


def _dummy_storage_worker(journal_path: str, worker_index: int):
    optuna = target._import_optuna()
    study = target._study(
        optuna,
        "dummy_multi_process",
        journal_path,
        population_size=4,
        dividing_parameter=2,
        seed=100 + worker_index,
    )

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        target.set_trial_constraints(trial, -1.0, -1.0)
        trial.set_user_attr("annual_budget_constraint", -1.0)
        trial.set_user_attr("total_budget_constraint", -1.0)
        return x, x, x, 1.0 - x, 1.0 - x

    study.optimize(objective, n_trials=2)


class PolicyRepresentationTests(unittest.TestCase):
    def test_round_trip_has_unique_intervals(self):
        params = _example_params()
        decision = target.policy_params_to_decision(params)
        self.assertEqual(decision["house_elevation_end_year"], 6)
        self.assertEqual(decision["forest_management_end_year"], 11)
        self.assertEqual(decision["breeding_end_year"], 14)
        self.assertEqual(target.decision_to_policy_params(decision), params)

    def test_disabled_breeding_rejects_meaningless_start(self):
        params = _example_params(breeding=False)
        decision = target.policy_params_to_decision(params)
        self.assertEqual(decision["breeding_enabled"], 0)
        params["breeding_start_year"] = 3
        with self.assertRaisesRegex(ValueError, "must be absent"):
            target.policy_params_to_decision(params)

    def test_interval_cannot_extend_past_2023(self):
        params = _example_params()
        params["migration_start_year"] = 14
        params["migration_duration_years"] = 2
        with self.assertRaisesRegex(ValueError, "finish"):
            target.policy_params_to_decision(params)


class ObjectiveAndConstraintTests(unittest.TestCase):
    def test_objective_ratios_keep_directional_values(self):
        baseline = {
            "flood_damage": 10.0,
            "innundation_damage": 20.0,
            "landslide_risk": 5.0,
            "crop_revenue": 100.0,
            "biodiversity": 0.5,
        }
        metrics = {
            "flood_damage": 8.0,
            "innundation_damage": 10.0,
            "landslide_risk": 4.0,
            "crop_revenue": 110.0,
            "biodiversity": 0.6,
        }
        self.assertEqual(
            target.objective_ratios(metrics, baseline),
            (0.8, 0.5, 0.8, 1.1, 1.2),
        )

    def test_budget_constraint_sign(self):
        feasible = {
            "max_annual_policy_additional_cost": target.ANNUAL_BUDGET,
            "policy_additional_cost": target.TOTAL_BUDGET,
        }
        self.assertEqual(target.budget_constraints(feasible), (0.0, 0.0))
        infeasible = dict(feasible)
        infeasible["policy_additional_cost"] *= 1.1
        self.assertGreater(target.budget_constraints(infeasible)[1], 0.0)

    def test_pareto_export_filter_removes_infeasible_trial(self):
        complete = SimpleNamespace(name="COMPLETE")
        feasible = SimpleNamespace(
            number=2,
            state=complete,
            user_attrs={
                "annual_budget_constraint": 0.0,
                "total_budget_constraint": -0.1,
            },
        )
        infeasible = SimpleNamespace(
            number=1,
            state=complete,
            user_attrs={
                "annual_budget_constraint": 0.1,
                "total_budget_constraint": -0.1,
            },
        )
        study = SimpleNamespace(best_trials=[infeasible, feasible])
        self.assertEqual(target._pareto_trials(study), [feasible])


class ResumeAndInputTests(unittest.TestCase):
    def test_trial_quota_is_exact(self):
        quotas = target.split_trial_quota(10, 3)
        self.assertEqual(quotas, [4, 3, 3])
        self.assertEqual(sum(quotas), 10)

    def test_de_files_require_all_twelve_cases_and_years(self):
        patterns = (
            "balanced",
            "disaster_only",
            "agriculture_only",
            "ecosystem_only",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "run_summary.csv"
            decisions = root / "run_decisions.csv"
            yearly = root / "run_yearly.csv"
            keyed = [
                {"scenario": scenario, "pattern": pattern, "seed": "42"}
                for scenario in target.SCENARIO_ORDER
                for pattern in patterns
            ]
            for path, rows, fields in (
                (summary, keyed, ["scenario", "pattern", "seed"]),
                (decisions, keyed, ["scenario", "pattern", "seed"]),
                (
                    yearly,
                    [
                        {**row, "year": str(target.CALENDAR_START_YEAR + year)}
                        for row in keyed
                        for year in range(target.CALENDAR_NUM_YEARS)
                    ],
                    ["scenario", "pattern", "seed", "year"],
                ),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            target._validate_de_files(summary, decisions, yearly)
            with yearly.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            rows.pop()
            with yearly.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["scenario", "pattern", "seed", "year"]
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "12 cases x 15 years"):
                target._validate_de_files(summary, decisions, yearly)


@unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is not installed")
class JournalMultiprocessTests(unittest.TestCase):
    def test_optuna_constraint_api_excludes_infeasible_dominator(self):
        optuna = target._import_optuna()
        study = optuna.create_study(
            directions=list(target.DIRECTIONS),
            sampler=target._make_sampler(
                optuna, population_size=2, dividing_parameter=1, seed=42
            ),
        )

        def objective(trial):
            violation = 1.0 if trial.number == 0 else -1.0
            target.set_trial_constraints(trial, violation, violation)
            trial.set_user_attr("annual_budget_constraint", violation)
            trial.set_user_attr("total_budget_constraint", violation)
            if violation > 0:
                return 0.0, 0.0, 0.0, 2.0, 2.0
            return 1.0, 1.0, 1.0, 1.0, 1.0

        study.optimize(objective, n_trials=2)
        self.assertEqual([trial.number for trial in study.best_trials], [1])

    def test_two_processes_share_one_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = str(Path(temporary) / "journal.log")
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(target=_dummy_storage_worker, args=(journal, index))
                for index in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            optuna = target._import_optuna()
            study = target._study(
                optuna,
                "dummy_multi_process",
                journal,
                population_size=4,
                dividing_parameter=2,
                seed=100,
            )
            self.assertEqual(len(target._completed_trials(study)), 4)


if __name__ == "__main__":
    unittest.main()
