"""Numerical checks against known areas and trial-filtering semantics."""

import unittest

import numpy as np

import plot_optuna_to7_hypervolume as target


class HypervolumeTests(unittest.TestCase):
    def test_known_rectangle_union_and_replacement(self):
        # Favourable coordinates A=(.8,.3), B=(.5,.6), C=(.3,.8).
        original = 1 - np.array([[.8, .3], [.5, .6], [.3, .8]])
        self.assertAlmostEqual(target.exact_hypervolume(original, np.ones(2)), .45)
        updated = target.add_to_front(original, np.array([.3, .3]))
        self.assertEqual(len(updated), 3)
        self.assertAlmostEqual(target.exact_hypervolume(updated, np.ones(2)), .55)

    def test_infeasible_trial_counts_on_axis_but_not_in_volume(self):
        losses = np.array([[.2, .7], [0., 0.], [.5, .4], [.7, .2], [.3, .3]])
        eligible = np.array([True, False, True, True, True])
        history = target.history_from_losses(losses, eligible, np.ones(5, dtype=bool), np.ones(2), 2)
        self.assertEqual(history["trial_count"].tolist(), [0, 2, 4, 5])
        np.testing.assert_allclose(history["hypervolume"], [0., .24, .45, .55])
        self.assertEqual(history["complete_count"].tolist(), [0, 2, 4, 5])
        self.assertEqual(history["feasible_count"].tolist(), [0, 1, 3, 4])

    def test_failed_trial_and_duplicate_do_not_improve_front(self):
        losses = np.array([[.2, .7], [np.nan, np.nan], [.2, .7]])
        mask = np.array([True, False, True])
        history = target.history_from_losses(losses, mask, mask, np.ones(2), 1)
        np.testing.assert_allclose(history["hypervolume"], [0., .24, .24, .24])
        self.assertEqual(history.iloc[-1]["complete_count"], 2)
        self.assertEqual(history.iloc[-1]["nondominated_vectors"], 1)

    def test_reference_excludes_noncontributing_vector(self):
        front = np.array([[.2, .7], [2., 0.]])
        self.assertAlmostEqual(target.exact_hypervolume(front, np.ones(2)), .24)
        self.assertEqual(target.exact_hypervolume(np.empty((0, 2)), np.ones(2)), 0.)

    def test_fixed_scaling_respects_both_objective_directions(self):
        values = np.array([[.8, .7, .9, 1.02, 1.04], [.9, .8, .95, 1.01, 1.03]])
        scaling = target.make_scaling(values, 1.1)
        target.validate_scaling(scaling)
        ideal = np.array(scaling["ideal_losses"])
        scales = np.array(scaling["scales"])
        np.testing.assert_allclose((values[0] * target.MINIMIZATION_SIGNS - ideal) / scales, 0.)
        np.testing.assert_allclose((target.MINIMIZATION_SIGNS - ideal) / scales, 1.)
        # A later better solution must lie beyond the original zero without rescaling.
        improved = np.array([.7, .6, .8, 1.03, 1.05])
        self.assertTrue(np.all((improved * target.MINIMIZATION_SIGNS - ideal) / scales < 0))

    def test_no_improvement_has_explicit_finite_scale(self):
        scaling = target.make_scaling(np.ones((2, 5)), 1.1)
        self.assertEqual(scaling["fallback_scale_objectives"], target.OBJECTIVES)
        np.testing.assert_array_equal(scaling["scales"], np.ones(5))
        target.validate_scaling(scaling)


if __name__ == "__main__":
    unittest.main()
