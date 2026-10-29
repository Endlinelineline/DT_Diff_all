"""Exact formula and layout checks for the Guangning kW evaluation."""

import math
import unittest

import numpy as np

from utils.wind_metrics import (
    LayeredWindMetrics,
    better_validation_score,
    to_sample_turbine_horizon,
)


class WindMetricsTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([
            [[100, 200], [400, 0]],
            [[150, 500], [50, 300]],
        ], dtype=np.float64)
        self.prediction = np.array([
            [[100, 260], [600, 50]],
            [[200, 650], [50, 390]],
        ], dtype=np.float64)

    def test_layout_and_formula(self):
        model_layout = np.swapaxes(self.prediction, 1, 2)
        converted = to_sample_turbine_horizon(model_layout, "NHT")
        np.testing.assert_array_equal(converted, self.prediction)
        scores = LayeredWindMetrics(["01", "02"], 2)
        scores.update(converted[:1], self.truth[:1])
        scores.update(converted[1:], self.truth[1:])
        result = scores.result()
        overall = result["curve_overall"]
        self.assertEqual(result["shape"], [2, 2, 2])
        self.assertEqual(overall["valid_power_points"], 5)  # 100 is excluded.
        self.assertEqual(overall["total_points"], 8)
        self.assertEqual(overall["acc30_passed_points"], 3)
        self.assertEqual(overall["strict_acc30_pct"], 60)
        self.assertEqual(overall["MAE_kW"], 75)
        self.assertEqual(overall["MSE_kW2"], 9900)
        self.assertAlmostEqual(overall["RMSE_kW"], math.sqrt(9900))
        self.assertAlmostEqual(overall["MAPE_pct"], 34.66666666666667)
        self.assertEqual(result["endpoint_overall"]["strict_acc30_pct"], 100)
        self.assertEqual(result["site_overall"]["curve"]["strict_acc30_pct"], 50)
        self.assertEqual(len(result["per_horizon"]), 2)
        self.assertEqual(result["per_horizon"][0]["prediction_step"], 1)

    def test_invalid_shapes_and_ties(self):
        with self.assertRaises(ValueError):
            to_sample_turbine_horizon(np.zeros((2, 3)), "NHT")
        scores = LayeredWindMetrics(["01", "02"], 2)
        with self.assertRaises(ValueError):
            scores.update(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
        self.assertTrue(better_validation_score(60, 90, 60, 100))
        self.assertFalse(better_validation_score(60, 110, 60, 100))
        self.assertTrue(better_validation_score(60.01, 1000, 60, 1))
        self.assertFalse(better_validation_score(59.99, 1, 60, 1000))


if __name__ == "__main__":
    unittest.main()
