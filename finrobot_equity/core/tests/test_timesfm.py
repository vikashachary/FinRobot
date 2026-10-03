#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for Google TimesFM integration and forecasting utilities.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from finrobot.functional.timesfm_utils import TimesFMForecaster, get_timesfm_forecaster
from finrobot.functional.forecasting import TimeSeriesForecastingUtils
from modules.timesfm_financial_predictor import predict_financial_projections_with_timesfm


class TestTimesFMIntegration(unittest.TestCase):

    def setUp(self):
        # Deterministic synthetic series
        self.series = np.sin(np.linspace(0, 10, 64)) + 50.0

    def test_timesfm_forecaster_initialization(self):
        forecaster = get_timesfm_forecaster()
        self.assertIsNotNone(forecaster)

    def test_forecast_series(self):
        forecaster = get_timesfm_forecaster()
        result = forecaster.forecast_series(self.series, horizon=10)
        self.assertIn("point_forecast", result)
        self.assertIn("bear_case_p10", result)
        self.assertIn("bull_case_p90", result)
        self.assertEqual(len(result["point_forecast"]), 10)
        self.assertEqual(len(result["bear_case_p10"]), 10)
        self.assertEqual(len(result["bull_case_p90"]), 10)

    def test_evaluate_accuracy_metrics(self):
        actual = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        predicted = np.array([10.5, 11.5, 14.2, 15.8, 18.1])

        metrics = TimeSeriesForecastingUtils.evaluate_forecast_accuracy(actual, predicted)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("directional_accuracy_pct", metrics)
        self.assertGreater(metrics["directional_accuracy_pct"], 0)

    def test_financial_statement_projections(self):
        df = pd.DataFrame({
            "metrics": ["Revenue", "Cost of Operations", "SG&A", "EBITDA"],
            "2022A": [1000.0, 400.0, 200.0, 400.0],
            "2023A": [1200.0, 480.0, 240.0, 480.0],
            "2024A": [1500.0, 600.0, 300.0, 600.0],
            "2025A": [1900.0, 750.0, 380.0, 770.0]
        })

        proj = predict_financial_projections_with_timesfm(df, forecast_years=["2026E", "2027E", "2028E"])
        self.assertIn("revenue_growth_assumptions", proj)
        self.assertIn("projections", proj)
        self.assertIn("scenarios", proj)
        self.assertIn("Revenue", proj["projections"])
        self.assertEqual(len(proj["projections"]["Revenue"]), 3)


if __name__ == "__main__":
    unittest.main()
