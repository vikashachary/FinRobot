#!/usr/bin/env python
# coding: utf-8
"""
Time Series Forecasting High-Level Functional Module for FinRobot.
Integrates TimesFM, statistical models, and evaluation tools for financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from .timesfm_utils import get_timesfm_forecaster, TimesFMForecaster


class TimeSeriesForecastingUtils:
    """
    High-level utilities for stock price prediction, financial statement forecasting,
    and quantitative forecast evaluation metrics.
    """

    @staticmethod
    def forecast_stock_price(
        ticker: str,
        days_ahead: int = 30,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Forecasts future stock price trajectory with probabilistic confidence intervals using TimesFM.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'COIN')
            days_ahead: Prediction horizon in trading days (default: 30)
            lookback_days: Historical window size (default: 365)

        Returns:
            Dictionary containing current price, predicted price, expected return, and quantile bounds.
        """
        forecaster = get_timesfm_forecaster()
        return forecaster.forecast_stock_trajectory(
            ticker=ticker,
            horizon_days=days_ahead,
            lookback_days=lookback_days
        )

    @staticmethod
    def forecast_financial_metrics(
        historical_values: List[float],
        years_ahead: int = 3,
        metric_name: str = "Revenue"
    ) -> Dict[str, Any]:
        """
        Forecasts future financial metrics (e.g. Revenue, EBITDA) across multi-year horizons.

        Args:
            historical_values: List of historical annual/quarterly metric values
            years_ahead: Number of forward periods to predict
            metric_name: Name of the metric for reporting

        Returns:
            Dictionary containing point projections and bear/base/bull scenario values.
        """
        forecaster = get_timesfm_forecaster()
        result = forecaster.forecast_series(
            series=historical_values,
            horizon=years_ahead,
            freq=2  # Quarterly / Annual frequency
        )

        return {
            "metric": metric_name,
            "historical_last": historical_values[-1] if historical_values else None,
            "projections": result["point_forecast"].tolist(),
            "bear_case": result["bear_case_p10"].tolist(),
            "base_case": result["base_case_p50"].tolist(),
            "bull_case": result["bull_case_p90"].tolist(),
            "model_used": result["model_used"]
        }

    @staticmethod
    def evaluate_forecast_accuracy(
        actual: Union[np.ndarray, List[float], pd.Series],
        predicted: Union[np.ndarray, List[float], pd.Series]
    ) -> Dict[str, float]:
        """
        Calculates standard quantitative time-series forecasting accuracy metrics:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
        - Directional Accuracy / Hit Ratio (% of correctly predicted sign changes)
        - Normalized RMSE (NRMSE)
        """
        act = np.array(actual, dtype=np.float64)
        pred = np.array(predicted, dtype=np.float64)

        if len(act) != len(pred):
            min_len = min(len(act), len(pred))
            act = act[:min_len]
            pred = pred[:min_len]

        errors = act - pred
        mse = np.mean(errors ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))

        # Percentage error (avoid division by zero)
        denom = np.where(np.abs(act) == 0, 1e-8, np.abs(act))
        mape = float(np.mean(np.abs(errors) / denom) * 100.0)

        # Directional Accuracy (Hit Ratio)
        if len(act) > 1:
            actual_direction = np.sign(np.diff(act))
            pred_direction = np.sign(np.diff(pred))
            hit_ratio = float(np.mean(actual_direction == pred_direction) * 100.0)
        else:
            hit_ratio = 100.0 if np.sign(act[0]) == np.sign(pred[0]) else 0.0

        nrmse = float(rmse / (np.max(act) - np.min(act))) if (np.max(act) - np.min(act)) > 0 else 0.0

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "directional_accuracy_pct": round(hit_ratio, 2),
            "nrmse": round(nrmse, 4),
            "sample_size": len(act)
        }
