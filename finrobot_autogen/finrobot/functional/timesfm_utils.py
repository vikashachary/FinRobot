#!/usr/bin/env python
# coding: utf-8
"""
Google TimesFM (Time Series Foundation Model) Utility Module for FinRobot.
Provides zero-shot multi-horizon point and probabilistic quantile forecasting
for stock prices, financial statement metrics, and scenario analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Global singleton for TimesFM model instance to prevent duplicate memory loading
_GLOBAL_TIMESFM_INSTANCE = None


class TimesFMForecaster:
    """
    Wrapper for Google TimesFM 2.5 200M time series foundation model.
    Provides robust zero-shot point forecasts, uncertainty quantiles (p10..p90),
    and scenario bands (Bear, Base, Bull).
    """

    def __init__(
        self,
        max_context: int = 512,
        max_horizon: int = 128,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        backend: str = "cpu",
        torch_compile: bool = False
    ):
        global _GLOBAL_TIMESFM_INSTANCE
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.repo_id = repo_id
        self.backend = backend
        self.torch_compile = torch_compile
        self.is_available = False
        self.model = None

        if _GLOBAL_TIMESFM_INSTANCE is not None:
            self.model = _GLOBAL_TIMESFM_INSTANCE.model
            self.is_available = _GLOBAL_TIMESFM_INSTANCE.is_available
            return

        self._initialize_model()
        if self.is_available:
            _GLOBAL_TIMESFM_INSTANCE = self

    def _initialize_model(self):
        """Initializes and compiles the TimesFM model."""
        try:
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            from timesfm import ForecastConfig

            logger.info(f"Loading TimesFM model from {self.repo_id}...")
            self.model = TimesFM_2p5_200M_torch.from_pretrained(
                self.repo_id,
                torch_compile=self.torch_compile
            )

            config = ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                infer_is_positive=False
            )
            self.model.compile(config)
            self.is_available = True
            logger.info("✅ Google TimesFM model loaded and compiled successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load TimesFM model ({e}). Using statistical fallback engine.")
            self.model = None
            self.is_available = False

    def forecast_series(
        self,
        series: Union[pd.Series, np.ndarray, List[float]],
        horizon: int = 30,
        freq: int = 0
    ) -> Dict[str, Any]:
        """
        Generates zero-shot point forecast and probabilistic quantile bands for a 1D time series.

        Args:
            series: Historical time series data (pandas Series, numpy array, or list of numbers).
            horizon: Number of time steps to forecast ahead.
            freq: Time frequency (0: daily/high-freq, 1: weekly, 2: monthly/quarterly).

        Returns:
            Dict containing:
                - point_forecast: Expected values (np.ndarray)
                - quantiles: 10 quantiles p10..p90 (np.ndarray)
                - bear_case_p10: 10th percentile conservative estimate
                - base_case_p50: 50th percentile (median) estimate
                - bull_case_p90: 90th percentile optimistic estimate
                - model_used: 'timesfm-2.5-200m' or 'statistical_fallback'
        """
        # Clean and validate input
        if isinstance(series, pd.Series):
            clean_values = series.dropna().values.astype(np.float32)
        elif isinstance(series, np.ndarray):
            clean_values = series[~np.isnan(series)].astype(np.float32)
        else:
            clean_values = np.array([x for x in series if x is not None and not np.isnan(x)], dtype=np.float32)

        if len(clean_values) < 3:
            raise ValueError("Time series requires at least 3 historical data points for forecasting.")

        if self.is_available and self.model is not None:
            try:
                # TimesFM requires list of 1D numpy arrays
                inputs = [clean_values[-self.max_context:]]
                point_fc, full_fc = self.model.forecast(horizon=horizon, inputs=inputs)

                # Point forecast: shape (1, horizon) -> 1D array
                point_pred = point_fc[0]

                # Full forecast: shape (1, horizon, 10) representing quantiles
                quantiles_matrix = full_fc[0]  # (horizon, 10)

                # Quantile indices: 0: p10, 4: p50 (median), 8: p90 (or 9: mean depending on configuration)
                p10 = quantiles_matrix[:, 0]
                p50 = quantiles_matrix[:, 4] if quantiles_matrix.shape[1] > 4 else point_pred
                p90 = quantiles_matrix[:, 8] if quantiles_matrix.shape[1] > 8 else quantiles_matrix[:, -1]

                return {
                    "point_forecast": point_pred,
                    "quantiles": quantiles_matrix,
                    "bear_case_p10": p10,
                    "base_case_p50": p50,
                    "bull_case_p90": p90,
                    "model_used": "timesfm-2.5-200m",
                    "history_length": len(clean_values),
                    "horizon": horizon
                }
            except Exception as e:
                logger.warning(f"TimesFM inference failed ({e}), switching to robust statistical fallback.")

        # Robust Statistical Fallback Engine (Holt's Exponential Smoothing + Drift + Uncertainty Bounds)
        return self._statistical_fallback_forecast(clean_values, horizon)

    def _statistical_fallback_forecast(
        self,
        values: np.ndarray,
        horizon: int
    ) -> Dict[str, Any]:
        """High-quality statistical baseline fallback with exponential smoothing and calibrated bands."""
        n = len(values)
        last_val = values[-1]

        # Calculate historical drift and volatility
        returns = np.diff(values) / np.where(values[:-1] == 0, 1.0, values[:-1])
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 1 else 0.02
        volatility = max(volatility, 0.01)

        # Exponential smoothing trend
        alpha = 0.3
        smoothed = np.zeros(n)
        smoothed[0] = values[0]
        for t in range(1, n):
            smoothed[t] = alpha * values[t] + (1 - alpha) * smoothed[t - 1]

        recent_trend = (smoothed[-1] - smoothed[max(0, n - 10)]) / max(1, min(10, n))
        
        # Project forward
        time_steps = np.arange(1, horizon + 1)
        point_pred = last_val + recent_trend * time_steps

        # Standard error increases with sqrt(time)
        cum_std = volatility * last_val * np.sqrt(time_steps)
        p10 = point_pred - 1.28 * cum_std
        p90 = point_pred + 1.28 * cum_std
        p50 = point_pred

        # Build simulated 10-quantile matrix
        quantiles = np.zeros((horizon, 10))
        z_scores = np.linspace(-1.28, 1.28, 10)
        for q_idx, z in enumerate(z_scores):
            quantiles[:, q_idx] = point_pred + z * cum_std

        return {
            "point_forecast": point_pred,
            "quantiles": quantiles,
            "bear_case_p10": p10,
            "base_case_p50": p50,
            "bull_case_p90": p90,
            "model_used": "statistical_fallback",
            "history_length": n,
            "horizon": horizon
        }

    def forecast_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int = 30,
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Forecasts a pandas DataFrame target column and returns an extended DataFrame with forecast steps and bounds.
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")

        series = df[target_col]
        result = self.forecast_series(series, horizon=horizon)

        # Build future date index if date column exists
        future_dates = None
        if date_col and date_col in df.columns:
            last_date = pd.to_datetime(df[date_col].iloc[-1])
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]

        forecast_df = pd.DataFrame({
            "point_forecast": result["point_forecast"],
            "bear_case_p10": result["bear_case_p10"],
            "base_case_p50": result["base_case_p50"],
            "bull_case_p90": result["bull_case_p90"]
        })

        if future_dates is not None:
            forecast_df["date"] = future_dates

        return forecast_df

    def forecast_stock_trajectory(
        self,
        ticker: str,
        horizon_days: int = 30,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Fetches stock historical data via yfinance and generates TimesFM price forecasts.
        """
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{lookback_days}d")
        if hist.empty or "Close" not in hist:
            raise ValueError(f"No price history returned for ticker {ticker}")

        close_series = hist["Close"]
        fc_result = self.forecast_series(close_series, horizon=horizon_days, freq=0)

        current_price = float(close_series.iloc[-1])
        predicted_end_price = float(fc_result["point_forecast"][-1])
        expected_return_pct = ((predicted_end_price - current_price) / current_price) * 100.0

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price_horizon": predicted_end_price,
            "expected_return_pct": round(expected_return_pct, 2),
            "point_forecast": fc_result["point_forecast"],
            "bear_case_p10": fc_result["bear_case_p10"],
            "base_case_p50": fc_result["base_case_p50"],
            "bull_case_p90": fc_result["bull_case_p90"],
            "model_used": fc_result["model_used"],
            "horizon_days": horizon_days
        }


# Convenience factory function
def get_timesfm_forecaster(
    max_context: int = 512,
    max_horizon: int = 128
) -> TimesFMForecaster:
    """Returns a singleton or initialized instance of TimesFMForecaster."""
    global _GLOBAL_TIMESFM_INSTANCE
    if _GLOBAL_TIMESFM_INSTANCE is None:
        _GLOBAL_TIMESFM_INSTANCE = TimesFMForecaster(
            max_context=max_context,
            max_horizon=max_horizon
        )
    return _GLOBAL_TIMESFM_INSTANCE
