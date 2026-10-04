#!/usr/bin/env python
# coding: utf-8
"""
TimesFM Financial Statement Predictor for finrobot_equity.
Empirically forecasts Revenue, Costs, SG&A, and EBITDA using Google TimesFM foundation model.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_autogen_dir = os.path.join(_repo_root, "finrobot_autogen")
if _autogen_dir not in sys.path:
    sys.path.insert(0, _autogen_dir)


def predict_financial_projections_with_timesfm(
    historical_metrics_df: pd.DataFrame,
    forecast_years: List[str] = None
) -> Dict[str, Any]:
    """
    Generates data-driven multi-year projections for financial metrics using TimesFM.

    Args:
        historical_metrics_df: DataFrame with 'metrics' column and historical year columns (e.g., '2022A', '2023A', '2024A', '2025A')
        forecast_years: List of forecast years (default: ['2026E', '2027E', '2028E'])

    Returns:
        Dict containing projected growth rates, metric point forecasts, and Bear/Base/Bull quantile bands.
    """
    if forecast_years is None:
        forecast_years = ["2026E", "2027E", "2028E"]

    num_years = len(forecast_years)

    try:
        from finrobot.functional.timesfm_utils import get_timesfm_forecaster
        forecaster = get_timesfm_forecaster()
    except Exception as e:
        logger.warning(f"TimesFM module import error ({e}), using fallback forecaster.")
        from finrobot.functional.timesfm_utils import TimesFMForecaster
        forecaster = TimesFMForecaster()

    # Identify historical year columns in order
    year_cols = [c for c in historical_metrics_df.columns if c.endswith("A") and c != "metrics"]
    year_cols = sorted(year_cols, key=lambda x: int(x.replace("A", "")))

    projections = {}
    scenarios = {
        "bear_case": {},
        "base_case": {},
        "bull_case": {}
    }

    for _, row in historical_metrics_df.iterrows():
        metric_name = row["metrics"]
        raw_vals = [row[y] for y in year_cols]

        # Clean numerical values
        clean_vals = []
        for v in raw_vals:
            if isinstance(v, (int, float)) and not np.isnan(v):
                clean_vals.append(float(v))
            elif isinstance(v, str):
                try:
                    clean_vals.append(float(v.replace(",", "").replace("$", "").replace("%", "")))
                except ValueError:
                    pass

        if len(clean_vals) >= 2:
            try:
                fc_res = forecaster.forecast_series(clean_vals, horizon=num_years, freq=2)
                projections[metric_name] = fc_res["point_forecast"].tolist()
                scenarios["bear_case"][metric_name] = fc_res["bear_case_p10"].tolist()
                scenarios["base_case"][metric_name] = fc_res["base_case_p50"].tolist()
                scenarios["bull_case"][metric_name] = fc_res["bull_case_p90"].tolist()
            except Exception as e:
                logger.warning(f"TimesFM forecast for {metric_name} failed ({e}), using trend fallback.")
                last_val = clean_vals[-1]
                avg_growth = (clean_vals[-1] / clean_vals[0]) ** (1.0 / max(1, len(clean_vals) - 1)) - 1.0
                projected = [last_val * ((1.0 + avg_growth) ** i) for i in range(1, num_years + 1)]
                projections[metric_name] = projected

    # Compute derived TimesFM Revenue Growth Assumptions
    revenue_projections = projections.get("Revenue", [])
    revenue_growth_assumptions = {}
    if revenue_projections and "Revenue" in historical_metrics_df["metrics"].values:
        rev_row = historical_metrics_df[historical_metrics_df["metrics"] == "Revenue"].iloc[0]
        last_hist_rev = float(rev_row[year_cols[-1]]) if year_cols else None
        if last_hist_rev and last_hist_rev > 0:
            prev = last_hist_rev
            for idx, fy in enumerate(forecast_years):
                if idx < len(revenue_projections):
                    cur = revenue_projections[idx]
                    growth = (cur - prev) / prev if prev > 0 else 0.05
                    revenue_growth_assumptions[fy] = round(float(growth), 4)
                    prev = cur

    return {
        "forecast_years": forecast_years,
        "projections": projections,
        "scenarios": scenarios,
        "revenue_growth_assumptions": revenue_growth_assumptions,
        "model_used": "TimesFM-2.5-200M" if forecaster.is_available else "Statistical-Baseline"
    }
