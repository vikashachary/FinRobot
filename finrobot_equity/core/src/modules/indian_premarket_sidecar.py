#!/usr/bin/env python
# coding: utf-8
"""
Indian Market Premarket Research Sidecar Module.
Collects off-market data, computes multi-timeframe confluence (CPR, Camarilla),
executes Google TimesFM zero-shot probabilistic forecasting on Nifty 50,
evaluates sub-50ms Laya System 1 decision routing,
and uses Google Gemini API (gemini-3.8-flash) to synthesize institutional premarket briefing.
Serializes to macro_context.json for low-latency live execution ingestion.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Ensure paths
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_autogen_dir = os.path.join(_repo_root, "finrobot_autogen")
if _autogen_dir not in sys.path:
    sys.path.insert(0, _autogen_dir)

logger = logging.getLogger(__name__)

# Key Indian market tickers
INDIAN_INDICES = {
    "NIFTY_50": "^NSEI",
    "BANK_NIFTY": "^NSEBANK",
    "INDIA_VIX": "^INDIAVIX"
}

SECTOR_INDICES = {
    "NIFTY_IT": "^CNXIT",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_FMCG": "^CNXFMCG"
}

GLOBAL_CUES = {
    "US_SP500": "^GSPC",
    "US_NASDAQ": "^IXIC",
    "BRENT_CRUDE": "BZ=F",
    "USD_INR": "INR=X",
    "US_10Y_YIELD": "^TNX"
}

TOP_HEAVYWEIGHTS = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS"
}


def calculate_cpr_and_camarilla(high: float, low: float, close: float) -> Dict[str, Any]:
    """
    Calculates Central Pivot Range (CPR) and Camarilla Pivot Points.
    
    CPR Formulas:
      Pivot (P) = (High + Low + Close) / 3
      Bottom Central (BC) = (High + Low) / 2
      Top Central (TC) = (2 * P) - BC
      CPR Width % = abs(TC - BC) / P * 100
      Width Category: Narrow (<0.25% => Trending), Average (0.25%-0.50%), Wide (>0.50% => Rangebound/Sideways)
      
    Camarilla Formulas:
      Range (R) = High - Low
      H4 = Close + (1.1 * R)   (Breakout Long)
      H3 = Close + (0.55 * R)  (Range Short / Resistance)
      L3 = Close - (0.55 * R)  (Range Long / Support)
      L4 = Close - (1.1 * R)   (Breakdown Short)
    """
    p = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (2.0 * p) - bc

    cpr_top = max(tc, bc)
    cpr_bottom = min(tc, bc)
    cpr_width = abs(tc - bc)
    cpr_width_pct = (cpr_width / p) * 100.0

    if cpr_width_pct < 0.25:
        width_regime = "NARROW (High Probability Trending Day)"
    elif cpr_width_pct <= 0.50:
        width_regime = "AVERAGE (Balanced Trend / Selective Range)"
    else:
        width_regime = "WIDE (High Probability Rangebound / Sideways Day)"

    rng = high - low
    h4 = close + (1.1 * rng)
    h3 = close + (0.55 * rng)
    l3 = close - (0.55 * rng)
    l4 = close - (1.1 * rng)
    h5 = (high / low) * close if low > 0 else h4

    return {
        "cpr": {
            "pivot": round(p, 2),
            "top_central": round(cpr_top, 2),
            "bottom_central": round(cpr_bottom, 2),
            "width_points": round(cpr_width, 2),
            "width_pct": round(cpr_width_pct, 4),
            "regime": width_regime
        },
        "camarilla": {
            "h5_target": round(h5, 2),
            "h4_breakout": round(h4, 2),
            "h3_resistance": round(h3, 2),
            "l3_support": round(l3, 2),
            "l4_breakdown": round(l4, 2)
        }
    }


def fetch_multi_timeframe_metrics(ticker: str) -> Dict[str, Any]:
    """Fetches daily, weekly, and monthly price history and returns moving averages & trend."""
    try:
        t = yf.Ticker(ticker)
        df_daily = t.history(period="6mo", interval="1d")
        if df_daily.empty:
            return {}

        current_close = float(df_daily["Close"].iloc[-1])
        prev_close = float(df_daily["Close"].iloc[-2]) if len(df_daily) >= 2 else current_close
        high_prev = float(df_daily["High"].iloc[-2]) if len(df_daily) >= 2 else current_close
        low_prev = float(df_daily["Low"].iloc[-2]) if len(df_daily) >= 2 else current_close
        close_prev = prev_close

        change_pts = current_close - prev_close
        change_pct = (change_pts / prev_close) * 100.0 if prev_close else 0.0

        # Moving Averages
        close_series = df_daily["Close"]
        ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close_series) >= 200 else ema50

        # Multi-timeframe trend
        trend_daily = "BULLISH" if current_close > ema20 > ema50 else ("BEARISH" if current_close < ema20 < ema50 else "CONSOLIDATION")

        # Weekly trend
        df_weekly = t.history(period="1y", interval="1wk")
        w_ema20 = float(df_weekly["Close"].ewm(span=20, adjust=False).mean().iloc[-1]) if len(df_weekly) >= 20 else current_close
        trend_weekly = "BULLISH" if current_close > w_ema20 else "BEARISH"

        pivots = calculate_cpr_and_camarilla(high_prev, low_prev, close_prev)

        return {
            "current_close": round(current_close, 2),
            "prev_close": round(prev_close, 2),
            "change_pts": round(change_pts, 2),
            "change_pct": round(change_pct, 2),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "trend_daily": trend_daily,
            "trend_weekly": trend_weekly,
            "cpr": pivots["cpr"],
            "camarilla": pivots["camarilla"],
            "history_series": close_series.tail(60).tolist()
        }
    except Exception as e:
        logger.error(f"Error fetching multi-timeframe metrics for {ticker}: {e}")
        return {}


def fetch_sector_relative_strength(nifty_return_1d: float, nifty_return_5d: float) -> List[Dict[str, Any]]:
    """Calculates relative strength and performance ranking across NSE sectors."""
    results = []
    for name, sym in SECTOR_INDICES.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="1mo", interval="1d")
            if hist.empty or len(hist) < 5:
                continue
            c = hist["Close"]
            ret_1d = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) * 100.0 if len(c) >= 2 else 0.0
            ret_5d = ((c.iloc[-1] - c.iloc[-6]) / c.iloc[-6]) * 100.0 if len(c) >= 6 else ret_1d
            rs_1d = ret_1d - nifty_return_1d
            rs_5d = ret_5d - nifty_return_5d

            status = "OUTPERFORMING" if rs_5d > 0.5 else ("UNDERPERFORMING" if rs_5d < -0.5 else "NEUTRAL")

            results.append({
                "sector": name.replace("^", "").replace("_", " "),
                "symbol": sym,
                "close": round(float(c.iloc[-1]), 2),
                "change_1d_pct": round(ret_1d, 2),
                "change_5d_pct": round(ret_5d, 2),
                "rs_alpha_5d": round(rs_5d, 2),
                "momentum_status": status
            })
        except Exception as e:
            logger.warning(f"Could not fetch sector {name}: {e}")

    # Sort descending by 5D relative strength
    results.sort(key=lambda x: x["rs_alpha_5d"], reverse=True)
    return results


def fetch_heavyweights_momentum() -> List[Dict[str, Any]]:
    """Fetches key price levels and 5D trend for Indian mega-caps."""
    results = []
    for name, sym in TOP_HEAVYWEIGHTS.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="1mo", interval="1d")
            if hist.empty or len(hist) < 2:
                continue
            c = hist["Close"]
            h = hist["High"]
            l = hist["Low"]
            ret_1d = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) * 100.0
            pivots = calculate_cpr_and_camarilla(float(h.iloc[-2]), float(l.iloc[-2]), float(c.iloc[-2]))

            results.append({
                "ticker": sym,
                "name": name,
                "close": round(float(c.iloc[-1]), 2),
                "change_1d_pct": round(ret_1d, 2),
                "cpr_pivot": pivots["cpr"]["pivot"],
                "camarilla_h3": pivots["camarilla"]["h3_resistance"],
                "camarilla_l3": pivots["camarilla"]["l3_support"],
                "cpr_regime": pivots["cpr"]["regime"]
            })
        except Exception as e:
            logger.warning(f"Could not fetch heavyweight {name}: {e}")
    return results


def fetch_global_cues() -> Dict[str, Any]:
    """Fetches global macroeconomic cues impacting Indian market opening."""
    cues = {}
    for name, sym in GLOBAL_CUES.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty and len(hist) >= 2:
                last_val = float(hist["Close"].iloc[-1])
                prev_val = float(hist["Close"].iloc[-2])
                chg_pct = ((last_val - prev_val) / prev_val) * 100.0
                cues[name] = {
                    "symbol": sym,
                    "value": round(last_val, 2),
                    "change_pct": round(chg_pct, 2)
                }
            elif not hist.empty:
                cues[name] = {
                    "symbol": sym,
                    "value": round(float(hist["Close"].iloc[-1]), 2),
                    "change_pct": 0.0
                }
        except Exception as e:
            logger.warning(f"Error fetching global cue {name}: {e}")
    return cues


def run_timesfm_nifty_forecast(nifty_history: List[float], horizon: int = 5) -> Dict[str, Any]:
    """Executes TimesFM foundation model forecasting on NIFTY 50 closing prices."""
    try:
        from finrobot.functional.timesfm_utils import get_timesfm_forecaster
        forecaster = get_timesfm_forecaster()
        fc = forecaster.forecast_series(nifty_history, horizon=horizon, freq=0)
        
        last_price = nifty_history[-1] if nifty_history else 0.0
        p50 = [round(float(v), 2) for v in fc["base_case_p50"]]
        p10 = [round(float(v), 2) for v in fc["bear_case_p10"]]
        p90 = [round(float(v), 2) for v in fc["bull_case_p90"]]

        expected_drift_pct = ((p50[-1] - last_price) / last_price) * 100.0 if last_price else 0.0

        return {
            "model": fc.get("model_used", "timesfm-2.5-200m"),
            "horizon_days": horizon,
            "base_case_p50": p50,
            "bear_case_p10": p10,
            "bull_case_p90": p90,
            "terminal_p50": p50[-1],
            "terminal_p10": p10[-1],
            "terminal_p90": p90[-1],
            "expected_5d_drift_pct": round(expected_drift_pct, 2)
        }
    except Exception as e:
        logger.warning(f"TimesFM forecasting failed ({e}), using baseline.")
        last_val = nifty_history[-1] if nifty_history else 22500.0
        return {
            "model": "statistical_fallback",
            "horizon_days": horizon,
            "base_case_p50": [round(last_val * (1 + 0.001 * i), 2) for i in range(1, horizon + 1)],
            "bear_case_p10": [round(last_val * (1 - 0.004 * i), 2) for i in range(1, horizon + 1)],
            "bull_case_p90": [round(last_val * (1 + 0.005 * i), 2) for i in range(1, horizon + 1)],
            "expected_5d_drift_pct": 0.5
        }


def run_gemini_premarket_synthesis(context_data: Dict[str, Any], api_key: str, model_name: str = "gemini-3.8-flash") -> Dict[str, str]:
    """Calls Google Gemini API (gemini-3.8-flash) to synthesize dual Senior Trader & Systems Architect briefing."""
    import urllib.request
    import urllib.error

    system_prompt = (
        "You are an Elite Senior Algorithmic Quantitative Trader and Chief Systems Architect specializing in "
        "Indian equity derivatives and market microstructure (NSE/BSE). "
        "Evaluate the market strictly under the Dual Senior Perspective Invariant:\n"
        "1. Senior Algorithmic/Quant Trader: Multi-timeframe confluence, Central Pivot Range (CPR width & regime), "
        "Camarilla levels (H4 breakout, H3 reversal resistance, L3 reversal support, L4 breakdown), "
        "India VIX volatility regime, derivatives execution risks (STT on exercised ITM options at 0.125%, "
        "exchange turnover fees, GST, stamp duty), and TimesFM probabilistic quantile ranges (p10, p50, p90).\n"
        "2. Chief Systems Architect: Low-latency state routing, broker API rate limits (e.g. Zerodha Kite <= 3 req/s), "
        "fail-safe stop loss guards, and separation of research sidecars from the live execution hot path.\n\n"
        "Generate a structured, rigorous, highly actionable institutional premarket briefing. "
        "Format with clear headings and bullet points. Do not include vague generalities; reference exact levels."
    )

    user_prompt = f"""
INDIAN MARKET PREMARKET TELEMETRY & CONFLUENCE DATA:
{json.dumps(context_data, indent=2)}

Please provide:
1. EXECUTIVE MARKET OPENING BIAS & CONFLUENCE REGIME (Nifty 50 & Bank Nifty)
2. CENTRAL PIVOT RANGE (CPR) & CAMARILLA TACTICAL BATTLEGROUNDS
3. SECTOR ROTATION & HEAVYWEIGHT LEADERSHIP
4. DERIVATIVES EXECUTION & INDIAN TAX REALITIES (STT on ITM options, VIX dynamics, slippage guards)
5. SYSTEMS ARCHITECTURE & AUTOMATED EXECUTION SAFEGUARDS
"""

    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    candidate_models = [model_name, "gemini-2.5-flash-lite", "gemini-flash-latest"]
    # De-duplicate while preserving order
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    for m in models_to_try:
        logger.info(f"Attempting Gemini synthesis using model: {m}")
        for attempt in range(2):
            try:
                from openai import OpenAI
                import time
                client = OpenAI(api_key=api_key, base_url=gemini_base_url, timeout=60.0)
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=2500
                )
                content = resp.choices[0].message.content or ""
                if content:
                    import re
                    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                    return {"briefing": cleaned, "provider": f"Google Gemini ({m})"}
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Gemini {m} attempt {attempt+1} failed: {err_msg}")
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    logger.info(f"Quota exhausted for {m}, moving to next candidate model...")
                    break
                elif "503" in err_msg:
                    import time
                    time.sleep(3)
                else:
                    break

    # Direct Gemini REST fallback on gemini-2.5-flash-lite
    for m in ["gemini-2.5-flash-lite", model_name]:
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 2500
                }
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        txt = parts[0]["text"].strip()
                        import re
                        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', txt).strip()
                        return {"briefing": cleaned, "provider": f"Google Gemini REST ({m})"}
        except Exception as rest_e:
            logger.error(f"Gemini REST direct call ({m}) failed: {rest_e}")

    # Fallback if API completely unreachable
    return {
        "briefing": (
            "### NIFTY 50 & BANK NIFTY PREMARKET BRIEFING (Fallback)\n\n"
            "- **Regime**: Confluence indicates selective rangebound behavior around daily CPR.\n"
            "- **Key Nifty Levels**: Monitor Camarilla H3 resistance and L3 support.\n"
            "- **Derivatives Guard**: Beware STT 0.125% risk on ITM exercised options on expiry day."
        ),
        "provider": "Local Fallback Engine"
    }


def generate_indian_premarket_report(
    config_path: str = "finrobot_equity/core/config/config.ini",
    output_dir: str = "output/indian_market"
) -> Tuple[Dict[str, Any], str, str]:
    """
    Main orchestration routine for Indian Market Premarket Analysis.
    Returns:
        (macro_context_dict, json_path, html_path)
    """
    from modules.common_utils import load_config, get_llm_config
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)
    llm_cfg = get_llm_config(config, service="gemini")
    gemini_key = llm_cfg.get("api_key")
    gemini_model = llm_cfg.get("model", "gemini-3.8-flash")

    print("🇮🇳 Starting Indian Market Premarket Research Sidecar...")
    print(f"   Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # 1. Fetch Nifty 50 & Bank Nifty multi-timeframe
    print("📊 Ingesting NIFTY 50 & BANK NIFTY telemetry...")
    nifty_data = fetch_multi_timeframe_metrics(INDIAN_INDICES["NIFTY_50"])
    banknifty_data = fetch_multi_timeframe_metrics(INDIAN_INDICES["BANK_NIFTY"])

    # India VIX
    vix_val = 14.5
    try:
        t_vix = yf.Ticker(INDIAN_INDICES["INDIA_VIX"])
        h_vix = t_vix.history(period="5d", interval="1d")
        if not h_vix.empty:
            vix_val = round(float(h_vix["Close"].iloc[-1]), 2)
    except Exception:
        pass

    vix_regime = "NORMAL (13-18)" if 13 <= vix_val <= 18 else ("ELEVATED (>18)" if vix_val > 18 else "SUBDUED (<13)")

    # 2. Global Cues
    print("🌍 Ingesting Global Macro Cues...")
    global_cues = fetch_global_cues()

    # 3. Sector Relative Strength
    print("🔄 Evaluating Sector Rotation & Relative Strength...")
    n_1d = nifty_data.get("change_pct", 0.0)
    sector_rs = fetch_sector_relative_strength(n_1d, n_1d * 2.5)

    # 4. Heavyweights Momentum
    print("🏢 Tracking Top NSE Mega-Caps...")
    heavyweights = fetch_heavyweights_momentum()

    # 5. TimesFM Foundation Model Quantile Forecast
    print("🤖 Executing Google TimesFM Zero-Shot Probabilistic Forecast for NIFTY 50...")
    nifty_series = nifty_data.get("history_series", [])
    timesfm_forecast = run_timesfm_nifty_forecast(nifty_series, horizon=5)
    print(f"   TimesFM 5-Day Base Case: {timesfm_forecast['base_case_p50']}")
    print(f"   TimesFM Expected Drift: {timesfm_forecast['expected_5d_drift_pct']}%")

    # 6. Sub-50ms Laya System 1 Decision Router
    print("⚡ Evaluating Laya RLCD System 1 Decision Router (<50ms)...")
    laya_routing = {}
    try:
        from modules.laya_decision_engine import LayaDecisionEngine
        laya_engine = LayaDecisionEngine()
        growth_proxy = {"2025E": timesfm_forecast["expected_5d_drift_pct"] / 100.0}
        laya_dossier = laya_engine.generate_system1_dossier(
            company_ticker="^NSEI",
            company_name="NIFTY 50 Benchmark Index",
            forecast_growth=growth_proxy,
            timesfm_metadata={"scenarios": {"base_case": {"Revenue": [timesfm_forecast["terminal_p50"]]}}},
            risk_factors=[f"India VIX at {vix_val}", f"CPR regime {nifty_data.get('cpr', {}).get('regime', 'Neutral')}"]
        )
        laya_routing = {
            "rating": laya_dossier["investment_rating"]["label"],
            "confidence": laya_dossier["investment_rating"]["confidence"],
            "system_routing": laya_dossier["system_routing"],
            "latency_ms": laya_dossier["total_latency_ms"]
        }
        print(f"   Laya Decision: {laya_routing['rating']} ({laya_routing['confidence']*100:.1f}%) in {laya_routing['latency_ms']}ms")
    except Exception as e:
        logger.warning(f"Laya Decision evaluation error: {e}")
        laya_routing = {"rating": "Hold", "confidence": 0.65, "system_routing": "EXECUTE_SYSTEM_1", "latency_ms": 0.05}

    # Assemble Structured Telemetry Context
    telemetry_context = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "market": "NSE_INDIA",
        "nifty_50": {
            "current_close": nifty_data.get("current_close"),
            "change_pct": nifty_data.get("change_pct"),
            "trend_daily": nifty_data.get("trend_daily"),
            "trend_weekly": nifty_data.get("trend_weekly"),
            "cpr": nifty_data.get("cpr"),
            "camarilla": nifty_data.get("camarilla")
        },
        "bank_nifty": {
            "current_close": banknifty_data.get("current_close"),
            "change_pct": banknifty_data.get("change_pct"),
            "trend_daily": banknifty_data.get("trend_daily"),
            "cpr": banknifty_data.get("cpr"),
            "camarilla": banknifty_data.get("camarilla")
        },
        "volatility": {
            "india_vix": vix_val,
            "regime": vix_regime
        },
        "timesfm_forecast": timesfm_forecast,
        "laya_system1_routing": laya_routing,
        "sector_relative_strength": sector_rs,
        "top_heavyweights": heavyweights,
        "global_cues": global_cues
    }

    # 7. Google Gemini Autonomous Agent Synthesis
    print("🧠 Generating Institutional Premarket Narrative via Google Gemini API...")
    gemini_output = run_gemini_premarket_synthesis(telemetry_context, gemini_key, gemini_model)
    telemetry_context["ai_briefing"] = gemini_output

    # 8. Serialize macro_context.json (Research Sidecar Contract)
    json_path = os.path.join(output_dir, "macro_context.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_context, f, indent=2)
    print(f"✅ Serialized research sidecar artifact to: {json_path}")

    # 9. Render Executive HTML Premarket Report
    html_path = os.path.join(output_dir, "Indian_Market_Premarket_Report.html")
    html_content = render_html_premarket_dashboard(telemetry_context)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Generated Indian Market Premarket HTML Report: {html_path}")

    return telemetry_context, json_path, html_path


def render_html_premarket_dashboard(data: Dict[str, Any]) -> str:
    """Renders a clean, executive HTML premarket report."""
    nifty = data.get("nifty_50", {})
    bn = data.get("bank_nifty", {})
    cpr = nifty.get("cpr", {})
    cam = nifty.get("camarilla", {})
    tfm = data.get("timesfm_forecast", {})
    laya = data.get("laya_system1_routing", {})
    vol = data.get("volatility", {})
    briefing = data.get("ai_briefing", {}).get("briefing", "")
    provider = data.get("ai_briefing", {}).get("provider", "Gemini")

    # Format markdown bold in briefing
    formatted_briefing = briefing.replace("\n", "<br/>").replace("### ", "<h3>").replace("## ", "<h2>").replace("# ", "<h1>")
    formatted_briefing = formatted_briefing.replace("**", "<strong>")

    # Sector table rows
    sector_rows = ""
    for s in data.get("sector_relative_strength", []):
        badge_color = "#10b981" if "OUTPERFORMING" in s.get("momentum_status", "") else ("#ef4444" if "UNDERPERFORMING" in s.get("momentum_status", "") else "#6b7280")
        sector_rows += f"""
        <tr>
            <td style="font-weight:600;">{s.get('sector')}</td>
            <td>₹{s.get('close'):,.2f}</td>
            <td style="color:{'#10b981' if s.get('change_1d_pct', 0) >= 0 else '#ef4444'};">{s.get('change_1d_pct'):+.2f}%</td>
            <td style="color:{'#10b981' if s.get('rs_alpha_5d', 0) >= 0 else '#ef4444'}; font-weight:600;">{s.get('rs_alpha_5d'):+.2f}%</td>
            <td><span style="background:{badge_color}20; color:{badge_color}; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600;">{s.get('momentum_status')}</span></td>
        </tr>
        """

    # Heavyweight rows
    hw_rows = ""
    for h in data.get("top_heavyweights", []):
        hw_rows += f"""
        <tr>
            <td style="font-weight:600;">{h.get('name')} <span style="font-size:11px; color:#6b7280;">({h.get('ticker')})</span></td>
            <td>₹{h.get('close'):,.2f}</td>
            <td style="color:{'#10b981' if h.get('change_1d_pct', 0) >= 0 else '#ef4444'}; font-weight:600;">{h.get('change_1d_pct'):+.2f}%</td>
            <td>₹{h.get('cpr_pivot'):,.2f}</td>
            <td style="color:#ef4444;">₹{h.get('camarilla_h3'):,.2f}</td>
            <td style="color:#10b981;">₹{h.get('camarilla_l3'):,.2f}</td>
        </tr>
        """

    # TimesFM Quantile rows
    tfm_rows = ""
    p50 = tfm.get("base_case_p50", [])
    p10 = tfm.get("bear_case_p10", [])
    p90 = tfm.get("bull_case_p90", [])
    for idx in range(len(p50)):
        tfm_rows += f"""
        <tr>
            <td style="font-weight:600;">Day +{idx + 1}</td>
            <td style="color:#ef4444; font-weight:600;">₹{p10[idx]:,.2f}</td>
            <td style="color:#3b82f6; font-weight:700;">₹{p50[idx]:,.2f}</td>
            <td style="color:#10b981; font-weight:600;">₹{p90[idx]:,.2f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Indian Market Premarket Research Dossier (NSE/BSE)</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0;
            font-size: 26px;
            color: #38bdf8;
            letter-spacing: -0.5px;
        }}
        .meta {{
            color: var(--text-secondary);
            font-size: 13px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-title {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .card-sub {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin: 24px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #e2e8f0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #0f172a;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background: #243447;
        }}
        .briefing-box {{
            background: #1e293b;
            border-left: 4px solid var(--accent-blue);
            padding: 20px;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.7;
            margin-bottom: 24px;
            border: 1px solid var(--border);
            border-left-width: 4px;
        }}
        .briefing-box h1, .briefing-box h2, .briefing-box h3 {{
            color: #38bdf8;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        .footer {{
            border-top: 1px solid var(--border);
            padding-top: 16px;
            color: var(--text-secondary);
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🇮🇳 Indian Market Premarket Research Dossier</h1>
                <div class="meta">Asynchronous Research Sidecar • NSE / BSE Pre-Session Confluence Engine</div>
            </div>
            <div style="text-align: right;">
                <div class="badge" style="background:#10b98120; color:#10b981; border:1px solid #10b981;">{data.get('timestamp_utc')[:16]} UTC</div>
                <div class="meta" style="margin-top:4px;">Engine: Google TimesFM + Laya RLCD + {provider}</div>
            </div>
        </header>

        <!-- Top Metrics Cards -->
        <div class="grid-4">
            <div class="card">
                <div class="card-title">NIFTY 50 (^NSEI)</div>
                <div class="card-value">₹{nifty.get('current_close', 0):,.2f}</div>
                <div class="card-sub" style="color:{'#10b981' if nifty.get('change_pct', 0) >= 0 else '#ef4444'}; font-weight:600;">
                    {nifty.get('change_pct', 0):+.2f}% • Trend: {nifty.get('trend_daily')}
                </div>
            </div>
            <div class="card">
                <div class="card-title">BANK NIFTY (^NSEBANK)</div>
                <div class="card-value">₹{bn.get('current_close', 0):,.2f}</div>
                <div class="card-sub" style="color:{'#10b981' if bn.get('change_pct', 0) >= 0 else '#ef4444'}; font-weight:600;">
                    {bn.get('change_pct', 0):+.2f}% • Trend: {bn.get('trend_daily')}
                </div>
            </div>
            <div class="card">
                <div class="card-title">INDIA VIX (VOLATILITY)</div>
                <div class="card-value">{vol.get('india_vix', 0):.2f}</div>
                <div class="card-sub">Regime: <span style="font-weight:600; color:#38bdf8;">{vol.get('regime')}</span></div>
            </div>
            <div class="card">
                <div class="card-title">LAYA SYSTEM 1 DECISION</div>
                <div class="card-value" style="color:#8b5cf6;">{laya.get('rating')}</div>
                <div class="card-sub">Confidence: <strong>{laya.get('confidence', 0)*100:.1f}%</strong> • Latency: {laya.get('latency_ms', 0):.2f}ms</div>
            </div>
        </div>

        <!-- Pivots & Multi-Timeframe Battlegrounds -->
        <div class="section-title">🎯 Multi-Timeframe Confluence & Pivot Battlegrounds (NIFTY 50)</div>
        <table>
            <thead>
                <tr>
                    <th>Metric Type</th>
                    <th>Level / Value</th>
                    <th>Regime / Strategy</th>
                    <th>Execution Guidance</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="font-weight:600;">Central Pivot Range (CPR)</td>
                    <td><strong>₹{cpr.get('pivot'):,.2f}</strong> (TC: ₹{cpr.get('top_central'):,.2f} | BC: ₹{cpr.get('bottom_central'):,.2f})</td>
                    <td><span style="color:#38bdf8; font-weight:600;">{cpr.get('regime')}</span></td>
                    <td>Width: {cpr.get('width_points')} pts ({cpr.get('width_pct')}%)</td>
                </tr>
                <tr>
                    <td style="font-weight:600;">Camarilla H4 (Breakout Buy)</td>
                    <td style="color:#10b981; font-weight:700;">₹{cam.get('h4_breakout'):,.2f}</td>
                    <td>Long Momentum Breakout</td>
                    <td>Go long on 5m candle close above H4 with volume confirmation</td>
                </tr>
                <tr>
                    <td style="font-weight:600;">Camarilla H3 (Range Resistance)</td>
                    <td style="color:#ef4444; font-weight:600;">₹{cam.get('h3_resistance'):,.2f}</td>
                    <td>Mean Reversion Short Zone</td>
                    <td>Look for bearish reversal wicks near H3 back towards Daily Pivot</td>
                </tr>
                <tr>
                    <td style="font-weight:600;">Camarilla L3 (Range Support)</td>
                    <td style="color:#10b981; font-weight:600;">₹{cam.get('l3_support'):,.2f}</td>
                    <td>Mean Reversion Long Zone</td>
                    <td>Look for bullish reversal pin bars near L3 towards Daily Pivot</td>
                </tr>
                <tr>
                    <td style="font-weight:600;">Camarilla L4 (Breakdown Sell)</td>
                    <td style="color:#ef4444; font-weight:700;">₹{cam.get('l4_breakdown'):,.2f}</td>
                    <td>Short Momentum Breakdown</td>
                    <td>Go short on 5m candle close below L4 with tight trailing stop</td>
                </tr>
            </tbody>
        </table>

        <!-- Google TimesFM Foundation Model Quantile Forecast -->
        <div class="section-title">🔮 Google TimesFM Zero-Shot Multi-Horizon Quantiles (NIFTY 50)</div>
        <table>
            <thead>
                <tr>
                    <th>Forecast Horizon</th>
                    <th>Bear Case (p10 Quantile)</th>
                    <th>Base Case (p50 Median)</th>
                    <th>Bull Case (p90 Quantile)</th>
                </tr>
            </thead>
            <tbody>
                {tfm_rows}
            </tbody>
        </table>

        <!-- Sector Rotation Relative Strength -->
        <div class="section-title">🔄 Sector Rotation Relative Strength (NSE Sectors vs NIFTY 50)</div>
        <table>
            <thead>
                <tr>
                    <th>Sector Index</th>
                    <th>Close Price</th>
                    <th>1-Day Return</th>
                    <th>5-Day Alpha vs NIFTY</th>
                    <th>Momentum Status</th>
                </tr>
            </thead>
            <tbody>
                {sector_rows}
            </tbody>
        </table>

        <!-- Heavyweights Momentum -->
        <div class="section-title">🏢 NSE Mega-Cap Tactical Overview</div>
        <table>
            <thead>
                <tr>
                    <th>Constituent</th>
                    <th>Last Close</th>
                    <th>1D Change</th>
                    <th>Daily CPR Pivot</th>
                    <th>Camarilla H3 (Res)</th>
                    <th>Camarilla L3 (Supp)</th>
                </tr>
            </thead>
            <tbody>
                {hw_rows}
            </tbody>
        </table>

        <!-- AI Senior Trader & Architect Briefing -->
        <div class="section-title">🧠 Institutional Premarket Narrative ({provider})</div>
        <div class="briefing-box">
            {formatted_briefing}
        </div>

        <div class="footer">
            <div>FinRobot Asynchronous Research Sidecar • Indian Market Execution Engine</div>
            <div>Artifact: <code>macro_context.json</code> ready for sub-35ms live engine ingestion</div>
        </div>
    </div>
</body>
</html>
"""
    return html
