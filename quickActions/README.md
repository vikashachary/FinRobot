# FinRobot QuickActions CLI Suite

A streamlined, unified automation suite for executing FinRobot's quantitative models, premarket analysis sidecars, and institutional equity research pipelines with automated virtual environment management (`.venv` / `venv`).

---

## 🚀 Quick Start

### 1. Interactive Menu Mode

Simply run the master launcher without arguments:

```bash
./quickActions/run.sh
```

This presents a terminal menu:
```text
╔══════════════════════════════════════════════════════════════════════╗
║                    FINROBOT QUICK ACTIONS CLI                        ║
║        AI Decision Intelligence & Quantitative Research Platform     ║
╚══════════════════════════════════════════════════════════════════════╝

• Virtualenv:  /Users/vikashachary/WIP/pythonCode/FinRobot/venv
• Python:      Python 3.12.8
• Workspace:   /Users/vikashachary/WIP/pythonCode/FinRobot

Please select a FinRobot deliverable to execute:

  [1] 🇮🇳 Indian Market Premarket Analysis
      • Multi-timeframe confluence: NIFTY 50 & BANK NIFTY
      • Central Pivot Range (CPR) & Camarilla Levels (H4, H3, L3, L4)
      • TimesFM Zero-Shot Forecast + Laya System 1 Routing + Gemini Briefing
      • Outputs: macro_context.json & Indian_Market_Premarket_Report.html

  [2] 📊 Scrip-Specific Equity Research Report
      • 2-Step deterministic ingestion + TimesFM + Laya + Gemini multi-agent synthesis
      • Supports US (NVDA, COIN, AAPL) and Indian Equities (RELIANCE.NS, TCS.NS, INFY.NS)
      • Outputs: Professional_Equity_Report_<TICKER>.html & charts

  [3] ⚡ TimesFM vs Baselines Benchmark Suite
      • Evaluates zero-shot TimesFM foundation model against ARIMA, Linear, ExpSmoothing
      • Outputs: timesfm_comparison_report.html & benchmark metrics

  [4] 🧪 Laya RLCD Decision Engine Test Suite
      • Executes 15 pytest unit tests verifying sub-50ms non-autoregressive decision heads

  [5] 🌐 Launch FinRobot Equity Web Application
      • Starts self-hosted FastAPI browser service on http://127.0.0.1:8001

  [6] 📁 Open Output Artifacts Folder
  [0] 🚪 Exit
```

---

## ⚡ Direct CLI One-Liners

You can invoke each offering directly via flags or dedicated scripts without navigating through the menu:

### 1. Premarket Research Sidecar (NSE / BSE)

Executes multi-timeframe confluence on NIFTY 50 & BANK NIFTY, CPR & Camarilla pivots, India VIX, Sector Relative Strength, Google TimesFM zero-shot quantile forecasting, Laya System 1 routing, and Google Gemini AI executive briefing:

```bash
# Using master launcher
./quickActions/run.sh --premarket

# Or using dedicated script
./quickActions/premarket.sh
```

**Artifacts Produced:**
- **Macro Context Sidecar JSON:** `output/indian_market/macro_context.json` *(for sub-35ms live engine boot ingestion)*
- **Executive Premarket HTML Report:** `output/indian_market/Indian_Market_Premarket_Report.html`

---

### 2. Scrip-Specific Institutional Equity Research Report

Runs deterministic financial statement ingestion, Google TimesFM empirical forecasts, Laya RLCD System 1 decision engine, and multi-agent AI narrative writing via Gemini API:

```bash
# Analyze Indian Mega-Caps (NSE)
./quickActions/run.sh --scrip RELIANCE.NS
./quickActions/run.sh --scrip TCS.NS
./quickActions/run.sh --scrip INFY.NS

# Analyze US Tech & Equities
./quickActions/run.sh --scrip NVDA
./quickActions/run.sh --scrip COIN
./quickActions/run.sh --scrip AAPL

# Custom ticker with explicit company name and peer basket
./quickActions/scrip_report.sh COIN "Coinbase Global, Inc." "HOOD SQ"
```

**Artifacts Produced:**
- **Professional Paged HTML Report:** `output/<TICKER>/report/Professional_Equity_Report_<TICKER>.html`
- **Combined Single-Page Report:** `output/<TICKER>/report/Combined_Equity_Report_<TICKER>.html`
- **Charts & Multiples:** `output/<TICKER>/report/*_chart.png`
- **Empirical Projections & Decision Dossier:** `output/<TICKER>/analysis/timesfm_projections.json` & `laya_decision.json`

---

### 3. Google TimesFM vs Baselines Benchmark Suite

Evaluates Google TimesFM-2.5-200M zero-shot foundation model against classical quantitative baselines (ARIMA, Linear Regression, Exponential Smoothing) across MAPE, Directional Accuracy, and Latency:

```bash
# Using master launcher
./quickActions/run.sh --benchmark

# Or using dedicated script
./quickActions/benchmark_timesfm.sh
```

**Artifacts Produced:**
- **Visual Comparison Report:** `output/benchmark/timesfm_comparison_report.html`
- **Benchmark Metrics JSON:** `output/benchmark/timesfm_benchmark_results.json`

---

### 4. Laya RLCD Decision Engine Test Suite

Executes the unit and latency test suite verifying sub-50ms ModernBERT typed decision heads, calibrated confidence scores, and dual-system escalation routing:

```bash
# Using master launcher
./quickActions/run.sh --test-laya

# Or using dedicated script
./quickActions/test_laya.sh
```

---

### 5. Launch FinRobot Equity Web App

Starts the self-hosted FastAPI browser service at `http://127.0.0.1:8001`:

```bash
# Using master launcher
./quickActions/run.sh --web-app

# Or using dedicated script
./quickActions/web_app.sh
```

---

## 🛠 Script Directory Overview

| Script | Executable Command | Purpose |
| :--- | :--- | :--- |
| [`run.sh`](./run.sh) | `./quickActions/run.sh` | Master interactive CLI menu & flag dispatcher |
| [`premarket.sh`](./premarket.sh) | `./quickActions/premarket.sh` | Premarket Research Sidecar runner (NSE/BSE) |
| [`scrip_report.sh`](./scrip_report.sh) | `./quickActions/scrip_report.sh [TICKER]` | End-to-end 2-step Equity Research Report pipeline |
| [`benchmark_timesfm.sh`](./benchmark_timesfm.sh) | `./quickActions/benchmark_timesfm.sh` | TimesFM foundation model benchmark suite |
| [`test_laya.sh`](./test_laya.sh) | `./quickActions/test_laya.sh` | Laya Decision Engine pytest suite |
| [`web_app.sh`](./web_app.sh) | `./quickActions/web_app.sh` | FinRobot Web Application server launcher |

---

## 🔒 Virtual Environment & Security

- **Auto-Detection:** Every script automatically resolves and activates `.venv` or `venv` located at the workspace root.
- **Private Configuration:** API keys configured in `finrobot_equity/core/config/config.ini` are strictly isolated and verified gitignored.
