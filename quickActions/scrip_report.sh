#!/usr/bin/env bash
# ==============================================================================
# FinRobot QuickActions: Scrip-Specific Equity Research Report
# Runs Step 1 (Financial Ingestion + TimesFM + Laya + Gemini Text) + Step 2 (HTML Report)
# ==============================================================================

set -e

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve Python environment (.venv or venv)
if [ -x "$WORKSPACE_ROOT/venv/bin/python" ]; then
    PYTHON_EXEC="$WORKSPACE_ROOT/venv/bin/python"
    VENV_NAME="venv"
elif [ -x "$WORKSPACE_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$WORKSPACE_ROOT/.venv/bin/python"
    VENV_NAME=".venv"
else
    echo "❌ Error: Virtual environment not found in $WORKSPACE_ROOT/.venv or $WORKSPACE_ROOT/venv"
    exit 1
fi

if [ -f "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" 2>/dev/null || true
fi

echo "======================================================================"
echo "📊 FINROBOT SCRIP-SPECIFIC EQUITY RESEARCH REPORT"
echo "======================================================================"

# Determine Ticker from argument or prompt
TICKER="$1"
if [ -z "$TICKER" ]; then
    if [ -t 0 ]; then
        echo "Popular options: NVDA, COIN, AAPL, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"
        read -p "Enter Stock Ticker [Default: RELIANCE.NS]: " -r INPUT_TICKER
        TICKER=${INPUT_TICKER:-RELIANCE.NS}
    else
        TICKER="RELIANCE.NS"
    fi
fi
TICKER="$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')"

# Default company names & peers
COMPANY_NAME="$2"
PEERS="$3"
if [ -z "$COMPANY_NAME" ]; then
    case "$TICKER" in
        "RELIANCE.NS") COMPANY_NAME="Reliance Industries Limited"; PEERS="TCS.NS INFY.NS" ;;
        "TCS.NS")       COMPANY_NAME="Tata Consultancy Services"; PEERS="INFY.NS WIPRO.NS" ;;
        "INFY.NS")      COMPANY_NAME="Infosys Limited";           PEERS="TCS.NS WIPRO.NS" ;;
        "HDFCBANK.NS")  COMPANY_NAME="HDFC Bank Limited";         PEERS="ICICIBANK.NS SBIN.NS" ;;
        "NVDA")         COMPANY_NAME="NVIDIA Corporation";        PEERS="AMD INTC" ;;
        "COIN")         COMPANY_NAME="Coinbase Global, Inc.";     PEERS="HOOD SQ" ;;
        "AAPL")         COMPANY_NAME="Apple Inc.";                PEERS="MSFT GOOGL" ;;
        "MSFT")         COMPANY_NAME="Microsoft Corporation";     PEERS="AAPL GOOGL" ;;
        *)              COMPANY_NAME="$TICKER Equity";            PEERS="" ;;
    esac
fi

echo "• Scrip / Ticker: $TICKER"
echo "• Company Name:   $COMPANY_NAME"
echo "• Peer Tickers:   ${PEERS:-None}"
echo "• Python Env:     $WORKSPACE_ROOT/$VENV_NAME"
echo "----------------------------------------------------------------------"

ANALYSIS_DIR="$WORKSPACE_ROOT/output/$TICKER/analysis"
REPORT_DIR="$WORKSPACE_ROOT/output/$TICKER/report"
CONFIG_FILE="$WORKSPACE_ROOT/finrobot_equity/core/config/config.ini"

cd "$WORKSPACE_ROOT"

echo "⏳ STEP 1/2: Generating financial metrics, TimesFM forecast & narrative..."
# Build peer args
PEER_ARGS=()
if [ -n "$PEERS" ]; then
    # shellcheck disable=SC2206
    PEER_ARGS=(--peer-tickers $PEERS)
fi

"$PYTHON_EXEC" "$WORKSPACE_ROOT/finrobot_equity/core/src/generate_financial_analysis.py" \
    --company-ticker "$TICKER" \
    --company-name "$COMPANY_NAME" \
    --config-file "$CONFIG_FILE" \
    "${PEER_ARGS[@]}" \
    --generate-text-sections \
    --use-timesfm \
    --enable-laya \
    --parallel-text-workers 1

echo "----------------------------------------------------------------------"
echo "⏳ STEP 2/2: Rendering charts, multiples & professional HTML reports..."

"$PYTHON_EXEC" "$WORKSPACE_ROOT/finrobot_equity/core/src/create_equity_report.py" \
    --company-ticker "$TICKER" \
    --company-name "$COMPANY_NAME" \
    --analysis-csv "$ANALYSIS_DIR/financial_metrics_and_forecasts.csv" \
    --ratios-csv "$ANALYSIS_DIR/ratios_raw_data.csv" \
    --peer-ebitda-csv "$ANALYSIS_DIR/peer_ebitda_comparison.csv" \
    --peer-ev-ebitda-csv "$ANALYSIS_DIR/peer_ev_ebitda_comparison.csv" \
    --tagline-file "$ANALYSIS_DIR/tagline.txt" \
    --company-overview-file "$ANALYSIS_DIR/company_overview.txt" \
    --investment-overview-file "$ANALYSIS_DIR/investment_overview.txt" \
    --valuation-overview-file "$ANALYSIS_DIR/valuation_overview.txt" \
    --risks-file "$ANALYSIS_DIR/risks.txt" \
    --competitor-analysis-file "$ANALYSIS_DIR/competitor_analysis.txt" \
    --major-takeaways-file "$ANALYSIS_DIR/major_takeaways.txt" \
    --news-summary-file "$ANALYSIS_DIR/news_summary.txt" \
    --config-file "$CONFIG_FILE"

PROFESSIONAL_HTML="$REPORT_DIR/Professional_Equity_Report_$TICKER.html"
COMBINED_HTML="$REPORT_DIR/Combined_Equity_Report_$TICKER.html"

echo "======================================================================"
echo "🎉 Scrip Research Report Generated Successfully for $TICKER!"
echo ""
echo "📁 Generated Deliverables:"
echo "  1. Professional HTML Report (Paged / PDF Structure):"
echo "     👉 $PROFESSIONAL_HTML"
echo "  2. Combined Single-Page HTML Report:"
echo "     👉 $COMBINED_HTML"
echo "  3. Analysis Directory (CSVs, TimesFM JSON, Laya System 1 Decision):"
echo "     👉 $ANALYSIS_DIR"
echo "======================================================================"

if [[ "$OSTYPE" == "darwin"* ]] && [ -t 0 ]; then
    read -p "🌐 Open Professional HTML report in default browser? [Y/n]: " -r OPEN_BROWSER
    OPEN_BROWSER=${OPEN_BROWSER:-Y}
    if [[ "$OPEN_BROWSER" =~ ^[Yy]$ ]]; then
        open "$PROFESSIONAL_HTML"
    fi
fi
