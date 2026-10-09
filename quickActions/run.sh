#!/usr/bin/env bash
# ==============================================================================
# FinRobot Master QuickActions CLI
# Unified entry point for Premarket Analysis, Scrip Reports & Deliverables
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
    echo "Please create a virtual environment first: python3 -m venv venv"
    exit 1
fi

# Activate virtualenv
if [ -f "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" 2>/dev/null || true
fi

# ANSI Colors for terminal UI
BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RESET="\033[0m"

# Print banner
show_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                    FINROBOT QUICK ACTIONS CLI                        ║"
    echo "║        AI Decision Intelligence & Quantitative Research Platform     ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo -e "• ${BOLD}Virtualenv:${RESET}  $WORKSPACE_ROOT/$VENV_NAME"
    echo -e "• ${BOLD}Python:${RESET}      $($PYTHON_EXEC --version 2>&1)"
    echo -e "• ${BOLD}Workspace:${RESET}   $WORKSPACE_ROOT"
    echo ""
}

# Help message
show_help() {
    show_banner
    echo -e "${BOLD}USAGE:${RESET}"
    echo "  ./quickActions/run.sh [OPTION] [ARGUMENTS]"
    echo ""
    echo -e "${BOLD}DIRECT CLI FLAGS:${RESET}"
    echo "  --premarket, -p            Run Indian Market Premarket Analysis Sidecar"
    echo "  --scrip, -s [TICKER]       Run Scrip-Specific Equity Research Report (e.g. RELIANCE.NS, NVDA, COIN)"
    echo "  --benchmark, -b            Run Google TimesFM vs Baselines Benchmark Suite"
    echo "  --test-laya, -t            Run Laya RLCD Decision Engine test suite (pytest)"
    echo "  --web-app, -w              Launch FinRobot Equity Web App Server (http://127.0.0.1:8001)"
    echo "  --help, -h                 Show this help screen"
    echo ""
    echo -e "${BOLD}INTERACTIVE MODE:${RESET}"
    echo "  ./quickActions/run.sh      Launch interactive menu"
    echo ""
}

# Check command line flags
if [ $# -gt 0 ]; then
    case "$1" in
        --premarket|-p)
            bash "$SCRIPT_DIR/premarket.sh"
            exit 0
            ;;
        --scrip|-s)
            shift
            bash "$SCRIPT_DIR/scrip_report.sh" "$@"
            exit 0
            ;;
        --benchmark|-b)
            bash "$SCRIPT_DIR/benchmark_timesfm.sh"
            exit 0
            ;;
        --test-laya|-t)
            bash "$SCRIPT_DIR/test_laya.sh"
            exit 0
            ;;
        --web-app|-w)
            bash "$SCRIPT_DIR/web_app.sh"
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
fi

# Interactive Menu Loop
clear 2>/dev/null || true
show_banner

echo -e "${BOLD}Please select a FinRobot deliverable to execute:${RESET}"
echo ""
echo -e "  ${GREEN}${BOLD}[1]${RESET} 🇮🇳 ${BOLD}Indian Market Premarket Analysis${RESET}"
echo -e "      • Multi-timeframe confluence: NIFTY 50 & BANK NIFTY"
echo -e "      • Central Pivot Range (CPR) & Camarilla Levels (H4, H3, L3, L4)"
echo -e "      • TimesFM Zero-Shot Forecast + Laya System 1 Routing + Gemini Briefing"
echo -e "      • Outputs: ${CYAN}macro_context.json${RESET} & ${CYAN}Indian_Market_Premarket_Report.html${RESET}"
echo ""
echo -e "  ${BLUE}${BOLD}[2]${RESET} 📊 ${BOLD}Scrip-Specific Equity Research Report${RESET}"
echo -e "      • 2-Step deterministic ingestion + TimesFM + Laya + Gemini multi-agent synthesis"
echo -e "      • Supports US (NVDA, COIN, AAPL) and Indian Equities (RELIANCE.NS, TCS.NS, INFY.NS)"
echo -e "      • Outputs: ${CYAN}Professional_Equity_Report_<TICKER>.html${RESET} & charts"
echo ""
echo -e "  ${YELLOW}${BOLD}[3]${RESET} ⚡ ${BOLD}TimesFM vs Baselines Benchmark Suite${RESET}"
echo -e "      • Evaluates zero-shot TimesFM foundation model against ARIMA, Linear, ExpSmoothing"
echo -e "      • Outputs: ${CYAN}timesfm_comparison_report.html${RESET} & benchmark metrics"
echo ""
echo -e "  ${MAGENTA}${BOLD}[4]${RESET} 🧪 ${BOLD}Laya RLCD Decision Engine Test Suite${RESET}"
echo -e "      • Executes 15 pytest unit tests verifying sub-50ms non-autoregressive decision heads"
echo ""
echo -e "  ${CYAN}${BOLD}[5]${RESET} 🌐 ${BOLD}Launch FinRobot Equity Web Application${RESET}"
echo -e "      • Starts self-hosted FastAPI browser service on http://127.0.0.1:8001"
echo ""
echo -e "  ${BOLD}[6]${RESET} 📁 ${BOLD}Open Output Artifacts Folder${RESET}"
echo -e "  ${BOLD}[0]${RESET} 🚪 ${BOLD}Exit${RESET}"
echo ""

read -p "Enter selection [1-6, 0]: " -r CHOICE

case "$CHOICE" in
    1)
        bash "$SCRIPT_DIR/premarket.sh"
        ;;
    2)
        bash "$SCRIPT_DIR/scrip_report.sh"
        ;;
    3)
        bash "$SCRIPT_DIR/benchmark_timesfm.sh"
        ;;
    4)
        bash "$SCRIPT_DIR/test_laya.sh"
        ;;
    5)
        bash "$SCRIPT_DIR/web_app.sh"
        ;;
    6)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "$WORKSPACE_ROOT/output"
        else
            ls -la "$WORKSPACE_ROOT/output"
        fi
        ;;
    0|q|Q)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid selection."
        exit 1
        ;;
esac
