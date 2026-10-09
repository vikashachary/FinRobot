#!/usr/bin/env bash
# ==============================================================================
# FinRobot QuickActions: Indian Market Premarket Research Sidecar
# Runs multi-timeframe confluence, TimesFM forecast, Laya routing & Gemini narrative
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

# Optional activation
if [ -f "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE_ROOT/$VENV_NAME/bin/activate" 2>/dev/null || true
fi

echo "======================================================================"
echo "🇮🇳 FINROBOT PREMARKET RESEARCH SIDECAR (NSE / BSE)"
echo "======================================================================"
echo "• Environment: $WORKSPACE_ROOT/$VENV_NAME (Python: $($PYTHON_EXEC --version))"
echo "• Sidecar Script: finrobot_equity/core/src/run_indian_premarket.py"
echo "• Models: Google TimesFM (Foundation Forecast) + Laya RLCD + Gemini AI"
echo "----------------------------------------------------------------------"

# Execute premarket analysis
cd "$WORKSPACE_ROOT"
"$PYTHON_EXEC" "$WORKSPACE_ROOT/finrobot_equity/core/src/run_indian_premarket.py" \
    --config-file "$WORKSPACE_ROOT/finrobot_equity/core/config/config.ini" \
    --output-dir "$WORKSPACE_ROOT/output/indian_market"

HTML_REPORT="$WORKSPACE_ROOT/output/indian_market/Indian_Market_Premarket_Report.html"
MACRO_JSON="$WORKSPACE_ROOT/output/indian_market/macro_context.json"

echo "----------------------------------------------------------------------"
echo "🎉 Premarket Analysis Completed Successfully!"
echo ""
echo "📁 Generated Deliverables:"
echo "  1. Macro Context Sidecar JSON:"
echo "     👉 $MACRO_JSON"
echo "  2. Executive Premarket HTML Report:"
echo "     👉 $HTML_REPORT"
echo "----------------------------------------------------------------------"

# If running on macOS and interactive, offer to open the HTML report
if [[ "$OSTYPE" == "darwin"* ]] && [ -t 0 ]; then
    read -p "🌐 Open HTML report in default browser? [Y/n]: " -r OPEN_BROWSER
    OPEN_BROWSER=${OPEN_BROWSER:-Y}
    if [[ "$OPEN_BROWSER" =~ ^[Yy]$ ]]; then
        open "$HTML_REPORT"
    fi
fi
