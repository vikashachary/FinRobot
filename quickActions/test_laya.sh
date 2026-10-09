#!/usr/bin/env bash
# ==============================================================================
# FinRobot QuickActions: Laya RLCD Decision Engine Test Suite
# Runs pytest on non-autoregressive ModernBERT typed decision heads (<50ms)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
echo "🧪 LAYA RLCD DECISION ENGINE UNIT & LATENCY TEST SUITE"
echo "======================================================================"
echo "• Environment: $WORKSPACE_ROOT/$VENV_NAME"
echo "• Test file: finrobot_equity/core/tests/test_laya_decision_engine.py"
echo "----------------------------------------------------------------------"

cd "$WORKSPACE_ROOT"
"$PYTHON_EXEC" -m unittest "$WORKSPACE_ROOT/finrobot_equity/core/tests/test_laya_decision_engine.py" -v

echo "----------------------------------------------------------------------"
echo "✅ All Laya System 1 Decision Router Tests Passed!"
echo "======================================================================"
