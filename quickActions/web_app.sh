#!/usr/bin/env bash
# ==============================================================================
# FinRobot QuickActions: Launch Equity Web Application
# Deploys self-hosted FastAPI + browser web dashboard at http://127.0.0.1:8001
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
echo "🌐 LAUNCHING FINROBOT EQUITY WEB APPLICATION"
echo "======================================================================"
echo "• Environment: $WORKSPACE_ROOT/$VENV_NAME"
echo "• Web App URL: http://127.0.0.1:8001"
echo "----------------------------------------------------------------------"

cd "$WORKSPACE_ROOT/finrobot_equity"
if [ -f "deploy.sh" ]; then
    ./deploy.sh start
else
    "$PYTHON_EXEC" run_web_app.py --port 8001
fi
