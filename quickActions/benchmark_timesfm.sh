#!/usr/bin/env bash
# ==============================================================================
# FinRobot QuickActions: Google TimesFM vs Baselines Benchmark Suite
# Evaluates zero-shot foundation model against statistical baselines (MAPE, latency)
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
echo "⚡ GOOGLE TIMESFM VS BASELINES BENCHMARK SUITE"
echo "======================================================================"
echo "• Environment: $WORKSPACE_ROOT/$VENV_NAME"
echo "• Benchmark Script: benchmarks/benchmark_timesfm_vs_baselines.py"
echo "----------------------------------------------------------------------"

cd "$WORKSPACE_ROOT"
"$PYTHON_EXEC" "$WORKSPACE_ROOT/benchmarks/benchmark_timesfm_vs_baselines.py"

BENCHMARK_HTML="$WORKSPACE_ROOT/output/benchmark/timesfm_comparison_report.html"
BENCHMARK_JSON="$WORKSPACE_ROOT/output/benchmark/timesfm_benchmark_results.json"

echo "----------------------------------------------------------------------"
echo "🎉 Benchmark Evaluation Completed!"
echo ""
echo "📁 Generated Deliverables:"
echo "  1. Benchmark Visual Comparison Report:"
echo "     👉 $BENCHMARK_HTML"
echo "  2. Benchmark Metrics JSON:"
echo "     👉 $BENCHMARK_JSON"
echo "----------------------------------------------------------------------"

if [[ "$OSTYPE" == "darwin"* ]] && [ -t 0 ]; then
    read -p "🌐 Open Benchmark report in default browser? [Y/n]: " -r OPEN_BROWSER
    OPEN_BROWSER=${OPEN_BROWSER:-Y}
    if [[ "$OPEN_BROWSER" =~ ^[Yy]$ ]]; then
        open "$BENCHMARK_HTML"
    fi
fi
