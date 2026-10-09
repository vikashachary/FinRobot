#!/usr/bin/env python
# coding: utf-8
"""
CLI Runner for Indian Market Premarket Analysis & Macro Research Sidecar.
Executes multi-timeframe confluence, TimesFM foundation forecasting, Laya System 1 routing,
and Google Gemini API (gemini-3.8-flash) executive synthesis.
Generates macro_context.json and an institutional HTML report.
"""

import os
import sys
import argparse

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_core_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)
_repo_root = os.path.abspath(os.path.join(_core_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_autogen_dir = os.path.join(_repo_root, "finrobot_autogen")
if _autogen_dir not in sys.path:
    sys.path.insert(0, _autogen_dir)

from modules.indian_premarket_sidecar import generate_indian_premarket_report


def main():
    parser = argparse.ArgumentParser(
        description="Run FinRobot Indian Market Premarket Research Sidecar (NSE/BSE) using Google Gemini API & TimesFM."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=os.path.join(_core_dir, "config", "config.ini"),
        help="Path to config.ini file (contains Gemini API key)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_repo_root, "output", "indian_market"),
        help="Output directory for macro_context.json and HTML report."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 FINROBOT INDIAN MARKET PREMARKET RESEARCH SIDECAR")
    print("=" * 70)
    print(f"Config: {args.config_file}")
    print(f"Output: {args.output_dir}")
    print("-" * 70)

    telemetry, json_path, html_path = generate_indian_premarket_report(
        config_path=args.config_file,
        output_dir=args.output_dir
    )

    print("-" * 70)
    print("✅ PREMARKET ANALYSIS COMPLETE!")
    print(f"📁 Macro Context JSON (Sidecar): {json_path}")
    print(f"📄 Executive HTML Report:       {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
