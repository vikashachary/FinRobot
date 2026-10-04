#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for LayaDecisionEngine (System 1 RLCD Decision Model).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.laya_decision_engine import (
    LayaDecisionEngine,
    ChoiceDecision,
    ScoreDecision,
    NoulDecision
)


class TestLayaDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LayaDecisionEngine(confidence_escalation_threshold=0.70)

    def test_investment_rating_choice(self):
        """Test sub-50ms investment rating choice head."""
        decision = self.engine.predict_investment_rating(
            company_ticker="COIN",
            forecast_growth={"2026E": 0.18, "2027E": 0.12},
            sentiment_score=0.75
        )
        self.assertIsInstance(decision, ChoiceDecision)
        self.assertIn(decision.label, ["Strong Buy", "Buy", "Hold", "Underperform", "Sell"])
        self.assertGreater(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertIn(decision.system_routing, ["FAST_PATH", "ESCALATE_TO_SYSTEM_2"])
        # Fast latency test (<50ms target)
        self.assertLess(decision.latency_ms, 50.0)
        # Sum of probabilities should be approx 1.0
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0, places=2)

    def test_risk_severity_score(self):
        """Test Score decision head for risk triage."""
        risks = ["SEC regulatory litigation", "Extreme price volatility", "Halving volume drop"]
        decision = self.engine.evaluate_risk_severity(risks, volatility_flag=True)
        self.assertIsInstance(decision, ScoreDecision)
        self.assertIn(decision.score, [1, 2, 3, 4, 5])
        self.assertEqual(decision.scale_max, 5)
        self.assertGreater(decision.confidence, 0.0)
        self.assertLess(decision.latency_ms, 50.0)

    def test_materiality_noul(self):
        """Test Noul (binary calibrated decision) head."""
        stmt = "The company faces an active SEC investigation and restructuring."
        decision = self.engine.evaluate_materiality(stmt)
        self.assertIsInstance(decision, NoulDecision)
        self.assertTrue(decision.decision)
        self.assertGreaterEqual(decision.probability_yes, 0.5)

    def test_system1_dossier_generation(self):
        """Test generation of the full sub-50ms System 1 dossier."""
        dossier = self.engine.generate_system1_dossier(
            company_ticker="COIN",
            company_name="Coinbase Global, Inc.",
            forecast_growth={"2026E": -0.17, "2027E": -0.10, "2028E": 0.11},
            timesfm_metadata={"projections": {"Revenue": [6.2e9, 5.5e9, 6.1e9]}}
        )
        self.assertEqual(dossier["company_ticker"], "COIN")
        self.assertIn("investment_rating", dossier)
        self.assertIn("risk_triage", dossier)
        self.assertLess(dossier["total_latency_ms"], 50.0)
        self.assertIn(dossier["system_routing"], ["FAST_PATH", "ESCALATE_TO_SYSTEM_2"])


if __name__ == "__main__":
    unittest.main()
