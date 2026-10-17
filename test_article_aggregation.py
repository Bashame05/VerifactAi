"""
test_article_aggregation.py
───────────────────────────
Unit tests for article-level risk score and tier aggregation.

Tests:
1. Low mean + 1 High-risk claim  → floored to "Medium", tier_floor_applied is True.
2. Medium mean + 2 High-risk claims → floored to "High", tier_floor_applied is True.
3. No High-risk claims (Low & Medium means) → unchanged, tier_floor_applied is False.
4. Single claim that is itself High → tier is "High", tier_floor_applied is False (no double application).
5. Empty claims list sanity check.
6. Large claim count independence (e.g. 20 claims with 1 High claim).

Run:
    python test_article_aggregation.py
"""

from __future__ import annotations

import unittest
from pipeline import aggregate_article_risk


class TestArticleRiskAggregation(unittest.TestCase):
    """Test suite for aggregate_article_risk floor rules and score fidelity."""

    def test_low_mean_with_single_high_claim_floored_to_medium(self):
        """Mean alone is Low (<34), but 1 claim is High -> final tier floored at Medium."""
        claims = [
            {"risk_score": 75.0, "risk_tier": "High"},
            {"risk_score": 10.0, "risk_tier": "Low"},
            {"risk_score": 10.0, "risk_tier": "Low"},
        ]
        # Mean = (75 + 10 + 10) / 3 = 31.67 -> normally "Low"
        result = aggregate_article_risk(claims)

        self.assertEqual(result["overall_risk_score"], 31.67)
        self.assertEqual(result["overall_risk_tier"], "Medium")
        self.assertTrue(result["tier_floor_applied"])
        self.assertIsNotNone(result["tier_override_reason"])
        self.assertIn("floored to Medium", result["tier_override_reason"])
        self.assertIn("1 high-risk claim", result["tier_override_reason"])

    def test_medium_mean_with_two_high_claims_floored_to_high(self):
        """Mean alone is Medium (34-66), but 2 claims are High -> final tier floored at High."""
        claims = [
            {"risk_score": 80.0, "risk_tier": "High"},
            {"risk_score": 70.0, "risk_tier": "High"},
            {"risk_score": 5.0, "risk_tier": "Low"},
            {"risk_score": 5.0, "risk_tier": "Low"},
        ]
        # Mean = (80 + 70 + 5 + 5) / 4 = 40.0 -> normally "Medium"
        result = aggregate_article_risk(claims)

        self.assertEqual(result["overall_risk_score"], 40.0)
        self.assertEqual(result["overall_risk_tier"], "High")
        self.assertTrue(result["tier_floor_applied"])
        self.assertIsNotNone(result["tier_override_reason"])
        self.assertIn("floored to High", result["tier_override_reason"])
        self.assertIn("2 high-risk claims", result["tier_override_reason"])

    def test_no_high_claims_behavior_unchanged(self):
        """When no claims are High, article tier strictly follows the arithmetic mean."""
        # Low case: mean 20.0 -> "Low"
        claims_low = [
            {"risk_score": 15.0, "risk_tier": "Low"},
            {"risk_score": 25.0, "risk_tier": "Low"},
        ]
        result_low = aggregate_article_risk(claims_low)
        self.assertEqual(result_low["overall_risk_score"], 20.0)
        self.assertEqual(result_low["overall_risk_tier"], "Low")
        self.assertFalse(result_low["tier_floor_applied"])
        self.assertIsNone(result_low["tier_override_reason"])

        # Medium case: mean 50.0 -> "Medium"
        claims_med = [
            {"risk_score": 45.0, "risk_tier": "Medium"},
            {"risk_score": 55.0, "risk_tier": "Medium"},
        ]
        result_med = aggregate_article_risk(claims_med)
        self.assertEqual(result_med["overall_risk_score"], 50.0)
        self.assertEqual(result_med["overall_risk_tier"], "Medium")
        self.assertFalse(result_med["tier_floor_applied"])
        self.assertIsNone(result_med["tier_override_reason"])

    def test_single_high_claim_article_no_double_floor(self):
        """An article with a single claim that is High has mean 85.0 -> already High, no floor needed."""
        claims = [
            {"risk_score": 85.0, "risk_tier": "High"},
        ]
        result = aggregate_article_risk(claims)

        self.assertEqual(result["overall_risk_score"], 85.0)
        self.assertEqual(result["overall_risk_tier"], "High")
        self.assertFalse(result["tier_floor_applied"])
        self.assertIsNone(result["tier_override_reason"])

    def test_empty_claims_list(self):
        """Sanity check: empty claims list yields 0.0 Low with no floor applied."""
        result = aggregate_article_risk([])
        self.assertEqual(result["overall_risk_score"], 0.0)
        self.assertEqual(result["overall_risk_tier"], "Low")
        self.assertFalse(result["tier_floor_applied"])
        self.assertIsNone(result["tier_override_reason"])

    def test_claim_count_independence(self):
        """Floor applies consistently across larger articles (e.g. 20 claims)."""
        # 1 High claim (80.0) + 19 Low claims (10.0 each)
        claims = [{"risk_score": 80.0, "risk_tier": "High"}] + [
            {"risk_score": 10.0, "risk_tier": "Low"} for _ in range(19)
        ]
        # Mean = (80 + 190) / 20 = 13.5 -> would be "Low", but 1 High claim floors to "Medium"
        result = aggregate_article_risk(claims)

        self.assertEqual(result["overall_risk_score"], 13.5)
        self.assertEqual(result["overall_risk_tier"], "Medium")
        self.assertTrue(result["tier_floor_applied"])
        self.assertIn("1 high-risk claim", result["tier_override_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
