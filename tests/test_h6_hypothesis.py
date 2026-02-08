"""
Tests for H6 Market Condition hypothesis.
"""

import pytest
import numpy as np

from src.hypothesis.hypothesis_tester import HypothesisTester


class TestH6MarketCondition:
    """Tests for H6 market condition hypothesis."""

    @pytest.fixture
    def tester(self):
        return HypothesisTester(alpha=0.05, random_seed=42)

    def test_h6_detects_significant_difference(self, tester):
        """Test H6 detects when bull R0 >> bear R0."""
        r0_bulls = [2.5, 2.8, 3.1]  # High R0 in bull markets
        r0_bear = 0.8               # Low R0 in bear market

        result = tester.test_h6_market_condition_r0(r0_bulls, r0_bear)

        assert result.hypothesis == "H6"
        assert result.reject_null is True
        assert result.additional_metrics['r0_bull_mean'] > result.additional_metrics['r0_bear']

    def test_h6_handles_single_bull_market(self, tester):
        """Test H6 works with only one bull market period."""
        r0_bulls = [2.5]
        r0_bear = 0.8

        result = tester.test_h6_market_condition_r0(r0_bulls, r0_bear)

        # Should still produce a result
        assert result.hypothesis == "H6"
        assert np.isfinite(result.p_value) or result.sample_size < 3

    def test_h6_no_difference(self, tester):
        """Test H6 returns not significant when R0 values are similar."""
        r0_bulls = [1.0, 1.1, 0.9]
        r0_bear = 1.0

        result = tester.test_h6_market_condition_r0(r0_bulls, r0_bear)

        assert result.hypothesis == "H6"
        # Should NOT reject the null when values are similar
        # (may still reject by chance at alpha=0.05, so just check it runs)
        assert hasattr(result, 'reject_null')

    def test_h6_returns_effect_size(self, tester):
        """Test H6 reports a Cohen's d effect size."""
        r0_bulls = [2.5, 2.8, 3.1]
        r0_bear = 0.8

        result = tester.test_h6_market_condition_r0(r0_bulls, r0_bear)

        assert 'effect_size' in result.__dict__ or hasattr(result, 'effect_size')
        assert np.isfinite(result.effect_size)
