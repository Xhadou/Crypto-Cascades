"""
Tests for Epidemic Early Warning Indicators

Tests the EpidemicEarlyWarning class which computes variance,
lag-1 autocorrelation, and skewness as critical slowing down indicators.
"""

import pytest
import numpy as np
import pandas as pd

from src.network_analysis.early_warning import EpidemicEarlyWarning


@pytest.fixture
def ews():
    """Create an EpidemicEarlyWarning instance."""
    return EpidemicEarlyWarning()


@pytest.fixture
def synthetic_seir_data():
    """Create synthetic SEIR data with a known transition point.

    The infected fraction ramps up around t=30 to simulate a
    tipping point where EWS should trigger alarms.
    """
    np.random.seed(42)
    t = np.arange(100)
    # Sigmoid rise around t=30 with small noise
    I_frac = 0.01 + 0.3 / (1 + np.exp(-0.3 * (t - 30)))
    I_frac += np.random.normal(0, 0.005, len(t))
    I_frac = np.clip(I_frac, 0, 1)
    return pd.DataFrame({'t': t, 'I_frac': I_frac})


class TestComputeEWSIndicators:
    """Tests for compute_ews_indicators method."""

    def test_returns_dataframe(self, ews, synthetic_seir_data):
        """Test that output is a DataFrame with expected columns."""
        result = ews.compute_ews_indicators(synthetic_seir_data)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'t', 'variance', 'autocorrelation', 'skewness', 'alarm'}

    def test_output_length(self, ews, synthetic_seir_data):
        """Test that output length matches input minus window."""
        window_size = 7
        result = ews.compute_ews_indicators(synthetic_seir_data, window_size=window_size)
        expected_len = len(synthetic_seir_data) - window_size
        assert len(result) == expected_len

    def test_variance_increases_during_transition(self, ews, synthetic_seir_data):
        """Test that variance increases during the transition phase."""
        result = ews.compute_ews_indicators(synthetic_seir_data, window_size=7)
        # Compare early vs late variance
        early_var = result.loc[result['t'] < 20, 'variance'].mean()
        late_var = result.loc[result['t'] > 40, 'variance'].mean()
        # Variance should be higher during/after transition
        assert late_var > early_var

    def test_alarm_flags_present(self, ews, synthetic_seir_data):
        """Test that alarm flags are boolean values."""
        result = ews.compute_ews_indicators(synthetic_seir_data)
        assert result['alarm'].dtype == bool

    def test_empty_when_no_i_frac(self, ews):
        """Test returns empty DataFrame when I_frac column is missing."""
        df = pd.DataFrame({'t': [1, 2, 3], 'S_frac': [0.9, 0.8, 0.7]})
        result = ews.compute_ews_indicators(df)
        assert result.empty

    def test_empty_when_series_too_short(self, ews):
        """Test returns empty DataFrame when time series is shorter than window."""
        df = pd.DataFrame({'t': [1, 2, 3], 'I_frac': [0.01, 0.02, 0.03]})
        result = ews.compute_ews_indicators(df, window_size=10)
        assert result.empty

    def test_custom_thresholds(self, ews, synthetic_seir_data):
        """Test with custom alarm thresholds."""
        result = ews.compute_ews_indicators(
            synthetic_seir_data,
            alarm_autocorr_threshold=0.9,
            alarm_variance_factor=1.0
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestDetectTransitionPoint:
    """Tests for detect_transition_point method."""

    def test_returns_int_or_none(self, ews, synthetic_seir_data):
        """Test that method returns an int or None."""
        ews_df = ews.compute_ews_indicators(synthetic_seir_data)
        result = ews.detect_transition_point(ews_df)
        assert result is None or isinstance(result, int)

    def test_returns_none_for_empty_df(self, ews):
        """Test returns None for empty DataFrame."""
        result = ews.detect_transition_point(pd.DataFrame())
        assert result is None

    def test_returns_none_for_no_alarms(self, ews):
        """Test returns None when no alarms are present."""
        df = pd.DataFrame({
            't': range(10),
            'alarm': [False] * 10
        })
        result = ews.detect_transition_point(df)
        assert result is None

    def test_detects_consecutive_alarms(self, ews):
        """Test detection of consecutive alarm sequences."""
        df = pd.DataFrame({
            't': range(10),
            'alarm': [False, False, False, True, True, True, True, False, False, False]
        })
        result = ews.detect_transition_point(df, consecutive_alarms=3)
        assert result == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
