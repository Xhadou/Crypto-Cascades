"""Tests for AICc small-sample correction."""
import pytest
import numpy as np


class TestAICc:
    def test_aicc_correction_applied(self):
        """AICc should add 2k(k+1)/(n-k-1) to AIC."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        tester = HypothesisTester()
        aic = tester._compute_aic(sse=10.0, n_params=3, n_obs=20)
        aicc = tester._compute_aicc(sse=10.0, n_params=3, n_obs=20)
        correction = 2 * 3 * 4 / (20 - 3 - 1)
        assert abs(aicc - (aic + correction)) < 1e-10

    def test_aicc_converges_to_aic_for_large_n(self):
        """AICc correction vanishes as n grows."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        tester = HypothesisTester()
        aic = tester._compute_aic(sse=100.0, n_params=3, n_obs=10000)
        aicc = tester._compute_aicc(sse=100.0, n_params=3, n_obs=10000)
        assert abs(aicc - aic) < 0.01

    def test_aicc_returns_inf_when_n_too_small(self):
        """AICc should return inf when n <= k+1."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        tester = HypothesisTester()
        aicc = tester._compute_aicc(sse=10.0, n_params=3, n_obs=4)
        assert aicc == np.inf

    def test_estimator_uses_aicc(self):
        """EstimationResult should have aicc field."""
        from src.estimation.estimator import EstimationResult
        r = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1)
        assert hasattr(r, 'aicc')

    def test_model_comparison_returns_aicc(self):
        """ModelComparison.compute_information_criteria should return aicc."""
        from src.estimation.model_comparison import ModelComparison
        import pandas as pd
        mc = ModelComparison()
        obs = pd.DataFrame({'I_frac': np.random.rand(50)})
        pred = pd.DataFrame({'I_frac': np.random.rand(50)})
        result = mc.compute_information_criteria(obs, pred, n_params=3)
        assert 'aicc' in result

    def test_model_comparison_aicc_equals_formula(self):
        """ModelComparison AICc should match AIC + 2k(k+1)/(n-k-1)."""
        from src.estimation.model_comparison import ModelComparison
        import pandas as pd
        mc = ModelComparison()
        np.random.seed(42)
        obs = pd.DataFrame({'I_frac': np.random.rand(30)})
        pred = pd.DataFrame({'I_frac': np.random.rand(30)})
        result = mc.compute_information_criteria(obs, pred, n_params=3)
        n = 30
        k = 3
        expected_correction = 2 * k * (k + 1) / (n - k - 1)
        assert abs(result['aicc'] - (result['aic'] + expected_correction)) < 1e-10
