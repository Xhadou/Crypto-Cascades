"""
Unit Tests for Hypothesis Testing Module

Tests statistical hypothesis testing functionality including:
- H1: FOMO follows epidemic dynamics
- H2: Network amplification
- H3: FGI correlation
- H4: Centrality effects
- H5: Community clustering
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.hypothesis.hypothesis_tester import HypothesisTester, HypothesisResult
from src.estimation.estimator import EstimationResult


class TestHypothesisResult:
    """Tests for HypothesisResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a hypothesis result."""
        result = HypothesisResult(
            hypothesis="H1",
            description="Test hypothesis",
            test_statistic=2.5,
            p_value=0.01,
            effect_size=0.6,
            confidence_interval=(0.4, 0.8),
            reject_null=True,
            alpha=0.05,
            sample_size=100,
            additional_metrics={}
        )
        
        assert result.hypothesis == "H1"
        assert result.reject_null == True
        assert result.p_value == 0.01
    
    def test_result_string_representation(self):
        """Test string representation includes key info."""
        result = HypothesisResult(
            hypothesis="H1",
            description="Test",
            test_statistic=2.5,
            p_value=0.01,
            effect_size=0.6,
            confidence_interval=(0.4, 0.8),
            reject_null=True,
            alpha=0.05,
            sample_size=100,
            additional_metrics={}
        )
        
        str_repr = str(result)
        assert "H1" in str_repr
        assert "REJECTED" in str_repr
        assert "0.01" in str_repr


class TestHypothesisTester:
    """Tests for HypothesisTester class."""
    
    @pytest.fixture
    def tester(self):
        """Create a hypothesis tester."""
        return HypothesisTester(alpha=0.05, random_seed=42)
    
    @pytest.fixture
    def test_graph(self):
        """Create a test graph."""
        return nx.barabasi_albert_graph(500, 3, seed=42)
    
    @pytest.fixture
    def test_state_history(self):
        """Create test state history."""
        return pd.DataFrame({
            't': list(range(100)) * 5,
            'I': [np.random.poisson(50 + t * 0.5) for t in range(100)] * 5
        })
    
    @pytest.fixture
    def test_fgi(self):
        """Create test FGI values."""
        return np.random.uniform(30, 70, 100)
    
    @pytest.fixture
    def test_params(self):
        """Create test estimation result."""
        return EstimationResult(
            beta=0.3, sigma=0.2, gamma=0.1,
            r_squared=0.85, loss=0.001
        )
    
    def test_tester_creation(self, tester):
        """Test tester creation."""
        assert tester.alpha == 0.05
    
    def test_h1_returns_result(self, tester, test_state_history, test_params):
        """Test H1 returns HypothesisResult."""
        result = tester.test_h1_epidemic_dynamics(
            test_state_history, test_params, None
        )
        assert isinstance(result, HypothesisResult)
        assert result.hypothesis == "H1"
    
    def test_h2_returns_result(self, tester, test_graph, test_params):
        """Test H2 returns HypothesisResult."""
        result = tester.test_h2_network_amplification(test_graph, test_params)
        assert isinstance(result, HypothesisResult)
        assert result.hypothesis == "H2"
    
    def test_h2_detects_amplification(self, tester, test_params):
        """Test H2 detects network amplification in scale-free graph."""
        G = nx.barabasi_albert_graph(1000, 3, seed=42)
        result = tester.test_h2_network_amplification(G, test_params)
        
        # Scale-free networks should show amplification (network factor > 1)
        assert result.additional_metrics['network_factor'] > 1
    
    def test_h3_returns_result(self, tester, test_state_history, test_fgi):
        """Test H3 returns HypothesisResult."""
        result = tester.test_h3_fgi_correlation(test_state_history, test_fgi)
        assert isinstance(result, HypothesisResult)
        assert result.hypothesis == "H3"
    
    def test_h3_detects_correlation(self, tester):
        """Test H3 detects correlation when present."""
        # Create data with positive correlation
        fgi = np.linspace(30, 70, 100)
        infections = fgi * 2 + np.random.normal(0, 5, 100)  # Correlated
        
        state_history = pd.DataFrame({
            't': range(100),
            'I': infections
        })
        
        result = tester.test_h3_fgi_correlation(state_history, fgi)
        
        # Should detect positive correlation
        assert result.test_statistic > 0
    
    def test_h4_returns_result(self, tester, test_graph, test_state_history):
        """Test H4 returns HypothesisResult."""
        result = tester.test_h4_centrality_effect(test_graph, test_state_history)
        assert isinstance(result, HypothesisResult)
        assert result.hypothesis == "H4"
    
    def test_h5_returns_result(self, tester, test_graph, test_state_history):
        """Test H5 returns HypothesisResult."""
        result = tester.test_h5_community_clustering(test_graph, test_state_history)
        assert isinstance(result, HypothesisResult)
        assert result.hypothesis == "H5"
    
    def test_h5_detects_community_structure(self, tester):
        """Test H5 detects community clustering."""
        # Create a graph with clear community structure
        G1 = nx.complete_graph(50)
        G2 = nx.complete_graph(50)
        G = nx.disjoint_union(G1, G2)
        
        # Add a few inter-community edges
        G.add_edge(0, 50)
        G.add_edge(25, 75)
        
        state_history = pd.DataFrame({'t': range(10), 'I': range(10)})
        
        result = tester.test_h5_community_clustering(G, state_history)
        
        # Should detect strong within-community connectivity
        assert result.additional_metrics['n_communities'] >= 2
    
    def test_test_all_returns_dict(
        self, tester, test_graph, test_state_history, test_fgi, test_params
    ):
        """Test that test_all returns dict of results."""
        results = tester.test_all(
            test_graph, test_state_history, test_fgi, test_params
        )
        
        assert isinstance(results, dict)
        assert all(h in results for h in ['H1', 'H2', 'H3', 'H4', 'H5'])
    
    def test_generate_report(self, tester, test_graph, test_state_history, test_fgi, test_params):
        """Test report generation."""
        results = tester.test_all(
            test_graph, test_state_history, test_fgi, test_params
        )
        
        report = tester.generate_report(results)
        
        assert isinstance(report, str)
        assert "HYPOTHESIS TESTING REPORT" in report
        assert "SUMMARY" in report


class TestStatisticalValidity:
    """Tests for statistical validity of hypothesis tests."""
    
    @pytest.fixture
    def tester(self):
        return HypothesisTester(alpha=0.05, random_seed=42)
    
    def test_pvalue_in_valid_range(self, tester):
        """Test that p-values are in [0, 1]."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.8)
        
        result = tester.test_h2_network_amplification(G, params)
        
        assert 0 <= result.p_value <= 1
    
    def test_confidence_interval_ordering(self, tester):
        """Test that CI lower <= CI upper."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.8)
        
        result = tester.test_h2_network_amplification(G, params)
        
        assert result.confidence_interval[0] <= result.confidence_interval[1]
    
    def test_reject_null_consistency(self, tester):
        """Test that reject_null is consistent with p-value and alpha."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.8)
        
        result = tester.test_h2_network_amplification(G, params)
        
        if result.p_value < tester.alpha:
            assert result.reject_null == True
        else:
            assert result.reject_null == False


class TestH2NullModel:
    """Tests for H2 null model comparison (replacing trivial >1 test)."""

    @pytest.fixture
    def tester(self):
        return HypothesisTester(alpha=0.05, random_seed=42)

    @pytest.fixture
    def test_params(self):
        return EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.8)

    def test_h2_compares_against_null_models(self, tester, test_params):
        """H2 should test network factor against null models, not against 1."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        result = tester.test_h2_network_amplification(G, test_params)
        # Result should contain null model comparison info
        assert 'null_model_factors' in result.additional_metrics
        assert 'observed_vs_null_p' in result.additional_metrics

    def test_h2_null_model_p_value_used(self, tester, test_params):
        """H2 p-value should come from null model comparison, not bootstrap >1."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        result = tester.test_h2_network_amplification(G, test_params)
        # The p-value should equal the null model comparison p-value
        assert result.p_value == result.additional_metrics['observed_vs_null_p']

    def test_h2_regular_graph_not_significant(self, tester, test_params):
        """A k-regular graph should NOT show significant amplification vs
        configuration-model null networks.

        A regular graph has uniform degree distribution, so the configuration
        model null (which preserves the degree sequence) should produce very
        similar network factors. The test should not reject.
        """
        G = nx.random_regular_graph(6, 200, seed=42)
        result = tester.test_h2_network_amplification(G, test_params, n_null=200)
        # For a regular graph the observed factor should be close to the null
        # distribution, so the effect size should be small.
        assert result.additional_metrics['n_null_models'] > 0
        # The null factors should be close to observed (within reason)
        null_mean = result.additional_metrics['null_model_mean']
        observed = result.additional_metrics['network_factor']
        # Relative difference should be small (< 5%)
        assert abs(observed - null_mean) / observed < 0.05

    def test_h2_null_model_factors_list(self, tester, test_params):
        """null_model_factors should be a non-empty list of floats."""
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        result = tester.test_h2_network_amplification(G, test_params)
        factors = result.additional_metrics['null_model_factors']
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert all(isinstance(f, float) for f in factors)

    def test_h2_network_factor_still_in_metrics(self, tester, test_params):
        """The observed network factor should still be reported."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        result = tester.test_h2_network_amplification(G, test_params)
        assert 'network_factor' in result.additional_metrics
        assert result.additional_metrics['network_factor'] > 1


class TestVuongTest:
    """Tests for Vuong test replacing fabricated H1 p-value."""

    def test_vuong_returns_real_p_value(self):
        """H1 p-value should be computed, not hard-coded."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
        from src.estimation.estimator import EstimationResult

        tester = HypothesisTester(alpha=0.05, random_seed=42)
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        observed = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est_params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.85)
        result = tester.test_h1_epidemic_dynamics(
            state_history=observed,
            estimated_params=est_params,
            observed_data=observed,
        )
        # p-value must NOT be exactly 0.01 or 0.5 (the old hard-coded values)
        assert result.p_value not in (0.01, 0.5)
        assert 0 <= result.p_value <= 1

    def test_vuong_h1_confidence_interval_is_statistical(self):
        """H1 CI should not be R^2 +/- 0.1."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
        from src.estimation.estimator import EstimationResult

        tester = HypothesisTester(alpha=0.05, random_seed=42)
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        observed = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est_params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.85)
        result = tester.test_h1_epidemic_dynamics(
            state_history=observed,
            estimated_params=est_params,
            observed_data=observed,
        )
        lo, hi = result.confidence_interval
        # CI width should not be exactly 0.2 (the old +/-0.1)
        assert abs(hi - lo - 0.2) > 0.001

    def test_vuong_test_statistic_stored(self):
        """The Vuong test statistic should be stored in additional_metrics."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
        from src.estimation.estimator import EstimationResult

        tester = HypothesisTester(alpha=0.05, random_seed=42)
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        observed = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est_params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.85)
        result = tester.test_h1_epidemic_dynamics(
            state_history=observed,
            estimated_params=est_params,
            observed_data=observed,
        )
        assert 'vuong_statistic' in result.additional_metrics
        assert isinstance(result.additional_metrics['vuong_statistic'], float)

    def test_vuong_seir_r2_in_additional_metrics(self):
        """R^2 should still be available in additional_metrics for reference."""
        from src.hypothesis.hypothesis_tester import HypothesisTester
        from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
        from src.estimation.estimator import EstimationResult

        tester = HypothesisTester(alpha=0.05, random_seed=42)
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        observed = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est_params = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1, r_squared=0.85)
        result = tester.test_h1_epidemic_dynamics(
            state_history=observed,
            estimated_params=est_params,
            observed_data=observed,
        )
        assert 'seir_r_squared' in result.additional_metrics


class TestH3OneTailed:
    """Tests for H3 one-tailed p-value fix."""

    def test_h3_uses_one_tailed_p(self):
        """H3 should use one-tailed p-value, not two-tailed with direction check."""
        tester = HypothesisTester(alpha=0.05, random_seed=42)
        np.random.seed(42)
        fgi = np.linspace(30, 80, 50)
        infections = fgi * 0.5 + np.random.normal(0, 2, 50)

        state_history = pd.DataFrame({
            't': range(50),
            'I': infections
        })

        result = tester.test_h3_fgi_correlation(state_history, fgi)
        assert result.additional_metrics.get('one_tailed', False)

    def test_h3_one_tailed_p_smaller_for_positive_corr(self):
        """One-tailed p should be half of two-tailed when correlation is positive."""
        from scipy.stats import spearmanr
        tester = HypothesisTester(alpha=0.05, random_seed=42)
        np.random.seed(42)
        fgi = np.linspace(30, 80, 50)
        infections = fgi * 0.5 + np.random.normal(0, 2, 50)

        state_history = pd.DataFrame({
            't': range(50),
            'I': infections
        })

        result = tester.test_h3_fgi_correlation(state_history, fgi)
        # For positive correlation, one-tailed p = two-tailed p / 2
        # So it should be less than or equal to the two-tailed p
        two_tailed_p = result.additional_metrics.get('two_tailed_p_value')
        assert two_tailed_p is not None
        assert result.p_value <= two_tailed_p

    def test_h3_negative_corr_not_significant(self):
        """Negative correlation should not be significant for one-tailed test."""
        tester = HypothesisTester(alpha=0.05, random_seed=42)
        np.random.seed(42)
        fgi = np.linspace(30, 80, 50)
        infections = -fgi * 0.5 + np.random.normal(0, 2, 50)  # Negative correlation

        state_history = pd.DataFrame({
            't': range(50),
            'I': infections
        })

        result = tester.test_h3_fgi_correlation(state_history, fgi)
        # One-tailed p for wrong direction should be > 0.5
        assert result.p_value > 0.5
        assert not result.reject_null


class TestH5Permutation:
    """Tests for H5 permutation test fix."""

    def test_h5_uses_permutation_test(self):
        """H5 should use network permutation, not just chi-square."""
        tester = HypothesisTester(alpha=0.05, random_seed=42)

        # Create graph with clear community structure
        G1 = nx.complete_graph(50)
        G2 = nx.complete_graph(50)
        G = nx.disjoint_union(G1, G2)
        G.add_edge(0, 50)
        G.add_edge(25, 75)

        state_history = pd.DataFrame({'t': range(10), 'I': range(10)})

        result = tester.test_h5_community_clustering(G, state_history)
        assert 'permutation_p_value' in result.additional_metrics

    def test_h5_permutation_p_used_for_rejection(self):
        """H5 rejection should be based on permutation p-value."""
        tester = HypothesisTester(alpha=0.05, random_seed=42)

        G = nx.barabasi_albert_graph(200, 3, seed=42)
        state_history = pd.DataFrame({'t': range(10), 'I': range(10)})

        result = tester.test_h5_community_clustering(G, state_history)
        perm_p = result.additional_metrics['permutation_p_value']
        # The main p_value should be the permutation p-value
        assert result.p_value == perm_p


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
