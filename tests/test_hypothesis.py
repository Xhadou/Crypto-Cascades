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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
