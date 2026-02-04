"""
Integration Tests for End-to-End Pipeline

Tests the complete data flow from raw data to hypothesis testing results.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import tempfile
import os

from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
from src.estimation.estimator import ParameterEstimator, EstimationResult
from src.hypothesis.hypothesis_tester import HypothesisTester
from src.state_engine.state_assigner import StateAssigner, State
from src.preprocessing.graph_builder import GraphBuilder
from src.network_analysis.metrics import NetworkMetrics
from src.network_analysis.community_detection import CommunityDetector


class TestEndToEndPipeline:
    """Integration tests for the complete analysis pipeline."""
    
    @pytest.fixture
    def synthetic_transactions(self):
        """Create synthetic transaction data."""
        np.random.seed(42)
        
        n_transactions = 5000
        n_wallets = 500
        
        # Create transactions with power-law degree distribution
        sources = np.random.pareto(1.5, n_transactions).astype(int) % n_wallets
        targets = np.random.pareto(1.5, n_transactions).astype(int) % n_wallets
        
        # Remove self-loops
        mask = sources != targets
        sources = sources[mask]
        targets = targets[mask]
        
        # Generate timestamps
        start = pd.Timestamp('2017-01-01')
        timestamps = start + pd.to_timedelta(
            np.random.uniform(0, 365, len(sources)), unit='D'
        )
        
        return pd.DataFrame({
            'source_id': sources,
            'target_id': targets,
            'timestamp': timestamps,
            'btc_value': np.random.lognormal(0, 2, len(sources)),
            'usd_value': np.random.lognormal(8, 2, len(sources))
        })
    
    @pytest.fixture
    def synthetic_fgi(self):
        """Create synthetic Fear & Greed Index data."""
        np.random.seed(42)
        return np.random.uniform(25, 75, 365)
    
    def test_graph_building_from_transactions(self, synthetic_transactions):
        """Test building graph from transaction data."""
        builder = GraphBuilder()
        G = builder.build_transaction_graph(
            synthetic_transactions,
            directed=False,
            weight_column='usd_value',
            aggregate_multi_edges=True
        )
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
    
    def test_network_analysis_pipeline(self, synthetic_transactions):
        """Test network analysis from transactions."""
        # Build graph
        builder = GraphBuilder()
        G = builder.build_transaction_graph(synthetic_transactions, directed=False)
        
        # Verify graph was built
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        
        # Compute centrality metrics
        metrics = NetworkMetrics()
        centrality = metrics.compute_centrality_measures(G, measures=['degree'])
        
        assert 'degree' in centrality
        assert len(centrality['degree']) == G.number_of_nodes()
        
        # Community detection
        detector = CommunityDetector()
        result = detector.detect_communities_louvain(G)
        
        assert result['n_communities'] > 0
        assert result['modularity'] >= -1  # modularity can be negative
    
    def test_seir_simulation_pipeline(self, synthetic_transactions, synthetic_fgi):
        """Test SEIR simulation from network."""
        # Build graph
        builder = GraphBuilder()
        G = builder.build_transaction_graph(synthetic_transactions, directed=False)
        
        # Initialize SEIR model
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        # Run mean-field simulation
        N = G.number_of_nodes()
        results = model.simulate_meanfield(
            N=N,
            initial_infected=max(1, int(N * 0.01)),
            t_max=100,
            fgi_values=synthetic_fgi[:100]
        )
        
        assert len(results) == 100
        assert all(col in results.columns for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac'])
    
    def test_parameter_estimation_pipeline(self, synthetic_transactions, synthetic_fgi):
        """Test parameter estimation from simulation data."""
        # Generate simulation data with known parameters
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(true_params, random_seed=42)
        
        sim_data = model.simulate_meanfield(
            N=5000, initial_infected=10, t_max=100
        )
        
        # Estimate parameters
        estimator = ParameterEstimator(method='lsq', random_seed=42)
        result = estimator.estimate(
            sim_data, N=5000,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=10
        )
        
        # Should recover approximately correct parameters
        assert result.beta == pytest.approx(0.3, rel=0.3)
        assert result.success == True
    
    def test_hypothesis_testing_pipeline(self, synthetic_transactions, synthetic_fgi):
        """Test complete hypothesis testing pipeline."""
        # Build graph
        builder = GraphBuilder()
        G = builder.build_transaction_graph(synthetic_transactions, directed=False)
        
        # Run SEIR simulation
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        sim_data = model.simulate_meanfield(
            N=G.number_of_nodes(),
            initial_infected=10,
            t_max=100,
            fgi_values=synthetic_fgi[:100]
        )
        
        # Estimate parameters
        estimator = ParameterEstimator(method='lsq')
        est_result = estimator.estimate(sim_data, N=G.number_of_nodes(), n_bootstrap=0)
        
        # Run hypothesis tests
        tester = HypothesisTester(alpha=0.05, random_seed=42)
        
        h2_result = tester.test_h2_network_amplification(G, est_result)
        
        assert h2_result.hypothesis == "H2"
        assert 0 <= h2_result.p_value <= 1
    
    def test_full_pipeline_with_all_hypotheses(self, synthetic_transactions, synthetic_fgi):
        """Test running all hypotheses in sequence."""
        # Build graph
        builder = GraphBuilder()
        G = builder.build_transaction_graph(synthetic_transactions, directed=False)
        
        # Run simulation
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        sim_data = model.simulate_meanfield(
            N=G.number_of_nodes(),
            initial_infected=10,
            t_max=100,
            fgi_values=synthetic_fgi[:100]
        )
        
        # Estimate parameters
        estimator = ParameterEstimator(method='lsq')
        est_result = estimator.estimate(sim_data, N=G.number_of_nodes(), n_bootstrap=0)
        
        # Test all hypotheses
        tester = HypothesisTester(alpha=0.05, random_seed=42)
        results = tester.test_all(
            G, sim_data, synthetic_fgi[:100], est_result, observed_data=sim_data
        )
        
        # Verify all hypotheses were tested
        assert all(f'H{i}' in results for i in range(1, 6))
        
        # Verify all results are valid
        for h_name, result in results.items():
            assert 0 <= result.p_value <= 1 or np.isnan(result.p_value)
            # CI may have NaN for edge cases (e.g., empty samples)
            if not np.isnan(result.confidence_interval[0]) and not np.isnan(result.confidence_interval[1]):
                assert result.confidence_interval[0] <= result.confidence_interval[1]


class TestDataConsistency:
    """Tests for data consistency throughout the pipeline."""
    
    def test_population_conservation_through_pipeline(self):
        """Test that population is conserved at all stages."""
        N = 1000
        
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        results = model.simulate_meanfield(N=N, initial_infected=10, t_max=100)
        
        # Check at every timestep
        for _, row in results.iterrows():
            total = row['S'] + row['E'] + row['I'] + row['R']
            assert total == pytest.approx(N, rel=0.01)
    
    def test_state_fractions_sum_to_one(self):
        """Test that state fractions always sum to 1."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        results = model.simulate_meanfield(N=1000, initial_infected=10, t_max=100)
        
        for _, row in results.iterrows():
            total_frac = row['S_frac'] + row['E_frac'] + row['I_frac'] + row['R_frac']
            assert total_frac == pytest.approx(1.0, rel=0.01)
    
    def test_monotonicity_constraints(self):
        """Test monotonicity where expected."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.0)
        model = NetworkSEIR(params, random_seed=42)
        
        results = model.simulate_meanfield(N=10000, initial_infected=10, t_max=200)
        
        # Without immunity waning, R should be monotonically increasing (mostly)
        R_values = results['R'].values
        
        # Check that R increases overall
        assert R_values[-1] > R_values[0]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        G = nx.Graph()
        
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params)
        
        r0 = model.compute_network_r0(G)
        assert r0 == 0
    
    def test_single_node_graph(self):
        """Test with single node graph."""
        G = nx.Graph()
        G.add_node(0)
        
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params)
        
        r0 = model.compute_network_r0(G)
        assert r0 == 0  # No edges, no transmission
    
    def test_very_high_r0(self):
        """Test behavior with very high R0."""
        params = SEIRParameters(beta=0.9, sigma=0.5, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        # R0 = 9, should see rapid epidemic
        results = model.simulate_meanfield(N=10000, initial_infected=10, t_max=50)
        
        # Peak should occur early and be high
        peak_idx = results['I_frac'].idxmax()
        assert peak_idx < 30
        assert results['I_frac'].max() > 0.3
    
    def test_very_low_r0(self):
        """Test behavior with R0 < 1."""
        params = SEIRParameters(beta=0.05, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        
        # R0 = 0.5, epidemic should die out
        results = model.simulate_meanfield(N=10000, initial_infected=100, t_max=100)
        
        # Infections should decrease
        assert results['I'].iloc[-1] < results['I'].iloc[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
