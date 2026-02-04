"""
Unit Tests for SEIR Epidemic Model

Tests the core SEIR model functionality including:
- Parameter validation
- Mean-field simulation
- Network-based simulation
- R0 calculations
- FOMO factor integration
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
from src.state_engine.state_assigner import State


class TestSEIRParameters:
    """Tests for SEIRParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = SEIRParameters()
        assert params.beta == 0.3
        assert params.sigma == 0.2
        assert params.gamma == 0.1
        assert params.omega == 0.01
        assert params.fomo_alpha == 1.0
        assert params.fomo_enabled == True
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = SEIRParameters(beta=0.5, sigma=0.3, gamma=0.2, omega=0.05)
        assert params.beta == 0.5
        assert params.sigma == 0.3
        assert params.gamma == 0.2
        assert params.omega == 0.05
    
    def test_invalid_beta_raises(self):
        """Test that invalid beta raises assertion error."""
        with pytest.raises(AssertionError):
            SEIRParameters(beta=0)  # beta must be > 0
        
        with pytest.raises(AssertionError):
            SEIRParameters(beta=1.5)  # beta must be <= 1
    
    def test_invalid_gamma_raises(self):
        """Test that invalid gamma raises assertion error."""
        with pytest.raises(AssertionError):
            SEIRParameters(gamma=0)  # gamma must be > 0
    
    def test_r0_calculation(self):
        """Test basic reproduction number calculation."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        assert params.r0() == pytest.approx(3.0, rel=1e-6)
    
    def test_effective_beta_neutral_fgi(self):
        """Test effective beta at neutral FGI (50)."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        assert params.effective_beta(50) == pytest.approx(0.3, rel=1e-6)
    
    def test_effective_beta_high_fgi(self):
        """Test effective beta at high FGI (greed)."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        beta_eff = params.effective_beta(75)  # FGI=75 -> factor = 1.5
        assert beta_eff == pytest.approx(0.45, rel=1e-6)
    
    def test_effective_beta_low_fgi(self):
        """Test effective beta at low FGI (fear)."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        beta_eff = params.effective_beta(25)  # FGI=25 -> factor = 0.5
        assert beta_eff == pytest.approx(0.15, rel=1e-6)
    
    def test_effective_beta_fomo_disabled(self):
        """Test effective beta when FOMO is disabled."""
        params = SEIRParameters(beta=0.3, fomo_enabled=False)
        assert params.effective_beta(100) == pytest.approx(0.3, rel=1e-6)
        assert params.effective_beta(0) == pytest.approx(0.3, rel=1e-6)


class TestNetworkSEIRMeanField:
    """Tests for mean-field SEIR simulation."""
    
    @pytest.fixture
    def model(self):
        """Create a standard SEIR model for testing."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.0)
        return NetworkSEIR(params, random_seed=42)
    
    def test_simulation_returns_dataframe(self, model):
        """Test that simulation returns a DataFrame."""
        result = model.simulate_meanfield(N=1000, initial_infected=10, t_max=50)
        assert isinstance(result, pd.DataFrame)
    
    def test_simulation_columns(self, model):
        """Test that result has required columns."""
        result = model.simulate_meanfield(N=1000, initial_infected=10, t_max=50)
        required_cols = ['t', 'S', 'E', 'I', 'R', 'S_frac', 'E_frac', 'I_frac', 'R_frac']
        for col in required_cols:
            assert col in result.columns
    
    def test_population_conservation(self, model):
        """Test that total population is conserved."""
        N = 1000
        result = model.simulate_meanfield(N=N, initial_infected=10, t_max=50)
        
        total = result['S'] + result['E'] + result['I'] + result['R']
        assert np.allclose(total, N, rtol=1e-3)
    
    def test_fraction_sum_to_one(self, model):
        """Test that state fractions sum to 1."""
        result = model.simulate_meanfield(N=1000, initial_infected=10, t_max=50)
        
        total_frac = result['S_frac'] + result['E_frac'] + result['I_frac'] + result['R_frac']
        assert np.allclose(total_frac, 1.0, rtol=1e-3)
    
    def test_initial_conditions(self, model):
        """Test that initial conditions are correct."""
        N = 1000
        I0 = 10
        result = model.simulate_meanfield(N=N, initial_infected=I0, t_max=50)
        
        assert result['I'].iloc[0] == pytest.approx(I0, rel=0.1)
        assert result['E'].iloc[0] == pytest.approx(0, abs=1)
        assert result['R'].iloc[0] == pytest.approx(0, abs=1)
        assert result['S'].iloc[0] == pytest.approx(N - I0, rel=0.01)
    
    def test_epidemic_grows_with_r0_above_1(self, model):
        """Test that epidemic grows when R0 > 1."""
        result = model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)
        
        # With R0=3, infections should grow initially
        max_I = result['I'].max()
        initial_I = result['I'].iloc[0]
        
        assert max_I > initial_I * 5  # Should see significant growth
    
    def test_epidemic_peaks_and_declines(self, model):
        """Test that epidemic eventually peaks and declines."""
        result = model.simulate_meanfield(N=10000, initial_infected=10, t_max=200)
        
        max_idx = result['I'].idxmax()
        final_I = result['I'].iloc[-1]
        max_I = result['I'].iloc[max_idx]
        
        # Final infected should be less than peak
        assert final_I < max_I
    
    def test_fgi_affects_dynamics(self, model):
        """Test that FGI values affect epidemic dynamics."""
        # High FGI (greed) should lead to faster spread
        fgi_high = np.ones(100) * 80
        result_high = model.simulate_meanfield(N=1000, initial_infected=10, t_max=100, fgi_values=fgi_high)
        
        # Low FGI (fear) should lead to slower spread
        fgi_low = np.ones(100) * 20
        result_low = model.simulate_meanfield(N=1000, initial_infected=10, t_max=100, fgi_values=fgi_low)
        
        # Peak infected should be higher with high FGI
        assert result_high['I'].max() > result_low['I'].max()


class TestNetworkSEIRStochastic:
    """Tests for stochastic network SEIR simulation."""
    
    @pytest.fixture
    def model(self):
        """Create a standard SEIR model for testing."""
        params = SEIRParameters(beta=0.4, sigma=0.3, gamma=0.15, omega=0.0)
        return NetworkSEIR(params, random_seed=42)
    
    @pytest.fixture
    def test_graph(self):
        """Create a test graph."""
        return nx.barabasi_albert_graph(200, 3, seed=42)
    
    def test_network_simulation_returns_dataframe(self, model, test_graph):
        """Test that network simulation returns a DataFrame."""
        result = model.simulate_network_stochastic(
            test_graph, initial_infected=[0, 1, 2], t_max=30
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_network_simulation_columns(self, model, test_graph):
        """Test result has required columns."""
        result = model.simulate_network_stochastic(
            test_graph, initial_infected=[0, 1], t_max=30
        )
        for col in ['t', 'S', 'E', 'I', 'R', 'S_frac', 'E_frac', 'I_frac', 'R_frac']:
            assert col in result.columns
    
    def test_network_population_conservation(self, model, test_graph):
        """Test population conservation in network simulation."""
        N = test_graph.number_of_nodes()
        result = model.simulate_network_stochastic(
            test_graph, initial_infected=[0], t_max=30
        )
        
        total = result['S'] + result['E'] + result['I'] + result['R']
        assert np.all(total == N)
    
    def test_network_initial_infected(self, model, test_graph):
        """Test initial infected nodes."""
        initial = [0, 1, 2]
        result = model.simulate_network_stochastic(
            test_graph, initial_infected=initial, t_max=10
        )
        
        assert result['I'].iloc[0] == len(initial)


class TestNetworkR0:
    """Tests for network R0 calculation."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        return NetworkSEIR(params)
    
    def test_r0_regular_graph(self, model):
        """Test R0 for regular graph (all nodes same degree)."""
        G = nx.random_regular_graph(4, 100, seed=42)
        r0_network = model.compute_network_r0(G)
        
        # For regular graph, <k²>/<k> ≈ k, so R0_network ≈ R0_basic * k
        # R0_basic = 3, k = 4, so R0_network ≈ 12
        assert 10 < r0_network < 14
    
    def test_r0_scale_free_graph(self, model):
        """Test that scale-free graphs have higher network R0."""
        G_regular = nx.random_regular_graph(6, 500, seed=42)
        G_scalefree = nx.barabasi_albert_graph(500, 3, seed=42)
        
        r0_regular = model.compute_network_r0(G_regular)
        r0_scalefree = model.compute_network_r0(G_scalefree)
        
        # Scale-free networks should have higher R0 due to degree variance
        assert r0_scalefree > r0_regular
    
    def test_r0_empty_graph_is_zero(self, model):
        """Test that empty graph has R0 of 0."""
        G = nx.Graph()
        r0 = model.compute_network_r0(G)
        assert r0 == 0


class TestMonteCarloSimulations:
    """Tests for Monte Carlo ensemble simulations."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        params = SEIRParameters(beta=0.4, sigma=0.3, gamma=0.15)
        return NetworkSEIR(params, random_seed=42)
    
    @pytest.fixture
    def test_graph(self):
        """Create test graph."""
        return nx.barabasi_albert_graph(100, 3, seed=42)
    
    def test_monte_carlo_returns_dict(self, model, test_graph):
        """Test Monte Carlo returns a dictionary."""
        result = model.run_monte_carlo(
            test_graph, 
            initial_infected_count=3, 
            t_max=20, 
            n_simulations=5
        )
        assert isinstance(result, dict)
    
    def test_monte_carlo_statistics(self, model, test_graph):
        """Test Monte Carlo contains required statistics."""
        result = model.run_monte_carlo(
            test_graph,
            initial_infected_count=3,
            t_max=20,
            n_simulations=5
        )
        
        for state in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
            assert state in result
            assert 'mean' in result[state]
            assert 'std' in result[state]
            assert 'q05' in result[state]
            assert 'q95' in result[state]
    
    def test_monte_carlo_uncertainty_bounds(self, model, test_graph):
        """Test that uncertainty bounds are sensible."""
        result = model.run_monte_carlo(
            test_graph,
            initial_infected_count=3,
            t_max=20,
            n_simulations=10
        )
        
        for state in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
            # 5th percentile should be <= 95th percentile
            assert np.all(result[state]['q05'] <= result[state]['q95'])
            
            # Mean should be between percentiles
            mean = result[state]['mean']
            q05 = result[state]['q05']
            q95 = result[state]['q95']
            
            # Allow some tolerance for edge cases
            assert np.all(mean >= q05 - 0.1)
            assert np.all(mean <= q95 + 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
