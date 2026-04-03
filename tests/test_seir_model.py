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
import igraph as ig

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
        """Test effective beta at neutral FGI (50) -- sigmoid midpoint."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        # At FGI=50: expit(0)=0.5, factor = 1 + 1.0*0.5 = 1.5, beta_eff = 0.45
        assert params.effective_beta(50) == pytest.approx(0.3 * 1.5, rel=1e-6)

    def test_effective_beta_high_fgi(self):
        """Test effective beta at high FGI (greed) is amplified."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        beta_eff = params.effective_beta(75)
        # With sigmoid, FGI=75 gives more than midpoint but less than linear
        assert beta_eff > params.effective_beta(50)
        assert beta_eff <= 0.99  # bounded

    def test_effective_beta_low_fgi(self):
        """Test effective beta at low FGI (fear) is reduced."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        beta_eff = params.effective_beta(25)
        # With sigmoid, FGI=25 gives less than midpoint
        assert beta_eff < params.effective_beta(50)
        assert beta_eff > 0  # always positive
    
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
        g = ig.Graph.Barabasi(200, 3)
        g.vs['name'] = list(range(200))
        return g
    
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
        N = test_graph.vcount()
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
        G = ig.Graph.K_Regular(100, 4)
        r0_network = model.compute_network_r0(G)

        # For regular graph, <k²>/<k> ≈ k, so R0_network ≈ R0_basic * k
        # R0_basic = 3, k = 4, so R0_network ≈ 12
        assert 10 < r0_network < 14

    def test_r0_scale_free_graph(self, model):
        """Test that scale-free graphs have higher network R0."""
        G_regular = ig.Graph.K_Regular(500, 6)
        G_scalefree = ig.Graph.Barabasi(500, 3)

        r0_regular = model.compute_network_r0(G_regular)
        r0_scalefree = model.compute_network_r0(G_scalefree)

        # Scale-free networks should have higher R0 due to degree variance
        assert r0_scalefree > r0_regular

    def test_r0_empty_graph_is_zero(self, model):
        """Test that empty graph has R0 of 0."""
        G = ig.Graph()
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
        g = ig.Graph.Barabasi(100, 3)
        g.vs['name'] = list(range(100))
        return g
    
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


class TestSigmoidalFOMO:
    """Tests for sigmoidal FOMO coupling (replaces linear coupling)."""

    def test_fomo_is_bounded(self):
        """Effective beta should never exceed 0.99."""
        params = SEIRParameters(beta=0.5, fomo_alpha=3.0)
        beta_eff = params.effective_beta(100.0)
        assert beta_eff <= 0.99

    def test_fomo_saturates_at_extremes(self):
        """Sigmoid should produce diminishing returns at extreme FGI."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        b60 = params.effective_beta(60.0)
        b80 = params.effective_beta(80.0)
        b100 = params.effective_beta(100.0)
        # Marginal increase should decrease (sigmoid saturation)
        assert (b80 - b60) > (b100 - b80)

    def test_fomo_symmetric_around_50(self):
        """FGI=50 should give predictable amplification (sigmoid midpoint)."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        b50 = params.effective_beta(50.0)
        # At midpoint, expit(0)=0.5, so factor = 1 + alpha*0.5 = 1.5
        # beta_eff = 0.3 * 1.5 = 0.45
        assert b50 == pytest.approx(0.3 * 1.5, rel=1e-6)

    def test_fomo_disabled_returns_base_beta(self):
        """With FOMO disabled, effective_beta should return base beta."""
        params = SEIRParameters(beta=0.3, fomo_enabled=False)
        assert params.effective_beta(100.0) == 0.3

    def test_fomo_k_controls_steepness(self):
        """Higher fomo_k should produce steeper sigmoid transition."""
        params_gentle = SEIRParameters(beta=0.3, fomo_alpha=1.0, fomo_k=1.0)
        params_steep = SEIRParameters(beta=0.3, fomo_alpha=1.0, fomo_k=5.0)
        # At FGI=70 (moderately above midpoint), steep should be closer to saturation
        b_gentle = params_gentle.effective_beta(70.0)
        b_steep = params_steep.effective_beta(70.0)
        assert b_steep > b_gentle

    def test_fomo_low_fgi_reduces_beta(self):
        """Low FGI (fear) should produce beta below the midpoint value."""
        params = SEIRParameters(beta=0.3, fomo_alpha=1.0)
        b20 = params.effective_beta(20.0)
        b50 = params.effective_beta(50.0)
        assert b20 < b50

    def test_fomo_never_negative(self):
        """Effective beta should never go negative even at FGI=0."""
        params = SEIRParameters(beta=0.3, fomo_alpha=2.0)
        b0 = params.effective_beta(0.0)
        assert b0 > 0


class TestSolveIVP:
    """Tests for solve_ivp migration (replacing legacy odeint)."""

    def test_meanfield_uses_solve_ivp(self):
        """simulate_meanfield should use solve_ivp, not odeint."""
        import inspect
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        source = inspect.getsource(model.simulate_meanfield)
        assert 'solve_ivp' in source, "simulate_meanfield should use solve_ivp"
        assert 'odeint' not in source, "simulate_meanfield should not use odeint"

    def test_solve_ivp_population_conservation(self):
        """solve_ivp integration should conserve total population."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.0)
        model = NetworkSEIR(params, random_seed=42)
        N = 10000
        result = model.simulate_meanfield(N=N, initial_infected=10, t_max=100)
        total = result['S'] + result['E'] + result['I'] + result['R']
        assert np.allclose(total, N, rtol=1e-6), (
            f"Population not conserved: min={total.min()}, max={total.max()}"
        )

    def test_solve_ivp_with_fgi_values(self):
        """solve_ivp should handle time-varying beta via FGI correctly."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.01)
        model = NetworkSEIR(params, random_seed=42)
        fgi_values = np.linspace(30, 80, 100)
        result = model.simulate_meanfield(
            N=5000, initial_infected=10, t_max=100, fgi_values=fgi_values
        )
        # Should have time-varying beta_eff
        assert 'beta_eff' in result.columns
        assert result['beta_eff'].nunique() > 1, "beta_eff should vary over time"
        # Population should still be conserved
        total = result['S'] + result['E'] + result['I'] + result['R']
        assert np.allclose(total, 5000, rtol=1e-4)

    def test_solve_ivp_matches_epidemic_dynamics(self):
        """solve_ivp results should show proper SEIR epidemic dynamics."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.0)
        model = NetworkSEIR(params, random_seed=42)
        result = model.simulate_meanfield(N=10000, initial_infected=10, t_max=200)
        # Epidemic should grow (R0=3)
        assert result['I'].max() > result['I'].iloc[0] * 5
        # Should peak and decline
        max_idx = result['I'].idxmax()
        assert result['I'].iloc[-1] < result['I'].iloc[max_idx]
        # Non-negative values
        for col in ['S', 'E', 'I', 'R']:
            assert (result[col] >= 0).all(), f"Negative values in {col}"

    def test_solve_ivp_no_odeint_import(self):
        """The module should import solve_ivp, not odeint."""
        import inspect
        import src.epidemic_model.network_seir as module
        source = inspect.getsource(module)
        assert 'from scipy.integrate import solve_ivp' in source
        assert 'from scipy.integrate import odeint' not in source


class TestRNG:
    """Tests for instance-level RNG (np.random.default_rng) instead of global seed."""

    def test_model_has_rng_instance(self):
        """Model should use np.random.default_rng, not global seed."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        assert hasattr(model, 'rng')
        assert isinstance(model.rng, np.random.Generator)

    def test_rng_reproducibility(self):
        """Two models with same seed should produce identical results."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        m1 = NetworkSEIR(params, random_seed=42)
        m2 = NetworkSEIR(params, random_seed=42)
        r1 = m1.simulate_meanfield(N=1000, initial_infected=10, t_max=50)
        r2 = m2.simulate_meanfield(N=1000, initial_infected=10, t_max=50)
        np.testing.assert_array_equal(r1['I'].values, r2['I'].values)

    def test_rng_different_seeds_differ(self):
        """Two models with different seeds should produce different stochastic results."""
        params = SEIRParameters(beta=0.4, sigma=0.3, gamma=0.15, omega=0.0)
        G = ig.Graph.Barabasi(100, 3)
        G.vs['name'] = list(range(100))
        m1 = NetworkSEIR(params, random_seed=42)
        m2 = NetworkSEIR(params, random_seed=99)
        r1 = m1.simulate_network_stochastic(G, initial_infected=[0, 1, 2], t_max=20)
        r2 = m2.simulate_network_stochastic(G, initial_infected=[0, 1, 2], t_max=20)
        # Stochastic simulations with different seeds should diverge
        assert not np.array_equal(r1['I'].values, r2['I'].values)

    def test_rng_stochastic_reproducibility(self):
        """Same seed should produce identical stochastic simulation results."""
        params = SEIRParameters(beta=0.4, sigma=0.3, gamma=0.15, omega=0.0)
        G = ig.Graph.Barabasi(100, 3)
        G.vs['name'] = list(range(100))
        m1 = NetworkSEIR(params, random_seed=42)
        m2 = NetworkSEIR(params, random_seed=42)
        r1 = m1.simulate_network_stochastic(G, initial_infected=[0, 1, 2], t_max=20)
        r2 = m2.simulate_network_stochastic(G, initial_infected=[0, 1, 2], t_max=20)
        np.testing.assert_array_equal(r1['I'].values, r2['I'].values)

    def test_no_global_random_seed_in_source(self):
        """Source code should not contain np.random.seed calls."""
        import inspect
        import src.epidemic_model.network_seir as module
        source = inspect.getsource(module)
        assert 'np.random.seed(' not in source, (
            "Module should not use np.random.seed(); use self.rng instead"
        )


class TestCoriRt:
    """Tests for Cori et al. (2013) R(t) estimation."""

    @pytest.fixture
    def seir_data(self):
        """Generate SEIR data with a clear epidemic for R(t) estimation."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.01)
        model = NetworkSEIR(params, random_seed=42)
        return model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)

    def test_compute_rt_returns_dataframe(self, seir_data):
        """compute_time_varying_r0 should return a DataFrame."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data)
        assert isinstance(rt_df, pd.DataFrame)
        assert len(rt_df) > 0

    def test_compute_rt_has_rt_column(self, seir_data):
        """Result should contain R_t column."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data)
        assert 'R_t' in rt_df.columns

    def test_cori_method_available(self, seir_data):
        """compute_time_varying_r0 should accept method='cori'."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data, method='cori')
        assert isinstance(rt_df, pd.DataFrame)
        assert len(rt_df) > 0

    def test_cori_rt_has_rt_column(self, seir_data):
        """Cori method should produce R_t column."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data, method='cori')
        assert 'R_t' in rt_df.columns

    def test_ratio_method_still_works(self, seir_data):
        """Ratio method should continue to work as before."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data, method='ratio')
        assert isinstance(rt_df, pd.DataFrame)
        assert 'R_t' in rt_df.columns
        assert len(rt_df) > 0

    def test_cori_fallback_on_import_error(self, seir_data):
        """If epyestim is unavailable, Cori method should fall back to ratio."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        # Even if epyestim is not installed, method='cori' should not raise
        rt_df = model.compute_time_varying_r0(seir_data, method='cori')
        assert isinstance(rt_df, pd.DataFrame)
        assert len(rt_df) > 0

    def test_rt_values_are_reasonable(self, seir_data):
        """R(t) estimates should be non-negative and not astronomically large."""
        params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(params, random_seed=42)
        rt_df = model.compute_time_varying_r0(seir_data)
        valid_rt = rt_df['R_t'].dropna()
        if len(valid_rt) > 0:
            assert (valid_rt >= 0).all(), "R(t) should be non-negative"
            assert valid_rt.max() < 100, "R(t) should not be astronomically large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
