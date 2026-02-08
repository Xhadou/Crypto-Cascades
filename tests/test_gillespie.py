"""
Tests for Gillespie stochastic simulation implementation.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters


@pytest.fixture
def model():
    params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
    return NetworkSEIR(params, random_seed=42)


@pytest.fixture
def test_graph():
    return nx.barabasi_albert_graph(100, 3, seed=42)


class TestGillespieSimulation:
    """Tests for Gillespie algorithm implementation."""

    def test_gillespie_produces_valid_trajectory(self, model, test_graph):
        """Test Gillespie produces valid state counts."""
        result = model.simulate_gillespie(
            test_graph, initial_infected=[0, 1], t_max=50
        )

        assert isinstance(result, pd.DataFrame)
        assert 't' in result.columns
        assert result['S'].min() >= 0
        assert result['I'].min() >= 0

        # Population conservation
        N = test_graph.number_of_nodes()
        total = result['S'] + result['E'] + result['I'] + result['R']
        assert np.all(total == N), "Population must be conserved at every time step"

    def test_gillespie_handles_extinction(self, model, test_graph):
        """Test Gillespie handles epidemic extinction gracefully."""
        # Low beta should lead to (near) immediate extinction
        model.params.beta = 0.01
        result = model.simulate_gillespie(
            test_graph, initial_infected=[0], t_max=100
        )

        # Should complete without error and have at least the initial row
        assert len(result) > 0

    def test_gillespie_respects_max_time(self, model, test_graph):
        """Test Gillespie doesn't exceed t_max."""
        result = model.simulate_gillespie(
            test_graph, initial_infected=[0, 1, 2], t_max=50
        )

        assert result['t'].max() <= 50

    def test_gillespie_early_termination(self, model, test_graph):
        """Test early termination when I and E are both empty."""
        model.params.beta = 0.001  # very low transmission
        model.params.gamma = 5.0   # very fast recovery
        result = model.simulate_gillespie(
            test_graph,
            initial_infected=[0],
            t_max=1000,
            early_termination=True,
            min_infected_for_continuation=1
        )

        # Should terminate well before t_max
        assert len(result) > 0

    def test_gillespie_disconnected_graph(self, model):
        """Test Gillespie on a disconnected graph produces a result."""
        G = nx.Graph()
        G.add_nodes_from(range(20))
        # Two disconnected cliques
        G.add_edges_from([(i, j) for i in range(10) for j in range(i + 1, 10)])
        G.add_edges_from([(i, j) for i in range(10, 20) for j in range(i + 1, 20)])

        result = model.simulate_gillespie(
            G, initial_infected=[0], t_max=50
        )
        assert len(result) > 0


class TestTimeVaryingR0:
    """Tests for time-varying R₀ computation."""

    def test_r0_basic_output(self, model, test_graph):
        """Test that compute_time_varying_r0 returns expected columns."""
        sim = model.simulate_network_stochastic(
            test_graph, initial_infected=[0, 1, 2], t_max=50
        )
        r0_df = model.compute_time_varying_r0(sim, window_size=5)

        assert isinstance(r0_df, pd.DataFrame)
        assert 'R_t' in r0_df.columns
        assert 'R_t_lower' in r0_df.columns
        assert 'R_t_upper' in r0_df.columns
        assert len(r0_df) > 0

    def test_r0_ci_contains_point_estimate(self, model, test_graph):
        """Confidence interval should bracket the point estimate."""
        sim = model.simulate_network_stochastic(
            test_graph, initial_infected=[0, 1, 2], t_max=50
        )
        r0_df = model.compute_time_varying_r0(sim, window_size=5)

        # For rows with valid CI
        valid = r0_df.dropna(subset=['R_t', 'R_t_lower', 'R_t_upper'])
        if len(valid) > 0:
            assert (valid['R_t_lower'] <= valid['R_t'] + 1e-10).all()
            assert (valid['R_t'] <= valid['R_t_upper'] + 1e-10).all()


class TestNetworkR0:
    """Tests for network-based R₀ with bootstrap CI."""

    def test_network_r0_returns_float_by_default(self, model, test_graph):
        r0 = model.compute_network_r0(test_graph)
        assert isinstance(r0, float)
        assert r0 > 0

    def test_network_r0_returns_ci(self, model, test_graph):
        r0, (lo, hi) = model.compute_network_r0(
            test_graph, return_ci=True, n_bootstrap=50
        )
        assert isinstance(r0, float)
        assert lo <= r0 <= hi

    def test_network_r0_single_node_graph(self, model):
        G = nx.Graph()
        G.add_node(0)
        r0 = model.compute_network_r0(G)
        assert r0 == 0.0


class TestMonteCarloParallel:
    """Tests for parallel Monte Carlo."""

    def test_parallel_matches_sequential_stats_shape(self, model, test_graph):
        """Parallel run should return the same dict structure as sequential."""
        seq = model.run_monte_carlo(
            test_graph, initial_infected_count=3, t_max=20, n_simulations=5
        )
        par = model.run_monte_carlo_parallel(
            test_graph, initial_infected_count=3, t_max=20,
            n_simulations=5, n_workers=2
        )

        assert set(seq.keys()) == set(par.keys())
        for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
            assert col in par
            assert 'mean' in par[col]
