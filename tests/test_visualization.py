"""
Smoke Tests for Visualization Module

Basic tests to ensure visualization functions:
- Return matplotlib Figure objects
- Save to file without error
- Handle edge cases gracefully
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; must be set before importing pyplot
from matplotlib.figure import Figure

from src.visualization.plots import SEIRVisualizer, STATE_COLORS
from src.estimation.estimator import EstimationResult
from src.hypothesis.hypothesis_tester import HypothesisResult
from src.state_engine.state_assigner import State


@pytest.fixture
def visualizer(tmp_path):
    """Create a visualizer with temporary output directory."""
    return SEIRVisualizer(output_dir=str(tmp_path))


@pytest.fixture
def sample_seir_results():
    """Create sample SEIR trajectory data."""
    t = np.arange(100)
    return pd.DataFrame({
        't': t,
        'S_frac': 0.9 * np.exp(-0.03 * t),
        'E_frac': 0.1 * (1 - np.exp(-0.05 * t)) * np.exp(-0.02 * t),
        'I_frac': 0.2 * (1 - np.exp(-0.03 * t)) * np.exp(-0.01 * t),
        'R_frac': np.clip(
            1 - 0.9 * np.exp(-0.03 * t)
            - 0.1 * (1 - np.exp(-0.05 * t)) * np.exp(-0.02 * t)
            - 0.2 * (1 - np.exp(-0.03 * t)) * np.exp(-0.01 * t),
            0, 1
        ),
    })


@pytest.fixture
def sample_fgi():
    """Create sample FGI values."""
    np.random.seed(42)
    return np.random.uniform(25, 75, 100)


@pytest.fixture
def sample_graph():
    """Create a small test graph."""
    return nx.barabasi_albert_graph(100, 3, seed=42)


@pytest.fixture
def sample_node_states(sample_graph):
    """Create sample node states."""
    all_states = [State.SUSCEPTIBLE, State.EXPOSED, State.INFECTED, State.RECOVERED]
    probs = [0.6, 0.1, 0.2, 0.1]
    states = {}
    random.seed(42)
    for node in sample_graph.nodes():
        states[node] = random.choices(all_states, weights=probs, k=1)[0]
    return states


@pytest.fixture
def sample_hypothesis_results():
    """Create sample hypothesis results."""
    results = {}
    for h in ['H1', 'H2', 'H3', 'H4', 'H5']:
        results[h] = HypothesisResult(
            hypothesis=h,
            description=f"Test {h}",
            test_statistic=np.random.uniform(0, 5),
            p_value=np.random.uniform(0, 0.1),
            effect_size=np.random.uniform(0, 1),
            confidence_interval=(0.1, 0.9),
            reject_null=np.random.random() > 0.5,
            alpha=0.05,
            sample_size=100,
            additional_metrics={}
        )
    return results


@pytest.fixture
def sample_estimated_params():
    """Create sample estimated parameters."""
    return EstimationResult(
        beta=0.3, sigma=0.2, gamma=0.1,
        beta_ci=(0.25, 0.35),
        sigma_ci=(0.15, 0.25),
        gamma_ci=(0.08, 0.12),
        r_squared=0.85
    )


class TestSEIRTrajectoryPlot:
    """Tests for SEIR trajectory plotting."""

    def test_returns_figure(self, visualizer, sample_seir_results):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = visualizer.plot_seir_trajectory(sample_seir_results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_fgi_overlay(self, visualizer, sample_seir_results, sample_fgi):
        """Test plot with FGI secondary axis."""
        import matplotlib.pyplot as plt
        fig = visualizer.plot_seir_trajectory(
            sample_seir_results, fgi_values=sample_fgi
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, visualizer, sample_seir_results, tmp_path):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        save_path = str(tmp_path / "test_seir.png")
        fig = visualizer.plot_seir_trajectory(
            sample_seir_results, save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


class TestEpidemicCurvePlot:
    """Tests for epidemic curve plotting."""

    def test_returns_figure(self, visualizer, sample_seir_results):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = visualizer.plot_epidemic_curve(sample_seir_results, r0_value=3.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, visualizer, sample_seir_results, tmp_path):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        save_path = str(tmp_path / "test_epi_curve.png")
        fig = visualizer.plot_epidemic_curve(
            sample_seir_results, save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


class TestNetworkStatesPlot:
    """Tests for network state visualization."""

    def test_returns_figure(self, visualizer, sample_graph, sample_node_states):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = visualizer.plot_network_states(sample_graph, sample_node_states)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(
        self, visualizer, sample_graph, sample_node_states, tmp_path
    ):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        save_path = str(tmp_path / "test_network.png")
        fig = visualizer.plot_network_states(
            sample_graph, sample_node_states, save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


class TestFGICorrelationPlot:
    """Tests for FGI correlation plotting."""

    def test_returns_figure(self, visualizer):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        fgi = np.random.uniform(30, 70, 50)
        rate = fgi * 0.5 + np.random.normal(0, 3, 50)
        fig = visualizer.plot_fgi_correlation(
            fgi, rate, correlation=0.65, p_value=0.001
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestParameterSensitivityPlot:
    """Tests for parameter sensitivity plotting."""

    def test_returns_figure(self, visualizer):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        df = pd.DataFrame({
            'parameter': ['beta', 'sigma', 'gamma'],
            'base_value': [0.3, 0.2, 0.1],
            'sensitivity': [0.5, -0.3, 0.8],
            'elasticity': [1.2, -0.6, 0.9],
            'mse_up': [0.01, 0.02, 0.015],
            'mse_down': [0.008, 0.012, 0.005],
        })
        fig = visualizer.plot_parameter_sensitivity(df)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestHypothesisResultsPlot:
    """Tests for hypothesis results plotting."""

    def test_returns_figure(self, visualizer, sample_hypothesis_results):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = visualizer.plot_hypothesis_results(sample_hypothesis_results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(
        self, visualizer, sample_hypothesis_results, tmp_path
    ):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        save_path = str(tmp_path / "test_hypothesis.png")
        fig = visualizer.plot_hypothesis_results(
            sample_hypothesis_results, save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


class TestHypothesisResultsInconclusiveNaN:
    """Tests for inconclusive (NaN p-value) handling."""

    def test_nan_pvalue_renders_gray_bar(self, visualizer):
        """Test that NaN p-values render as gray bars with N/A annotation."""
        import matplotlib.pyplot as plt
        results = {}
        for h in ['H1', 'H2', 'H4', 'H5']:
            results[h] = HypothesisResult(
                hypothesis=h, description=f"Test {h}",
                test_statistic=2.0, p_value=0.01, effect_size=0.5,
                confidence_interval=(0.1, 0.9), reject_null=True,
                alpha=0.05, sample_size=100, additional_metrics={}
            )
        # H3 is inconclusive with NaN p-value
        results['H3'] = HypothesisResult(
            hypothesis='H3', description='Inconclusive',
            test_statistic=float('nan'), p_value=float('nan'),
            effect_size=float('nan'),
            confidence_interval=(float('nan'), float('nan')),
            reject_null=False, alpha=0.05, sample_size=0,
            additional_metrics={'inconclusive': True}
        )
        fig = visualizer.plot_hypothesis_results(results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_dashboard_nan_pvalue(
        self, visualizer, sample_seir_results, sample_graph,
        sample_node_states, sample_estimated_params, sample_fgi
    ):
        """Test dashboard handles NaN p-values in hypothesis results."""
        import matplotlib.pyplot as plt
        results = {}
        for h in ['H1', 'H2', 'H4', 'H5']:
            results[h] = HypothesisResult(
                hypothesis=h, description=f"Test {h}",
                test_statistic=2.0, p_value=0.01, effect_size=0.5,
                confidence_interval=(0.1, 0.9), reject_null=True,
                alpha=0.05, sample_size=100, additional_metrics={}
            )
        results['H3'] = HypothesisResult(
            hypothesis='H3', description='Inconclusive',
            test_statistic=float('nan'), p_value=float('nan'),
            effect_size=float('nan'),
            confidence_interval=(float('nan'), float('nan')),
            reject_null=False, alpha=0.05, sample_size=0,
            additional_metrics={'inconclusive': True}
        )
        fig = visualizer.create_summary_dashboard(
            seir_results=sample_seir_results,
            G=sample_graph,
            node_states=sample_node_states,
            hypothesis_results=results,
            estimated_params=sample_estimated_params,
            fgi_values=sample_fgi,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSummaryDashboard:
    """Tests for the summary dashboard."""

    def test_returns_figure(
        self, visualizer, sample_seir_results, sample_graph,
        sample_node_states, sample_hypothesis_results, sample_estimated_params,
        sample_fgi
    ):
        """Test that dashboard returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = visualizer.create_summary_dashboard(
            seir_results=sample_seir_results,
            G=sample_graph,
            node_states=sample_node_states,
            hypothesis_results=sample_hypothesis_results,
            estimated_params=sample_estimated_params,
            fgi_values=sample_fgi,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPriceSentimentOverview:
    """Tests for price-sentiment overview plot."""

    def test_returns_figure(self, visualizer):
        """Test that plot returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'price': np.cumsum(np.random.randn(100)) + 10000,
            'fear_greed_value': np.random.uniform(20, 80, 100),
            'returns': np.random.randn(100) * 0.02,
        })
        fig = visualizer.plot_price_sentiment_overview(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, visualizer, tmp_path):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'price': np.cumsum(np.random.randn(100)) + 10000,
            'fear_greed_value': np.random.uniform(20, 80, 100),
            'returns': np.random.randn(100) * 0.02,
        })
        save_path = str(tmp_path / "test_price_sentiment.png")
        fig = visualizer.plot_price_sentiment_overview(df, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)


class TestR0ComparisonPlot:
    """Tests for R₀ comparison bar chart."""

    def test_returns_figure(self, visualizer):
        """Test that R₀ comparison plot returns a Figure."""
        import matplotlib.pyplot as plt
        period_results = {
            'training': {
                'r0': 2.5, 'r0_ci': (2.0, 3.0),
                'market_type': 'bull', 'description': '2017 Bull'
            },
            'control': {
                'r0': 0.8, 'r0_ci': (0.6, 1.0),
                'market_type': 'bear', 'description': '2018 Bear'
            },
            'validation': {
                'r0': 2.1, 'r0_ci': (1.7, 2.5),
                'market_type': 'bull', 'description': '2020-21 Bull'
            },
        }
        fig = visualizer.plot_r0_comparison_by_period(period_results)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestCommunityInfectionHeatmap:
    """Tests for community infection heatmap."""

    def test_returns_figure_with_data(self, visualizer):
        """Test heatmap with valid infection data."""
        import matplotlib.pyplot as plt
        communities = {0: [1, 2, 3], 1: [4, 5, 6]}
        infection_times = {1: 5.0, 2: 7.0, 4: 3.0, 5: 10.0}
        fig = visualizer.plot_community_infection_heatmap(
            communities, infection_times
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_empty_data(self, visualizer):
        """Test heatmap with no infection data."""
        import matplotlib.pyplot as plt
        communities = {0: [1, 2], 1: [3, 4]}
        infection_times = {}
        fig = visualizer.plot_community_infection_heatmap(
            communities, infection_times
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestEarlyWarningSignalsPlot:
    """Tests for early warning signals plot."""

    def test_returns_figure(self, visualizer):
        """Test that EWS plot returns a Figure."""
        import matplotlib.pyplot as plt
        ews_df = pd.DataFrame({
            't': range(10, 100),
            'variance': np.random.uniform(0, 0.01, 90),
            'autocorrelation': np.random.uniform(0, 1, 90),
            'skewness': np.random.uniform(-1, 1, 90),
            'alarm': [False] * 70 + [True] * 20,
        })
        fig = visualizer.plot_early_warning_signals(ews_df, transition_point=80)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, visualizer, tmp_path):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        ews_df = pd.DataFrame({
            't': range(10, 50),
            'variance': np.random.uniform(0, 0.01, 40),
            'autocorrelation': np.random.uniform(0, 1, 40),
            'skewness': np.random.uniform(-1, 1, 40),
            'alarm': [False] * 40,
        })
        save_path = str(tmp_path / "test_ews.png")
        fig = visualizer.plot_early_warning_signals(ews_df, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)


class TestTopologyComparison:
    """Tests for SNAP vs ORBITAAL topology comparison plot."""

    def test_returns_figure(self, visualizer):
        """Test that topology comparison plot returns a Figure."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        snap_deg = np.random.pareto(1.5, 500).astype(float)
        orb_deg = np.random.pareto(1.5, 800).astype(float)
        fig = visualizer.plot_topology_comparison(
            snap_deg, orb_deg, ks_statistic=0.12, ks_pvalue=0.03
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, visualizer, tmp_path):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        snap_deg = np.random.pareto(1.5, 500).astype(float)
        orb_deg = np.random.pareto(1.5, 800).astype(float)
        save_path = str(tmp_path / "test_topo.png")
        fig = visualizer.plot_topology_comparison(
            snap_deg, orb_deg, ks_statistic=0.12, ks_pvalue=0.03,
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


class TestTrustInfectionBoxplot:
    """Tests for trust infection boxplot."""

    def test_returns_figure(self, visualizer):
        """Test that trust infection boxplot returns a Figure."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        df = pd.DataFrame({
            'trust_category': ['trusted'] * 20 + ['distrusted'] * 20 + ['neutral'] * 10,
            'infection_time': np.concatenate([
                np.random.uniform(5, 15, 20),
                np.random.uniform(10, 25, 20),
                np.random.uniform(8, 20, 10),
            ]),
        })
        fig = visualizer.plot_trust_infection_boxplot(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_insufficient_data(self, visualizer):
        """Test boxplot with too few data points."""
        import matplotlib.pyplot as plt
        df = pd.DataFrame({
            'trust_category': ['trusted'],
            'infection_time': [5.0],
        })
        fig = visualizer.plot_trust_infection_boxplot(df)
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
