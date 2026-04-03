"""
Tests for SNAP trust network validation module.
"""

import math
import pytest
import numpy as np
import pandas as pd
import igraph as ig

from src.validation.trust_network_validator import TrustNetworkValidator


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def validator():
    return TrustNetworkValidator()


@pytest.fixture
def snap_df():
    """Synthetic SNAP-like DataFrame with known trust patterns."""
    np.random.seed(42)
    rows = []
    # Trusted users (targets 1-10): receive mostly positive ratings
    for target in range(1, 11):
        for _ in range(10):
            rows.append({
                'source': np.random.randint(100, 200),
                'target': target,
                'rating': np.random.choice([1, 2, 3, 4, 5]),
                'time': 1_500_000_000 + np.random.randint(0, 100_000),
            })
    # Distrusted users (targets 11-20): receive mostly negative ratings
    for target in range(11, 21):
        for _ in range(10):
            rows.append({
                'source': np.random.randint(100, 200),
                'target': target,
                'rating': np.random.choice([-5, -4, -3, -2, -1]),
                'time': 1_500_000_000 + np.random.randint(0, 100_000),
            })
    # Neutral user (target 50): mixed ratings averaging ~0
    for _ in range(10):
        rows.append({
            'source': np.random.randint(100, 200),
            'target': 50,
            'rating': np.random.choice([-1, 0, 1]),
            'time': 1_500_000_000 + np.random.randint(0, 100_000),
        })

    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    return df


@pytest.fixture
def orbitaal_graph():
    """Small BA graph with integer node IDs overlapping some SNAP IDs."""
    G = ig.Graph.Barabasi(100, 3)
    G.vs['name'] = list(range(100))
    return G


@pytest.fixture
def node_states():
    """Dict mapping node IDs 0..99 to dummy state strings."""
    from src.state_engine.state_assigner import State
    return {i: State.INFECTED for i in range(100)}


@pytest.fixture
def infection_times_df():
    """DataFrame with infection times for nodes 0..99."""
    np.random.seed(42)
    return pd.DataFrame({
        'node': list(range(100)),
        'infection_time': np.random.uniform(0, 50, 100),
    })


# ------------------------------------------------------------------ #
# Tests: compute_trust_scores
# ------------------------------------------------------------------ #

class TestComputeTrustScores:
    def test_returns_correct_columns(self, validator, snap_df):
        scores = validator.compute_trust_scores(snap_df)
        assert 'mean_rating' in scores.columns
        assert 'n_ratings' in scores.columns
        assert 'trust_category' in scores.columns

    def test_trusted_users_categorised(self, validator, snap_df):
        scores = validator.compute_trust_scores(snap_df)
        for uid in range(1, 11):
            assert scores.loc[uid, 'trust_category'] == 'trusted'

    def test_distrusted_users_categorised(self, validator, snap_df):
        scores = validator.compute_trust_scores(snap_df)
        for uid in range(11, 21):
            assert scores.loc[uid, 'trust_category'] == 'distrusted'

    def test_empty_input(self, validator):
        empty = pd.DataFrame(columns=['source', 'target', 'rating'])
        scores = validator.compute_trust_scores(empty)
        assert scores.empty

    def test_rating_counts(self, validator, snap_df):
        scores = validator.compute_trust_scores(snap_df)
        # Each trusted/distrusted user has 10 ratings
        for uid in range(1, 21):
            assert scores.loc[uid, 'n_ratings'] == 10


# ------------------------------------------------------------------ #
# Tests: compare_network_topology
# ------------------------------------------------------------------ #

class TestCompareNetworkTopology:
    def test_returns_expected_keys(self, validator, snap_df, orbitaal_graph):
        result = validator.compare_network_topology(snap_df, orbitaal_graph)
        expected_keys = {
            'snap_nodes', 'snap_edges',
            'orbitaal_nodes', 'orbitaal_edges',
            'snap_mean_degree', 'orbitaal_mean_degree',
            'degree_ks_statistic', 'degree_ks_pvalue',
            'snap_clustering', 'orbitaal_clustering',
            'snap_density', 'orbitaal_density',
        }
        assert expected_keys.issubset(result.keys())

    def test_ks_statistic_valid(self, validator, snap_df, orbitaal_graph):
        result = validator.compare_network_topology(snap_df, orbitaal_graph)
        assert 0.0 <= result['degree_ks_statistic'] <= 1.0
        assert 0.0 <= result['degree_ks_pvalue'] <= 1.0

    def test_density_in_range(self, validator, snap_df, orbitaal_graph):
        result = validator.compare_network_topology(snap_df, orbitaal_graph)
        assert 0.0 <= result['snap_density'] <= 1.0
        assert 0.0 <= result['orbitaal_density'] <= 1.0

    def test_clustering_in_range(self, validator, snap_df, orbitaal_graph):
        result = validator.compare_network_topology(snap_df, orbitaal_graph)
        assert 0.0 <= result['snap_clustering'] <= 1.0
        assert 0.0 <= result['orbitaal_clustering'] <= 1.0


# ------------------------------------------------------------------ #
# Tests: validate_trust_transmission
# ------------------------------------------------------------------ #

class TestValidateTrustTransmission:
    def test_no_overlap_returns_inconclusive(self, validator, snap_df):
        """Nodes in SNAP (1-20, 50, 100-200) don't overlap with 500-600."""
        states = {i: 'I' for i in range(500, 600)}
        inf_df = pd.DataFrame({
            'node': list(range(500, 600)),
            'infection_time': np.random.uniform(0, 50, 100),
        })
        result = validator.validate_trust_transmission(
            snap_df, states, inf_df
        )
        assert result['inconclusive'] is True

    def test_small_overlap_returns_inconclusive(self, validator):
        """Fewer than 20 overlapping nodes → inconclusive."""
        snap_small = pd.DataFrame({
            'source': [100, 101, 102],
            'target': [1, 2, 3],
            'rating': [5, 5, -5],
        })
        states = {1: 'I', 2: 'I', 3: 'I'}
        inf_df = pd.DataFrame({
            'node': [1, 2, 3],
            'infection_time': [5.0, 10.0, 15.0],
        })
        result = validator.validate_trust_transmission(
            snap_small, states, inf_df
        )
        assert result['inconclusive'] is True

    def test_sufficient_overlap_returns_stats(self, validator):
        """With enough overlap and distinct groups, returns test results."""
        np.random.seed(42)
        rows = []
        # 15 trusted targets
        for t in range(1, 16):
            for _ in range(5):
                rows.append({'source': 500, 'target': t, 'rating': 5})
        # 15 distrusted targets
        for t in range(16, 31):
            for _ in range(5):
                rows.append({'source': 500, 'target': t, 'rating': -5})
        snap_big = pd.DataFrame(rows)

        states = {i: 'I' for i in range(1, 31)}
        # Trusted users infected earlier, distrusted later
        inf_df = pd.DataFrame({
            'node': list(range(1, 31)),
            'infection_time': (
                list(np.random.uniform(1, 5, 15))
                + list(np.random.uniform(20, 30, 15))
            ),
        })
        result = validator.validate_trust_transmission(
            snap_big, states, inf_df
        )
        assert result['inconclusive'] is False
        assert 'p_value' in result
        assert 0.0 <= result['p_value'] <= 1.0
        assert result['n_trusted'] == 15
        assert result['n_distrusted'] == 15

    def test_empty_snap_returns_inconclusive(self, validator):
        empty = pd.DataFrame(columns=['source', 'target', 'rating'])
        result = validator.validate_trust_transmission(empty, {}, pd.DataFrame())
        assert result['inconclusive'] is True

    def test_empty_infection_times_returns_inconclusive(
        self, validator, snap_df, node_states
    ):
        result = validator.validate_trust_transmission(
            snap_df, node_states, pd.DataFrame()
        )
        assert result['inconclusive'] is True


# ------------------------------------------------------------------ #
# Tests: run_all_validations
# ------------------------------------------------------------------ #

class TestRunAllValidations:
    def test_returns_all_keys(
        self, validator, snap_df, orbitaal_graph,
        node_states, infection_times_df, tmp_path,
    ):
        """Write SNAP CSVs to tmp dir and run full validation."""
        snap_dir = tmp_path / 'snap'
        snap_dir.mkdir()
        # Write synthetic CSV files (SNAPDownloader format)
        for name in ('otc', 'alpha'):
            csv_path = snap_dir / f'bitcoin_{name}.csv'
            subset = snap_df[['source', 'target', 'rating', 'time']].copy()
            subset.to_csv(csv_path, index=False, header=False)

        results = validator.run_all_validations(
            snap_dir=str(snap_dir),
            orbitaal_graph=orbitaal_graph,
            node_states=node_states,
            infection_times_df=infection_times_df,
        )

        assert 'topology' in results
        assert 'trust_scores' in results
        assert 'trust_transmission' in results

    def test_missing_dir_returns_empty(self, validator, orbitaal_graph, tmp_path):
        missing = tmp_path / 'does_not_exist'
        results = validator.run_all_validations(
            snap_dir=str(missing),
            orbitaal_graph=orbitaal_graph,
        )
        assert results == {}

    def test_empty_dir_returns_empty(self, validator, orbitaal_graph, tmp_path):
        empty_dir = tmp_path / 'empty_snap'
        empty_dir.mkdir()
        results = validator.run_all_validations(
            snap_dir=str(empty_dir),
            orbitaal_graph=orbitaal_graph,
        )
        assert results == {}


# ------------------------------------------------------------------ #
# Tests: generate_validation_report
# ------------------------------------------------------------------ #

class TestGenerateValidationReport:
    def test_report_contains_header(self, validator):
        report = validator.generate_validation_report({})
        assert "SNAP TRUST NETWORK VALIDATION REPORT" in report

    def test_report_with_full_results(self, validator):
        results = {
            'topology': {
                'snap_nodes': 100,
                'snap_edges': 500,
                'orbitaal_nodes': 200,
                'orbitaal_edges': 800,
                'snap_mean_degree': 5.0,
                'orbitaal_mean_degree': 4.0,
                'degree_ks_statistic': 0.15,
                'degree_ks_pvalue': 0.3,
                'snap_clustering': 0.2,
                'orbitaal_clustering': 0.3,
                'snap_density': 0.05,
                'orbitaal_density': 0.04,
            },
            'trust_scores': {'trusted': 40, 'distrusted': 30, 'neutral': 10},
            'trust_transmission': {
                'inconclusive': False,
                'test': 'Mann-Whitney U',
                'test_statistic': 150.0,
                'p_value': 0.03,
                'effect_size': 0.25,
                'n_trusted': 20,
                'n_distrusted': 15,
                'mean_infection_time_trusted': 5.0,
                'mean_infection_time_distrusted': 12.0,
            },
        }
        report = validator.generate_validation_report(results)
        assert "TOPOLOGY" in report
        assert "TRUST SCORE" in report
        assert "TRUST-TRANSMISSION" in report
        assert "Mann-Whitney" in report

    def test_report_inconclusive_transmission(self, validator):
        results = {
            'trust_transmission': {
                'inconclusive': True,
                'reason': 'Too few nodes',
            },
        }
        report = validator.generate_validation_report(results)
        assert "INCONCLUSIVE" in report
        assert "Too few nodes" in report


# ------------------------------------------------------------------ #
# Tests: _inconclusive helper
# ------------------------------------------------------------------ #

class TestInconclusiveHelper:
    def test_returns_dict_with_required_keys(self, validator):
        result = TrustNetworkValidator._inconclusive("test reason")
        assert result['inconclusive'] is True
        assert result['reason'] == "test reason"
        assert math.isnan(result['test_statistic'])
        assert math.isnan(result['p_value'])
        assert math.isnan(result['effect_size'])
