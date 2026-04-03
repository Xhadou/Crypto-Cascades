"""
Unit Tests for Community Detection

Tests the community detection module including:
- Leiden algorithm (leidenalg + python-igraph)
- Louvain fallback when Leiden is unavailable
- Result format consistency across algorithms
"""

import random
import pytest
import numpy as np
import igraph as ig
from unittest.mock import patch

from src.network_analysis.community_detection import CommunityDetector


@pytest.fixture
def detector():
    """Create a CommunityDetector instance."""
    return CommunityDetector()


@pytest.fixture
def barabasi_graph():
    """Create a Barabasi-Albert graph with clear community structure."""
    rng = random.Random(42)
    ig.set_random_number_generator(rng)
    g = ig.Graph.Barabasi(200, 3)
    # Assign string node names to match project convention
    g.vs['name'] = [str(i) for i in range(g.vcount())]
    g['_name_to_idx'] = {name: i for i, name in enumerate(g.vs['name'])}
    return g


@pytest.fixture
def directed_graph():
    """Create a directed graph for testing directed-to-undirected conversion."""
    rng = random.Random(42)
    ig.set_random_number_generator(rng)
    g = ig.Graph.Barabasi(100, 3)
    # Convert to directed (each undirected edge becomes two directed edges)
    g = g.as_directed()
    g.vs['name'] = [str(i) for i in range(g.vcount())]
    g['_name_to_idx'] = {name: i for i, name in enumerate(g.vs['name'])}
    return g


@pytest.fixture
def planted_partition_graph():
    """Create a graph with known planted community structure.

    4 communities of 50 nodes each; p_in >> p_out ensures clear partition.
    """
    rng = random.Random(42)
    n_per_group = 50
    n_groups = 4
    n = n_per_group * n_groups
    p_in = 0.3
    p_out = 0.01

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            same_group = (i // n_per_group) == (j // n_per_group)
            p = p_in if same_group else p_out
            if rng.random() < p:
                edges.append((i, j))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.vs['name'] = [str(i) for i in range(n)]
    g['_name_to_idx'] = {str(i): i for i in range(n)}
    return g


class TestLeidenCommunity:
    """Tests for Leiden community detection."""

    def test_leiden_returns_partition(self, detector, barabasi_graph):
        """Leiden should return a partition dict mapping every node to a community."""
        result = detector.detect_communities_leiden(barabasi_graph)
        assert 'partition' in result
        assert 'modularity' in result
        assert 'n_communities' in result
        assert 'community_sizes' in result

    def test_leiden_finds_multiple_communities(self, detector, barabasi_graph):
        """Leiden should find more than one community in a Barabasi-Albert graph."""
        result = detector.detect_communities_leiden(barabasi_graph)
        assert result['n_communities'] > 1

    def test_leiden_partition_covers_all_nodes(self, detector, barabasi_graph):
        """Every node in the graph should appear in the partition."""
        result = detector.detect_communities_leiden(barabasi_graph)
        assert set(result['partition'].keys()) == set(barabasi_graph.vs['name'])

    def test_leiden_modularity_is_positive(self, detector, barabasi_graph):
        """Modularity should be positive for a graph with community structure."""
        result = detector.detect_communities_leiden(barabasi_graph)
        assert result['modularity'] > 0

    def test_leiden_community_sizes_sum(self, detector, barabasi_graph):
        """Sum of community sizes should equal total number of nodes."""
        result = detector.detect_communities_leiden(barabasi_graph)
        total = sum(result['community_sizes'].values())
        assert total == barabasi_graph.vcount()

    def test_leiden_handles_directed_graph(self, detector, directed_graph):
        """Leiden should handle directed graphs by converting to undirected."""
        result = detector.detect_communities_leiden(directed_graph)
        assert 'partition' in result
        assert result['n_communities'] >= 1

    def test_leiden_resolution_parameter(self, detector, barabasi_graph):
        """Higher resolution should produce more (or equal) communities."""
        result_low = detector.detect_communities_leiden(
            barabasi_graph, resolution=0.5
        )
        result_high = detector.detect_communities_leiden(
            barabasi_graph, resolution=2.0
        )
        # Higher resolution generally yields more or equal communities
        assert result_high['n_communities'] >= result_low['n_communities']

    def test_leiden_reproducible_with_seed(self, detector, barabasi_graph):
        """Running Leiden twice with the same seed should give identical results."""
        result1 = detector.detect_communities_leiden(
            barabasi_graph, random_state=42
        )
        result2 = detector.detect_communities_leiden(
            barabasi_graph, random_state=42
        )
        assert result1['partition'] == result2['partition']
        assert result1['modularity'] == result2['modularity']

    def test_leiden_planted_partition(self, detector, planted_partition_graph):
        """Leiden should recover planted partition with high modularity."""
        result = detector.detect_communities_leiden(planted_partition_graph)
        # With strong planted structure, modularity should be high
        assert result['modularity'] > 0.3
        # Should find approximately the right number of communities
        assert result['n_communities'] >= 3


class TestLeidenFallback:
    """Tests for Leiden fallback to Louvain when dependencies are missing."""

    def test_falls_back_to_louvain_when_leiden_unavailable(
        self, detector, barabasi_graph
    ):
        """When HAS_LEIDEN is False, detect_communities_leiden should
        fall back to Louvain and still return a valid result."""
        with patch(
            'src.network_analysis.community_detection.HAS_LEIDEN', False
        ):
            result = detector.detect_communities_leiden(barabasi_graph)
            assert 'partition' in result
            assert 'modularity' in result
            assert result['n_communities'] > 1

    def test_fallback_covers_all_nodes(self, detector, barabasi_graph):
        """Fallback result should still cover every node."""
        with patch(
            'src.network_analysis.community_detection.HAS_LEIDEN', False
        ):
            result = detector.detect_communities_leiden(barabasi_graph)
            assert set(result['partition'].keys()) == set(
                barabasi_graph.vs['name']
            )


class TestResultFormatConsistency:
    """Verify that Leiden and Louvain return identically-structured results."""

    def test_same_keys_returned(self, detector, barabasi_graph):
        """Leiden and Louvain should return dicts with the same keys."""
        leiden = detector.detect_communities_leiden(barabasi_graph)
        louvain = detector.detect_communities_louvain(barabasi_graph)
        assert set(leiden.keys()) == set(louvain.keys())

    def test_partition_value_types(self, detector, barabasi_graph):
        """Partition values should be integers (community IDs)."""
        result = detector.detect_communities_leiden(barabasi_graph)
        for node, comm_id in result['partition'].items():
            assert isinstance(comm_id, int)
