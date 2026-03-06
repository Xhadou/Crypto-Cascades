"""Tests for graph construction from transaction data."""
import pytest
import pandas as pd
import numpy as np
import networkx as nx

from src.preprocessing.graph_builder import GraphBuilder


class TestGraphBuilderInit:
    """Tests for GraphBuilder initialization."""

    def test_creates_instance(self):
        builder = GraphBuilder()
        assert builder is not None

    def test_has_logger(self):
        builder = GraphBuilder()
        assert builder.logger is not None


class TestBuildTransactionGraph:
    """Tests for build_transaction_graph method."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def simple_tx(self):
        return pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [2, 3, 4],
            "btc_value": [1.0, 2.0, 0.5],
            "usd_value": [5000.0, 10000.0, 2500.0],
        })

    @pytest.fixture
    def multi_edge_tx(self):
        """Transactions with multiple edges between the same pair."""
        return pd.DataFrame({
            "source_id": [1, 1, 1, 2],
            "target_id": [2, 2, 3, 3],
            "btc_value": [1.0, 0.5, 2.0, 0.1],
            "usd_value": [5000.0, 2500.0, 10000.0, 500.0],
        })

    def test_builds_directed_graph_by_default(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx)
        assert isinstance(G, nx.DiGraph)

    def test_builds_undirected_graph(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx, directed=False)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_correct_node_count(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx)
        assert G.number_of_nodes() == 4  # nodes 1, 2, 3, 4

    def test_correct_edge_count(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx)
        assert G.number_of_edges() == 3

    def test_edges_have_weight(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx)
        for u, v, data in G.edges(data=True):
            assert "weight" in data

    def test_edges_have_count(self, builder, simple_tx):
        G = builder.build_transaction_graph(simple_tx)
        for u, v, data in G.edges(data=True):
            assert "count" in data

    def test_aggregates_multi_edges_by_default(self, builder, multi_edge_tx):
        G = builder.build_transaction_graph(multi_edge_tx)
        # 1->2 appears twice, should be aggregated into one edge
        assert G.has_edge(1, 2)
        assert G[1][2]["weight"] == pytest.approx(1.5)  # 1.0 + 0.5
        assert G[1][2]["count"] == 2

    def test_aggregates_usd_value(self, builder, multi_edge_tx):
        G = builder.build_transaction_graph(multi_edge_tx)
        assert G[1][2]["usd_value"] == pytest.approx(7500.0)  # 5000 + 2500

    def test_non_aggregated_still_aggregates_in_simple_graph(self, builder, multi_edge_tx):
        """When aggregate_multi_edges=False, _build_multigraph still aggregates
        because it uses a simple DiGraph (not a MultiDiGraph)."""
        G = builder.build_transaction_graph(
            multi_edge_tx, aggregate_multi_edges=False
        )
        assert G.has_edge(1, 2)
        # The multigraph path still aggregates onto the same edge
        assert G[1][2]["weight"] == pytest.approx(1.5)
        assert G[1][2]["count"] == 2

    def test_custom_weight_column(self, builder):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [2, 3],
            "btc_value": [1.0, 2.0],
            "usd_value": [5000.0, 10000.0],
        })
        G = builder.build_transaction_graph(df, weight_column="usd_value")
        assert G[1][2]["weight"] == pytest.approx(5000.0)

    def test_preserves_all_nodes(self, builder):
        df = pd.DataFrame({
            "source_id": [1, 2, 3, 4, 5],
            "target_id": [6, 7, 8, 9, 10],
            "btc_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        G = builder.build_transaction_graph(df)
        assert G.number_of_nodes() == 10

    def test_handles_empty_dataframe(self, builder):
        df = pd.DataFrame({
            "source_id": pd.Series([], dtype=int),
            "target_id": pd.Series([], dtype=int),
            "btc_value": pd.Series([], dtype=float),
        })
        G = builder.build_transaction_graph(df)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestBuildTemporalGraphs:
    """Tests for building temporal graphs from snapshots."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def snapshots(self):
        return {
            "2017-10-01": pd.DataFrame({
                "source_id": [1, 2],
                "target_id": [2, 3],
                "btc_value": [1.0, 2.0],
            }),
            "2017-10-02": pd.DataFrame({
                "source_id": [3, 4],
                "target_id": [4, 5],
                "btc_value": [0.5, 1.5],
            }),
        }

    def test_returns_dict_of_graphs(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots)
        assert isinstance(graphs, dict)
        assert len(graphs) == 2

    def test_each_value_is_a_graph(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots)
        for key, G in graphs.items():
            assert isinstance(G, (nx.Graph, nx.DiGraph))

    def test_directed_by_default(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots)
        for G in graphs.values():
            assert G.is_directed()

    def test_undirected_option(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots, directed=False)
        for G in graphs.values():
            assert not G.is_directed()

    def test_keys_match_snapshot_keys(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots)
        assert set(graphs.keys()) == set(snapshots.keys())


class TestAddNodeAttributes:
    """Tests for adding node attributes from activity data."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def graph(self):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        return G

    @pytest.fixture
    def activity_df(self):
        return pd.DataFrame({
            "wallet_id": [1, 2, 3, 4],
            "net_btc": [10.0, -5.0, 3.0, -8.0],
            "total_tx": [50, 30, 20, 10],
            "btc_in": [15.0, 10.0, 5.0, 2.0],
            "btc_out": [5.0, 15.0, 2.0, 10.0],
        })

    def test_adds_attributes_to_nodes(self, builder, graph, activity_df):
        G = builder.add_node_attributes(graph, activity_df)
        assert G.nodes[1]["net_btc"] == 10.0
        assert G.nodes[2]["total_tx"] == 30

    def test_returns_same_graph_object(self, builder, graph, activity_df):
        G = builder.add_node_attributes(graph, activity_df)
        assert G is graph  # modifies in place

    def test_handles_missing_nodes_gracefully(self, builder, activity_df):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2)])  # Only nodes 1 and 2
        G = builder.add_node_attributes(G, activity_df)
        assert G.nodes[1].get("net_btc") == 10.0
        # Node 3 and 4 are in activity_df but not in graph -- no error

    def test_custom_attribute_list(self, builder, graph, activity_df):
        G = builder.add_node_attributes(
            graph, activity_df, attributes=["net_btc"]
        )
        assert "net_btc" in G.nodes[1]
        assert "total_tx" not in G.nodes[1]

    def test_handles_nodes_not_in_activity(self, builder, activity_df):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (99, 100)])  # 99, 100 not in activity_df
        G = builder.add_node_attributes(G, activity_df)
        assert G.nodes[1].get("net_btc") == 10.0
        assert "net_btc" not in G.nodes[99]


class TestFilterGraph:
    """Tests for graph filtering."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def weighted_graph(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=5.0, count=3)
        G.add_edge(2, 3, weight=0.1, count=1)
        G.add_edge(3, 4, weight=10.0, count=5)
        G.add_edge(4, 1, weight=2.0, count=2)
        return G

    def test_returns_copy(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_weight=1.0)
        assert filtered is not weighted_graph

    def test_filters_by_min_weight(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_weight=3.0)
        # Only edges with weight >= 3.0: (1,2)=5.0, (3,4)=10.0
        assert filtered.has_edge(1, 2)
        assert filtered.has_edge(3, 4)
        assert not filtered.has_edge(2, 3)  # weight=0.1

    def test_filters_by_min_count(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_count=3)
        # Only edges with count >= 3: (1,2)=3, (3,4)=5
        assert filtered.has_edge(1, 2)
        assert filtered.has_edge(3, 4)
        assert not filtered.has_edge(2, 3)  # count=1

    def test_filters_by_min_degree(self, builder):
        G = nx.DiGraph()
        # Create a star with hub node 1
        for i in range(2, 7):
            G.add_edge(1, i, weight=1.0, count=1)
        # Add isolated pair
        G.add_edge(10, 11, weight=1.0, count=1)
        filtered = builder.filter_graph(G, min_degree=3)
        assert 1 in filtered.nodes()
        assert 10 not in filtered.nodes()  # degree=1

    def test_no_filter_returns_all(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph)
        assert filtered.number_of_nodes() == weighted_graph.number_of_nodes()
        assert filtered.number_of_edges() == weighted_graph.number_of_edges()


class TestGetLargestComponent:
    """Tests for extracting the largest connected component."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def test_extracts_largest_weakly_connected(self, builder):
        G = nx.DiGraph()
        # Component 1: 3 nodes
        G.add_edges_from([(1, 2), (2, 3)])
        # Component 2: 2 nodes
        G.add_edges_from([(10, 11)])
        result = builder.get_largest_component(G)
        assert result.number_of_nodes() == 3

    def test_extracts_largest_strongly_connected(self, builder):
        G = nx.DiGraph()
        # Strongly connected: 1->2->3->1
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        # Not strongly connected to above
        G.add_edge(4, 1)
        result = builder.get_largest_component(G, strongly_connected=True)
        assert result.number_of_nodes() == 3
        assert set(result.nodes()) == {1, 2, 3}

    def test_undirected_graph(self, builder):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        G.add_edge(10, 11)
        result = builder.get_largest_component(G)
        assert result.number_of_nodes() == 4

    def test_returns_copy(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3)])
        result = builder.get_largest_component(G)
        # Should be a copy, not a view
        result.add_node(999)
        assert 999 not in G.nodes()

    def test_single_component_graph(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        result = builder.get_largest_component(G)
        assert result.number_of_nodes() == G.number_of_nodes()

    def test_empty_graph_returns_original(self, builder):
        G = nx.DiGraph()
        result = builder.get_largest_component(G)
        assert result.number_of_nodes() == 0


class TestComputeGraphStats:
    """Tests for graph statistics computation."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def test_returns_dict(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3)])
        stats = builder.compute_graph_stats(G)
        assert isinstance(stats, dict)

    def test_basic_stats_present(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        stats = builder.compute_graph_stats(G)
        assert "nodes" in stats
        assert "edges" in stats
        assert "density" in stats
        assert "avg_degree" in stats
        assert "is_directed" in stats

    def test_node_and_edge_count_correct(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        stats = builder.compute_graph_stats(G)
        assert stats["nodes"] == 4
        assert stats["edges"] == 3

    def test_degree_stats_present(self, builder):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        stats = builder.compute_graph_stats(G)
        assert "min_degree" in stats
        assert "max_degree" in stats
        assert "median_degree" in stats

    def test_directed_flag(self, builder):
        G_dir = nx.DiGraph()
        G_dir.add_edge(1, 2)
        G_undir = nx.Graph()
        G_undir.add_edge(1, 2)
        assert builder.compute_graph_stats(G_dir)["is_directed"] is True
        assert builder.compute_graph_stats(G_undir)["is_directed"] is False

    def test_empty_graph_stats(self, builder):
        G = nx.DiGraph()
        stats = builder.compute_graph_stats(G)
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["density"] == 0
        assert stats["avg_degree"] == 0

    def test_clustering_for_undirected(self, builder):
        G = nx.Graph()
        # Triangle
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        stats = builder.compute_graph_stats(G)
        assert "avg_clustering" in stats
        assert stats["avg_clustering"] == pytest.approx(1.0)

    def test_path_length_for_connected_undirected(self, builder):
        G = nx.path_graph(5)  # 0-1-2-3-4
        stats = builder.compute_graph_stats(G)
        assert "avg_path_length" in stats
        assert stats["avg_path_length"] > 0


class TestCreateSubgraphByTime:
    """Tests for time-windowed subgraph creation."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def full_graph_and_df(self):
        df = pd.DataFrame({
            "source_id": [1, 2, 3, 4, 5],
            "target_id": [2, 3, 4, 5, 1],
            "btc_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "datetime": pd.to_datetime([
                "2017-10-01",
                "2017-10-05",
                "2017-10-10",
                "2017-10-15",
                "2017-10-20",
            ]),
        })
        builder = GraphBuilder()
        G = builder.build_transaction_graph(df)
        return G, df

    def test_creates_subgraph_in_time_window(self, builder, full_graph_and_df):
        G, df = full_graph_and_df
        start = pd.Timestamp("2017-10-03")
        end = pd.Timestamp("2017-10-12")
        sub = builder.create_subgraph_by_time(G, df, start, end)
        # Only edges on 2017-10-05 and 2017-10-10 fall in range
        assert sub.number_of_edges() == 2

    def test_preserves_directedness(self, builder, full_graph_and_df):
        G, df = full_graph_and_df
        start = pd.Timestamp("2017-10-01")
        end = pd.Timestamp("2017-10-31")
        sub = builder.create_subgraph_by_time(G, df, start, end)
        assert sub.is_directed() == G.is_directed()


class TestMergeGraphs:
    """Tests for merging multiple graphs."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def test_merges_two_disjoint_graphs(self, builder):
        G1 = nx.DiGraph()
        G1.add_edge(1, 2, weight=1.0, count=1)
        G2 = nx.DiGraph()
        G2.add_edge(3, 4, weight=2.0, count=1)
        merged = builder.merge_graphs([G1, G2])
        assert merged.number_of_nodes() == 4
        assert merged.number_of_edges() == 2

    def test_aggregates_overlapping_edges(self, builder):
        G1 = nx.DiGraph()
        G1.add_edge(1, 2, weight=1.0, count=1)
        G2 = nx.DiGraph()
        G2.add_edge(1, 2, weight=3.0, count=2)
        merged = builder.merge_graphs([G1, G2], aggregate_weights=True)
        assert merged.number_of_edges() == 1
        assert merged[1][2]["weight"] == pytest.approx(4.0)
        assert merged[1][2]["count"] == 3

    def test_no_aggregation_replaces_edge(self, builder):
        G1 = nx.DiGraph()
        G1.add_edge(1, 2, weight=1.0, count=1)
        G2 = nx.DiGraph()
        G2.add_edge(1, 2, weight=3.0, count=2)
        merged = builder.merge_graphs([G1, G2], aggregate_weights=False)
        assert merged.number_of_edges() == 1
        # Without aggregation, the second graph's edge data overwrites the first
        assert merged[1][2]["weight"] == pytest.approx(3.0)

    def test_empty_list_returns_empty_digraph(self, builder):
        merged = builder.merge_graphs([])
        assert isinstance(merged, nx.DiGraph)
        assert merged.number_of_nodes() == 0

    def test_preserves_directed_type(self, builder):
        G1 = nx.DiGraph()
        G1.add_edge(1, 2, weight=1.0, count=1)
        merged = builder.merge_graphs([G1])
        assert isinstance(merged, nx.DiGraph)

    def test_preserves_undirected_type(self, builder):
        G1 = nx.Graph()
        G1.add_edge(1, 2, weight=1.0, count=1)
        merged = builder.merge_graphs([G1])
        assert isinstance(merged, nx.Graph)

    def test_merges_three_graphs(self, builder):
        graphs = []
        for i in range(3):
            G = nx.DiGraph()
            G.add_edge(i * 10, i * 10 + 1, weight=float(i + 1), count=1)
            graphs.append(G)
        merged = builder.merge_graphs(graphs)
        assert merged.number_of_nodes() == 6
        assert merged.number_of_edges() == 3
