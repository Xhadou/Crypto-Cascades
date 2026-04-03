"""Tests for graph construction from transaction data."""
import pytest
import pandas as pd
import numpy as np
import igraph as ig

from src.preprocessing.graph_builder import GraphBuilder
from src.utils.graph_adapter import name_to_idx


class TestGraphBuilderInit:
    """Tests for GraphBuilder initialization."""

    def test_creates_instance(self):
        builder = GraphBuilder()
        assert builder is not None

    def test_has_logger(self):
        builder = GraphBuilder()
        assert builder.logger is not None


# ---------------------------------------------------------------------------
# Helper to look up edge data by node names (replaces G[u][v] syntax)
# ---------------------------------------------------------------------------

def _edge_data(g: ig.Graph, src_name, tgt_name) -> dict:
    """Return edge attribute dict for the edge between two node names."""
    n2i = name_to_idx(g)
    eid = g.get_eid(n2i[src_name], n2i[tgt_name])
    return {a: g.es[eid][a] for a in g.es.attributes()}


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
        g = builder.build_transaction_graph(simple_tx)
        assert isinstance(g, ig.Graph)
        assert g.is_directed()

    def test_builds_undirected_graph(self, builder, simple_tx):
        g = builder.build_transaction_graph(simple_tx, directed=False)
        assert isinstance(g, ig.Graph)
        assert not g.is_directed()

    def test_correct_node_count(self, builder, simple_tx):
        g = builder.build_transaction_graph(simple_tx)
        assert g.vcount() == 4  # nodes 1, 2, 3, 4

    def test_correct_edge_count(self, builder, simple_tx):
        g = builder.build_transaction_graph(simple_tx)
        assert g.ecount() == 3

    def test_edges_have_weight(self, builder, simple_tx):
        g = builder.build_transaction_graph(simple_tx)
        for e in g.es:
            assert "weight" in e.attributes()

    def test_edges_have_count(self, builder, simple_tx):
        g = builder.build_transaction_graph(simple_tx)
        for e in g.es:
            assert "count" in e.attributes()

    def test_aggregates_multi_edges_by_default(self, builder, multi_edge_tx):
        g = builder.build_transaction_graph(multi_edge_tx)
        data = _edge_data(g, 1, 2)
        assert data["weight"] == pytest.approx(1.5)  # 1.0 + 0.5
        assert data["count"] == 2

    def test_aggregates_usd_value(self, builder, multi_edge_tx):
        g = builder.build_transaction_graph(multi_edge_tx)
        data = _edge_data(g, 1, 2)
        assert data["usd_value"] == pytest.approx(7500.0)  # 5000 + 2500

    def test_non_aggregated_still_aggregates_in_simple_graph(self, builder, multi_edge_tx):
        """When aggregate_multi_edges=False, _build_multigraph still aggregates
        because igraph does not use a MultiGraph."""
        g = builder.build_transaction_graph(
            multi_edge_tx, aggregate_multi_edges=False
        )
        data = _edge_data(g, 1, 2)
        # The multigraph path still aggregates onto the same edge
        assert data["weight"] == pytest.approx(1.5)
        assert data["count"] == 2

    def test_custom_weight_column(self, builder):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [2, 3],
            "btc_value": [1.0, 2.0],
            "usd_value": [5000.0, 10000.0],
        })
        g = builder.build_transaction_graph(df, weight_column="usd_value")
        data = _edge_data(g, 1, 2)
        assert data["weight"] == pytest.approx(5000.0)

    def test_preserves_all_nodes(self, builder):
        df = pd.DataFrame({
            "source_id": [1, 2, 3, 4, 5],
            "target_id": [6, 7, 8, 9, 10],
            "btc_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        g = builder.build_transaction_graph(df)
        assert g.vcount() == 10

    def test_handles_empty_dataframe(self, builder):
        df = pd.DataFrame({
            "source_id": pd.Series([], dtype=int),
            "target_id": pd.Series([], dtype=int),
            "btc_value": pd.Series([], dtype=float),
        })
        g = builder.build_transaction_graph(df)
        assert g.vcount() == 0
        assert g.ecount() == 0


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
        for key, g in graphs.items():
            assert isinstance(g, ig.Graph)

    def test_directed_by_default(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots)
        for g in graphs.values():
            assert g.is_directed()

    def test_undirected_option(self, builder, snapshots):
        graphs = builder.build_temporal_graphs(snapshots, directed=False)
        for g in graphs.values():
            assert not g.is_directed()

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
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)], directed=True)
        g.vs["name"] = [1, 2, 3, 4]
        g["_name_to_idx"] = {1: 0, 2: 1, 3: 2, 4: 3}
        return g

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
        g = builder.add_node_attributes(graph, activity_df)
        n2i = name_to_idx(g)
        assert g.vs[n2i[1]]["net_btc"] == 10.0
        assert g.vs[n2i[2]]["total_tx"] == 30

    def test_returns_same_graph_object(self, builder, graph, activity_df):
        g = builder.add_node_attributes(graph, activity_df)
        assert g is graph  # modifies in place

    def test_handles_missing_nodes_gracefully(self, builder, activity_df):
        g = ig.Graph(n=2, edges=[(0, 1)], directed=True)
        g.vs["name"] = [1, 2]
        g["_name_to_idx"] = {1: 0, 2: 1}
        g = builder.add_node_attributes(g, activity_df)
        n2i = name_to_idx(g)
        assert g.vs[n2i[1]]["net_btc"] == 10.0
        # Node 3 and 4 are in activity_df but not in graph -- no error

    def test_custom_attribute_list(self, builder, graph, activity_df):
        g = builder.add_node_attributes(
            graph, activity_df, attributes=["net_btc"]
        )
        n2i = name_to_idx(g)
        assert g.vs[n2i[1]]["net_btc"] == 10.0
        assert "total_tx" not in g.vs.attributes()

    def test_handles_nodes_not_in_activity(self, builder, activity_df):
        g = ig.Graph(n=4, edges=[(0, 1), (2, 3)], directed=True)
        g.vs["name"] = [1, 2, 99, 100]
        g["_name_to_idx"] = {1: 0, 2: 1, 99: 2, 100: 3}
        g = builder.add_node_attributes(g, activity_df)
        n2i = name_to_idx(g)
        assert g.vs[n2i[1]]["net_btc"] == 10.0
        assert g.vs[n2i[99]]["net_btc"] is None


class TestFilterGraph:
    """Tests for graph filtering."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    @pytest.fixture
    def weighted_graph(self):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)], directed=True)
        g.vs["name"] = [1, 2, 3, 4]
        g["_name_to_idx"] = {1: 0, 2: 1, 3: 2, 4: 3}
        g.es["weight"] = [5.0, 0.1, 10.0, 2.0]
        g.es["count"] = [3, 1, 5, 2]
        return g

    def test_returns_copy(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_weight=1.0)
        assert filtered is not weighted_graph

    def test_filters_by_min_weight(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_weight=3.0)
        names_in = set(filtered.vs["name"])
        n2i = name_to_idx(filtered)
        # Only edges with weight >= 3.0: (1,2)=5.0, (3,4)=10.0
        assert 1 in names_in and 2 in names_in
        assert 3 in names_in and 4 in names_in
        try:
            filtered.get_eid(n2i[2], n2i[3])
            has_2_3 = True
        except ig.InternalError:
            has_2_3 = False
        assert not has_2_3  # weight=0.1

    def test_filters_by_min_count(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph, min_count=3)
        n2i = name_to_idx(filtered)
        # Only edges with count >= 3: (1,2)=3, (3,4)=5
        assert 1 in n2i and 2 in n2i
        assert 3 in n2i and 4 in n2i
        try:
            filtered.get_eid(n2i[2], n2i[3])
            has_2_3 = True
        except ig.InternalError:
            has_2_3 = False
        assert not has_2_3  # count=1

    def test_filters_by_min_degree(self, builder):
        # Create a star with hub node 1
        nodes = [1, 2, 3, 4, 5, 6, 10, 11]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [(n2i[1], n2i[i]) for i in range(2, 7)]
        edges.append((n2i[10], n2i[11]))
        g = ig.Graph(n=len(nodes), edges=edges, directed=True)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i
        g.es["weight"] = [1.0] * g.ecount()
        g.es["count"] = [1] * g.ecount()

        filtered = builder.filter_graph(g, min_degree=3)
        assert 1 in set(filtered.vs["name"])
        assert 10 not in set(filtered.vs["name"])  # degree=1

    def test_no_filter_returns_all(self, builder, weighted_graph):
        filtered = builder.filter_graph(weighted_graph)
        assert filtered.vcount() == weighted_graph.vcount()
        assert filtered.ecount() == weighted_graph.ecount()


class TestGetLargestComponent:
    """Tests for extracting the largest connected component."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def test_extracts_largest_weakly_connected(self, builder):
        # Component 1: nodes 1,2,3  |  Component 2: nodes 10,11
        nodes = [1, 2, 3, 10, 11]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [(n2i[1], n2i[2]), (n2i[2], n2i[3]), (n2i[10], n2i[11])]
        g = ig.Graph(n=len(nodes), edges=edges, directed=True)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i

        result = builder.get_largest_component(g)
        assert result.vcount() == 3

    def test_extracts_largest_strongly_connected(self, builder):
        # Strongly connected: 1->2->3->1  |  Not strongly connected: 4->1
        nodes = [1, 2, 3, 4]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [
            (n2i[1], n2i[2]), (n2i[2], n2i[3]), (n2i[3], n2i[1]),
            (n2i[4], n2i[1]),
        ]
        g = ig.Graph(n=len(nodes), edges=edges, directed=True)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i

        result = builder.get_largest_component(g, strongly_connected=True)
        assert result.vcount() == 3
        assert set(result.vs["name"]) == {1, 2, 3}

    def test_undirected_graph(self, builder):
        nodes = [1, 2, 3, 4, 10, 11]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [
            (n2i[1], n2i[2]), (n2i[2], n2i[3]), (n2i[3], n2i[4]),
            (n2i[10], n2i[11]),
        ]
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i

        result = builder.get_largest_component(g)
        assert result.vcount() == 4

    def test_returns_new_graph(self, builder):
        nodes = [1, 2, 3]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [(n2i[1], n2i[2]), (n2i[2], n2i[3])]
        g = ig.Graph(n=len(nodes), edges=edges, directed=True)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i

        result = builder.get_largest_component(g)
        # induced_subgraph returns a new graph
        result.add_vertices(1)
        result.vs[result.vcount() - 1]["name"] = 999
        assert 999 not in g.vs["name"]

    def test_single_component_graph(self, builder):
        nodes = [1, 2, 3]
        n2i = {n: i for i, n in enumerate(nodes)}
        edges = [(n2i[1], n2i[2]), (n2i[2], n2i[3]), (n2i[3], n2i[1])]
        g = ig.Graph(n=len(nodes), edges=edges, directed=True)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i

        result = builder.get_largest_component(g)
        assert result.vcount() == g.vcount()

    def test_empty_graph_returns_original(self, builder):
        g = ig.Graph(directed=True)
        result = builder.get_largest_component(g)
        assert result.vcount() == 0


class TestComputeGraphStats:
    """Tests for graph statistics computation."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def test_returns_dict(self, builder):
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
        g.vs["name"] = [1, 2, 3]
        stats = builder.compute_graph_stats(g)
        assert isinstance(stats, dict)

    def test_basic_stats_present(self, builder):
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2), (2, 0)], directed=True)
        g.vs["name"] = [1, 2, 3]
        stats = builder.compute_graph_stats(g)
        assert "nodes" in stats
        assert "edges" in stats
        assert "density" in stats
        assert "avg_degree" in stats
        assert "is_directed" in stats

    def test_node_and_edge_count_correct(self, builder):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)], directed=True)
        g.vs["name"] = [1, 2, 3, 4]
        stats = builder.compute_graph_stats(g)
        assert stats["nodes"] == 4
        assert stats["edges"] == 3

    def test_degree_stats_present(self, builder):
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2), (2, 0)], directed=True)
        g.vs["name"] = [1, 2, 3]
        stats = builder.compute_graph_stats(g)
        assert "min_degree" in stats
        assert "max_degree" in stats
        assert "median_degree" in stats

    def test_directed_flag(self, builder):
        g_dir = ig.Graph(n=2, edges=[(0, 1)], directed=True)
        g_dir.vs["name"] = [1, 2]
        g_undir = ig.Graph(n=2, edges=[(0, 1)], directed=False)
        g_undir.vs["name"] = [1, 2]
        assert builder.compute_graph_stats(g_dir)["is_directed"] is True
        assert builder.compute_graph_stats(g_undir)["is_directed"] is False

    def test_empty_graph_stats(self, builder):
        g = ig.Graph(directed=True)
        stats = builder.compute_graph_stats(g)
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["density"] == 0
        assert stats["avg_degree"] == 0

    def test_clustering_for_undirected(self, builder):
        # Triangle
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2), (2, 0)], directed=False)
        g.vs["name"] = [1, 2, 3]
        stats = builder.compute_graph_stats(g)
        assert "avg_clustering" in stats
        assert stats["avg_clustering"] == pytest.approx(1.0)

    def test_path_length_for_connected_undirected(self, builder):
        # Path graph: 0-1-2-3-4
        g = ig.Graph(n=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)], directed=False)
        g.vs["name"] = [0, 1, 2, 3, 4]
        stats = builder.compute_graph_stats(g)
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
        g = builder.build_transaction_graph(df)
        return g, df

    def test_creates_subgraph_in_time_window(self, builder, full_graph_and_df):
        g, df = full_graph_and_df
        start = pd.Timestamp("2017-10-03")
        end = pd.Timestamp("2017-10-12")
        sub = builder.create_subgraph_by_time(g, df, start, end)
        # Only edges on 2017-10-05 and 2017-10-10 fall in range
        assert sub.ecount() == 2

    def test_preserves_directedness(self, builder, full_graph_and_df):
        g, df = full_graph_and_df
        start = pd.Timestamp("2017-10-01")
        end = pd.Timestamp("2017-10-31")
        sub = builder.create_subgraph_by_time(g, df, start, end)
        assert sub.is_directed() == g.is_directed()


class TestMergeGraphs:
    """Tests for merging multiple graphs."""

    @pytest.fixture
    def builder(self):
        return GraphBuilder()

    def _make_graph(self, edges_with_data, directed=True):
        """Helper to build an igraph from (src, tgt, weight, count) tuples."""
        all_nodes = set()
        for src, tgt, *_ in edges_with_data:
            all_nodes.add(src)
            all_nodes.add(tgt)
        nodes = sorted(all_nodes)
        n2i = {n: i for i, n in enumerate(nodes)}
        ig_edges = [(n2i[s], n2i[t]) for s, t, *_ in edges_with_data]
        g = ig.Graph(n=len(nodes), edges=ig_edges, directed=directed)
        g.vs["name"] = nodes
        g["_name_to_idx"] = n2i
        g.es["weight"] = [e[2] for e in edges_with_data]
        g.es["count"] = [e[3] for e in edges_with_data]
        return g

    def test_merges_two_disjoint_graphs(self, builder):
        g1 = self._make_graph([(1, 2, 1.0, 1)])
        g2 = self._make_graph([(3, 4, 2.0, 1)])
        merged = builder.merge_graphs([g1, g2])
        assert merged.vcount() == 4
        assert merged.ecount() == 2

    def test_aggregates_overlapping_edges(self, builder):
        g1 = self._make_graph([(1, 2, 1.0, 1)])
        g2 = self._make_graph([(1, 2, 3.0, 2)])
        merged = builder.merge_graphs([g1, g2], aggregate_weights=True)
        assert merged.ecount() == 1
        data = _edge_data(merged, 1, 2)
        assert data["weight"] == pytest.approx(4.0)
        assert data["count"] == 3

    def test_no_aggregation_replaces_edge(self, builder):
        g1 = self._make_graph([(1, 2, 1.0, 1)])
        g2 = self._make_graph([(1, 2, 3.0, 2)])
        merged = builder.merge_graphs([g1, g2], aggregate_weights=False)
        assert merged.ecount() == 1
        # Without aggregation, the second graph's edge data overwrites the first
        data = _edge_data(merged, 1, 2)
        assert data["weight"] == pytest.approx(3.0)

    def test_empty_list_returns_empty_digraph(self, builder):
        merged = builder.merge_graphs([])
        assert isinstance(merged, ig.Graph)
        assert merged.is_directed()
        assert merged.vcount() == 0

    def test_preserves_directed_type(self, builder):
        g1 = self._make_graph([(1, 2, 1.0, 1)], directed=True)
        merged = builder.merge_graphs([g1])
        assert merged.is_directed()

    def test_preserves_undirected_type(self, builder):
        g1 = self._make_graph([(1, 2, 1.0, 1)], directed=False)
        merged = builder.merge_graphs([g1])
        assert not merged.is_directed()

    def test_merges_three_graphs(self, builder):
        graphs = []
        for i in range(3):
            g = self._make_graph([(i * 10, i * 10 + 1, float(i + 1), 1)])
            graphs.append(g)
        merged = builder.merge_graphs(graphs)
        assert merged.vcount() == 6
        assert merged.ecount() == 3
