"""
Graph Builder

Constructs igraph graphs from ORBITAAL transaction data and other sources.
Supports both directed and undirected graphs with edge aggregation.
"""

import igraph as ig
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.graph_adapter import build_igraph_from_df, name_to_idx


class GraphBuilder:
    """Build igraph graphs from transaction data."""

    def __init__(self):
        """Initialize the graph builder."""
        self.logger = get_logger(__name__)

    def build_transaction_graph(
        self,
        df: pd.DataFrame,
        directed: bool = True,
        weight_column: str = 'btc_value',
        aggregate_multi_edges: bool = True
    ) -> ig.Graph:
        """
        Build transaction graph from edge list.

        Args:
            df: DataFrame with source_id, target_id, and optional weight column
            directed: Whether to create directed graph
            weight_column: Column to use as edge weight
            aggregate_multi_edges: Whether to aggregate multiple edges between same nodes

        Returns:
            igraph Graph with edge attributes (weight, count, usd_value)
        """
        self.logger.info(f"Building {'directed' if directed else 'undirected'} transaction graph...")

        if aggregate_multi_edges:
            g = self._build_aggregated_graph(df, directed, weight_column)
        else:
            g = self._build_multigraph(df, directed, weight_column)

        self.logger.info(f"Built graph: {g.vcount():,} nodes, {g.ecount():,} edges")
        return g

    def _build_aggregated_graph(
        self,
        df: pd.DataFrame,
        directed: bool,
        weight_column: str
    ) -> ig.Graph:
        """Build graph with aggregated edges (optimized for large datasets)."""
        self.logger.info("Aggregating edges...")

        # Aggregate edges
        agg_dict = {weight_column: 'sum'}
        if 'usd_value' in df.columns and weight_column != 'usd_value':
            agg_dict['usd_value'] = 'sum'

        grouped = df.groupby(['source_id', 'target_id']).agg(agg_dict).reset_index()

        # Count edges
        edge_counts = df.groupby(['source_id', 'target_id']).size().reset_index(name='count')
        grouped = grouped.merge(edge_counts, on=['source_id', 'target_id'])

        self.logger.info(f"Aggregated to {len(grouped):,} unique edges")

        # Handle empty DataFrames
        if len(grouped) == 0:
            g = ig.Graph(directed=directed)
            g["_name_to_idx"] = {}
            return g

        # Build igraph from aggregated DataFrame
        # Sorted unique node list for deterministic indexing.
        nodes = np.union1d(
            grouped["source_id"].unique(), grouped["target_id"].unique()
        )
        nodes.sort()
        n2i: dict = {n: i for i, n in enumerate(nodes)}

        src_idx = grouped["source_id"].map(n2i).values
        tgt_idx = grouped["target_id"].map(n2i).values

        g = ig.Graph(
            n=len(nodes),
            edges=list(zip(src_idx, tgt_idx)),
            directed=directed,
        )
        g.vs["name"] = nodes.tolist()

        # Set edge attributes
        g.es["weight"] = grouped[weight_column].values.tolist()
        g.es["count"] = grouped["count"].values.tolist()
        if "usd_value" in grouped.columns:
            g.es["usd_value"] = grouped["usd_value"].values.tolist()

        # Cache reverse mapping
        g["_name_to_idx"] = n2i

        return g

    def _build_multigraph(
        self,
        df: pd.DataFrame,
        directed: bool,
        weight_column: str
    ) -> ig.Graph:
        """Build graph allowing multiple edges (aggregated onto same edge)."""
        # Aggregate via pandas first, then build igraph
        agg_cols = [weight_column]
        if 'usd_value' in df.columns and weight_column != 'usd_value':
            agg_cols.append('usd_value')

        agg_dict = {c: 'sum' for c in agg_cols if c in df.columns}
        grouped = df.groupby(['source_id', 'target_id']).agg(agg_dict).reset_index()
        edge_counts = df.groupby(['source_id', 'target_id']).size().reset_index(name='count')
        grouped = grouped.merge(edge_counts, on=['source_id', 'target_id'])

        if len(grouped) == 0:
            g = ig.Graph(directed=directed)
            g["_name_to_idx"] = {}
            return g

        nodes = np.union1d(
            grouped["source_id"].unique(), grouped["target_id"].unique()
        )
        nodes.sort()
        n2i: dict = {n: i for i, n in enumerate(nodes)}

        src_idx = grouped["source_id"].map(n2i).values
        tgt_idx = grouped["target_id"].map(n2i).values

        g = ig.Graph(
            n=len(nodes),
            edges=list(zip(src_idx, tgt_idx)),
            directed=directed,
        )
        g.vs["name"] = nodes.tolist()

        g.es["weight"] = grouped[weight_column].values.tolist()
        g.es["count"] = grouped["count"].values.tolist()
        if "usd_value" in grouped.columns:
            g.es["usd_value"] = grouped["usd_value"].values.tolist()

        g["_name_to_idx"] = n2i
        return g

    def build_temporal_graphs(
        self,
        snapshots: Dict[str, pd.DataFrame],
        directed: bool = True
    ) -> Dict[str, ig.Graph]:
        """
        Build graphs for each temporal snapshot.

        Args:
            snapshots: Dict mapping time period to transaction DataFrame
            directed: Whether to create directed graphs

        Returns:
            Dict mapping time period to igraph Graph
        """
        self.logger.info(f"Building {len(snapshots)} temporal graphs...")

        graphs = {}
        for period, df in tqdm(snapshots.items(), desc="Building temporal graphs"):
            graphs[period] = self.build_transaction_graph(df, directed=directed)

        return graphs

    def add_node_attributes(
        self,
        g: ig.Graph,
        activity_df: pd.DataFrame,
        attributes: Optional[List[str]] = None
    ) -> ig.Graph:
        """
        Add wallet activity attributes to nodes.

        Args:
            g: igraph Graph
            activity_df: DataFrame with wallet_id and activity metrics
            attributes: List of columns to add as attributes (default: all available)

        Returns:
            Graph with node attributes (modified in place)
        """
        if attributes is None:
            attributes = ['net_btc', 'net_usd', 'total_tx', 'btc_in', 'btc_out',
                         'usd_in', 'usd_out', 'tx_in_count', 'tx_out_count']

        # Filter to available columns
        available = [a for a in attributes if a in activity_df.columns]

        activity_dict = activity_df.set_index('wallet_id').to_dict('index')
        n2i = name_to_idx(g)

        added_count = 0
        for attr_name in available:
            # Initialize attribute with None for all vertices
            values = [None] * g.vcount()
            for wallet_id, row_data in activity_dict.items():
                if wallet_id in n2i and attr_name in row_data:
                    values[n2i[wallet_id]] = row_data[attr_name]
                    added_count += 1
            g.vs[attr_name] = values

        self.logger.info(f"Added {added_count} node attributes")
        return g

    def filter_graph(
        self,
        g: ig.Graph,
        min_degree: int = 0,
        min_weight: float = 0.0,
        min_count: int = 0
    ) -> ig.Graph:
        """
        Filter graph by node degree and edge weight.

        Args:
            g: Input graph
            min_degree: Minimum node degree to keep
            min_weight: Minimum edge weight to keep
            min_count: Minimum edge count to keep

        Returns:
            Filtered graph (new graph)
        """
        original_nodes = g.vcount()
        original_edges = g.ecount()

        # Collect edges that pass the filter
        keep_eids = []
        for e in g.es:
            if min_weight > 0 and e["weight"] < min_weight:
                continue
            if min_count > 0 and e["count"] < min_count:
                continue
            keep_eids.append(e.index)

        # Build a new graph from kept edges
        if len(keep_eids) == g.ecount():
            g_filtered = g.copy()
        else:
            # Extract edge data for kept edges
            names = g.vs["name"]
            edge_attrs = g.es.attributes()
            new_edges_src = []
            new_edges_tgt = []
            attr_values = {a: [] for a in edge_attrs}

            for eid in keep_eids:
                e = g.es[eid]
                new_edges_src.append(names[e.source])
                new_edges_tgt.append(names[e.target])
                for a in edge_attrs:
                    attr_values[a].append(e[a])

            # Unique nodes from kept edges
            node_set = set(new_edges_src) | set(new_edges_tgt)
            new_nodes = sorted(node_set)
            n2i = {n: i for i, n in enumerate(new_nodes)}

            mapped_edges = [
                (n2i[s], n2i[t])
                for s, t in zip(new_edges_src, new_edges_tgt)
            ]

            g_filtered = ig.Graph(
                n=len(new_nodes),
                edges=mapped_edges,
                directed=g.is_directed(),
            )
            g_filtered.vs["name"] = new_nodes
            for a in edge_attrs:
                g_filtered.es[a] = attr_values[a]
            g_filtered["_name_to_idx"] = n2i

        # Filter nodes by degree (single pass, matching original NetworkX behaviour)
        if min_degree > 0:
            low_degree = [
                v.index for v in g_filtered.vs
                if g_filtered.degree(v.index) < min_degree
            ]
            if low_degree:
                g_filtered.delete_vertices(low_degree)
                # Rebuild name mapping after deletion
                if g_filtered.vcount() > 0:
                    g_filtered["_name_to_idx"] = {
                        name: idx for idx, name in enumerate(g_filtered.vs["name"])
                    }

        self.logger.info(
            f"Filtered graph: {original_nodes:,} -> {g_filtered.vcount():,} nodes, "
            f"{original_edges:,} -> {g_filtered.ecount():,} edges"
        )
        return g_filtered

    def get_largest_component(
        self,
        g: ig.Graph,
        strongly_connected: bool = False
    ) -> ig.Graph:
        """
        Extract largest connected component.

        Args:
            g: Input graph
            strongly_connected: For directed graphs, use strongly connected component

        Returns:
            Subgraph of largest component
        """
        original_size = g.vcount()

        if original_size == 0:
            self.logger.warning("No connected components found")
            return g

        if g.is_directed():
            if strongly_connected:
                mode = "strong"
            else:
                mode = "weak"
        else:
            mode = "weak"

        components = g.connected_components(mode=mode)

        if len(components) == 0:
            self.logger.warning("No connected components found")
            return g

        # Find largest component
        largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
        largest = components[largest_idx]

        result = g.induced_subgraph(largest)
        # Rebuild the name-to-idx cache
        if result.vcount() > 0:
            result["_name_to_idx"] = {
                name: idx for idx, name in enumerate(result.vs["name"])
            }

        self.logger.info(
            f"Extracted largest component: {len(largest):,} nodes "
            f"({len(largest)/original_size*100:.1f}% of original)"
        )

        return result

    def compute_graph_stats(self, g: ig.Graph) -> Dict:
        """
        Compute basic graph statistics.

        Args:
            g: igraph Graph

        Returns:
            Dictionary of statistics
        """
        n_nodes = g.vcount()
        n_edges = g.ecount()

        stats = {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': g.density() if n_nodes > 0 else 0,
            'avg_degree': sum(g.degree()) / n_nodes if n_nodes > 0 else 0,
            'is_directed': g.is_directed(),
        }

        # Degree distribution stats
        if n_nodes > 0:
            degrees = g.degree()
            stats['min_degree'] = min(degrees)
            stats['max_degree'] = max(degrees)
            stats['median_degree'] = float(np.median(degrees))

        # For smaller graphs, compute more expensive metrics
        if n_nodes < 10000 and n_nodes > 0:
            try:
                # Clustering
                if not g.is_directed():
                    stats['avg_clustering'] = g.transitivity_avglocal_undirected(
                        mode="zero"
                    )

                # Path length (only for connected graphs)
                if g.is_directed():
                    if g.is_connected(mode="weak"):
                        # Use undirected version for path length
                        g_undirected = g.as_undirected()
                        stats['avg_path_length'] = g_undirected.average_path_length()
                else:
                    if g.is_connected():
                        stats['avg_path_length'] = g.average_path_length()
            except Exception as e:
                self.logger.debug(f"Could not compute some metrics: {e}")

        return stats

    def create_subgraph_by_time(
        self,
        g: ig.Graph,
        df: pd.DataFrame,
        start_time,
        end_time,
        time_column: str = 'datetime'
    ) -> ig.Graph:
        """
        Create subgraph containing only edges within a time window.

        Args:
            g: Full graph
            df: Original transaction DataFrame with timestamps
            start_time: Start of time window
            end_time: End of time window
            time_column: Timestamp column name

        Returns:
            Subgraph with edges in time window
        """
        # Filter transactions
        mask = (df[time_column] >= start_time) & (df[time_column] <= end_time)
        filtered_df = df[mask]

        # Build new graph
        return self.build_transaction_graph(filtered_df, directed=g.is_directed())

    def merge_graphs(
        self,
        graphs: List[ig.Graph],
        aggregate_weights: bool = True
    ) -> ig.Graph:
        """
        Merge multiple graphs into one.

        Args:
            graphs: List of graphs to merge
            aggregate_weights: Whether to sum edge weights

        Returns:
            Merged graph
        """
        if not graphs:
            g = ig.Graph(directed=True)
            g["_name_to_idx"] = {}
            return g

        directed = graphs[0].is_directed()

        # Collect all edges with attributes across graphs
        edge_data: dict = {}  # (src_name, tgt_name) -> {attr: value}
        all_nodes: set = set()

        for g in graphs:
            names = g.vs["name"]
            edge_attrs = g.es.attributes()
            for e in g.es:
                src_name = names[e.source]
                tgt_name = names[e.target]
                all_nodes.add(src_name)
                all_nodes.add(tgt_name)
                key = (src_name, tgt_name)

                if key in edge_data and aggregate_weights:
                    for a in edge_attrs:
                        val = e[a]
                        if isinstance(val, (int, float)):
                            edge_data[key][a] = edge_data[key].get(a, 0) + val
                        else:
                            edge_data[key][a] = val
                else:
                    edge_data[key] = {a: e[a] for a in edge_attrs}

        sorted_nodes = sorted(all_nodes)
        n2i = {n: i for i, n in enumerate(sorted_nodes)}

        merged_edges = [(n2i[s], n2i[t]) for s, t in edge_data]
        merged = ig.Graph(
            n=len(sorted_nodes),
            edges=merged_edges,
            directed=directed,
        )
        merged.vs["name"] = sorted_nodes
        merged["_name_to_idx"] = n2i

        # Set edge attributes
        if edge_data:
            all_attrs = set()
            for attrs in edge_data.values():
                all_attrs.update(attrs.keys())
            for a in all_attrs:
                merged.es[a] = [
                    edge_data[key].get(a, 0) for key in edge_data
                ]

        self.logger.info(f"Merged {len(graphs)} graphs into one with {merged.vcount():,} nodes")
        return merged


def main():
    """Test the graph builder."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser

    parser = OrbitaalParser()
    builder = GraphBuilder()

    # Load sample data
    sample_path = Path("data/raw/orbitaal/orbitaal-snapshot-2016_07_08.csv")

    if sample_path.exists():
        df = parser.load_snapshot(str(sample_path))

        # Build graph
        g = builder.build_transaction_graph(df)

        # Get stats
        stats = builder.compute_graph_stats(g)
        print("\nGraph Statistics:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:,.4f}")
            else:
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

        # Get largest component
        g_lcc = builder.get_largest_component(g)
        print(f"\nLargest component: {g_lcc.vcount():,} nodes")

        # Filter graph
        g_filtered = builder.filter_graph(g, min_degree=2)
        print(f"Filtered (min_degree=2): {g_filtered.vcount():,} nodes")
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
