"""
Graph Builder

Constructs NetworkX graphs from ORBITAAL transaction data and other sources.
Supports both directed and undirected graphs with edge aggregation.
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
import logging

from tqdm import tqdm

from src.utils.logger import get_logger


class GraphBuilder:
    """Build NetworkX graphs from transaction data."""
    
    def __init__(self):
        """Initialize the graph builder."""
        self.logger = get_logger(__name__)
        
    def build_transaction_graph(
        self,
        df: pd.DataFrame,
        directed: bool = True,
        weight_column: str = 'btc_value',
        aggregate_multi_edges: bool = True
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Build transaction graph from edge list.
        
        Args:
            df: DataFrame with source_id, target_id, and optional weight column
            directed: Whether to create directed graph
            weight_column: Column to use as edge weight
            aggregate_multi_edges: Whether to aggregate multiple edges between same nodes
            
        Returns:
            NetworkX graph with edge attributes (weight, count, usd_value)
        """
        self.logger.info(f"Building {'directed' if directed else 'undirected'} transaction graph...")
        
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
            
        if aggregate_multi_edges:
            G = self._build_aggregated_graph(df, G, weight_column)
        else:
            G = self._build_multigraph(df, G, weight_column)
                
        self.logger.info(f"Built graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def _build_aggregated_graph(
        self,
        df: pd.DataFrame,
        G: Union[nx.DiGraph, nx.Graph],
        weight_column: str
    ) -> Union[nx.DiGraph, nx.Graph]:
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

        # OPTIMIZED: Use bulk edge addition instead of iterating
        # Prepare edge list with attributes
        if 'usd_value' in grouped.columns:
            edges = [
                (row['source_id'], row['target_id'], {
                    'weight': row.get(weight_column, 1),
                    'count': row.get('count', 1),
                    'usd_value': row['usd_value']
                })
                for row in grouped.to_dict('records')
            ]
        else:
            edges = [
                (row['source_id'], row['target_id'], {
                    'weight': row.get(weight_column, 1),
                    'count': row.get('count', 1)
                })
                for row in grouped.to_dict('records')
            ]
        
        self.logger.info("Adding edges to graph...")
        G.add_edges_from(edges)
                
        return G
    
    def _build_multigraph(
        self,
        df: pd.DataFrame,
        G: Union[nx.DiGraph, nx.Graph],
        weight_column: str
    ) -> Union[nx.DiGraph, nx.Graph]:
        """Build graph allowing multiple edges."""
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
            source = row['source_id']
            target = row['target_id']
            
            if G.has_edge(source, target):
                # Aggregate
                G[source][target]['weight'] += row.get(weight_column, 1)
                G[source][target]['count'] += 1
                if 'usd_value' in row:
                    G[source][target]['usd_value'] += row.get('usd_value', 0)
            else:
                G.add_edge(
                    source,
                    target,
                    weight=row.get(weight_column, 1),
                    count=1,
                    usd_value=row.get('usd_value', 0)
                )
                
        return G
        
    def build_temporal_graphs(
        self,
        snapshots: Dict[str, pd.DataFrame],
        directed: bool = True
    ) -> Dict[str, Union[nx.DiGraph, nx.Graph]]:
        """
        Build graphs for each temporal snapshot.
        
        Args:
            snapshots: Dict mapping time period to transaction DataFrame
            directed: Whether to create directed graphs
            
        Returns:
            Dict mapping time period to NetworkX graph
        """
        self.logger.info(f"Building {len(snapshots)} temporal graphs...")
        
        graphs = {}
        for period, df in tqdm(snapshots.items(), desc="Building temporal graphs"):
            graphs[period] = self.build_transaction_graph(df, directed=directed)
            
        return graphs
        
    def add_node_attributes(
        self,
        G: Union[nx.DiGraph, nx.Graph],
        activity_df: pd.DataFrame,
        attributes: Optional[List[str]] = None
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Add wallet activity attributes to nodes.
        
        Args:
            G: NetworkX graph
            activity_df: DataFrame with wallet_id and activity metrics
            attributes: List of columns to add as attributes (default: all available)
            
        Returns:
            Graph with node attributes
        """
        if attributes is None:
            attributes = ['net_btc', 'net_usd', 'total_tx', 'btc_in', 'btc_out',
                         'usd_in', 'usd_out', 'tx_in_count', 'tx_out_count']
            
        # Filter to available columns
        available = [a for a in attributes if a in activity_df.columns]
        
        activity_dict = activity_df.set_index('wallet_id').to_dict('index')
        
        added_count = 0
        for node in G.nodes():
            if node in activity_dict:
                for attr in available:
                    if attr in activity_dict[node]:
                        G.nodes[node][attr] = activity_dict[node][attr]
                        added_count += 1
        
        self.logger.info(f"Added {added_count} node attributes")
        return G
        
    def filter_graph(
        self,
        G: Union[nx.DiGraph, nx.Graph],
        min_degree: int = 0,
        min_weight: float = 0.0,
        min_count: int = 0
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Filter graph by node degree and edge weight.
        
        Args:
            G: Input graph
            min_degree: Minimum node degree to keep
            min_weight: Minimum edge weight to keep
            min_count: Minimum edge count to keep
            
        Returns:
            Filtered graph (copy of original)
        """
        G = G.copy()
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()
        
        # Filter edges by weight and count
        if min_weight > 0 or min_count > 0:
            edges_to_remove = [
                (u, v) for u, v, d in G.edges(data=True)
                if d.get('weight', 0) < min_weight or d.get('count', 0) < min_count
            ]
            G.remove_edges_from(edges_to_remove)
            
        # Filter nodes by degree
        if min_degree > 0:
            degree_view = G.degree()  # type: ignore[operator]
            nodes_to_remove = [
                node for node, degree in dict(degree_view).items()
                if degree < min_degree
            ]
            G.remove_nodes_from(nodes_to_remove)
        
        self.logger.info(
            f"Filtered graph: {original_nodes:,} -> {G.number_of_nodes():,} nodes, "
            f"{original_edges:,} -> {G.number_of_edges():,} edges"
        )
        return G
        
    def get_largest_component(
        self,
        G: Union[nx.DiGraph, nx.Graph],
        strongly_connected: bool = False
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Extract largest connected component.
        
        Args:
            G: Input graph
            strongly_connected: For directed graphs, use strongly connected component
            
        Returns:
            Subgraph of largest component
        """
        original_size = G.number_of_nodes()
        
        if G.is_directed():
            if strongly_connected:
                components = list(nx.strongly_connected_components(G))  # type: ignore[arg-type]
            else:
                components = list(nx.weakly_connected_components(G))  # type: ignore[arg-type]
        else:
            components = list(nx.connected_components(G))
            
        if not components:
            self.logger.warning("No connected components found")
            return G
            
        largest = max(components, key=len)
        result = G.subgraph(largest).copy()
        
        self.logger.info(
            f"Extracted largest component: {len(largest):,} nodes "
            f"({len(largest)/original_size*100:.1f}% of original)"
        )
        
        return result
        
    def compute_graph_stats(self, G: Union[nx.DiGraph, nx.Graph]) -> Dict:
        """
        Compute basic graph statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of statistics
        """
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        stats = {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': nx.density(G) if n_nodes > 0 else 0,
            'avg_degree': sum(dict(G.degree()).values()) / n_nodes if n_nodes > 0 else 0,  # type: ignore[operator]
            'is_directed': G.is_directed(),
        }
        
        # Degree distribution stats
        if n_nodes > 0:
            degree_view = G.degree()  # type: ignore[operator]
            degrees = [d for _, d in degree_view]
            stats['min_degree'] = min(degrees)
            stats['max_degree'] = max(degrees)
            stats['median_degree'] = np.median(degrees)
        
        # For smaller graphs, compute more expensive metrics
        if n_nodes < 10000 and n_nodes > 0:
            try:
                # Clustering
                if not G.is_directed():
                    stats['avg_clustering'] = nx.average_clustering(G)
                    
                # Path length (only for connected graphs)
                if G.is_directed():
                    if nx.is_weakly_connected(G):  # type: ignore[arg-type]
                        # Use undirected version for path length
                        G_undirected = G.to_undirected()
                        stats['avg_path_length'] = nx.average_shortest_path_length(G_undirected)
                else:
                    if nx.is_connected(G):
                        stats['avg_path_length'] = nx.average_shortest_path_length(G)
            except Exception as e:
                self.logger.debug(f"Could not compute some metrics: {e}")
                
        return stats
    
    def create_subgraph_by_time(
        self,
        G: Union[nx.DiGraph, nx.Graph],
        df: pd.DataFrame,
        start_time,
        end_time,
        time_column: str = 'datetime'
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Create subgraph containing only edges within a time window.
        
        Args:
            G: Full graph
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
        return self.build_transaction_graph(filtered_df, directed=G.is_directed())
    
    def merge_graphs(
        self,
        graphs: List[Union[nx.DiGraph, nx.Graph]],
        aggregate_weights: bool = True
    ) -> Union[nx.DiGraph, nx.Graph]:
        """
        Merge multiple graphs into one.
        
        Args:
            graphs: List of graphs to merge
            aggregate_weights: Whether to sum edge weights
            
        Returns:
            Merged graph
        """
        if not graphs:
            return nx.DiGraph()
            
        # Use first graph as template
        if graphs[0].is_directed():
            merged = nx.DiGraph()
        else:
            merged = nx.Graph()
            
        for G in graphs:
            for u, v, data in G.edges(data=True):
                if merged.has_edge(u, v) and aggregate_weights:
                    merged[u][v]['weight'] = merged[u][v].get('weight', 0) + data.get('weight', 0)
                    merged[u][v]['count'] = merged[u][v].get('count', 0) + data.get('count', 0)
                else:
                    merged.add_edge(u, v, **data)
                    
        self.logger.info(f"Merged {len(graphs)} graphs into one with {merged.number_of_nodes():,} nodes")
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
        G = builder.build_transaction_graph(df)
        
        # Get stats
        stats = builder.compute_graph_stats(G)
        print("\nGraph Statistics:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:,.4f}")
            else:
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
            
        # Get largest component
        G_lcc = builder.get_largest_component(G)
        print(f"\nLargest component: {G_lcc.number_of_nodes():,} nodes")
        
        # Filter graph
        G_filtered = builder.filter_graph(G, min_degree=2)
        print(f"Filtered (min_degree=2): {G_filtered.number_of_nodes():,} nodes")
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
