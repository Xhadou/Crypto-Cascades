"""
Community Detection Module

Implements community detection algorithms for identifying
groups of nodes in the network.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

from src.utils.logger import get_logger

# Try to import community detection libraries
try:
    import community as community_louvain  # python-louvain package
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class CommunityDetector:
    """Detect communities in networks using various algorithms."""
    
    def __init__(self):
        """Initialize the community detector."""
        self.logger = get_logger(__name__)
        
    def detect_communities_louvain(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        resolution: float = 1.0,
        random_state: int = 42
    ) -> Dict:
        """
        Detect communities using the Louvain algorithm.
        
        The Louvain algorithm is a greedy optimization method that attempts
        to optimize modularity of the partition.
        
        Args:
            G: NetworkX graph (will be converted to undirected)
            resolution: Resolution parameter. Higher values lead to more communities.
            random_state: Random seed for reproducibility
            
        Returns:
            Dict with keys:
                - 'partition': Dict mapping node to community ID
                - 'modularity': Modularity score of the partition
                - 'n_communities': Number of communities
                - 'community_sizes': Dict mapping community ID to size
        """
        if not HAS_LOUVAIN:
            self.logger.error("python-louvain package not installed")
            return {'error': 'python-louvain not installed'}
        
        self.logger.info("Detecting communities using Louvain algorithm...")
        
        # Convert to undirected
        if G.is_directed():
            G = G.to_undirected()
        
        # Run Louvain algorithm
        partition = community_louvain.best_partition(
            G,
            resolution=resolution,
            random_state=random_state
        )
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)
        
        # Count community sizes
        community_sizes = {}
        for node, comm in partition.items():
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        
        n_communities = len(community_sizes)
        
        self.logger.info(
            f"Found {n_communities} communities with modularity {modularity:.4f}"
        )
        
        return {
            'partition': partition,
            'modularity': modularity,
            'n_communities': n_communities,
            'community_sizes': community_sizes
        }
    
    def detect_communities_label_propagation(
        self,
        G: Union[nx.Graph, nx.DiGraph]
    ) -> Dict:
        """
        Detect communities using label propagation algorithm.
        
        This is a fast, near-linear time algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dict with partition and statistics
        """
        self.logger.info("Detecting communities using label propagation...")
        
        # Convert to undirected
        if G.is_directed():
            G = G.to_undirected()
        
        # Run label propagation
        communities = nx.community.label_propagation_communities(G)
        
        # Convert to partition dict
        partition = {}
        community_sizes = {}
        
        for i, comm in enumerate(communities):
            community_sizes[i] = len(comm)
            for node in comm:
                partition[node] = i
        
        # Calculate modularity
        try:
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = None
        
        n_communities = len(community_sizes)
        
        self.logger.info(f"Found {n_communities} communities")
        
        return {
            'partition': partition,
            'modularity': modularity,
            'n_communities': n_communities,
            'community_sizes': community_sizes
        }
    
    def get_community_subgraph(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        partition: Dict,
        community_id: int
    ) -> Union[nx.Graph, nx.DiGraph]:
        """
        Extract subgraph for a specific community.
        
        Args:
            G: Original graph
            partition: Dict mapping node to community
            community_id: ID of community to extract
            
        Returns:
            Subgraph containing only nodes in the specified community
        """
        nodes = [n for n, c in partition.items() if c == community_id]
        return G.subgraph(nodes).copy()
    
    def compute_community_metrics(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        partition: Dict
    ) -> pd.DataFrame:
        """
        Compute metrics for each community.
        
        Args:
            G: NetworkX graph
            partition: Dict mapping node to community
            
        Returns:
            DataFrame with community metrics
        """
        self.logger.info("Computing community metrics...")
        
        metrics = []
        
        community_ids = set(partition.values())
        
        for comm_id in community_ids:
            # Get community subgraph
            nodes = [n for n, c in partition.items() if c == comm_id]
            subgraph = G.subgraph(nodes).copy()
            
            # Compute metrics
            n_nodes = subgraph.number_of_nodes()
            n_edges = subgraph.number_of_edges()
            
            # Internal density
            if n_nodes > 1:
                max_edges = n_nodes * (n_nodes - 1)
                if not G.is_directed():
                    max_edges //= 2
                internal_density = n_edges / max_edges if max_edges > 0 else 0
            else:
                internal_density = 0
            
            # Average degree within community
            if n_nodes > 0:
                avg_degree = 2 * n_edges / n_nodes if not G.is_directed() else n_edges / n_nodes
            else:
                avg_degree = 0
            
            metrics.append({
                'community_id': comm_id,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'internal_density': internal_density,
                'avg_internal_degree': avg_degree
            })
        
        df = pd.DataFrame(metrics)
        df = df.sort_values('n_nodes', ascending=False).reset_index(drop=True)
        
        return df
    
    def compute_cross_community_edges(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        partition: Dict
    ) -> Tuple[int, int]:
        """
        Count edges within communities vs between communities.
        
        Args:
            G: NetworkX graph
            partition: Dict mapping node to community
            
        Returns:
            Tuple of (within_community_edges, between_community_edges)
        """
        within = 0
        between = 0
        
        for u, v in G.edges():
            if u in partition and v in partition:
                if partition[u] == partition[v]:
                    within += 1
                else:
                    between += 1
        
        return within, between
    
    def identify_bridge_nodes(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        partition: Dict,
        min_external_ratio: float = 0.5
    ) -> List[int]:
        """
        Identify nodes that bridge multiple communities.
        
        A bridge node is one that has significant connections to
        other communities (weak ties).
        
        Args:
            G: NetworkX graph
            partition: Dict mapping node to community
            min_external_ratio: Minimum ratio of external edges to be a bridge
            
        Returns:
            List of bridge node IDs
        """
        self.logger.info("Identifying bridge nodes...")
        
        bridge_nodes = []
        
        for node in G.nodes():
            if node not in partition:
                continue
                
            node_comm = partition[node]
            
            # Count internal vs external edges
            internal = 0
            external = 0
            
            for neighbor in G.neighbors(node):
                if neighbor in partition:
                    if partition[neighbor] == node_comm:
                        internal += 1
                    else:
                        external += 1
            
            total = internal + external
            if total > 0 and external / total >= min_external_ratio:
                bridge_nodes.append(node)
        
        self.logger.info(f"Found {len(bridge_nodes)} bridge nodes")
        return bridge_nodes
    
    def partition_to_dataframe(
        self,
        partition: Dict
    ) -> pd.DataFrame:
        """
        Convert partition dict to DataFrame.
        
        Args:
            partition: Dict mapping node to community
            
        Returns:
            DataFrame with columns [node_id, community_id]
        """
        return pd.DataFrame([
            {'node_id': node, 'community_id': comm}
            for node, comm in partition.items()
        ])


def main():
    """Test community detection."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    from src.preprocessing.graph_builder import GraphBuilder
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    detector = CommunityDetector()
    
    # Load sample data
    sample_path = "data/raw/orbitaal/orbitaal-snapshot-2016_07_08.csv"
    
    df = parser.load_snapshot(sample_path)
    G = builder.build_transaction_graph(df)
    
    # Get largest component for community detection
    G = builder.get_largest_component(G, strongly_connected=False)
    print(f"\nLargest component: {G.number_of_nodes():,} nodes")
    
    # Detect communities with Louvain
    result = detector.detect_communities_louvain(G)
    
    if 'error' not in result:
        print(f"\nLouvain Communities:")
        print(f"  Number of communities: {result['n_communities']}")
        print(f"  Modularity: {result['modularity']:.4f}")
        
        # Community metrics
        metrics = detector.compute_community_metrics(G, result['partition'])
        print("\nTop 5 communities by size:")
        print(metrics.head())
        
        # Cross-community edges
        within, between = detector.compute_cross_community_edges(G, result['partition'])
        print(f"\nEdges within communities: {within:,}")
        print(f"Edges between communities: {between:,}")
        print(f"Cross-community ratio: {between/(within+between):.2%}")
        
        # Bridge nodes
        bridges = detector.identify_bridge_nodes(G, result['partition'])
        print(f"\nBridge nodes (>50% external edges): {len(bridges)}")


if __name__ == "__main__":
    main()
