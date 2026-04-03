"""
Community Detection Module

Implements community detection algorithms for identifying
groups of nodes in the network.
"""

import igraph as ig
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from src.utils.logger import get_logger
from src.utils.graph_adapter import name_to_idx

# Leiden algorithm requires leidenalg
try:
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class CommunityDetector:
    """Detect communities in networks using various algorithms."""

    def __init__(self):
        """Initialize the community detector."""
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_undirected(g: ig.Graph) -> ig.Graph:
        """Return an undirected version of *g* (no-op if already undirected)."""
        return g.as_undirected() if g.is_directed() else g

    @staticmethod
    def _clustering_to_partition(g: ig.Graph, membership: list) -> Dict:
        """Convert an igraph membership vector to ``{node_name: community_id}``."""
        names = g.vs['name']
        return {names[v]: membership[v] for v in range(g.vcount())}

    @staticmethod
    def _community_sizes(partition: Dict) -> Dict[int, int]:
        sizes: Dict[int, int] = {}
        for comm in partition.values():
            sizes[comm] = sizes.get(comm, 0) + 1
        return sizes

    # ------------------------------------------------------------------
    # Louvain (igraph multilevel)
    # ------------------------------------------------------------------

    def detect_communities_louvain(
        self,
        g: ig.Graph,
        resolution: float = 1.0,
        random_state: int = 42
    ) -> Dict:
        """
        Detect communities using the Louvain (multilevel) algorithm.

        Args:
            g: igraph Graph (will be converted to undirected if needed)
            resolution: Resolution parameter. Higher values lead to more communities.
            random_state: Random seed for reproducibility

        Returns:
            Dict with keys:
                - 'partition': Dict mapping node name to community ID
                - 'modularity': Modularity score of the partition
                - 'n_communities': Number of communities
                - 'community_sizes': Dict mapping community ID to size
        """
        self.logger.info("Detecting communities using Louvain algorithm...")

        g_u = self._ensure_undirected(g)

        weights = 'weight' if 'weight' in g_u.es.attributes() else None
        clustering = g_u.community_multilevel(weights=weights)

        partition = self._clustering_to_partition(g_u, clustering.membership)
        modularity = clustering.modularity
        community_sizes = self._community_sizes(partition)
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

    def detect_communities_leiden(
        self,
        g: ig.Graph,
        resolution: float = 1.0,
        random_state: int = 42
    ) -> Dict:
        """
        Detect communities using the Leiden algorithm (Traag et al. 2019).

        Args:
            g: igraph Graph (will be converted to undirected if needed)
            resolution: Resolution parameter. Higher values lead to more communities.
            random_state: Random seed for reproducibility

        Returns:
            Dict with keys:
                - 'partition': Dict mapping node name to community ID
                - 'modularity': Modularity score of the partition
                - 'n_communities': Number of communities
                - 'community_sizes': Dict mapping community ID to size
        """
        if not HAS_LEIDEN:
            self.logger.warning(
                "leidenalg not installed, falling back to Louvain"
            )
            return self.detect_communities_louvain(g, resolution, random_state)

        self.logger.info(
            f"Detecting communities using Leiden algorithm "
            f"({g.vcount():,} nodes, {g.ecount():,} edges) — this may take hours for large graphs..."
        )

        g_u = self._ensure_undirected(g)

        partition = leidenalg.find_partition(
            g_u,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_state,
        )

        node_partition = self._clustering_to_partition(g_u, partition.membership)
        modularity = partition.modularity
        community_sizes = self._community_sizes(node_partition)
        n_communities = len(community_sizes)

        self.logger.info(
            f"Found {n_communities} communities with modularity {modularity:.4f}"
        )

        return {
            'partition': node_partition,
            'modularity': modularity,
            'n_communities': n_communities,
            'community_sizes': community_sizes,
        }

    def detect_communities_label_propagation(
        self,
        g: ig.Graph
    ) -> Dict:
        """
        Detect communities using label propagation algorithm.

        This is a fast, near-linear time algorithm.

        Args:
            g: igraph Graph

        Returns:
            Dict with partition and statistics
        """
        self.logger.info("Detecting communities using label propagation...")

        g_u = self._ensure_undirected(g)

        weights = 'weight' if 'weight' in g_u.es.attributes() else None
        clustering = g_u.community_label_propagation(weights=weights)

        partition = self._clustering_to_partition(g_u, clustering.membership)
        community_sizes = self._community_sizes(partition)
        n_communities = len(community_sizes)

        modularity = clustering.modularity

        self.logger.info(f"Found {n_communities} communities")

        return {
            'partition': partition,
            'modularity': modularity,
            'n_communities': n_communities,
            'community_sizes': community_sizes
        }

    def get_community_subgraph(
        self,
        g: ig.Graph,
        partition: Dict,
        community_id: int
    ) -> ig.Graph:
        """
        Extract subgraph for a specific community.

        Args:
            g: Original igraph Graph
            partition: Dict mapping node name to community
            community_id: ID of community to extract

        Returns:
            Subgraph containing only nodes in the specified community
        """
        node_names = [n for n, c in partition.items() if c == community_id]
        n2i = name_to_idx(g)
        indices = [n2i[n] for n in node_names if n in n2i]
        return g.induced_subgraph(indices)

    def compute_community_metrics(
        self,
        g: ig.Graph,
        partition: Dict
    ) -> pd.DataFrame:
        """
        Compute metrics for each community.

        Args:
            g: igraph Graph
            partition: Dict mapping node name to community

        Returns:
            DataFrame with community metrics
        """
        self.logger.info("Computing community metrics...")

        n2i = name_to_idx(g)
        is_directed = g.is_directed()
        metrics = []

        community_ids = set(partition.values())

        for comm_id in community_ids:
            # Get community subgraph
            node_names = [n for n, c in partition.items() if c == comm_id]
            indices = [n2i[n] for n in node_names if n in n2i]
            subgraph = g.induced_subgraph(indices)

            n_nodes = subgraph.vcount()
            n_edges = subgraph.ecount()

            # Internal density
            if n_nodes > 1:
                max_edges = n_nodes * (n_nodes - 1)
                if not is_directed:
                    max_edges //= 2
                internal_density = n_edges / max_edges if max_edges > 0 else 0
            else:
                internal_density = 0

            # Average degree within community
            if n_nodes > 0:
                avg_degree = 2 * n_edges / n_nodes if not is_directed else n_edges / n_nodes
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
        g: ig.Graph,
        partition: Dict
    ) -> Tuple[int, int]:
        """
        Count edges within communities vs between communities.

        Args:
            g: igraph Graph
            partition: Dict mapping node name to community

        Returns:
            Tuple of (within_community_edges, between_community_edges)
        """
        within = 0
        between = 0

        names = g.vs['name']
        for u_idx, v_idx in g.get_edgelist():
            u_name = names[u_idx]
            v_name = names[v_idx]
            if u_name in partition and v_name in partition:
                if partition[u_name] == partition[v_name]:
                    within += 1
                else:
                    between += 1

        return within, between

    def identify_bridge_nodes(
        self,
        g: ig.Graph,
        partition: Dict,
        min_external_ratio: float = 0.5
    ) -> List:
        """
        Identify nodes that bridge multiple communities.

        A bridge node is one that has significant connections to
        other communities (weak ties).

        Args:
            g: igraph Graph
            partition: Dict mapping node name to community
            min_external_ratio: Minimum ratio of external edges to be a bridge

        Returns:
            List of bridge node names
        """
        self.logger.info("Identifying bridge nodes...")

        n2i = name_to_idx(g)
        names = g.vs['name']
        bridge_nodes = []

        for node_name, node_comm in partition.items():
            if node_name not in n2i:
                continue
            node_idx = n2i[node_name]

            internal = 0
            external = 0

            for neighbor_idx in g.neighbors(node_idx):
                neighbor_name = names[neighbor_idx]
                if neighbor_name in partition:
                    if partition[neighbor_name] == node_comm:
                        internal += 1
                    else:
                        external += 1

            total = internal + external
            if total > 0 and external / total >= min_external_ratio:
                bridge_nodes.append(node_name)

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
    g = builder.build_transaction_graph(df)

    # Get largest component for community detection
    g = builder.get_largest_component(g, strongly_connected=False)
    print(f"\nLargest component: {g.vcount():,} nodes")

    # Detect communities with Louvain
    result = detector.detect_communities_louvain(g)

    print(f"\nLouvain Communities:")
    print(f"  Number of communities: {result['n_communities']}")
    print(f"  Modularity: {result['modularity']:.4f}")

    # Community metrics
    metrics = detector.compute_community_metrics(g, result['partition'])
    print("\nTop 5 communities by size:")
    print(metrics.head())

    # Cross-community edges
    within, between = detector.compute_cross_community_edges(g, result['partition'])
    print(f"\nEdges within communities: {within:,}")
    print(f"Edges between communities: {between:,}")
    print(f"Cross-community ratio: {between/(within+between):.2%}")

    # Bridge nodes
    bridges = detector.identify_bridge_nodes(g, result['partition'])
    print(f"\nBridge nodes (>50% external edges): {len(bridges)}")


if __name__ == "__main__":
    main()
