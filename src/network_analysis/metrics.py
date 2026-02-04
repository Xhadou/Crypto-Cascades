"""
Network Metrics Module

Computes various network metrics including centrality measures,
degree distributions, clustering coefficients, and small-world analysis.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter

from src.utils.logger import get_logger

# Try to import powerlaw for degree distribution fitting
try:
    import powerlaw  # type: ignore
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False


class NetworkMetrics:
    """Compute network metrics and centrality measures."""
    
    def __init__(self):
        """Initialize the network metrics calculator."""
        self.logger = get_logger(__name__)
        
    def compute_centrality_measures(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        measures: Optional[List[str]] = None,
        normalized: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Compute centrality measures for all nodes.
        
        Args:
            G: NetworkX graph
            measures: List of measures to compute. Options:
                     ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']
                     If None, computes all available.
            normalized: Whether to normalize centrality values
            sample_size: For expensive metrics, sample this many nodes
            
        Returns:
            Dict mapping measure name to dict of {node: value}
        """
        if measures is None:
            measures = ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']
            
        results = {}
        n_nodes = G.number_of_nodes()
        
        self.logger.info(f"Computing centrality measures for {n_nodes:,} nodes...")
        
        # Degree centrality (fast)
        if 'degree' in measures:
            self.logger.info("  Computing degree centrality...")
            if isinstance(G, nx.DiGraph):
                results['in_degree'] = dict(G.in_degree())
                results['out_degree'] = dict(G.out_degree())
                # Normalize if requested
                if normalized and n_nodes > 1:
                    max_possible = n_nodes - 1
                    results['in_degree_norm'] = {k: v/max_possible for k, v in results['in_degree'].items()}
                    results['out_degree_norm'] = {k: v/max_possible for k, v in results['out_degree'].items()}
            results['degree'] = nx.degree_centrality(G)
        
        # PageRank (relatively fast)
        if 'pagerank' in measures:
            self.logger.info("  Computing PageRank...")
            try:
                results['pagerank'] = nx.pagerank(G, max_iter=100)
            except Exception as e:
                self.logger.warning(f"PageRank failed: {e}")
        
        # Betweenness centrality (expensive - O(VE))
        if 'betweenness' in measures:
            if n_nodes > 10000 and sample_size is None:
                self.logger.warning(
                    f"Betweenness centrality on {n_nodes:,} nodes is expensive. "
                    "Using k=500 sample approximation."
                )
                sample_size = min(500, n_nodes)
            
            self.logger.info("  Computing betweenness centrality...")
            try:
                if sample_size:
                    results['betweenness'] = nx.betweenness_centrality(
                        G, k=sample_size, normalized=normalized
                    )
                else:
                    results['betweenness'] = nx.betweenness_centrality(G, normalized=normalized)
            except Exception as e:
                self.logger.warning(f"Betweenness failed: {e}")
        
        # Closeness centrality (expensive - O(V^2))
        if 'closeness' in measures:
            if n_nodes > 5000:
                self.logger.warning(
                    f"Closeness centrality on {n_nodes:,} nodes is expensive. Skipping."
                )
            else:
                self.logger.info("  Computing closeness centrality...")
                try:
                    results['closeness'] = nx.closeness_centrality(G)
                except Exception as e:
                    self.logger.warning(f"Closeness failed: {e}")
        
        # Eigenvector centrality (can fail on some graphs)
        if 'eigenvector' in measures:
            self.logger.info("  Computing eigenvector centrality...")
            try:
                if G.is_directed():
                    results['eigenvector'] = nx.eigenvector_centrality(
                        G, max_iter=1000, tol=1e-6
                    )
                else:
                    results['eigenvector'] = nx.eigenvector_centrality(
                        G, max_iter=1000
                    )
            except Exception as e:
                self.logger.warning(f"Eigenvector centrality failed: {e}")
        
        return results
    
    def centrality_to_dataframe(
        self,
        centrality_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Convert centrality measures to a DataFrame.
        
        Args:
            centrality_dict: Dict from compute_centrality_measures
            
        Returns:
            DataFrame with nodes as rows and centrality measures as columns
        """
        df = pd.DataFrame(centrality_dict)
        df.index.name = 'node_id'
        return df.reset_index()
    
    def compute_clustering_coefficients(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        sample_size: int = 10000
    ) -> Dict[str, float]:
        """
        Compute clustering coefficients.
        
        Args:
            G: NetworkX graph
            sample_size: For large graphs, sample this many nodes for clustering
            
        Returns:
            Dict with global and average local clustering coefficients
        """
        self.logger.info("Computing clustering coefficients...")
        
        # Convert to undirected for clustering
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G
            
        results = {}
        n_nodes = G_undirected.number_of_nodes()
        
        # Average local clustering coefficient
        # For large graphs, use sampling to avoid O(n*k^2) complexity
        try:
            if n_nodes > sample_size:
                self.logger.info(f"Large graph ({n_nodes:,} nodes) - sampling {sample_size:,} nodes for clustering")
                import random
                sample_nodes = random.sample(list(G_undirected.nodes()), sample_size)
                results['avg_local_clustering'] = nx.average_clustering(G_undirected, nodes=sample_nodes)
                results['clustering_sampled'] = True
            else:
                results['avg_local_clustering'] = nx.average_clustering(G_undirected)
                results['clustering_sampled'] = False
        except Exception as e:
            self.logger.warning(f"Local clustering failed: {e}")
            results['avg_local_clustering'] = None
            results['clustering_sampled'] = None
        
        # Global clustering (transitivity) - this is fast even for large graphs
        try:
            results['transitivity'] = nx.transitivity(G_undirected)
        except Exception as e:
            self.logger.warning(f"Transitivity failed: {e}")
            results['transitivity'] = None
        
        return results
    
    def fit_power_law(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        degree_type: str = 'total'
    ) -> Dict:
        """
        Fit power-law distribution to degree sequence.
        
        Args:
            G: NetworkX graph
            degree_type: 'total', 'in', or 'out' (for directed graphs)
            
        Returns:
            Dict with alpha (exponent), xmin, p-value, and comparison results
        """
        if not HAS_POWERLAW:
            self.logger.warning("powerlaw package not installed. Skipping power-law fit.")
            return {'error': 'powerlaw package not installed'}
        
        self.logger.info(f"Fitting power-law distribution to {degree_type} degrees...")
        
        # Get degree sequence
        if isinstance(G, nx.DiGraph):
            if degree_type == 'in':
                degrees = [d for _, d in G.in_degree()]
            elif degree_type == 'out':
                degrees = [d for _, d in G.out_degree()]
            else:
                degree_view = G.degree()  # type: ignore[operator]
                degrees = [d for _, d in degree_view]
        else:
            degree_view = G.degree()  # type: ignore[operator]
            degrees = [d for _, d in degree_view]
        
        # Filter out zeros (powerlaw can't handle them)
        degrees = [d for d in degrees if d > 0]
        
        if len(degrees) < 50:
            self.logger.warning("Not enough non-zero degrees for power-law fit")
            return {'error': 'insufficient data'}
        
        try:
            # Fit power law
            fit = powerlaw.Fit(degrees, discrete=True)
            
            results = {
                'alpha': fit.power_law.alpha,
                'xmin': fit.power_law.xmin,
                'sigma': fit.power_law.sigma,  # Standard error on alpha
            }
            
            # Compare to other distributions
            # Positive R means power law is better fit
            R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
            R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal')
            
            results['vs_exponential'] = {'R': R_exp, 'p': p_exp}
            results['vs_lognormal'] = {'R': R_ln, 'p': p_ln}
            
            self.logger.info(
                f"Power-law fit: alpha={results['alpha']:.3f}, "
                f"xmin={results['xmin']}, sigma={results['sigma']:.3f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Power-law fit failed: {e}")
            return {'error': str(e)}
    
    def compute_degree_distribution(
        self,
        G: Union[nx.Graph, nx.DiGraph]
    ) -> pd.DataFrame:
        """
        Compute degree distribution.
        
        Args:
            G: NetworkX graph
            
        Returns:
            DataFrame with columns [degree, count, probability, cumulative]
        """
        if isinstance(G, nx.DiGraph):
            in_degrees = [d for _, d in G.in_degree()]
            out_degrees = [d for _, d in G.out_degree()]
            degree_view = G.degree()  # type: ignore[operator]
            total_degrees = [d for _, d in degree_view]
            
            # Total degree distribution
            degree_counts = Counter(total_degrees)
        else:
            degree_view = G.degree()  # type: ignore[operator]
            degrees = [d for _, d in degree_view]
            degree_counts = Counter(degrees)
        
        # Create distribution DataFrame
        df = pd.DataFrame([
            {'degree': k, 'count': v}
            for k, v in sorted(degree_counts.items())
        ])
        
        total = df['count'].sum()
        df['probability'] = df['count'] / total
        df['cumulative'] = df['probability'].cumsum()
        
        return df
    
    def compute_small_world_coefficient(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        n_random: int = 10
    ) -> Dict:
        """
        Compute small-world coefficient.
        
        σ = (C/C_rand) / (L/L_rand)
        
        Where:
        - C is clustering coefficient
        - L is average path length
        - _rand indicates random graph equivalent
        
        Args:
            G: NetworkX graph
            n_random: Number of random graphs to generate for comparison
            
        Returns:
            Dict with small-world metrics
        """
        self.logger.info("Computing small-world coefficient...")
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n > 5000:
            self.logger.warning(
                f"Small-world calculation on {n:,} nodes is expensive. "
                "Consider using a smaller sample."
            )
        
        # Convert to undirected
        if G.is_directed():
            G = G.to_undirected()
        
        # Get largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            n = G.number_of_nodes()
            m = G.number_of_edges()
            self.logger.info(f"Using largest connected component: {n:,} nodes")
        
        # Compute metrics for actual graph
        try:
            C = nx.average_clustering(G)
            L = nx.average_shortest_path_length(G)
        except Exception as e:
            self.logger.error(f"Failed to compute graph metrics: {e}")
            return {'error': str(e)}
        
        # Generate random graphs and compute their metrics
        C_rand_list = []
        L_rand_list = []
        
        p = 2 * m / (n * (n - 1))  # Edge probability for ER graph
        
        for i in range(n_random):
            # Generate Erdos-Renyi random graph
            G_rand = nx.gnm_random_graph(n, m)
            
            # Ensure connected (may need to try again)
            if not nx.is_connected(G_rand):
                # Use largest component
                largest_cc = max(nx.connected_components(G_rand), key=len)
                G_rand = G_rand.subgraph(largest_cc).copy()
            
            try:
                C_rand_list.append(nx.average_clustering(G_rand))
                L_rand_list.append(nx.average_shortest_path_length(G_rand))
            except:
                continue
        
        if not C_rand_list:
            return {'error': 'Could not compute random graph metrics'}
        
        C_rand = np.mean(C_rand_list)
        L_rand = np.mean(L_rand_list)
        
        # Compute small-world coefficient
        if C_rand > 0 and L_rand > 0 and L > 0:
            sigma = (C / C_rand) / (L / L_rand)
        else:
            sigma = None
        
        # Omega coefficient (alternative measure)
        # ω = L_rand/L - C/C_lattice
        # Approximate C_lattice ≈ 3/4 for ring lattice
        C_lattice = 0.75
        if L > 0:
            omega = L_rand / L - C / C_lattice
        else:
            omega = None
        
        results = {
            'clustering': C,
            'path_length': L,
            'clustering_random': C_rand,
            'path_length_random': L_rand,
            'sigma': sigma,  # σ > 1 indicates small-world
            'omega': omega,  # -1 < ω < 1, near 0 is small-world
            'is_small_world': sigma > 1 if sigma else None
        }
        
        self.logger.info(
            f"Small-world: σ={sigma:.3f if sigma else 'N/A'}, "
            f"C={C:.4f}, L={L:.2f}"
        )
        
        return results
    
    def get_top_nodes_by_centrality(
        self,
        centrality_dict: Dict,
        metric: str,
        n: int = 10
    ) -> List[Tuple]:
        """
        Get top nodes by a centrality metric.
        
        Args:
            centrality_dict: Result from compute_centrality_measures
            metric: Metric name (e.g., 'pagerank', 'betweenness')
            n: Number of top nodes to return
            
        Returns:
            List of (node_id, centrality_value) tuples
        """
        if metric not in centrality_dict:
            raise ValueError(f"Metric {metric} not found in centrality dict")
        
        values = centrality_dict[metric]
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]


def main():
    """Test network metrics."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    from src.preprocessing.graph_builder import GraphBuilder
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    metrics = NetworkMetrics()
    
    # Load sample data
    sample_path = "data/raw/orbitaal/orbitaal-snapshot-2016_07_08.csv"
    
    df = parser.load_snapshot(sample_path)
    G = builder.build_transaction_graph(df)
    
    # Get largest component for analysis
    G = builder.get_largest_component(G, strongly_connected=False)
    
    # Compute centrality (just degree and pagerank for speed)
    centrality = metrics.compute_centrality_measures(
        G, measures=['degree', 'pagerank']
    )
    
    # Top nodes by PageRank
    top_pr = metrics.get_top_nodes_by_centrality(centrality, 'pagerank', n=5)
    print("\nTop 5 nodes by PageRank:")
    for node, value in top_pr:
        print(f"  Node {node}: {value:.6f}")
    
    # Clustering
    clustering = metrics.compute_clustering_coefficients(G)
    print(f"\nClustering: {clustering}")
    
    # Degree distribution
    degree_dist = metrics.compute_degree_distribution(G)
    print(f"\nDegree distribution summary:")
    print(f"  Min degree: {degree_dist['degree'].min()}")
    print(f"  Max degree: {degree_dist['degree'].max()}")
    print(f"  Median degree: {degree_dist['degree'].median()}")
    
    # Power law fit
    pl_fit = metrics.fit_power_law(G)
    if 'alpha' in pl_fit:
        print(f"\nPower-law fit: α={pl_fit['alpha']:.3f}")


if __name__ == "__main__":
    main()
