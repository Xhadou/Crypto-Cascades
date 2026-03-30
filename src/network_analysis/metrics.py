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

# Try to import NetworKit for fast centrality on large graphs
try:
    import networkit as nk  # type: ignore
    HAS_NETWORKIT = True
except ImportError:
    HAS_NETWORKIT = False

# Try to import igraph for fast clustering on large graphs
try:
    import igraph as ig  # type: ignore
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False


class NetworkMetrics:
    """Compute network metrics and centrality measures."""
    
    def __init__(self, config: dict = None):
        """Initialize the network metrics calculator.

        Args:
            config: Optional configuration dict. If provided, thresholds are
                    read from config['thresholds']. Otherwise defaults are used.
        """
        self.logger = get_logger(__name__)
        cfg = config or {}
        thresholds = cfg.get('thresholds', {})
        self.large_graph_nodes = thresholds.get('large_graph_nodes', 10000)
        self.max_nodes_centrality = thresholds.get('max_nodes_for_centrality', 5000)
        self.betweenness_sample = thresholds.get('betweenness_sample_size', 500)
        self.clustering_sample = thresholds.get('clustering_sample_size', 10000)
        self.min_degrees_powerlaw = thresholds.get('min_degrees_for_powerlaw', 50)
        
    def _compute_betweenness_networkit(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        normalized: bool = True
    ) -> Dict:
        """Use NetworKit for fast approximate betweenness on large graphs.

        Converts the NetworkX graph to a NetworKit graph, runs approximate
        betweenness centrality estimation, and maps scores back to the
        original NetworkX node identifiers.

        Args:
            G: NetworkX graph (directed or undirected).
            normalized: Whether to normalize scores (consistent with
                        NetworkX convention).

        Returns:
            Dict mapping NetworkX node IDs to betweenness centrality scores.
        """
        directed = G.is_directed()
        nk_graph, node_map = nk.nxadapter.nx2nk(G, weightAttr=None)
        # node_map: dict mapping nx node -> nk node index
        reverse_map = {v: k for k, v in node_map.items()}

        bc = nk.centrality.EstimateBetweenness(
            nk_graph,
            nSamples=self.betweenness_sample,
            normalized=normalized,
            parallel=True,
        )
        bc.run()
        scores = bc.scores()
        return {reverse_map[i]: scores[i] for i in range(len(scores))}

    def _nx_to_igraph(
        self,
        G: Union[nx.Graph, nx.DiGraph]
    ) -> 'ig.Graph':
        """Convert NetworkX graph to igraph via edge list.

        Building from an edge list is significantly faster than
        ``ig.Graph.from_networkx()`` for graphs with millions of nodes
        because it avoids serializing per-node/per-edge attribute dicts.

        Args:
            G: NetworkX graph (directed or undirected).

        Returns:
            An igraph Graph with the same topology (attributes are not copied).
        """
        node_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(
            n=len(node_list), edges=edges, directed=G.is_directed()
        )
        return ig_graph

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
            if HAS_NETWORKIT and n_nodes > self.large_graph_nodes:
                self.logger.info(
                    f"  Computing betweenness centrality via NetworKit "
                    f"({n_nodes:,} nodes, {self.betweenness_sample} samples)..."
                )
                try:
                    results['betweenness'] = self._compute_betweenness_networkit(
                        G, normalized=normalized
                    )
                except Exception as e:
                    self.logger.warning(
                        f"NetworKit betweenness failed, falling back to NetworkX: {e}"
                    )
                    results.pop('betweenness', None)

            if 'betweenness' not in results:
                if n_nodes > self.large_graph_nodes and sample_size is None:
                    self.logger.warning(
                        f"Betweenness centrality on {n_nodes:,} nodes is expensive. "
                        f"Using k={self.betweenness_sample} sample approximation."
                    )
                    sample_size = min(self.betweenness_sample, n_nodes)

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
            if n_nodes > self.max_nodes_centrality:
                self.logger.warning(
                    f"Closeness centrality on {n_nodes:,} nodes is expensive. Skipping."
                )
            else:
                self.logger.info("  Computing closeness centrality...")
                try:
                    results['closeness'] = nx.closeness_centrality(G)
                except Exception as e:
                    self.logger.warning(f"Closeness failed: {e}")
        
        # Eigenvector centrality (can fail on some graphs, hangs on large ones)
        if 'eigenvector' in measures:
            if n_nodes > self.large_graph_nodes:
                self.logger.warning(
                    f"Eigenvector centrality on {n_nodes:,} nodes is too "
                    f"expensive (power iteration scales poorly). Skipping."
                )
            else:
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
    
    def _compute_clustering_networkit(
        self,
        G: Union[nx.Graph, nx.DiGraph],
    ) -> Dict[str, float]:
        """Use NetworKit for clustering (parallel C++, fastest option).

        NetworKit uses OpenMP to parallelise triangle counting across all
        available CPU cores, making it the fastest option for very large
        graphs (30M+ nodes).
        """
        directed = G.is_directed()
        nk_graph, node_map = nk.nxadapter.nx2nk(G, weightAttr=None)

        # NetworKit needs undirected for clustering
        if directed:
            nk_graph = nk_graph.toUndirected()

        self.logger.info("  Computing global clustering (NetworKit)...")
        gc = nk.globals.ClusteringCoefficient.exactGlobal(nk_graph)

        self.logger.info("  Computing local clustering (NetworKit)...")
        lcc = nk.centrality.LocalClusteringCoefficient(nk_graph, turbo=True)
        lcc.run()
        scores = lcc.scores()
        # Average, treating NaN (isolated nodes) as 0
        valid = [s for s in scores if s == s]  # filter NaN
        avg_local = sum(valid) / len(valid) if valid else 0.0

        return {
            'transitivity': gc,
            'avg_local_clustering': avg_local,
            'clustering_sampled': False,
        }

    def _compute_clustering_igraph(
        self,
        G: Union[nx.Graph, nx.DiGraph],
    ) -> Dict[str, float]:
        """Use igraph for clustering (single-threaded C, fast fallback)."""
        ig_graph = self._nx_to_igraph(G)
        if ig_graph.is_directed():
            ig_graph = ig_graph.as_undirected()

        self.logger.info("  Computing transitivity (igraph)...")
        transitivity = ig_graph.transitivity_undirected()

        self.logger.info("  Computing avg local clustering (igraph)...")
        avg_local = ig_graph.transitivity_avglocal_undirected(mode="zero")

        return {
            'transitivity': transitivity,
            'avg_local_clustering': avg_local,
            'clustering_sampled': False,
        }

    def compute_clustering_coefficients(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute clustering coefficients.

        Uses the best available backend for performance:
          1. **NetworKit** (parallel C++, fastest) — if installed
          2. **igraph** (single-threaded C) — if installed
          3. **NetworkX** (pure Python, with sampling) — last resort

        Args:
            G: NetworkX graph
            sample_size: For large graphs, sample this many nodes for clustering
                         (only used in the NetworkX fallback path)

        Returns:
            Dict with global and average local clustering coefficients
        """
        if sample_size is None:
            sample_size = self.clustering_sample

        self.logger.info("Computing clustering coefficients...")
        n_nodes = G.number_of_nodes()
        results = {}

        # ── Tier 1: NetworKit (parallel C++, fastest) ──
        if HAS_NETWORKIT and n_nodes > self.large_graph_nodes:
            self.logger.info(
                f"Using NetworKit (parallel C++) for {n_nodes:,} node graph"
            )
            try:
                return self._compute_clustering_networkit(G)
            except Exception as e:
                self.logger.warning(
                    f"NetworKit clustering failed: {e}. Trying igraph..."
                )

        # ── Tier 2: igraph (single-threaded C) ──
        if HAS_IGRAPH and n_nodes > self.large_graph_nodes:
            self.logger.info(
                f"Using igraph (C backend) for {n_nodes:,} node graph"
            )
            try:
                return self._compute_clustering_igraph(G)
            except Exception as e:
                self.logger.warning(
                    f"igraph clustering failed, falling back to NetworkX: {e}"
                )

        # ── Tier 3: NetworkX (pure Python, fine for small graphs) ──
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G

        n_nodes = G_undirected.number_of_nodes()

        try:
            if n_nodes > sample_size:
                self.logger.info(
                    f"Large graph ({n_nodes:,} nodes) - sampling "
                    f"{sample_size:,} nodes for clustering"
                )
                import random
                sample_nodes = random.sample(
                    list(G_undirected.nodes()), sample_size
                )
                results['avg_local_clustering'] = nx.average_clustering(
                    G_undirected, nodes=sample_nodes
                )
                results['clustering_sampled'] = True
            else:
                results['avg_local_clustering'] = nx.average_clustering(
                    G_undirected
                )
                results['clustering_sampled'] = False
        except Exception as e:
            self.logger.warning(f"Local clustering failed: {e}")
            results['avg_local_clustering'] = None
            results['clustering_sampled'] = None

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
        
        if len(degrees) < self.min_degrees_powerlaw:
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

        # Hard guard: average_shortest_path_length is O(V*E) and will hang
        # on large graphs.  The existing warning at max_nodes_centrality
        # only logged but did not bail out.
        if n > self.large_graph_nodes:
            self.logger.warning(
                f"Small-world coefficient on {n:,} nodes requires all-pairs "
                f"shortest paths. Skipping "
                f"(max supported: {self.large_graph_nodes:,} nodes)."
            )
            return {
                'error': (
                    f'Graph too large ({n:,} nodes) for small-world '
                    f'computation'
                )
            }
        
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
            except (nx.NetworkXError, ValueError, ZeroDivisionError):
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
        
        # Omega coefficient via NetworkX (proper lattice + random comparison)
        if n <= 1000:
            try:
                omega = nx.omega(G, niter=3, nrand=5)
            except Exception as e:
                self.logger.warning(f"Omega computation failed: {e}")
                omega = None
        else:
            self.logger.info(f"Graph too large ({n} nodes) for exact omega; skipping")
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
