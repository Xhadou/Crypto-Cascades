"""
Network Metrics Module

Computes various network metrics including centrality measures,
degree distributions, clustering coefficients, and small-world analysis.
"""

import igraph as ig
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from collections import Counter

from src.utils.logger import get_logger
from src.utils.graph_adapter import name_to_idx, to_networkx

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
        g: ig.Graph,
        normalized: bool = True
    ) -> Dict:
        """Use NetworKit for fast approximate betweenness on large graphs.

        Converts the igraph graph to NetworkX, then to NetworKit via nx2nk,
        runs approximate betweenness centrality estimation, and maps scores
        back to the original node names.

        Args:
            g: igraph graph.
            normalized: Whether to normalize scores.

        Returns:
            Dict mapping node names to betweenness centrality scores.
        """
        # Convert igraph -> NetworkX -> NetworKit
        nx_graph = to_networkx(g)
        result = nk.nxadapter.nx2nk(nx_graph, weightAttr=None)
        if isinstance(result, tuple):
            nk_graph, node_map = result
        else:
            nk_graph = result
            node_map = {n: i for i, n in enumerate(nx_graph.nodes())}
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

    def compute_centrality_measures(
        self,
        g: ig.Graph,
        measures: Optional[List[str]] = None,
        normalized: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Compute centrality measures for all nodes.

        Args:
            g: igraph graph
            measures: List of measures to compute. Options:
                     ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']
                     If None, computes all available.
            normalized: Whether to normalize centrality values
            sample_size: For expensive metrics, sample this many nodes

        Returns:
            Dict mapping measure name to dict of {node_name: value}
        """
        if measures is None:
            measures = ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']

        results = {}
        n_nodes = g.vcount()
        names = g.vs['name']

        self.logger.info(f"Computing centrality measures for {n_nodes:,} nodes...")

        # Degree centrality (fast)
        if 'degree' in measures:
            self.logger.info("  Computing degree centrality...")
            if g.is_directed():
                in_deg = g.indegree()
                out_deg = g.outdegree()
                results['in_degree'] = {names[i]: v for i, v in enumerate(in_deg)}
                results['out_degree'] = {names[i]: v for i, v in enumerate(out_deg)}
                if normalized and n_nodes > 1:
                    max_possible = n_nodes - 1
                    results['in_degree_norm'] = {names[i]: v / max_possible for i, v in enumerate(in_deg)}
                    results['out_degree_norm'] = {names[i]: v / max_possible for i, v in enumerate(out_deg)}
            deg = g.degree()
            if n_nodes > 1:
                results['degree'] = {names[i]: v / (n_nodes - 1) for i, v in enumerate(deg)}
            else:
                results['degree'] = {names[i]: 0.0 for i in range(n_nodes)}

        # PageRank (relatively fast)
        if 'pagerank' in measures:
            self.logger.info("  Computing PageRank...")
            try:
                pr = g.pagerank()
                results['pagerank'] = {names[i]: v for i, v in enumerate(pr)}
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
                        g, normalized=normalized
                    )
                except Exception as e:
                    self.logger.warning(
                        f"NetworKit betweenness failed, falling back to igraph: {e}"
                    )
                    results.pop('betweenness', None)

            if 'betweenness' not in results:
                if n_nodes > self.large_graph_nodes and sample_size is None:
                    self.logger.warning(
                        f"Betweenness centrality on {n_nodes:,} nodes is expensive. "
                        f"Computing with igraph (no sampling available)."
                    )

                self.logger.info("  Computing betweenness centrality...")
                try:
                    bc = g.betweenness()
                    # igraph betweenness is not normalized by default; normalize
                    # to match NetworkX convention: divide by (n-1)(n-2)/2 for
                    # undirected or (n-1)(n-2) for directed.
                    if normalized and n_nodes > 2:
                        if g.is_directed():
                            norm_factor = (n_nodes - 1) * (n_nodes - 2)
                        else:
                            norm_factor = (n_nodes - 1) * (n_nodes - 2) / 2
                        results['betweenness'] = {
                            names[i]: v / norm_factor for i, v in enumerate(bc)
                        }
                    else:
                        results['betweenness'] = {names[i]: v for i, v in enumerate(bc)}
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
                    cc = g.closeness()
                    results['closeness'] = {names[i]: v for i, v in enumerate(cc)}
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
                    ec = g.eigenvector_centrality()
                    results['eigenvector'] = {names[i]: v for i, v in enumerate(ec)}
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
        g: ig.Graph,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute clustering coefficients using igraph.

        Args:
            g: igraph graph
            sample_size: Unused (kept for API compatibility)

        Returns:
            Dict with global and average local clustering coefficients
        """
        self.logger.info("Computing clustering coefficients...")
        n_nodes = g.vcount()

        # Work on undirected copy if needed
        if g.is_directed():
            g_undirected = g.as_undirected()
        else:
            g_undirected = g

        self.logger.info("  Computing transitivity (igraph)...")
        transitivity = g_undirected.transitivity_undirected()

        self.logger.info("  Computing avg local clustering (igraph)...")
        avg_local = g_undirected.transitivity_avglocal_undirected(mode="zero")

        return {
            'transitivity': transitivity,
            'avg_local_clustering': avg_local,
            'clustering_sampled': False,
        }

    def fit_power_law(
        self,
        g: ig.Graph,
        degree_type: str = 'total'
    ) -> Dict:
        """
        Fit power-law distribution to degree sequence.

        Args:
            g: igraph graph
            degree_type: 'total', 'in', or 'out' (for directed graphs)

        Returns:
            Dict with alpha (exponent), xmin, p-value, and comparison results
        """
        if not HAS_POWERLAW:
            self.logger.warning("powerlaw package not installed. Skipping power-law fit.")
            return {'error': 'powerlaw package not installed'}

        self.logger.info(f"Fitting power-law distribution to {degree_type} degrees...")

        # Get degree sequence
        if g.is_directed():
            if degree_type == 'in':
                degrees = g.indegree()
            elif degree_type == 'out':
                degrees = g.outdegree()
            else:
                degrees = g.degree()
        else:
            degrees = g.degree()

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
        g: ig.Graph
    ) -> pd.DataFrame:
        """
        Compute degree distribution.

        Args:
            g: igraph graph

        Returns:
            DataFrame with columns [degree, count, probability, cumulative]
        """
        if g.is_directed():
            in_degrees = g.indegree()
            out_degrees = g.outdegree()
            total_degrees = g.degree()

            # Total degree distribution
            degree_counts = Counter(total_degrees)
        else:
            degrees = g.degree()
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
        g: ig.Graph,
        n_random: int = 10
    ) -> Dict:
        """
        Compute small-world coefficient.

        sigma = (C/C_rand) / (L/L_rand)

        Where:
        - C is clustering coefficient
        - L is average path length
        - _rand indicates random graph equivalent

        Args:
            g: igraph graph
            n_random: Number of random graphs to generate for comparison

        Returns:
            Dict with small-world metrics
        """
        self.logger.info("Computing small-world coefficient...")

        n = g.vcount()
        m = g.ecount()

        # Hard guard: average_shortest_path_length is O(V*E) and will hang
        # on large graphs.
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
        if g.is_directed():
            g = g.as_undirected()

        # Get largest connected component
        if not g.is_connected():
            components = g.connected_components()
            largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
            g = g.subgraph(components[largest_idx])
            n = g.vcount()
            m = g.ecount()
            self.logger.info(f"Using largest connected component: {n:,} nodes")

        # Compute metrics for actual graph
        try:
            C = g.transitivity_avglocal_undirected(mode="zero")
            L = g.average_path_length()
        except Exception as e:
            self.logger.error(f"Failed to compute graph metrics: {e}")
            return {'error': str(e)}

        # Generate random graphs and compute their metrics
        C_rand_list = []
        L_rand_list = []

        for i in range(n_random):
            # Generate Erdos-Renyi random graph
            g_rand = ig.Graph.Erdos_Renyi(n, m=m)

            # Ensure connected (may need to use largest component)
            if not g_rand.is_connected():
                components = g_rand.connected_components()
                largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
                g_rand = g_rand.subgraph(components[largest_idx])

            try:
                C_rand_list.append(
                    g_rand.transitivity_avglocal_undirected(mode="zero")
                )
                L_rand_list.append(g_rand.average_path_length())
            except (Exception,):
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
                import networkx as nx
                nx_graph = to_networkx(g)
                omega = nx.omega(nx_graph, niter=3, nrand=5)
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
            'sigma': sigma,  # sigma > 1 indicates small-world
            'omega': omega,  # -1 < omega < 1, near 0 is small-world
            'is_small_world': sigma > 1 if sigma else None
        }

        self.logger.info(
            f"Small-world: sigma={sigma:.3f if sigma else 'N/A'}, "
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
        print(f"\nPower-law fit: alpha={pl_fit['alpha']:.3f}")


if __name__ == "__main__":
    main()
