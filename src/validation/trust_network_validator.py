"""
Trust Network Validation

Validates SEIR epidemic findings against SNAP Bitcoin trust networks.
The SNAP networks provide an independent signal: if FOMO spreads through
transaction networks, trust relationships should modulate transmission.

Validation checks:
1. Trust-weighted transmission: Do trusted users (high positive ratings)
   have different infection rates than distrusted users?
2. Network structure comparison: Do SNAP trust networks exhibit similar
   topological properties (degree distribution, clustering) to the
   ORBITAAL transaction subgraph?
3. Temporal overlap: Where SNAP and ORBITAAL time ranges overlap,
   do trust patterns correlate with transaction-based SEIR states?
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import igraph as ig
from scipy import stats

from src.utils.logger import get_logger
from src.data_acquisition.snap_downloader import SNAPDownloader


class TrustNetworkValidator:
    """Validate SEIR findings against SNAP Bitcoin trust networks."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_snap_networks(self, snap_dir: str) -> pd.DataFrame:
        """
        Load OTC and Alpha SNAP trust networks and combine them.

        Args:
            snap_dir: Directory containing SNAP CSV files.

        Returns:
            Combined DataFrame with columns
            [source, target, rating, time, datetime, network].
        """
        downloader = SNAPDownloader(data_dir=snap_dir)
        datasets = downloader.load_all()

        frames = []
        for name, df in datasets.items():
            df = df.copy()
            df['network'] = name
            frames.append(df)

        if not frames:
            self.logger.warning("No SNAP datasets loaded")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.logger.info(
            f"Loaded {len(combined):,} SNAP edges "
            f"({combined['network'].nunique()} networks)"
        )
        return combined

    # ------------------------------------------------------------------
    # Trust score computation
    # ------------------------------------------------------------------

    def compute_trust_scores(self, snap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-user trust scores from SNAP rating data.

        Args:
            snap_df: SNAP DataFrame with [source, target, rating] columns.

        Returns:
            DataFrame indexed by user with columns
            [mean_rating, n_ratings, trust_category].
        """
        if snap_df.empty:
            return pd.DataFrame(
                columns=['mean_rating', 'n_ratings', 'trust_category']
            )

        ratings_received = snap_df.groupby('target')['rating'].agg(
            mean_rating='mean',
            n_ratings='count',
        )
        ratings_received['trust_category'] = np.where(
            ratings_received['mean_rating'] > 0, 'trusted',
            np.where(ratings_received['mean_rating'] < 0, 'distrusted', 'neutral')
        )
        ratings_received.index.name = 'user'
        return ratings_received

    # ------------------------------------------------------------------
    # Topology comparison
    # ------------------------------------------------------------------

    def compare_network_topology(
        self,
        snap_df: pd.DataFrame,
        orbitaal_graph: ig.Graph,
    ) -> Dict:
        """
        Compare structural properties between SNAP trust network and
        ORBITAAL transaction graph.

        Uses a two-sample Kolmogorov-Smirnov test on the degree
        distributions and compares clustering coefficients and density.

        Args:
            snap_df: SNAP edge DataFrame.
            orbitaal_graph: ORBITAAL transaction graph.

        Returns:
            Dict with comparison metrics.
        """
        # Build SNAP graph (igraph)
        nodes = sorted(set(snap_df['source'].values) | set(snap_df['target'].values))
        n2i = {n: i for i, n in enumerate(nodes)}
        src_idx = snap_df['source'].map(n2i).values
        tgt_idx = snap_df['target'].map(n2i).values
        edges = list(zip(src_idx.tolist(), tgt_idx.tolist()))
        snap_graph = ig.Graph(n=len(nodes), edges=edges, directed=True)
        snap_graph.vs['name'] = nodes
        snap_graph['_name_to_idx'] = n2i

        snap_degrees = np.array(snap_graph.degree(), dtype=float)
        orb_degrees = np.array(orbitaal_graph.degree(), dtype=float)

        # KS test on degree distributions
        if len(snap_degrees) > 0 and len(orb_degrees) > 0:
            ks_stat, ks_p = stats.ks_2samp(snap_degrees, orb_degrees)
        else:
            ks_stat, ks_p = float('nan'), float('nan')

        # Use NetworkMetrics for clustering (auto-selects best backend)
        from src.network_analysis.metrics import NetworkMetrics
        _metrics = NetworkMetrics()
        snap_cc = _metrics.compute_clustering_coefficients(snap_graph)
        snap_clustering = snap_cc.get('avg_local_clustering', 0.0) or 0.0
        orb_cc = _metrics.compute_clustering_coefficients(orbitaal_graph)
        orb_clustering = orb_cc.get('avg_local_clustering', 0.0) or 0.0

        snap_density = snap_graph.density()
        orb_density = orbitaal_graph.density()

        results = {
            'snap_nodes': snap_graph.vcount(),
            'snap_edges': snap_graph.ecount(),
            'orbitaal_nodes': orbitaal_graph.vcount(),
            'orbitaal_edges': orbitaal_graph.ecount(),
            'snap_mean_degree': float(snap_degrees.mean()) if len(snap_degrees) else 0.0,
            'orbitaal_mean_degree': float(orb_degrees.mean()) if len(orb_degrees) else 0.0,
            'degree_ks_statistic': ks_stat,
            'degree_ks_pvalue': ks_p,
            'snap_clustering': snap_clustering,
            'orbitaal_clustering': orb_clustering,
            'snap_density': snap_density,
            'orbitaal_density': orb_density,
        }

        self.logger.info(
            f"Topology comparison — KS stat: {ks_stat:.4f}, "
            f"p-value: {ks_p:.4f}"
        )
        return results

    # ------------------------------------------------------------------
    # Trust-transmission validation
    # ------------------------------------------------------------------

    def validate_trust_transmission(
        self,
        snap_df: pd.DataFrame,
        node_states: Dict,
        infection_times_df: pd.DataFrame,
    ) -> Dict:
        """
        Test whether trusted vs distrusted users show different infection
        timing, using nodes that appear in both SNAP and ORBITAAL.

        Args:
            snap_df: SNAP edge DataFrame.
            node_states: Dict mapping node → SEIR state.
            infection_times_df: DataFrame with [node, infection_time].

        Returns:
            Dict with test statistic, p-value, effect size, and metadata.
        """
        trust_scores = self.compute_trust_scores(snap_df)
        if trust_scores.empty:
            return self._inconclusive(
                "No trust scores computed from SNAP data"
            )

        # Find overlapping node IDs
        snap_users = set(trust_scores.index)
        orbitaal_nodes = set(node_states.keys()) if node_states else set()
        overlap = snap_users & orbitaal_nodes

        self.logger.info(
            f"Node overlap: {len(overlap)} nodes in both SNAP and ORBITAAL "
            f"(SNAP: {len(snap_users)}, ORBITAAL: {len(orbitaal_nodes)})"
        )

        if len(overlap) < 20:
            return self._inconclusive(
                f"Insufficient node overlap ({len(overlap)} < 20) between "
                f"SNAP and ORBITAAL ID spaces"
            )

        # Merge trust scores with infection times
        if infection_times_df.empty:
            return self._inconclusive(
                "No infection time data available for comparison"
            )

        overlap_trust = trust_scores.loc[
            trust_scores.index.isin(overlap)
        ].copy()

        infection_map = dict(
            zip(infection_times_df['node'], infection_times_df['infection_time'])
        )
        overlap_trust['infection_time'] = overlap_trust.index.map(
            infection_map
        )
        overlap_trust = overlap_trust.dropna(subset=['infection_time'])

        trusted = overlap_trust.loc[
            overlap_trust['trust_category'] == 'trusted', 'infection_time'
        ]
        distrusted = overlap_trust.loc[
            overlap_trust['trust_category'] == 'distrusted', 'infection_time'
        ]

        if len(trusted) < 5 or len(distrusted) < 5:
            return self._inconclusive(
                f"Too few nodes per trust group "
                f"(trusted={len(trusted)}, distrusted={len(distrusted)})"
            )

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(
            trusted, distrusted, alternative='two-sided'
        )

        # Effect size: rank-biserial correlation r = 1 - 2U / (n1 * n2)
        n1, n2 = len(trusted), len(distrusted)
        effect_size = 1.0 - (2.0 * u_stat) / (n1 * n2)

        return {
            'test': 'Mann-Whitney U',
            'test_statistic': float(u_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'n_trusted': n1,
            'n_distrusted': n2,
            'mean_infection_time_trusted': float(trusted.mean()),
            'mean_infection_time_distrusted': float(distrusted.mean()),
            'n_overlap': len(overlap),
            'inconclusive': False,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_validation_report(self, results: Dict) -> str:
        """
        Format all validation results into a readable string report.

        Args:
            results: Dict from run_all_validations().

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "SNAP TRUST NETWORK VALIDATION REPORT",
            "=" * 60,
            "",
        ]

        # Topology
        topo = results.get('topology', {})
        if topo:
            lines.append("1. NETWORK TOPOLOGY COMPARISON")
            lines.append("-" * 40)
            lines.append(
                f"   SNAP network : {topo.get('snap_nodes', '?'):,} nodes, "
                f"{topo.get('snap_edges', '?'):,} edges"
            )
            lines.append(
                f"   ORBITAAL graph: {topo.get('orbitaal_nodes', '?'):,} nodes, "
                f"{topo.get('orbitaal_edges', '?'):,} edges"
            )
            lines.append(
                f"   SNAP mean degree    : {topo.get('snap_mean_degree', 0):.2f}"
            )
            lines.append(
                f"   ORBITAAL mean degree : {topo.get('orbitaal_mean_degree', 0):.2f}"
            )
            lines.append(
                f"   Degree KS statistic : {topo.get('degree_ks_statistic', float('nan')):.4f}"
            )
            lines.append(
                f"   Degree KS p-value   : {topo.get('degree_ks_pvalue', float('nan')):.4f}"
            )
            lines.append(
                f"   SNAP clustering     : {topo.get('snap_clustering', 0):.4f}"
            )
            lines.append(
                f"   ORBITAAL clustering  : {topo.get('orbitaal_clustering', 0):.4f}"
            )
            lines.append(
                f"   SNAP density        : {topo.get('snap_density', 0):.6f}"
            )
            lines.append(
                f"   ORBITAAL density     : {topo.get('orbitaal_density', 0):.6f}"
            )
            lines.append("")

        # Trust scores
        trust = results.get('trust_scores', {})
        if trust:
            lines.append("2. TRUST SCORE SUMMARY")
            lines.append("-" * 40)
            for cat in ('trusted', 'neutral', 'distrusted'):
                lines.append(f"   {cat}: {trust.get(cat, 0):,} users")
            lines.append("")

        # Transmission validation
        tx = results.get('trust_transmission', {})
        if tx:
            lines.append("3. TRUST-TRANSMISSION VALIDATION")
            lines.append("-" * 40)
            if tx.get('inconclusive'):
                lines.append(f"   INCONCLUSIVE: {tx.get('reason', 'unknown')}")
            else:
                lines.append(f"   Test           : {tx.get('test', '?')}")
                lines.append(f"   Test statistic : {tx.get('test_statistic', float('nan')):.4f}")
                lines.append(f"   p-value        : {tx.get('p_value', float('nan')):.4f}")
                lines.append(f"   Effect size    : {tx.get('effect_size', float('nan')):.4f}")
                lines.append(f"   Trusted (n)    : {tx.get('n_trusted', 0)}")
                lines.append(f"   Distrusted (n) : {tx.get('n_distrusted', 0)}")
                lines.append(
                    f"   Mean infection time (trusted)   : "
                    f"{tx.get('mean_infection_time_trusted', float('nan')):.2f}"
                )
                lines.append(
                    f"   Mean infection time (distrusted): "
                    f"{tx.get('mean_infection_time_distrusted', float('nan')):.2f}"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_all_validations(
        self,
        snap_dir: str,
        orbitaal_graph: ig.Graph,
        node_states: Optional[Dict] = None,
        infection_times_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run all SNAP-based validation checks.

        Args:
            snap_dir: Path to directory with SNAP CSV files.
            orbitaal_graph: ORBITAAL transaction graph.
            node_states: Dict mapping node → SEIR state.
            infection_times_df: DataFrame with [node, infection_time].

        Returns:
            Dict with keys 'topology', 'trust_scores', 'trust_transmission'.
        """
        results: Dict = {}

        snap_path = Path(snap_dir)
        if not snap_path.exists() or not any(snap_path.glob('*.csv')):
            self.logger.warning(
                f"SNAP directory missing or empty: {snap_dir}"
            )
            return results

        snap_df = self.load_snap_networks(snap_dir)
        if snap_df.empty:
            self.logger.warning("No SNAP data loaded — skipping validation")
            return results

        # 1. Topology comparison (always possible)
        results['topology'] = self.compare_network_topology(
            snap_df, orbitaal_graph
        )

        # 2. Trust score summary
        trust_scores = self.compute_trust_scores(snap_df)
        results['trust_scores'] = (
            trust_scores['trust_category']
            .value_counts()
            .to_dict()
            if not trust_scores.empty
            else {}
        )

        # 3. Trust-transmission validation
        if node_states is None:
            node_states = {}
        if infection_times_df is None:
            infection_times_df = pd.DataFrame()
        results['trust_transmission'] = self.validate_trust_transmission(
            snap_df, node_states, infection_times_df
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inconclusive(reason: str) -> Dict:
        """Return an inconclusive validation result."""
        return {
            'inconclusive': True,
            'reason': reason,
            'test_statistic': float('nan'),
            'p_value': float('nan'),
            'effect_size': float('nan'),
        }
