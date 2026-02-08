"""
Constants and Thresholds

Centralized location for all magic numbers and thresholds.
These can be overridden by config.yaml values.
"""

from typing import Optional
import yaml
from pathlib import Path


def _load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_threshold(key: str, default: float) -> float:
    """
    Get a threshold value from config or use default.

    Args:
        key: The threshold key (e.g., 'large_graph_nodes')
        default: Default value if not found in config

    Returns:
        The threshold value
    """
    config = _load_config()
    thresholds = config.get('thresholds', {})
    return thresholds.get(key, default)


# Graph size thresholds
def LARGE_GRAPH_NODES() -> int:
    """Threshold for considering a graph as 'large'."""
    return int(get_threshold('large_graph_nodes', 10000))


def VERY_LARGE_GRAPH_NODES() -> int:
    """Threshold for considering a graph as 'very large'."""
    return int(get_threshold('very_large_graph_nodes', 50000))


def MAX_NODES_FOR_CENTRALITY() -> int:
    """Maximum nodes for exact centrality computation."""
    return int(get_threshold('max_nodes_for_centrality', 5000))


# Sampling parameters
def BETWEENNESS_SAMPLE_SIZE() -> int:
    """Sample size for betweenness centrality approximation."""
    return int(get_threshold('betweenness_sample_size', 500))


def CLUSTERING_SAMPLE_SIZE() -> int:
    """Sample size for clustering coefficient estimation."""
    return int(get_threshold('clustering_sample_size', 10000))


# Statistical thresholds
def MIN_DEGREES_FOR_POWERLAW() -> int:
    """Minimum non-zero degrees for power-law fitting."""
    return int(get_threshold('min_degrees_for_powerlaw', 50))


def MIN_NODES_FOR_HYPOTHESIS() -> int:
    """Minimum nodes required for hypothesis testing."""
    return int(get_threshold('min_nodes_for_hypothesis', 20))


def MIN_TIME_POINTS() -> int:
    """Minimum time points for model fitting."""
    return int(get_threshold('min_time_points', 10))


# Monte Carlo parameters
def DEFAULT_BOOTSTRAP_SAMPLES() -> int:
    """Default number of bootstrap samples."""
    return int(get_threshold('default_bootstrap_samples', 100))


def DEFAULT_NULL_NETWORKS() -> int:
    """Default number of null networks for comparison."""
    return int(get_threshold('default_null_networks', 100))


def MAX_GILLESPIE_ITERATIONS_FACTOR() -> int:
    """Factor for max Gillespie iterations (max_iter = t_max * N * factor)."""
    return int(get_threshold('max_gillespie_iterations_factor', 10))
