"""
Pytest Configuration and Fixtures

Shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def random_seed():
    """Global random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Set random seed before each test."""
    np.random.seed(random_seed)


@pytest.fixture(scope="session")
def sample_graph():
    """Create a sample graph for testing."""
    return nx.barabasi_albert_graph(500, 3, seed=42)


@pytest.fixture(scope="session")
def sample_transactions():
    """Create sample transaction data."""
    np.random.seed(42)
    n = 1000
    
    return pd.DataFrame({
        'source_id': np.random.randint(0, 100, n),
        'target_id': np.random.randint(0, 100, n),
        'timestamp': pd.date_range('2017-01-01', periods=n, freq='H'),
        'btc_value': np.random.lognormal(0, 2, n),
        'usd_value': np.random.lognormal(8, 2, n)
    })


@pytest.fixture(scope="session")
def sample_fgi():
    """Create sample Fear & Greed Index data."""
    np.random.seed(42)
    return np.random.uniform(25, 75, 365)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_seir_params():
    """Create sample SEIR parameters."""
    from src.epidemic_model.network_seir import SEIRParameters
    return SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)


@pytest.fixture(scope="session")
def sample_seir_data(sample_seir_params):
    """Create sample SEIR simulation data."""
    from src.epidemic_model.network_seir import NetworkSEIR
    
    model = NetworkSEIR(sample_seir_params, random_seed=42)
    return model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
