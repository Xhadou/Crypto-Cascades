"""
Unit Tests for Parameter Estimation Module

Tests parameter fitting functionality including:
- Least squares estimation
- Maximum likelihood estimation
- Bootstrap confidence intervals
- Grid search
- Sensitivity analysis
"""

import pytest
import numpy as np
import pandas as pd

from src.estimation.estimator import ParameterEstimator, EstimationResult
from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters


class TestEstimationResult:
    """Tests for EstimationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating an estimation result."""
        result = EstimationResult(
            beta=0.3, sigma=0.2, gamma=0.1,
            loss=0.001, r_squared=0.95
        )
        
        assert result.beta == 0.3
        assert result.sigma == 0.2
        assert result.gamma == 0.1
    
    def test_r0_calculation(self):
        """Test R0 calculation from estimated parameters."""
        result = EstimationResult(beta=0.3, sigma=0.2, gamma=0.1)
        assert result.r0() == pytest.approx(3.0, rel=1e-6)
    
    def test_to_params_conversion(self):
        """Test conversion to SEIRParameters."""
        result = EstimationResult(beta=0.35, sigma=0.25, gamma=0.15)
        params = result.to_params()
        
        assert isinstance(params, SEIRParameters)
        assert params.beta == 0.35
        assert params.sigma == 0.25
        assert params.gamma == 0.15


class TestParameterEstimator:
    """Tests for ParameterEstimator class."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic SEIR data for testing."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(true_params, random_seed=42)
        
        data = model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)
        return data, 10000
    
    def test_lsq_estimator_creation(self):
        """Test creating a least squares estimator."""
        estimator = ParameterEstimator(method='lsq')
        assert estimator.method == 'lsq'
    
    def test_mle_estimator_creation(self):
        """Test creating a maximum likelihood estimator."""
        estimator = ParameterEstimator(method='mle')
        assert estimator.method == 'mle'
    
    def test_lsq_estimation_returns_result(self, synthetic_data):
        """Test that LSQ estimation returns EstimationResult."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        
        assert isinstance(result, EstimationResult)
    
    def test_lsq_recovers_true_parameters(self, synthetic_data):
        """Test that LSQ can recover true parameters from clean data."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq', random_seed=42)
        
        result = estimator.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        
        # Should recover parameters within 20%
        assert result.beta == pytest.approx(0.3, rel=0.2)
        assert result.sigma == pytest.approx(0.2, rel=0.2)
        assert result.gamma == pytest.approx(0.1, rel=0.2)
    
    def test_estimation_success_flag(self, synthetic_data):
        """Test that successful estimation sets success=True."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.estimate(
            data, N=N,
            initial_guess={'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            n_bootstrap=0
        )
        
        assert result.success == True
    
    def test_bootstrap_confidence_intervals(self, synthetic_data):
        """Test that bootstrap produces confidence intervals."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq', random_seed=42)
        
        result = estimator.estimate(
            data, N=N,
            initial_guess={'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            n_bootstrap=10  # Small number for speed
        )
        
        # CI should be defined and have proper structure
        assert result.beta_ci is not None
        assert len(result.beta_ci) == 2
        assert result.beta_ci[0] <= result.beta_ci[1]  # Lower <= upper
        # CI bounds should be positive (physical constraint)
        assert result.beta_ci[0] >= 0
        assert result.beta_ci[1] > 0
    
    def test_goodness_of_fit_metrics(self, synthetic_data):
        """Test that GOF metrics are computed."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.estimate(
            data, N=N,
            initial_guess={'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            n_bootstrap=0
        )
        
        # R² should be high for data without noise
        assert result.r_squared > 0.9


class TestGridSearch:
    """Tests for grid search functionality."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=5000, initial_infected=10, t_max=50)
        return data, 5000
    
    def test_grid_search_returns_dataframe(self, synthetic_data):
        """Test that grid search returns a DataFrame."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.grid_search(
            data, N=N,
            beta_range=(0.2, 0.4, 3),
            sigma_range=(0.1, 0.3, 3),
            gamma_range=(0.05, 0.15, 3)
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_grid_search_columns(self, synthetic_data):
        """Test that grid search has required columns."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.grid_search(
            data, N=N,
            beta_range=(0.2, 0.4, 3),
            sigma_range=(0.1, 0.3, 3),
            gamma_range=(0.05, 0.15, 3)
        )
        
        for col in ['beta', 'sigma', 'gamma', 'r0', 'mse']:
            assert col in result.columns
    
    def test_grid_search_sorted_by_mse(self, synthetic_data):
        """Test that grid search results are sorted by MSE."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.grid_search(
            data, N=N,
            beta_range=(0.2, 0.4, 5),
            sigma_range=(0.1, 0.3, 5),
            gamma_range=(0.05, 0.15, 5)
        )
        
        # Check that results are sorted (ascending MSE)
        assert result['mse'].is_monotonic_increasing


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=5000, initial_infected=10, t_max=50)
        return data, 5000
    
    def test_sensitivity_returns_dataframe(self, synthetic_data):
        """Test that sensitivity analysis returns DataFrame."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.sensitivity_analysis(
            {'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            data, N=N
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_sensitivity_all_parameters(self, synthetic_data):
        """Test that sensitivity covers all parameters."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.sensitivity_analysis(
            {'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            data, N=N
        )
        
        params_analyzed = set(result['parameter'].values)
        assert params_analyzed == {'beta', 'sigma', 'gamma'}
    
    def test_sensitivity_elasticity_values(self, synthetic_data):
        """Test that elasticity values are reasonable."""
        data, N = synthetic_data
        estimator = ParameterEstimator(method='lsq')
        
        result = estimator.sensitivity_analysis(
            {'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
            data, N=N
        )
        
        # Elasticities should be finite (or zero if MSE is zero/near-zero)
        assert all(np.isfinite(result['elasticity']) | (result['elasticity'] == 0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
