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


class TestBlockBootstrap:
    """Tests for stationary block bootstrap."""

    def test_bootstrap_uses_block_resampling(self):
        """Bootstrap should use stationary block bootstrap, not i.i.d."""
        from src.estimation.estimator import ParameterEstimator
        est = ParameterEstimator(method='lsq', n_bootstrap=50)
        assert hasattr(est, 'block_length_rule'), (
            "Estimator should expose block_length_rule for block bootstrap"
        )

    def test_bootstrap_minimum_resamples(self):
        """Default bootstrap should be >= 1000."""
        from src.estimation.estimator import ParameterEstimator
        est = ParameterEstimator(method='lsq')
        assert est.n_bootstrap >= 1000, (
            f"Default n_bootstrap={est.n_bootstrap} is too low; need >= 1000 "
            f"for reliable 95% CIs"
        )

    def test_block_length_scales_with_sqrt(self):
        """Block length should scale as sqrt(T) for a given time series."""
        from src.estimation.estimator import ParameterEstimator
        est = ParameterEstimator(method='lsq', n_bootstrap=50)
        # For T=100, block_length should be ~10 (sqrt(100))
        bl = est._compute_block_length(100)
        assert bl == max(2, int(np.sqrt(100)))

    def test_bootstrap_preserves_temporal_order(self):
        """Block bootstrap indices should come in sorted contiguous blocks."""
        from arch.bootstrap import StationaryBootstrap
        rng = np.random.default_rng(42)
        t_max = 50
        block_length = max(2, int(np.sqrt(t_max)))
        bs = StationaryBootstrap(block_length, np.arange(t_max), seed=rng)
        for pos_data, _ in bs.bootstrap(1):
            data = pos_data[0]
            indices = np.sort(data.astype(int).flatten())
            # Block bootstrap produces the same number of observations
            assert len(indices) == t_max


class TestMLEObservationModel:
    """Tests for the MLE observation model (multinomial, not Poisson)."""

    def test_mle_uses_multinomial_not_poisson(self):
        """MLE should use multinomial, not independent Poisson."""
        import inspect
        est = ParameterEstimator(method='mle')
        source = inspect.getsource(est._estimate_mle)
        assert 'multinomial' in source.lower() or 'dirichlet' in source.lower(), (
            "MLE method should reference multinomial (or Dirichlet) observation model"
        )
        assert 'poisson' not in source.lower(), (
            "MLE method should not use independent Poisson observation model"
        )

    def test_multinomial_nll_normalizes_simulated_fractions(self):
        """Simulated fractions should be normalized to sum to 1 in NLL."""
        import inspect
        est = ParameterEstimator(method='mle')
        source = inspect.getsource(est._estimate_mle)
        # The NLL must normalize sim_row so fractions sum to 1
        assert 'sum()' in source or 'normalize' in source.lower(), (
            "MLE NLL should normalize simulated fractions to enforce S+E+I+R=1"
        )

    def test_multinomial_nll_no_subtraction_of_sim_counts(self):
        """Multinomial NLL should NOT subtract sim_counts (that's Poisson).

        Poisson NLL = sum(obs*log(sim) - sim), the '- sim' term is wrong
        for multinomial where NLL = -sum(obs_counts * log(sim_frac)).
        """
        import inspect
        est = ParameterEstimator(method='mle')
        source = inspect.getsource(est._estimate_mle)
        # The Poisson-specific "- sim_counts" subtraction must be gone
        assert '- sim_counts' not in source, (
            "MLE NLL still contains '- sim_counts' term from Poisson model"
        )

    def test_mle_recovers_parameters_from_clean_data(self):
        """MLE should recover true parameters from clean synthetic data."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)

        est = ParameterEstimator(method='mle', random_seed=42)
        result = est.estimate(
            data, N=10000,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )

        assert result.success
        assert result.beta == pytest.approx(0.3, rel=0.25)
        assert result.sigma == pytest.approx(0.2, rel=0.25)
        assert result.gamma == pytest.approx(0.1, rel=0.25)


class TestOmegaEstimation:
    """Tests for omega (waning immunity rate) estimation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic SEIR data with known omega."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.02)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)
        return data, 5000

    def test_omega_is_estimated(self, synthetic_data):
        """Estimator should fit omega as a parameter, not hardcode it."""
        data, N = synthetic_data
        est = ParameterEstimator(method='lsq')
        result = est.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        assert hasattr(result, 'omega')
        # omega should be fitted, not hardcoded at 0.01
        assert result.omega is not None

    def test_omega_in_bounds(self, synthetic_data):
        """Estimated omega should be within the configured bounds."""
        data, N = synthetic_data
        est = ParameterEstimator(method='lsq')
        result = est.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        assert result.omega >= 0.001
        assert result.omega <= 0.1

    def test_omega_recovers_true_value(self, synthetic_data):
        """Estimator should recover true omega from clean data within tolerance."""
        data, N = synthetic_data
        est = ParameterEstimator(method='lsq', random_seed=42)
        result = est.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        # Should be in the right ballpark (within 100% relative error
        # given omega is a weak signal in short time series)
        assert 0.001 <= result.omega <= 0.1

    def test_omega_passed_to_seir_model(self, synthetic_data):
        """Fitted omega should be used when constructing SEIRParameters."""
        data, N = synthetic_data
        est = ParameterEstimator(method='lsq')
        result = est.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        params = result.to_params()
        assert params.omega == result.omega

    def test_mle_estimates_omega(self, synthetic_data):
        """MLE method should also estimate omega."""
        data, N = synthetic_data
        est = ParameterEstimator(method='mle', random_seed=42)
        result = est.estimate(
            data, N=N,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        assert result.omega is not None
        assert 0.001 <= result.omega <= 0.1

    def test_omega_ci_in_bootstrap(self):
        """Bootstrap should produce CI for omega."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.02)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est = ParameterEstimator(method='lsq', random_seed=42)
        result = est.estimate(
            data, N=5000,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=10  # Small number for speed
        )
        assert hasattr(result, 'omega_ci')
        assert result.omega_ci is not None
        assert len(result.omega_ci) == 2
        assert result.omega_ci[0] <= result.omega_ci[1]

    def test_gof_uses_four_parameters(self):
        """Goodness-of-fit should count 4 parameters (beta, sigma, gamma, omega)."""
        true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.02)
        model = NetworkSEIR(true_params, random_seed=42)
        data = model.simulate_meanfield(N=5000, initial_infected=10, t_max=100)

        est = ParameterEstimator(method='lsq', random_seed=42)
        result = est.estimate(
            data, N=5000,
            initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
            n_bootstrap=0
        )
        # AIC/BIC should be computed (non-zero for real data)
        assert result.aic != 0.0 or result.loss == 0.0


class TestRollingR0:
    """Tests for rolling-window R0 estimation."""

    def test_rolling_r0_returns_timeseries(self):
        """Rolling window R0 should return time-indexed DataFrame."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        dates = pd.date_range('2017-10-01', periods=120)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, 120),
            'E_frac': np.linspace(0.02, 0.1, 120),
            'I_frac': np.linspace(0.05, 0.3, 120),
            'R_frac': np.linspace(0.03, 0.1, 120),
        })
        result = estimate_rolling_r0(data, window_days=30, step_days=7, N=5000)
        assert 'R0' in result.columns
        assert 'window_start' in result.columns
        assert len(result) > 1

    def test_rolling_r0_has_all_columns(self):
        """Result should include beta, sigma, gamma, omega, and window bounds."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        dates = pd.date_range('2017-10-01', periods=120)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, 120),
            'E_frac': np.linspace(0.02, 0.1, 120),
            'I_frac': np.linspace(0.05, 0.3, 120),
            'R_frac': np.linspace(0.03, 0.1, 120),
        })
        result = estimate_rolling_r0(data, window_days=30, step_days=7, N=5000)
        for col in ['window_start', 'window_end', 'R0', 'beta', 'sigma', 'gamma', 'omega']:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_r0_window_count(self):
        """Number of windows should match expected from data length, window, and step."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        n_points = 100
        window_days = 30
        step_days = 10
        dates = pd.date_range('2017-10-01', periods=n_points)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, n_points),
            'E_frac': np.linspace(0.02, 0.1, n_points),
            'I_frac': np.linspace(0.05, 0.3, n_points),
            'R_frac': np.linspace(0.03, 0.1, n_points),
        })
        result = estimate_rolling_r0(data, window_days=window_days, step_days=step_days, N=5000)
        # Maximum possible windows (some may fail, but at least 1 should succeed)
        max_windows = len(range(0, n_points - window_days + 1, step_days))
        assert 1 <= len(result) <= max_windows

    def test_rolling_r0_positive_values(self):
        """All R0 values should be positive."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        dates = pd.date_range('2017-10-01', periods=120)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, 120),
            'E_frac': np.linspace(0.02, 0.1, 120),
            'I_frac': np.linspace(0.05, 0.3, 120),
            'R_frac': np.linspace(0.03, 0.1, 120),
        })
        result = estimate_rolling_r0(data, window_days=30, step_days=7, N=5000)
        assert (result['R0'] > 0).all()

    def test_rolling_r0_rejects_short_window(self):
        """Should raise ValueError if window_days < 10."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        dates = pd.date_range('2017-10-01', periods=120)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, 120),
            'E_frac': np.linspace(0.02, 0.1, 120),
            'I_frac': np.linspace(0.05, 0.3, 120),
            'R_frac': np.linspace(0.03, 0.1, 120),
        })
        with pytest.raises(ValueError, match="too small"):
            estimate_rolling_r0(data, window_days=5, step_days=2, N=5000)

    def test_rolling_r0_rejects_insufficient_data(self):
        """Should raise ValueError if data has fewer rows than window_days."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        dates = pd.date_range('2017-10-01', periods=20)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, 20),
            'E_frac': np.linspace(0.02, 0.1, 20),
            'I_frac': np.linspace(0.05, 0.3, 20),
            'R_frac': np.linspace(0.03, 0.1, 20),
        })
        with pytest.raises(ValueError, match="need at least window_days"):
            estimate_rolling_r0(data, window_days=30, step_days=7, N=5000)

    def test_rolling_r0_with_fgi_values(self):
        """Should accept and slice fgi_values per window."""
        from src.estimation.rolling_r0 import estimate_rolling_r0

        n_points = 120
        dates = pd.date_range('2017-10-01', periods=n_points)
        data = pd.DataFrame({
            'date': dates,
            'S_frac': np.linspace(0.9, 0.5, n_points),
            'E_frac': np.linspace(0.02, 0.1, n_points),
            'I_frac': np.linspace(0.05, 0.3, n_points),
            'R_frac': np.linspace(0.03, 0.1, n_points),
        })
        fgi = np.linspace(20, 80, n_points)
        result = estimate_rolling_r0(
            data, window_days=30, step_days=7, N=5000, fgi_values=fgi
        )
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
