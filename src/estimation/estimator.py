"""
Parameter Estimation Module

Estimates SEIR model parameters (β, σ, γ) from observed state transition data
using various optimization methods.

Methods:
1. Least Squares Fitting - Minimize sum of squared differences
2. Maximum Likelihood Estimation - Poisson observation model
3. Bayesian Inference - MCMC sampling with prior distributions
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
from src.utils.logger import get_logger


@dataclass
class EstimationResult:
    """Result of parameter estimation."""
    beta: float
    sigma: float
    gamma: float
    omega: float = 0.01
    
    # Uncertainty quantification
    beta_ci: Tuple[float, float] = (0.0, 1.0)
    sigma_ci: Tuple[float, float] = (0.0, 1.0)
    gamma_ci: Tuple[float, float] = (0.0, 1.0)
    
    # Goodness of fit
    loss: float = 0.0
    r_squared: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    
    # Convergence info
    success: bool = True
    message: str = ""
    n_iterations: int = 0
    
    def r0(self) -> float:
        """Compute R₀ from estimated parameters."""
        return self.beta / self.gamma
    
    def to_params(self) -> SEIRParameters:
        """Convert to SEIRParameters object."""
        return SEIRParameters(
            beta=self.beta,
            sigma=self.sigma,
            gamma=self.gamma,
            omega=self.omega
        )


class ParameterEstimator:
    """
    Estimates SEIR parameters from observed data.
    
    Supports multiple estimation methods:
    - 'lsq': Nonlinear least squares
    - 'mle': Maximum likelihood estimation
    - 'bayesian': Bayesian MCMC (if PyMC available)
    """
    
    def __init__(
        self,
        method: str = 'lsq',
        random_seed: int = 42
    ):
        """
        Initialize the estimator.
        
        Args:
            method: Estimation method ('lsq', 'mle', 'bayesian')
            random_seed: Random seed for reproducibility
        """
        self.method = method
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.logger = get_logger(__name__)
        
        # Parameter bounds
        self.bounds = {
            'beta': (0.01, 1.0),
            'sigma': (0.01, 1.0),
            'gamma': (0.01, 1.0),
        }
        
    def estimate(
        self,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray] = None,
        initial_guess: Optional[Dict[str, float]] = None,
        n_bootstrap: int = 100
    ) -> EstimationResult:
        """
        Estimate SEIR parameters from observed data.
        
        Args:
            observed_data: DataFrame with columns [t, S, E, I, R] or fractions
            N: Total population
            fgi_values: Fear & Greed Index values
            initial_guess: Initial parameter values
            n_bootstrap: Number of bootstrap samples for CI
            
        Returns:
            EstimationResult with fitted parameters and uncertainty
        """
        self.logger.info(f"Estimating parameters using {self.method} method...")
        
        if initial_guess is None:
            initial_guess = {'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1}
        
        if self.method == 'lsq':
            result = self._estimate_lsq(observed_data, N, fgi_values, initial_guess)
        elif self.method == 'mle':
            result = self._estimate_mle(observed_data, N, fgi_values, initial_guess)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Bootstrap confidence intervals
        if n_bootstrap > 0 and result.success:
            result = self._bootstrap_ci(
                result, observed_data, N, fgi_values, n_bootstrap
            )
        
        # Compute goodness of fit metrics
        result = self._compute_gof_metrics(result, observed_data, N, fgi_values)
        
        self.logger.info(
            f"Estimation complete: β={result.beta:.4f}, σ={result.sigma:.4f}, "
            f"γ={result.gamma:.4f}, R₀={result.r0():.3f}"
        )
        
        return result
    
    def _estimate_lsq(
        self,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray],
        initial_guess: Dict[str, float]
    ) -> EstimationResult:
        """Nonlinear least squares estimation."""
        
        # Normalize data
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        def residuals(params):
            """Compute residuals between observed and simulated."""
            beta, sigma, gamma = params
            
            seir_params = SEIRParameters(
                beta=beta, sigma=sigma, gamma=gamma, omega=0.01,
                fomo_enabled=fgi_values is not None
            )
            model = NetworkSEIR(seir_params)
            
            # Get initial conditions
            I0 = max(1, int(obs_fracs['I_frac'].iloc[0] * N))
            
            # Run simulation
            sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
            
            # Compute residuals for all compartments
            res = np.concatenate([
                np.asarray(sim['S_frac'].values) - np.asarray(obs_fracs['S_frac'].values),
                np.asarray(sim['E_frac'].values) - np.asarray(obs_fracs['E_frac'].values),
                np.asarray(sim['I_frac'].values) - np.asarray(obs_fracs['I_frac'].values),
                np.asarray(sim['R_frac'].values) - np.asarray(obs_fracs['R_frac'].values),
            ])
            
            return res
        
        # Initial guess
        x0 = [initial_guess['beta'], initial_guess['sigma'], initial_guess['gamma']]
        
        # Bounds
        bounds = (
            [self.bounds['beta'][0], self.bounds['sigma'][0], self.bounds['gamma'][0]],
            [self.bounds['beta'][1], self.bounds['sigma'][1], self.bounds['gamma'][1]],
        )
        
        # Optimize
        result = optimize.least_squares(
            residuals,
            x0,
            bounds=bounds,
            method='trf',
            loss='soft_l1',  # Robust to outliers
            verbose=0
        )
        
        return EstimationResult(
            beta=result.x[0],
            sigma=result.x[1],
            gamma=result.x[2],
            loss=result.cost,
            success=result.success,
            message=result.message,
            n_iterations=result.nfev
        )
    
    def _estimate_mle(
        self,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray],
        initial_guess: Dict[str, float]
    ) -> EstimationResult:
        """Maximum likelihood estimation with Poisson observation model."""
        
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        def neg_log_likelihood(params):
            """Compute negative log-likelihood."""
            beta, sigma, gamma = params
            
            seir_params = SEIRParameters(
                beta=beta, sigma=sigma, gamma=gamma, omega=0.01,
                fomo_enabled=fgi_values is not None
            )
            model = NetworkSEIR(seir_params)
            
            I0 = max(1, int(obs_fracs['I_frac'].iloc[0] * N))
            sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
            
            # Poisson log-likelihood
            eps = 1e-10
            nll = 0
            
            for col in ['S', 'E', 'I', 'R']:
                obs_counts = np.asarray(obs_fracs[f'{col}_frac'].values) * N
                sim_counts = np.asarray(sim[f'{col}_frac'].values) * N + eps
                
                # Log-likelihood: sum(obs * log(sim) - sim)
                nll -= np.sum(obs_counts * np.log(sim_counts) - sim_counts)
            
            return nll
        
        x0 = [initial_guess['beta'], initial_guess['sigma'], initial_guess['gamma']]
        bounds = [self.bounds['beta'], self.bounds['sigma'], self.bounds['gamma']]
        
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        return EstimationResult(
            beta=result.x[0],
            sigma=result.x[1],
            gamma=result.x[2],
            loss=result.fun,
            success=result.success,
            message=result.message if hasattr(result, 'message') else "",
            n_iterations=result.nfev
        )
    
    def _bootstrap_ci(
        self,
        result: EstimationResult,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray],
        n_bootstrap: int
    ) -> EstimationResult:
        """Compute bootstrap confidence intervals."""
        self.logger.info(f"Computing {n_bootstrap} bootstrap CIs...")
        
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        bootstrap_betas = []
        bootstrap_sigmas = []
        bootstrap_gammas = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(t_max, size=t_max, replace=True)
            indices.sort()
            
            bootstrap_df = obs_fracs.iloc[indices].reset_index(drop=True)
            
            # Re-estimate
            initial = {'beta': result.beta, 'sigma': result.sigma, 'gamma': result.gamma}
            
            try:
                boot_result = self._estimate_lsq(bootstrap_df, N, fgi_values, initial)
                
                if boot_result.success:
                    bootstrap_betas.append(boot_result.beta)
                    bootstrap_sigmas.append(boot_result.sigma)
                    bootstrap_gammas.append(boot_result.gamma)
            except Exception:
                continue
        
        # Compute 95% CIs
        if len(bootstrap_betas) >= 10:
            result.beta_ci = (
                float(np.percentile(bootstrap_betas, 2.5)),
                float(np.percentile(bootstrap_betas, 97.5))
            )
            result.sigma_ci = (
                float(np.percentile(bootstrap_sigmas, 2.5)),
                float(np.percentile(bootstrap_sigmas, 97.5))
            )
            result.gamma_ci = (
                float(np.percentile(bootstrap_gammas, 2.5)),
                float(np.percentile(bootstrap_gammas, 97.5))
            )
        
        return result
    
    def _compute_gof_metrics(
        self,
        result: EstimationResult,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray]
    ) -> EstimationResult:
        """Compute goodness of fit metrics."""
        
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        # Simulate with fitted parameters
        seir_params = result.to_params()
        model = NetworkSEIR(seir_params)
        I0 = max(1, int(obs_fracs['I_frac'].iloc[0] * N))
        sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
        
        # R-squared for Infected compartment
        I_obs = np.asarray(obs_fracs['I_frac'].values)
        I_sim = np.asarray(sim['I_frac'].values)
        
        ss_res = np.sum((I_obs - I_sim) ** 2)
        ss_tot = np.sum((I_obs - np.mean(I_obs)) ** 2)
        
        if ss_tot > 0:
            result.r_squared = 1 - ss_res / ss_tot
        
        # AIC and BIC
        n = t_max * 4  # 4 compartments
        k = 3  # 3 parameters
        
        if result.loss > 0:
            # Assuming loss is sum of squared residuals
            sigma2 = result.loss / n
            ll = -n / 2 * (1 + np.log(2 * np.pi * sigma2))
            
            result.aic = 2 * k - 2 * ll
            result.bic = k * np.log(n) - 2 * ll
        
        return result
    
    def _normalize_data(self, data: pd.DataFrame, N: int) -> pd.DataFrame:
        """Ensure data is in fraction form."""
        df = data.copy()
        
        if 'S_frac' not in df.columns:
            df['S_frac'] = df['S'] / N
            df['E_frac'] = df['E'] / N
            df['I_frac'] = df['I'] / N
            df['R_frac'] = df['R'] / N
        
        return df
    
    def grid_search(
        self,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray] = None,
        beta_range: Tuple[float, float, int] = (0.1, 0.5, 10),
        sigma_range: Tuple[float, float, int] = (0.1, 0.4, 10),
        gamma_range: Tuple[float, float, int] = (0.05, 0.2, 10)
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter space.
        
        Args:
            observed_data: Observed SEIR data
            N: Total population
            fgi_values: Fear & Greed Index values
            beta_range, sigma_range, gamma_range: (min, max, n_points)
            
        Returns:
            DataFrame with all parameter combinations and their loss
        """
        self.logger.info("Running grid search over parameter space...")
        
        beta_vals = np.linspace(*beta_range)
        sigma_vals = np.linspace(*sigma_range)
        gamma_vals = np.linspace(*gamma_range)
        
        results = []
        total = len(beta_vals) * len(sigma_vals) * len(gamma_vals)
        
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        for i, beta in enumerate(beta_vals):
            for sigma in sigma_vals:
                for gamma in gamma_vals:
                    seir_params = SEIRParameters(
                        beta=beta, sigma=sigma, gamma=gamma,
                        fomo_enabled=fgi_values is not None
                    )
                    model = NetworkSEIR(seir_params)
                    
                    I0 = max(1, int(obs_fracs['I_frac'].iloc[0] * N))
                    sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
                    
                    # Mean squared error
                    mse = 0.0
                    for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
                        mse += np.mean((np.asarray(sim[col].values) - np.asarray(obs_fracs[col].values)) ** 2)
                    
                    results.append({
                        'beta': beta,
                        'sigma': sigma,
                        'gamma': gamma,
                        'r0': beta / gamma,
                        'mse': mse
                    })
        
        df = pd.DataFrame(results)
        df = df.sort_values('mse')
        
        self.logger.info(f"Grid search complete. Best MSE: {df['mse'].min():.6f}")
        
        return df
    
    def sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        observed_data: pd.DataFrame,
        N: int,
        perturbation: float = 0.1,
        fgi_values: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Perform local sensitivity analysis.
        
        Args:
            base_params: Base parameter values
            observed_data: Observed data
            N: Population
            perturbation: Fraction to perturb each parameter
            fgi_values: Fear & Greed Index
            
        Returns:
            DataFrame with sensitivity indices
        """
        self.logger.info("Running sensitivity analysis...")
        
        obs_fracs = self._normalize_data(observed_data, N)
        t_max = len(observed_data)
        
        def compute_mse(params_dict):
            seir_params = SEIRParameters(**params_dict, fomo_enabled=fgi_values is not None)
            model = NetworkSEIR(seir_params)
            I0 = max(1, int(obs_fracs['I_frac'].iloc[0] * N))
            sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
            
            mse = sum(
                np.mean((np.asarray(sim[f'{col}_frac'].values) - np.asarray(obs_fracs[f'{col}_frac'].values)) ** 2)
                for col in ['S', 'E', 'I', 'R']
            )
            return mse
        
        # Base MSE
        base_mse = compute_mse(base_params)
        
        results = []
        for param_name in ['beta', 'sigma', 'gamma']:
            base_val = base_params[param_name]
            
            # Perturb up
            params_up = base_params.copy()
            params_up[param_name] = base_val * (1 + perturbation)
            mse_up = compute_mse(params_up)
            
            # Perturb down
            params_down = base_params.copy()
            params_down[param_name] = base_val * (1 - perturbation)
            mse_down = compute_mse(params_down)
            
            # Sensitivity index (normalized gradient)
            sensitivity = (mse_up - mse_down) / (2 * perturbation * base_val)
            # Normalized elasticity (handle zero MSE case)
            if base_mse > 1e-10:
                elasticity = sensitivity * base_val / base_mse
            else:
                elasticity = 0.0  # Perfect fit, no meaningful elasticity
            
            results.append({
                'parameter': param_name,
                'base_value': base_val,
                'sensitivity': sensitivity,
                'elasticity': elasticity,
                'mse_up': mse_up,
                'mse_down': mse_down
            })
        
        return pd.DataFrame(results)

    def cross_validate(
        self,
        observed_data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray] = None,
        train_fraction: float = 0.7,
        n_splits: int = 1,
        rolling_window: bool = False,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Temporal cross-validation for out-of-sample prediction quality.

        Fits parameters on a training portion of the time series and evaluates
        prediction accuracy on the held-out test portion.

        Args:
            observed_data: DataFrame with columns [t, S, E, I, R] or fractions
            N: Total population
            fgi_values: Optional Fear & Greed Index values
            train_fraction: Fraction of data to use for training (default 0.7)
            n_splits: Number of rolling splits (1 = single train/test split)
            rolling_window: If True, use rolling-window cross-validation
            window_size: Fixed training window size (for rolling; defaults to
                         int(train_fraction * len(data)))

        Returns:
            DataFrame with columns: split, train_mse, test_mse, train_r2,
            test_r2, beta, sigma, gamma, train_size, test_size
        """
        self.logger.info(
            f"Running temporal cross-validation "
            f"(train_frac={train_fraction}, n_splits={n_splits}, "
            f"rolling={rolling_window})..."
        )

        obs_fracs = self._normalize_data(observed_data, N)
        T = len(obs_fracs)

        if window_size is None:
            window_size = int(train_fraction * T)

        # Generate split indices
        splits: List[Tuple[int, int]] = []  # (train_end, test_end)
        if rolling_window and n_splits > 1:
            step = max(1, (T - window_size) // n_splits)
            for i in range(n_splits):
                train_end = window_size + i * step
                if train_end >= T:
                    break
                test_end = min(train_end + (T - window_size), T)
                splits.append((train_end, test_end))
        else:
            train_end = int(train_fraction * T)
            splits.append((train_end, T))

        results = []
        for split_idx, (train_end, test_end) in enumerate(splits):
            train_data = obs_fracs.iloc[:train_end].reset_index(drop=True)
            test_data = obs_fracs.iloc[train_end:test_end].reset_index(drop=True)

            if len(train_data) < 10 or len(test_data) < 5:
                self.logger.warning(
                    f"Split {split_idx}: insufficient data "
                    f"(train={len(train_data)}, test={len(test_data)}). Skipping."
                )
                continue

            fgi_train = fgi_values[:train_end] if fgi_values is not None else None
            fgi_test = fgi_values[train_end:test_end] if fgi_values is not None else None

            # Fit on training data
            try:
                est_result = self.estimate(
                    train_data, N=N, fgi_values=fgi_train,
                    initial_guess={'beta': 0.3, 'sigma': 0.2, 'gamma': 0.1},
                    n_bootstrap=0
                )
            except Exception as e:
                self.logger.warning(f"Split {split_idx} training failed: {e}")
                continue

            # Evaluate on training set
            train_mse, train_r2 = self._evaluate_prediction(
                est_result, train_data, N, fgi_train
            )

            # Predict on test set using fitted parameters
            test_mse, test_r2 = self._evaluate_prediction(
                est_result, test_data, N, fgi_test
            )

            results.append({
                'split': split_idx,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'beta': est_result.beta,
                'sigma': est_result.sigma,
                'gamma': est_result.gamma,
                'r0': est_result.r0(),
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'overfit_ratio': test_mse / train_mse if train_mse > 0 else float('inf'),
            })

            self.logger.info(
                f"Split {split_idx}: train_R²={train_r2:.4f}, "
                f"test_R²={test_r2:.4f}, "
                f"train_MSE={train_mse:.6f}, test_MSE={test_mse:.6f}"
            )

        df = pd.DataFrame(results)
        if len(df) > 0:
            self.logger.info(
                f"Cross-validation complete. "
                f"Mean test R²={df['test_r2'].mean():.4f}, "
                f"Mean test MSE={df['test_mse'].mean():.6f}"
            )
        else:
            self.logger.warning("Cross-validation produced no valid splits.")

        return df

    def _evaluate_prediction(
        self,
        est_result: 'EstimationResult',
        data: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Evaluate prediction accuracy for a fitted model against data.

        Args:
            est_result: Fitted parameters
            data: Observed data (normalised fractions)
            N: Population size
            fgi_values: Optional FGI values

        Returns:
            Tuple of (MSE, R²) for the Infected compartment
        """
        seir_params = est_result.to_params()
        model = NetworkSEIR(seir_params)
        t_max = len(data)
        I0 = max(1, int(data['I_frac'].iloc[0] * N))

        try:
            sim = model.simulate_meanfield(N, I0, t_max, fgi_values)
        except Exception:
            return float('inf'), float('-inf')

        I_obs = np.asarray(data['I_frac'].values)
        I_sim = np.asarray(sim['I_frac'].values[:len(I_obs)])

        mse = float(np.mean((I_obs - I_sim) ** 2))

        ss_res = np.sum((I_obs - I_sim) ** 2)
        ss_tot = np.sum((I_obs - np.mean(I_obs)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return mse, r2


def main():
    """Test parameter estimation."""
    import numpy as np
    
    # Generate synthetic data
    print("Generating synthetic SEIR data...")
    true_params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1)
    model = NetworkSEIR(true_params)
    
    synthetic = model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)
    
    # Add some noise
    noise_level = 0.02
    for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
        synthetic[col] += np.random.normal(0, noise_level, len(synthetic))
        synthetic[col] = synthetic[col].clip(0, 1)
    
    # Estimate parameters
    print("\nEstimating parameters from noisy data...")
    estimator = ParameterEstimator(method='lsq')
    
    result = estimator.estimate(
        synthetic, N=10000, 
        initial_guess={'beta': 0.2, 'sigma': 0.15, 'gamma': 0.05},
        n_bootstrap=20
    )
    
    print(f"\nTrue parameters: β={true_params.beta}, σ={true_params.sigma}, γ={true_params.gamma}")
    print(f"Estimated:       β={result.beta:.4f}, σ={result.sigma:.4f}, γ={result.gamma:.4f}")
    print(f"95% CI β: [{result.beta_ci[0]:.4f}, {result.beta_ci[1]:.4f}]")
    print(f"R²: {result.r_squared:.4f}")
    print(f"R₀ true: {true_params.r0():.3f}, R₀ est: {result.r0():.3f}")


if __name__ == "__main__":
    main()
