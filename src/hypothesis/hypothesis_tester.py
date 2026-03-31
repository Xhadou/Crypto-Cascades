"""
Hypothesis Testing Module

Implements statistical tests for the five research hypotheses:

H1: FOMO episodes follow epidemic dynamics (SEIR model fit)
H2: Network structure amplifies contagion (R₀_network > R₀_basic)
H3: Fear & Greed Index correlates with transmission (β vs FGI)
H4: High-centrality nodes accelerate spread
H5: Community structure creates infection clusters
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters
from src.estimation.estimator import ParameterEstimator, EstimationResult
from src.state_engine.state_assigner import State, StateAssigner
from src.network_analysis.metrics import NetworkMetrics
from src.network_analysis.community_detection import CommunityDetector
from src.utils.logger import get_logger


@dataclass
class HypothesisResult:
    """Result of a hypothesis test."""
    hypothesis: str
    description: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    reject_null: bool
    alpha: float
    sample_size: int
    additional_metrics: Dict
    
    def __str__(self) -> str:
        status = "REJECTED" if self.reject_null else "NOT REJECTED"
        return (
            f"{self.hypothesis}: {status} (p={self.p_value:.4f})\n"
            f"  {self.description}\n"
            f"  Test statistic: {self.test_statistic:.4f}\n"
            f"  Effect size: {self.effect_size:.4f}\n"
            f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]"
        )


class HypothesisTester:
    """
    Tests the five research hypotheses using statistical methods.
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize the hypothesis tester.
        
        Args:
            alpha: Significance level (default 0.05)
            random_seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.logger = get_logger(__name__)
        
    def test_all(
        self,
        G: nx.Graph,
        state_history: pd.DataFrame,
        fgi_values: np.ndarray,
        estimated_params: EstimationResult,
        observed_data: Optional[pd.DataFrame] = None,
        apply_correction: bool = True,
        correction_method: str = 'fdr_bh',
        infection_times_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, HypothesisResult]:
        """
        Run all hypothesis tests.

        Args:
            G: Transaction network
            state_history: DataFrame with state transitions over time
            fgi_values: Fear & Greed Index time series
            estimated_params: Estimated SEIR parameters
            observed_data: Optional observed SEIR data
            apply_correction: Whether to apply multiple testing correction
            correction_method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            infection_times_df: Optional DataFrame with per-node infection times for H4 test

        Returns:
            Dict mapping hypothesis name to result
        """
        results = {}

        self.logger.info("Running all hypothesis tests...")

        results['H1'] = self.test_h1_epidemic_dynamics(
            state_history, estimated_params, observed_data
        )

        results['H2'] = self.test_h2_network_amplification(
            G, estimated_params
        )

        results['H3'] = self.test_h3_fgi_correlation(
            state_history, fgi_values
        )

        results['H4'] = self.test_h4_centrality_effect(
            G, state_history, infection_times_df=infection_times_df
        )

        results['H5'] = self.test_h5_community_clustering(
            G, state_history
        )

        self.logger.info("All hypothesis tests complete.")

        # Add null model comparison as sanity check
        if G.number_of_nodes() < 5000:
            try:
                null_comparison = self.compare_against_null_networks(
                    G, estimated_params, n_null_networks=50
                )
                results['null_comparison'] = null_comparison
            except Exception as e:
                self.logger.warning(f"Null model comparison failed: {e}")

        # Apply multiple testing correction
        if apply_correction:
            results = self.apply_multiple_testing_correction(results, method=correction_method)

        return results

    def apply_multiple_testing_correction(
        self,
        results: Dict[str, HypothesisResult],
        method: str = 'fdr_bh'
    ) -> Dict[str, HypothesisResult]:
        """
        Apply multiple testing correction to hypothesis results.

        Args:
            results: Dict of hypothesis results
            method: Correction method:
                - 'bonferroni': Bonferroni correction (conservative)
                - 'holm': Holm-Bonferroni (less conservative)
                - 'fdr_bh': Benjamini-Hochberg FDR (recommended)

        Returns:
            Updated results dict with adjusted p-values
        """
        self.logger.info(f"Applying {method} multiple testing correction...")

        # Extract p-values (skip NaN/inconclusive and non-HypothesisResult entries)
        hypotheses = sorted(results.keys())
        p_values = []
        valid_hypotheses = []

        for h in hypotheses:
            if not isinstance(results[h], HypothesisResult):
                continue
            p = results[h].p_value
            if np.isfinite(p):
                p_values.append(p)
                valid_hypotheses.append(h)

        if len(p_values) == 0:
            self.logger.warning("No valid p-values to correct")
            return results

        n_tests = len(p_values)
        p_array = np.array(p_values)

        if method == 'bonferroni':
            adjusted_p = np.minimum(p_array * n_tests, 1.0)

        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_idx = np.argsort(p_array)
            adjusted_p = np.zeros(n_tests)
            cummax = 0
            for i, idx in enumerate(sorted_idx):
                adjusted = p_array[idx] * (n_tests - i)
                cummax = max(cummax, adjusted)
                adjusted_p[idx] = min(cummax, 1.0)

        elif method == 'fdr_bh':
            # Benjamini-Hochberg
            sorted_idx = np.argsort(p_array)
            adjusted_p = np.zeros(n_tests)
            cummin = 1.0
            for i in range(n_tests - 1, -1, -1):
                idx = sorted_idx[i]
                adjusted = p_array[idx] * n_tests / (i + 1)
                cummin = min(cummin, adjusted)
                adjusted_p[idx] = min(cummin, 1.0)
        else:
            raise ValueError(f"Unknown correction method: {method}")

        # Update results: store originals in additional_metrics, update primary fields
        for i, h in enumerate(valid_hypotheses):
            # Store original (unadjusted) values in additional_metrics
            results[h].additional_metrics['p_value_original'] = results[h].p_value
            results[h].additional_metrics['reject_null_original'] = results[h].reject_null
            results[h].additional_metrics['correction_method'] = method

            # Store adjusted values in additional_metrics for reference
            results[h].additional_metrics['p_value_adjusted'] = float(adjusted_p[i])
            results[h].additional_metrics['reject_null_adjusted'] = adjusted_p[i] < self.alpha

            # Update primary fields to use corrected values
            results[h].p_value = float(adjusted_p[i])
            results[h].reject_null = bool(adjusted_p[i] < self.alpha)

        self.logger.info(
            f"Correction applied. Primary p_value/reject_null updated to adjusted values. "
            f"Originals stored in additional_metrics['p_value_original'] and ['reject_null_original']."
        )

        return results
    
    def test_h6_market_condition_r0(
        self,
        r0_bull_markets: List[float],
        r0_bear_market: float
    ) -> HypothesisResult:
        """
        H6: R₀ is significantly higher during bull markets than bear markets.
        
        This validates that the model captures FOMO-specific behavior,
        not just general network activity.
        
        Test: One-sample t-test comparing bull R₀s against bear R₀
        Expected: Bull market R₀ > 1, Bear market R₀ < 1
        
        Args:
            r0_bull_markets: List of R₀ values from bull market periods
            r0_bear_market: R₀ value from bear market period
            
        Returns:
            HypothesisResult with test outcome
        """
        self.logger.info("Testing H6: R₀ differs between bull and bear markets...")
        
        bull_array = np.array(r0_bull_markets)

        # If we only have one or two bull market R0s, use permutation test
        # (exact for tiny n, no distributional assumptions needed)
        if len(bull_array) < 3:
            mean_bull = np.mean(bull_array)
            observed_diff = mean_bull - r0_bear_market

            # Pool all R0 values and permute group labels
            all_r0 = list(r0_bull_markets) + [r0_bear_market]
            n_bull = len(r0_bull_markets)

            n_perms = 10000
            rng = np.random.default_rng(self.random_seed)
            perm_diffs = []
            for _ in range(n_perms):
                shuffled = all_r0.copy()
                rng.shuffle(shuffled)
                perm_bull_mean = np.mean(shuffled[:n_bull])
                perm_bear_mean = np.mean(shuffled[n_bull:])
                perm_diffs.append(perm_bull_mean - perm_bear_mean)

            # One-tailed: proportion of permutations where diff >= observed
            p_value = float(np.mean([d >= observed_diff for d in perm_diffs]))
            t_stat = observed_diff / (np.std(perm_diffs) + 1e-10)

            # Effect size (standardized difference)
            if len(bull_array) > 1 and np.std(bull_array) > 0:
                effect_size = observed_diff / np.std(bull_array)
            else:
                effect_size = observed_diff  # Raw difference if can't standardize

            # Bootstrap CI for the difference
            n_bootstrap = 1000
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                boot_bulls = rng.choice(bull_array, size=len(bull_array), replace=True)
                bootstrap_diffs.append(np.mean(boot_bulls) - r0_bear_market)

            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)

            permutation_p_value = p_value
            n_permutations = n_perms

        else:
            # Standard t-test with enough samples
            ttest_result = stats.ttest_1samp(bull_array, r0_bear_market)
            t_stat = float(getattr(ttest_result, 'statistic', ttest_result[0]))  # type: ignore[arg-type]
            p_value_two = float(getattr(ttest_result, 'pvalue', ttest_result[1]))  # type: ignore[arg-type]
            
            # One-tailed test (bull > bear)
            p_value = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2
            
            # Effect size (Cohen's d)
            effect_size = (np.mean(bull_array) - r0_bear_market) / np.std(bull_array)
            
            # Bootstrap CI for difference
            n_bootstrap = 1000
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                boot_bulls = self.rng.choice(bull_array, size=len(bull_array), replace=True)
                bootstrap_diffs.append(np.mean(boot_bulls) - r0_bear_market)
            
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)

            permutation_p_value = None
            n_permutations = 0

        mean_bull = np.mean(bull_array)
        reject_null = p_value < self.alpha and mean_bull > r0_bear_market

        metrics = {
            'r0_bull_mean': mean_bull,
            'r0_bull_values': list(bull_array),
            'r0_bear': r0_bear_market,
            'r0_difference': mean_bull - r0_bear_market,
            'bull_above_threshold': mean_bull > 1,
            'bear_below_threshold': r0_bear_market < 1,
            'interpretation': f"Bull R₀ ({mean_bull:.2f}) vs Bear R₀ ({r0_bear_market:.2f})"
        }

        if permutation_p_value is not None:
            metrics['permutation_p_value'] = permutation_p_value
            metrics['n_permutations'] = n_permutations

        return HypothesisResult(
            hypothesis="H6",
            description="R₀ is higher during bull markets than bear markets",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(reject_null),
            alpha=self.alpha,
            sample_size=len(bull_array) + 1,
            additional_metrics=metrics
        )

    def test_h1_epidemic_dynamics(
        self,
        state_history: pd.DataFrame,
        estimated_params: EstimationResult,
        observed_data: Optional[pd.DataFrame] = None
    ) -> HypothesisResult:
        """
        H1: FOMO episodes follow epidemic dynamics.

        Test: Compare SEIR model fit against alternative growth models using AIC.
        Models compared:
            1. SEIR (epidemic dynamics)
            2. Exponential growth (unlimited growth)
            3. Logistic growth (saturation without recovery)
            4. Linear growth (constant rate)

        Null hypothesis: SEIR does not provide better fit than alternatives
        """
        self.logger.info("Testing H1: FOMO follows epidemic dynamics...")

        from scipy.optimize import curve_fit

        # Get observed infected fraction over time
        if observed_data is not None and 'I_frac' in observed_data.columns:
            t = observed_data['t'].values.astype(float) if 't' in observed_data.columns else np.arange(len(observed_data))
            I_obs = observed_data['I_frac'].values.astype(float)
        elif 'I_frac' in state_history.columns:
            t = state_history['t'].values.astype(float) if 't' in state_history.columns else np.arange(len(state_history))
            I_obs = state_history['I_frac'].values.astype(float)
        else:
            # Try to compute from I and total
            t = np.arange(len(state_history))
            if 'I' in state_history.columns and 'total' in state_history.columns:
                I_obs = (state_history['I'] / state_history['total']).values.astype(float)
            else:
                self.logger.error("Cannot extract infection data for H1 test")
                return self._inconclusive_result("H1", "Missing infection data")

        # Filter out NaN/Inf values
        valid_mask = np.isfinite(I_obs) & np.isfinite(t)
        t = t[valid_mask]
        I_obs = I_obs[valid_mask]

        if len(t) < 10:
            return self._inconclusive_result("H1", "Insufficient data points")

        # Normalize time to start at 0
        t = t - t.min()

        # Store model fits
        model_results = {}
        fitting_diagnostics = {}

        # --- Model 1: SEIR (use provided parameters) ---
        seir_params = estimated_params.to_params()
        seir_model = NetworkSEIR(seir_params)
        N = 10000  # Normalized population
        I0 = max(1, int(I_obs[0] * N))

        try:
            seir_sim = seir_model.simulate_meanfield(N=N, initial_infected=I0, t_max=len(t))
            I_seir = seir_sim['I_frac'].values[:len(t)]
            seir_residuals = I_obs - I_seir
            seir_sse = np.sum(seir_residuals**2)
            seir_aicc = self._compute_aicc(seir_sse, n_params=4, n_obs=len(t))
            model_results['SEIR'] = {
                'sse': seir_sse,
                'aicc': seir_aicc,
                'n_params': 4,
                'fitted': I_seir
            }
            fitting_diagnostics['SEIR'] = {'status': 'success', 'sse': float(seir_sse)}
        except Exception as e:
            self.logger.warning(f"SEIR fitting failed: {e}")
            model_results['SEIR'] = {'sse': np.inf, 'aicc': np.inf, 'n_params': 3}
            fitting_diagnostics['SEIR'] = {'status': 'failed', 'error': str(e)}

        # --- Model 2: Exponential growth ---
        def exponential(t: np.ndarray, a: float, r: float) -> np.ndarray:
            return a * np.exp(r * t)

        try:
            # Bound r to prevent overflow
            popt, _ = curve_fit(exponential, t, I_obs, p0=[I_obs[0] if I_obs[0] > 0 else 0.01, 0.01],
                               bounds=([0, -1], [1, 1]), maxfev=5000)
            I_exp = exponential(t, *popt)
            exp_sse = np.sum((I_obs - I_exp)**2)
            exp_aicc = self._compute_aicc(exp_sse, n_params=2, n_obs=len(t))
            model_results['Exponential'] = {'sse': exp_sse, 'aicc': exp_aicc, 'n_params': 2, 'fitted': I_exp}
            fitting_diagnostics['Exponential'] = {'status': 'success', 'sse': float(exp_sse)}
        except Exception as e:
            self.logger.warning(f"Exponential fitting failed: {e}")
            model_results['Exponential'] = {'sse': np.inf, 'aicc': np.inf, 'n_params': 2}
            fitting_diagnostics['Exponential'] = {'status': 'failed', 'error': str(e)}

        # --- Model 3: Logistic growth ---
        def logistic(t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:
            return K / (1 + np.exp(-r * (t - t0)))

        try:
            p0 = [max(I_obs.max(), 0.01), 0.1, t[len(t)//2]]
            popt, _ = curve_fit(logistic, t, I_obs, p0=p0,
                               bounds=([0, 0, 0], [1, 10, t.max()*2]), maxfev=5000)
            I_log = logistic(t, *popt)
            log_sse = np.sum((I_obs - I_log)**2)
            log_aicc = self._compute_aicc(log_sse, n_params=3, n_obs=len(t))
            model_results['Logistic'] = {'sse': log_sse, 'aicc': log_aicc, 'n_params': 3, 'fitted': I_log}
            fitting_diagnostics['Logistic'] = {'status': 'success', 'sse': float(log_sse)}
        except Exception as e:
            self.logger.warning(f"Logistic fitting failed: {e}")
            model_results['Logistic'] = {'sse': np.inf, 'aicc': np.inf, 'n_params': 3}
            fitting_diagnostics['Logistic'] = {'status': 'failed', 'error': str(e)}

        # --- Model 4: Linear growth ---
        def linear(t: np.ndarray, a: float, b: float) -> np.ndarray:
            return a + b * t

        try:
            popt, _ = curve_fit(linear, t, I_obs, maxfev=5000)
            I_lin = linear(t, *popt)
            lin_sse = np.sum((I_obs - I_lin)**2)
            lin_aicc = self._compute_aicc(lin_sse, n_params=2, n_obs=len(t))
            model_results['Linear'] = {'sse': lin_sse, 'aicc': lin_aicc, 'n_params': 2, 'fitted': I_lin}
            fitting_diagnostics['Linear'] = {'status': 'success', 'sse': float(lin_sse)}
        except Exception as e:
            self.logger.warning(f"Linear fitting failed: {e}")
            model_results['Linear'] = {'sse': np.inf, 'aicc': np.inf, 'n_params': 2}
            fitting_diagnostics['Linear'] = {'status': 'failed', 'error': str(e)}

        # --- Compare models ---
        valid_models = {k: v for k, v in model_results.items() if np.isfinite(v['aicc'])}

        if not valid_models or 'SEIR' not in valid_models:
            return self._inconclusive_result("H1", "Model fitting failed")

        # Find best model by AICc (small-sample corrected)
        best_model = min(valid_models.keys(), key=lambda x: valid_models[x]['aicc'])
        seir_aicc = valid_models['SEIR']['aicc']
        best_aicc = valid_models[best_model]['aicc']

        # Compute AICc weights (Akaike weights)
        aicc_values = [v['aicc'] for v in valid_models.values()]
        min_aicc = min(aicc_values)
        delta_aicc = {k: v['aicc'] - min_aicc for k, v in valid_models.items()}
        exp_delta = {k: np.exp(-0.5 * d) for k, d in delta_aicc.items()}
        sum_exp = sum(exp_delta.values())
        aicc_weights = {k: v / sum_exp for k, v in exp_delta.items()}

        # SEIR is supported if it has highest AICc weight or ΔAICc < 2 from best
        seir_delta_aicc = seir_aicc - best_aicc
        seir_supported = (best_model == 'SEIR') or (seir_delta_aicc < 2)

        # Compute R² for SEIR
        ss_tot = np.sum((I_obs - np.mean(I_obs))**2)
        seir_r2 = 1 - valid_models['SEIR']['sse'] / ss_tot if ss_tot > 0 else 0

        # Effect size: difference in AICc weights between SEIR and next best
        other_weights = [w for k, w in aicc_weights.items() if k != 'SEIR']
        aicc_effect_size = aicc_weights.get('SEIR', 0) - max(other_weights) if other_weights else 0

        # --- Vuong test for non-nested model comparison ---
        # Find the best non-SEIR model that has fitted values
        non_seir_with_fitted = [
            (name, info) for name, info in valid_models.items()
            if name != 'SEIR' and 'fitted' in info
        ]

        eps = 1e-10
        n_obs = len(I_obs)

        if non_seir_with_fitted:
            # Pick the best alternative (lowest AICc among non-SEIR)
            alt_name, alt_info = min(non_seir_with_fitted, key=lambda x: x[1]['aicc'])

            # Per-observation squared residuals
            seir_resid2 = (I_obs - valid_models['SEIR']['fitted'])**2
            alt_resid2 = (I_obs - alt_info['fitted'])**2

            # Per-observation log-likelihood ratio (Gaussian assumption)
            # lr_i > 0 means SEIR fits observation i better
            lr_i = 0.5 * (np.log(alt_resid2 + eps) - np.log(seir_resid2 + eps))

            # Vuong test statistic: mean(lr_i) / (std(lr_i) / sqrt(n))
            lr_std = np.std(lr_i, ddof=1)
            if lr_std > eps:
                vuong_stat = float(np.mean(lr_i) / (lr_std / np.sqrt(n_obs)))
            else:
                # If std is ~0, all observations agree; use sign of mean
                vuong_stat = float(np.sign(np.mean(lr_i)) * 10.0)

            # One-sided p-value: probability that SEIR is NOT better
            p_value = float(1.0 - stats.norm.cdf(vuong_stat))
        else:
            # No alternative model with fitted values; fall back to AIC-based assessment
            vuong_stat = float('nan')
            # Use chi2 survival function on AICc difference as approximate p-value
            p_value = float(np.exp(-0.5 * abs(seir_delta_aicc))) if seir_delta_aicc <= 0 else 0.5

        # --- NRMSE-based confidence interval (replaces ad-hoc R^2 +/- 0.1) ---
        seir_resid2 = (I_obs - valid_models['SEIR']['fitted'])**2
        seir_rmse = np.sqrt(np.mean(seir_resid2))
        obs_range = np.max(I_obs) - np.min(I_obs) + eps
        nrmse = seir_rmse / obs_range

        # Bootstrap NRMSE CI
        nrmse_boots = []
        rng = np.random.default_rng(self.random_seed)
        for _ in range(500):
            idx = rng.choice(n_obs, size=n_obs, replace=True)
            boot_rmse = np.sqrt(np.mean(seir_resid2[idx]))
            boot_nrmse = boot_rmse / obs_range
            nrmse_boots.append(boot_nrmse)
        ci_lo = float(np.percentile(nrmse_boots, 2.5))
        ci_hi = float(np.percentile(nrmse_boots, 97.5))

        # Use NRMSE as effect size (lower is better)
        effect_size = float(nrmse)

        return HypothesisResult(
            hypothesis="H1",
            description="FOMO episodes follow SEIR epidemic dynamics (vs alternative models)",
            test_statistic=seir_aicc,
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=(ci_lo, ci_hi),
            reject_null=bool(seir_supported and p_value < self.alpha),
            alpha=self.alpha,
            sample_size=len(t),
            additional_metrics={
                'model_comparison': {k: {'aicc': v['aicc'], 'sse': v['sse']}
                                    for k, v in valid_models.items()},
                'aicc_weights': aicc_weights,
                'best_model': best_model,
                'seir_r_squared': seir_r2,
                'seir_delta_aicc': seir_delta_aicc,
                'vuong_statistic': float(vuong_stat) if not np.isnan(vuong_stat) else float('nan'),
                'vuong_alternative': alt_name if non_seir_with_fitted else None,
                'nrmse': float(nrmse),
                'aicc_weight_effect_size': float(aicc_effect_size),
                'fitting_diagnostics': fitting_diagnostics,
                'data_quality': {
                    'n_observations': len(t),
                    'has_nan': bool(np.any(np.isnan(I_obs))),
                    'variance': float(np.var(I_obs)),
                    'range': (float(I_obs.min()), float(I_obs.max()))
                },
                'interpretation': f"SEIR {'is' if seir_supported else 'is NOT'} the best model (ΔAICc={seir_delta_aicc:.2f}, Vuong p={p_value:.4f})"
            }
        )

    def _compute_aic(self, sse: float, n_params: int, n_obs: int) -> float:
        """Compute Akaike Information Criterion."""
        if sse <= 0 or n_obs <= n_params:
            return np.inf
        # AIC = n*ln(SSE/n) + 2k
        return n_obs * np.log(sse / n_obs) + 2 * n_params

    def _compute_aicc(self, sse: float, n_params: int, n_obs: int) -> float:
        """Compute corrected Akaike Information Criterion (AICc).

        AICc = AIC + 2k(k+1)/(n-k-1) where k=number of parameters,
        n=number of observations. Returns inf when n <= k+1 (correction
        is undefined).
        """
        aic = self._compute_aic(sse, n_params, n_obs)
        if aic == np.inf or n_obs <= n_params + 1:
            return np.inf
        correction = 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
        return aic + correction

    def _inconclusive_result(self, hypothesis: str, reason: str) -> HypothesisResult:
        """Return an inconclusive hypothesis result."""
        return HypothesisResult(
            hypothesis=hypothesis,
            description=f"{hypothesis} test inconclusive: {reason}",
            test_statistic=float('nan'),
            p_value=float('nan'),
            effect_size=float('nan'),
            confidence_interval=(float('nan'), float('nan')),
            reject_null=False,
            alpha=self.alpha,
            sample_size=0,
            additional_metrics={'reason': reason, 'inconclusive': True}
        )

    def compare_against_null_networks(
        self,
        G: nx.Graph,
        estimated_params: EstimationResult,
        n_null_networks: int = 100,
        null_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare observed network R₀ against null network models.

        This validates that the observed network structure (not just topology)
        contributes to epidemic dynamics.

        Args:
            G: Observed network
            estimated_params: Estimated SEIR parameters
            n_null_networks: Number of null networks to generate per type
            null_types: Types of null models. Options:
                - 'erdos_renyi': Random graph with same n, m
                - 'configuration': Random graph with same degree sequence
                - 'rewired': Randomly rewired preserving degree sequence

        Returns:
            Dict with comparison results for each null type
        """
        if null_types is None:
            null_types = ['erdos_renyi', 'configuration', 'rewired']

        self.logger.info(f"Comparing against {n_null_networks} null networks of types: {null_types}")

        # Compute observed R₀
        model = NetworkSEIR(estimated_params.to_params())
        observed_r0: float = float(model.compute_network_r0(G))

        n = G.number_of_nodes()
        m = G.number_of_edges()
        degrees = [d for _, d in G.degree()]  # type: ignore[misc]

        null_r0s: Dict[str, List[float]] = {nt: [] for nt in null_types}

        for i in range(n_null_networks):
            seed = self.random_seed + i

            if 'erdos_renyi' in null_types:
                try:
                    G_er = nx.gnm_random_graph(n, m, seed=seed)
                    null_r0s['erdos_renyi'].append(float(model.compute_network_r0(G_er)))
                except Exception:
                    pass

            if 'configuration' in null_types:
                try:
                    # Configuration model preserves degree sequence
                    G_config = nx.configuration_model(degrees, seed=seed)
                    G_config = nx.Graph(G_config)  # Remove multi-edges
                    G_config.remove_edges_from(nx.selfloop_edges(G_config))
                    null_r0s['configuration'].append(float(model.compute_network_r0(G_config)))
                except Exception:
                    pass

            if 'rewired' in null_types:
                try:
                    # Double-edge swap preserves degree sequence exactly
                    G_rewired = G.copy()
                    nx.double_edge_swap(G_rewired, nswap=m*2, max_tries=m*20, seed=seed)
                    null_r0s['rewired'].append(float(model.compute_network_r0(G_rewired)))
                except Exception:
                    pass

        # Statistical comparison for each null type
        results: Dict = {
            'observed_r0': observed_r0,
            'n_nodes': n,
            'n_edges': m,
            'comparisons': {}
        }

        for null_type, r0_list in null_r0s.items():
            if len(r0_list) < 10:
                self.logger.warning(f"Insufficient null networks for {null_type}")
                continue

            r0_array = np.array(r0_list)
            null_mean = np.mean(r0_array)
            null_std = np.std(r0_array)

            # Z-score
            if null_std > 0:
                z_score = (observed_r0 - null_mean) / null_std
                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0 if observed_r0 == null_mean else np.inf
                p_value = 1.0 if observed_r0 == null_mean else 0.0

            # Effect size (Cohen's d)
            effect_size = (observed_r0 - null_mean) / null_std if null_std > 0 else 0

            # Percentile of observed in null distribution
            percentile = np.mean(r0_array <= observed_r0) * 100

            results['comparisons'][null_type] = {
                'null_mean': float(null_mean),
                'null_std': float(null_std),
                'null_min': float(np.min(r0_array)),
                'null_max': float(np.max(r0_array)),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'percentile': float(percentile),
                'n_samples': len(r0_list),
                'significant': p_value < self.alpha
            }

        return results
    
    def test_h2_network_amplification(
        self,
        G: nx.Graph,
        estimated_params: EstimationResult,
        n_null: int = 500
    ) -> HypothesisResult:
        """
        H2: Network structure amplifies contagion beyond what random topology predicts.

        Test: Compare the observed network amplification factor (<k^2>/<k>) against
        the distribution of the same statistic computed on configuration-model null
        networks that preserve the degree sequence.

        The old test compared <k^2>/<k> > 1, which is ALWAYS true by Jensen's
        inequality for any non-degenerate graph. The new test asks whether the
        observed topology produces a *higher* amplification factor than expected
        from random graphs with the same degree sequence.

        The configuration model preserves the exact degree sequence but randomizes
        which nodes connect to which, destroying degree-degree correlations,
        clustering, and community structure. If the observed network factor is
        significantly higher than the null distribution, it indicates that the
        network's higher-order structure (not just degree heterogeneity) amplifies
        contagion.

        Null hypothesis: Observed network factor is not greater than configuration-
        model null expectation.
        """
        self.logger.info("Testing H2: Network amplifies contagion (null model comparison)...")

        # Compute observed network metrics
        degree_view = G.degree()  # type: ignore[operator]
        degrees = [d for _, d in degree_view]
        n_nodes = len(degrees)
        k_mean = float(np.mean(degrees))
        k2_mean = float(np.mean([d**2 for d in degrees]))

        if k_mean > 0:
            network_factor = k2_mean / k_mean
        else:
            network_factor = 1.0

        r0_basic = estimated_params.r0()
        r0_network = r0_basic * network_factor

        # --- Null model comparison ---
        # Generate configuration-model null networks preserving the degree sequence.
        # The configuration model randomizes connections while keeping each node's
        # degree fixed, so the null tests whether higher-order structure (clustering,
        # assortativity, community structure) amplifies contagion beyond what the
        # degree distribution alone would predict.
        # Note: nx.configuration_model creates a multigraph; converting to simple
        # graph removes self-loops and multi-edges, slightly reducing effective
        # degrees. This is the standard approach in network epidemiology.
        rng = np.random.default_rng(self.random_seed)
        null_factors: List[float] = []

        for i in range(n_null):
            seed = int(rng.integers(0, 2**31))
            try:
                G_null = nx.configuration_model(degrees, seed=seed)
                G_null = nx.Graph(G_null)  # Remove multi-edges
                G_null.remove_edges_from(nx.selfloop_edges(G_null))
                null_degrees = [d for _, d in G_null.degree()]
                k_mean_null = float(np.mean(null_degrees)) if null_degrees else 1.0
                k2_mean_null = float(np.mean([d**2 for d in null_degrees])) if null_degrees else 1.0
                null_factors.append(k2_mean_null / max(k_mean_null, 1e-10))
            except (nx.NetworkXError, Exception):
                pass

        if len(null_factors) >= 10:
            # One-sided p-value: fraction of null factors >= observed
            p_value = float(np.mean([f >= network_factor for f in null_factors]))
            null_mean = float(np.mean(null_factors))
            null_std = float(np.std(null_factors))

            # Effect size: (observed - null_mean) / null_std  (Cohen's d)
            if null_std > 0:
                effect_size = (network_factor - null_mean) / null_std
            else:
                effect_size = 0.0 if network_factor == null_mean else float('inf')

            # 95% CI of the observed factor from bootstrap resampling of degrees
            bootstrap_factors = []
            for _ in range(1000):
                sample_idx = self.rng.choice(n_nodes, size=n_nodes, replace=True)
                sample_degrees = [degrees[i] for i in sample_idx]
                k_mean_boot = np.mean(sample_degrees)
                k2_mean_boot = np.mean([d**2 for d in sample_degrees])
                if k_mean_boot > 0:
                    bootstrap_factors.append(float(k2_mean_boot / k_mean_boot))

            if bootstrap_factors:
                ci_lower = float(np.percentile(bootstrap_factors, 2.5))
                ci_upper = float(np.percentile(bootstrap_factors, 97.5))
            else:
                ci_lower = float(network_factor)
                ci_upper = float(network_factor)
        else:
            # Fallback if null model generation largely failed
            self.logger.warning("Insufficient null models generated; falling back to bootstrap test")
            p_value = 1.0
            effect_size = 0.0
            ci_lower = float(network_factor)
            ci_upper = float(network_factor)

        return HypothesisResult(
            hypothesis="H2",
            description="Network structure amplifies FOMO contagion beyond null expectation",
            test_statistic=float(network_factor),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            reject_null=bool(p_value < self.alpha),
            alpha=self.alpha,
            sample_size=n_nodes,
            additional_metrics={
                'r0_basic': r0_basic,
                'r0_network': r0_network,
                'network_factor': network_factor,
                'mean_degree': k_mean,
                'degree_variance': float(np.var(degrees)),
                'null_model_factors': [float(f) for f in null_factors],
                'null_model_mean': float(np.mean(null_factors)) if null_factors else None,
                'null_model_std': float(np.std(null_factors)) if null_factors else None,
                'observed_vs_null_p': float(p_value),
                'n_null_models': len(null_factors),
            }
        )
    
    def test_h3_fgi_correlation(
        self,
        state_history: pd.DataFrame,
        fgi_values: np.ndarray,
        max_lag: int = 7
    ) -> HypothesisResult:
        """
        H3: Fear & Greed Index correlates with transmission.
        
        Test: Correlation between FGI and infection rate with lag analysis.
        Null hypothesis: ρ = 0 (no correlation)
        
        Args:
            state_history: DataFrame with state transitions over time
            fgi_values: Fear & Greed Index time series
            max_lag: Maximum number of lag days to test
        """
        self.logger.info("Testing H3: FGI correlates with transmission...")
        
        # Compute new infections per timestep
        if 'I' in state_history.columns:
            infection_counts = state_history.groupby('t')['I'].first().values
        elif 'I_count' in state_history.columns:
            infection_counts = state_history.groupby('t')['I_count'].first().values
        else:
            # Cannot derive infection counts from state_history
            self.logger.warning(
                "H3 test requires 'I' or 'I_count' column in state_history. "
                "Cannot compute meaningful correlation without infection data. "
                "Returning inconclusive result."
            )
            return self._inconclusive_result(
                "H3",
                "Missing infection count data in state_history "
                "(need 'I' or 'I_count' column)"
            )
        
        # Align lengths
        min_len = min(len(infection_counts), len(fgi_values))
        infections = infection_counts[:min_len]
        fgi = fgi_values[:min_len]
        
        # Compute change in infections
        delta_infections = np.diff(np.asarray(infections))
        
        # Lag analysis: test correlations at multiple lags
        best_lag = 0
        best_corr = 0.0
        lag_results = {}
        
        for lag in range(0, max_lag + 1):
            if lag > 0:
                fgi_lagged = fgi[:-lag]
                infections_lagged = delta_infections[lag:]
            else:
                fgi_lagged = fgi[:-1]  # Align with deltas
                infections_lagged = delta_infections
            
            # Align lengths
            lag_min_len = min(len(fgi_lagged), len(infections_lagged))
            if lag_min_len < 10:
                continue
            
            corr_result = spearmanr(fgi_lagged[:lag_min_len], infections_lagged[:lag_min_len])
            corr_val = float(corr_result.statistic if hasattr(corr_result, 'statistic') else corr_result[0])  # type: ignore[arg-type]
            p_val = float(corr_result.pvalue if hasattr(corr_result, 'pvalue') else corr_result[1])  # type: ignore[arg-type]
            lag_results[lag] = {'correlation': corr_val, 'p_value': p_val}
            
            if abs(corr_val) > abs(best_corr):
                best_corr = corr_val
                best_lag = lag
        
        # Use lag-0 for main result (backward compatible), but report best lag
        fgi_aligned = fgi[:-1]
        if len(delta_infections) > 10:
            corr_result = spearmanr(fgi_aligned, delta_infections[:len(fgi_aligned)])
            corr = float(corr_result.statistic if hasattr(corr_result, 'statistic') else corr_result[0])  # type: ignore[arg-type]
            p_value = float(corr_result.pvalue if hasattr(corr_result, 'pvalue') else corr_result[1])  # type: ignore[arg-type]
        else:
            corr, p_value = 0.0, 1.0
        
        # One-tailed test: we hypothesize positive correlation (higher FGI → more infections)
        two_tailed_p = float(p_value)
        if corr > 0:
            p_value_onetail = two_tailed_p / 2
        else:
            p_value_onetail = 1 - two_tailed_p / 2

        # Effect size (correlation is already standardized)
        effect_size = abs(corr)

        # Fisher's z transformation for CI
        n = len(fgi_aligned)
        if abs(corr) < 1:
            z = np.arctanh(corr)
            se = 1 / np.sqrt(n - 3)
            ci_lower = float(np.tanh(z - 1.96 * se))
            ci_upper = float(np.tanh(z + 1.96 * se))
        else:
            ci_lower, ci_upper = float(corr), float(corr)

        # Compute infection trend safely
        infections_arr = np.asarray(infections, dtype=float)
        try:
            infection_trend = float(np.polyfit(range(len(infections_arr)), infections_arr, 1)[0])
        except Exception:
            infection_trend = 0.0

        return HypothesisResult(
            hypothesis="H3",
            description="Fear & Greed Index correlates with FOMO transmission",
            test_statistic=float(corr),
            p_value=float(p_value_onetail),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            reject_null=bool(p_value_onetail < self.alpha and corr > 0),
            alpha=self.alpha,
            sample_size=n,
            additional_metrics={
                'spearman_rho': float(corr),
                'mean_fgi': float(np.mean(fgi)),
                'fgi_std': float(np.std(fgi)),
                'infection_trend': infection_trend,
                'optimal_lag_days': best_lag,
                'lag_analysis': lag_results,
                'best_lag_correlation': float(best_corr),
                'one_tailed': True,
                'two_tailed_p_value': two_tailed_p,
            }
        )
    
    def test_h4_centrality_effect(
        self,
        G: nx.Graph,
        state_history: pd.DataFrame,
        infection_times_df: Optional[pd.DataFrame] = None
    ) -> HypothesisResult:
        """
        H4: High-centrality nodes accelerate spread.

        Test: Compare infection time of high vs low centrality nodes.
        Null hypothesis: No difference in infection timing

        Args:
            G: Transaction network
            state_history: DataFrame with state transitions over time
            infection_times_df: Optional DataFrame with 'node' and 'infection_time' columns
                               from StateAssigner.get_infection_times_df()
        """
        self.logger.info("Testing H4: High-centrality nodes accelerate spread...")

        # Compute centrality
        metrics = NetworkMetrics()
        centrality = metrics.compute_centrality_measures(G, measures=['degree'], normalized=True)
        centrality = centrality.get('degree', {})
        if not centrality:
            centrality = nx.degree_centrality(G)

        # Get infection times: prefer separately-passed infection_times_df,
        # then check state_history columns
        has_infection_data = False
        infection_times: Dict = {}

        if infection_times_df is not None and not infection_times_df.empty:
            if 'node' in infection_times_df.columns and 'infection_time' in infection_times_df.columns:
                infection_times = dict(zip(infection_times_df['node'], infection_times_df['infection_time']))
                has_infection_data = True
                self.logger.info(f"Using {len(infection_times)} node infection times from StateAssigner")
        
        if not has_infection_data:
            if 'node' in state_history.columns and 'infection_time' in state_history.columns:
                infection_times = dict(zip(state_history['node'], state_history['infection_time']))
                has_infection_data = True

        if not has_infection_data:
            self.logger.warning(
                "H4 test requires infection time data. Pass infection_times_df from "
                "StateAssigner.get_infection_times_df() or include 'node' and 'infection_time' "
                "columns in state_history. Returning inconclusive result."
            )
            return HypothesisResult(
                hypothesis="H4",
                description="High-centrality nodes are infected earlier (INCONCLUSIVE - missing data)",
                test_statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                confidence_interval=(float('nan'), float('nan')),
                reject_null=False,
                alpha=self.alpha,
                sample_size=0,
                additional_metrics={
                    'reason': 'missing_infection_time_data',
                    'infection_times_df_provided': infection_times_df is not None,
                    'available_columns': list(state_history.columns)
                }
            )

        # Filter to nodes that exist in both graph and infection data
        nodes = [n for n in G.nodes() if n in infection_times and n in centrality]

        if len(nodes) < 20:
            self.logger.warning(f"Only {len(nodes)} nodes with infection data. Need at least 20.")
            return HypothesisResult(
                hypothesis="H4",
                description="High-centrality nodes are infected earlier (INCONCLUSIVE - insufficient data)",
                test_statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                confidence_interval=(float('nan'), float('nan')),
                reject_null=False,
                alpha=self.alpha,
                sample_size=len(nodes),
                additional_metrics={'reason': 'insufficient_data', 'n_nodes_with_data': len(nodes)}
            )

        # Normalize infection times to numeric (seconds from earliest)
        # StateAssigner produces datetime objects; tests may use numeric values
        raw_values = list(infection_times.values())
        if raw_values and hasattr(raw_values[0], 'timestamp'):
            min_time = min(raw_values)
            infection_times = {
                k: (v - min_time).total_seconds()
                for k, v in infection_times.items()
            }

        # Split into high/low centrality groups
        centrality_values = [centrality.get(n, 0) for n in nodes]
        median_centrality = np.median(centrality_values)

        high_centrality_times = []
        low_centrality_times = []

        for node in nodes:
            c = centrality.get(node, 0)
            if node not in infection_times:
                continue
            t = infection_times[node]
            if c >= median_centrality:
                high_centrality_times.append(t)
            else:
                low_centrality_times.append(t)

        # Mann-Whitney U test (non-parametric)
        if len(high_centrality_times) > 5 and len(low_centrality_times) > 5:
            stat, p_value = stats.mannwhitneyu(
                high_centrality_times,
                low_centrality_times,
                alternative='less'  # High centrality should have lower (earlier) times
            )
        else:
            stat, p_value = 0.0, 1.0

        # Effect size (Cliff's delta)
        n1, n2 = len(high_centrality_times), len(low_centrality_times)
        if n1 > 0 and n2 > 0:
            # Count dominance
            greater = sum(h < l for h in high_centrality_times for l in low_centrality_times)
            less = sum(h > l for h in high_centrality_times for l in low_centrality_times)
            effect_size = (greater - less) / (n1 * n2)
        else:
            effect_size = 0.0

        # Bootstrap CI for mean difference
        n_bootstrap = 1000
        mean_diffs = []

        for _ in range(n_bootstrap):
            h_sample = self.rng.choice(high_centrality_times, size=len(high_centrality_times), replace=True)
            l_sample = self.rng.choice(low_centrality_times, size=len(low_centrality_times), replace=True)
            mean_diffs.append(np.mean(h_sample) - np.mean(l_sample))

        ci_lower = np.percentile(mean_diffs, 2.5)
        ci_upper = np.percentile(mean_diffs, 97.5)

        # --- Cox Proportional Hazards survival analysis ---
        # Models time-to-infection as a function of node degree,
        # giving hazard ratios and proper p-values for the
        # centrality -> infection-time relationship.
        hazard_ratio = None
        cox_p_value = None
        cox_concordance = None
        try:
            from lifelines import CoxPHFitter

            max_time = max(infection_times.values()) + 1
            survival_data = []
            for node in G.nodes():
                deg = G.degree(node)
                infected = node in infection_times
                time = infection_times.get(node, max_time)
                survival_data.append({
                    'time': max(float(time), 0.01),
                    'event': int(infected),
                    'degree': float(deg)
                })
            df_surv = pd.DataFrame(survival_data)

            cph = CoxPHFitter()
            cph.fit(df_surv, duration_col='time', event_col='event')
            hazard_ratio = float(np.exp(cph.params_['degree']))
            cox_p_value = float(cph.summary.loc['degree', 'p'])
            cox_concordance = float(cph.concordance_index_)
            self.logger.info(
                f"H4 Cox PH: hazard_ratio={hazard_ratio:.4f}, "
                f"p={cox_p_value:.4e}, concordance={cox_concordance:.4f}"
            )
        except ImportError:
            self.logger.warning(
                "lifelines not installed; Cox PH analysis skipped for H4. "
                "Install with: pip install lifelines"
            )
        except Exception as e:
            self.logger.warning(f"Cox PH model failed for H4: {e}")

        # Use Cox p-value as primary if available (more powerful test)
        primary_p = cox_p_value if cox_p_value is not None else p_value
        primary_stat = hazard_ratio if hazard_ratio is not None else float(stat)

        additional_metrics = {
            'mean_time_high_centrality': np.mean(high_centrality_times) if high_centrality_times else float('nan'),
            'mean_time_low_centrality': np.mean(low_centrality_times) if low_centrality_times else float('nan'),
            'median_centrality': median_centrality,
            'n_high_centrality': n1,
            'n_low_centrality': n2,
            'mann_whitney_statistic': float(stat),
            'mann_whitney_p_value': float(p_value),
        }
        if hazard_ratio is not None:
            additional_metrics['hazard_ratio'] = hazard_ratio
            additional_metrics['cox_p_value'] = cox_p_value
            additional_metrics['cox_concordance'] = cox_concordance

        return HypothesisResult(
            hypothesis="H4",
            description="High-centrality nodes are infected earlier",
            test_statistic=float(primary_stat),
            p_value=float(primary_p),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(primary_p < self.alpha),
            alpha=self.alpha,
            sample_size=n1 + n2,
            additional_metrics=additional_metrics
        )
    
    def test_h5_community_clustering(
        self,
        G: nx.Graph,
        state_history: pd.DataFrame
    ) -> HypothesisResult:
        """
        H5: Community structure creates infection clusters.
        
        Test: Compare within-community vs between-community infection spread.
        Null hypothesis: Infections spread uniformly across communities
        """
        self.logger.info("Testing H5: Community structure creates clusters...")
        
        # Detect communities
        detector = CommunityDetector()
        communities_result = detector.detect_communities_louvain(G)
        partition = communities_result.get('partition', {})
        modularity_val = communities_result.get('modularity', 0.0)
        
        # Convert partition to communities dict
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Node to community mapping
        node_to_community = partition
        
        # Count within vs between community infection transmission events
        # Use state_history to identify S->E or S->I transitions and check
        # whether they cross community boundaries
        within_community = 0
        between_community = 0
        used_infection_data = False

        # Try to use actual infection transmission events from state_history
        if ('node' in state_history.columns and 'state' in state_history.columns
                and 'datetime' in state_history.columns):
            # Build per-node infection time from state transitions
            infection_events = state_history[
                state_history['state'].isin(['I', State.INFECTED.value if hasattr(State, 'INFECTED') else 'I'])
            ]
            if not infection_events.empty:
                infected_nodes = set(infection_events['node'].unique()) & set(G.nodes())
                # For each infected node, check if the infecting neighbor is in same community
                for node in infected_nodes:
                    comm_node = node_to_community.get(node, -1)
                    if comm_node == -1:
                        continue
                    # Check neighbors that were infected before this node
                    try:
                        neighbors = set(G.neighbors(node))
                        if G.is_directed():
                            neighbors.update(G.predecessors(node))  # type: ignore[union-attr]
                    except Exception:
                        continue
                    infected_neighbors = neighbors & infected_nodes
                    for nbr in infected_neighbors:
                        comm_nbr = node_to_community.get(nbr, -1)
                        if comm_nbr == -1:
                            continue
                        if comm_node == comm_nbr:
                            within_community += 1
                        else:
                            between_community += 1
                if within_community + between_community > 0:
                    used_infection_data = True

        # Fallback: count edges within vs between communities (static graph structure)
        if not used_infection_data:
            self.logger.info(
                "H5: No per-node infection data available in state_history. "
                "Falling back to static edge-based community analysis."
            )
            for u, v in G.edges():
                comm_u = node_to_community.get(u, -1)
                comm_v = node_to_community.get(v, -1)

                if comm_u == comm_v and comm_u != -1:
                    within_community += 1
                else:
                    between_community += 1
        
        total_edges = within_community + between_community
        
        # Expected under null: proportional to community sizes
        community_sizes = [len(nodes) for nodes in communities.values()]
        n_total = sum(community_sizes)
        
        # Expected within-community fraction under random assignment
        if n_total > 0:
            expected_within_frac = sum(s * (s - 1) for s in community_sizes) / (n_total * (n_total - 1))
        else:
            expected_within_frac = 0.5
        
        observed_within_frac = within_community / total_edges if total_edges > 0 else 0
        
        # Permutation test: shuffle community labels and recompute within-fraction
        n_permutations = 1000
        perm_fracs = []
        partition_values = list(partition.values())
        partition_keys = list(partition.keys())
        rng = np.random.default_rng(self.random_seed if hasattr(self, 'random_seed') else 42)
        for _ in range(n_permutations):
            shuffled_values = partition_values.copy()
            rng.shuffle(shuffled_values)
            shuffled_partition = dict(zip(partition_keys, shuffled_values))
            w = 0
            b = 0
            if used_infection_data:
                # Re-count using infection data with shuffled communities
                if ('node' in state_history.columns and 'state' in state_history.columns):
                    infection_events = state_history[
                        state_history['state'].isin(['I', State.INFECTED.value if hasattr(State, 'INFECTED') else 'I'])
                    ]
                    if not infection_events.empty:
                        infected_nodes = set(infection_events['node'].unique()) & set(G.nodes())
                        for node in infected_nodes:
                            comm_node = shuffled_partition.get(node, -1)
                            if comm_node == -1:
                                continue
                            try:
                                neighbors = set(G.neighbors(node))
                                if G.is_directed():
                                    neighbors.update(G.predecessors(node))
                            except Exception:
                                continue
                            infected_neighbors = neighbors & infected_nodes
                            for nbr in infected_neighbors:
                                comm_nbr = shuffled_partition.get(nbr, -1)
                                if comm_nbr == -1:
                                    continue
                                if comm_node == comm_nbr:
                                    w += 1
                                else:
                                    b += 1
            else:
                for u, v in G.edges():
                    comm_u = shuffled_partition.get(u, -1)
                    comm_v = shuffled_partition.get(v, -1)
                    if comm_u == comm_v and comm_u != -1:
                        w += 1
                    else:
                        b += 1
            total_perm = w + b
            perm_fracs.append(w / max(total_perm, 1))

        perm_p_value = float(np.mean([f >= observed_within_frac for f in perm_fracs]))

        # Effect size: ratio of observed to expected
        if expected_within_frac > 0:
            effect_size = (observed_within_frac - expected_within_frac) / expected_within_frac
        else:
            effect_size = 0.0

        # Modularity as additional metric (already computed)
        modularity = modularity_val

        # Bootstrap CI for within-community fraction
        n_bootstrap = 1000
        bootstrap_fracs = []

        # Pre-compute per-edge community match as a boolean array.
        # This avoids materialising a new list of edge tuples per bootstrap
        # iteration — only an integer index array is resampled.
        edge_within = np.array([
            node_to_community.get(u, -1) == node_to_community.get(v, -1)
            for u, v in G.edges()
        ])
        n_edges = len(edge_within)
        for _ in range(n_bootstrap):
            idx = self.rng.choice(n_edges, size=n_edges, replace=True)
            bootstrap_fracs.append(edge_within[idx].mean())

        ci_lower = np.percentile(bootstrap_fracs, 2.5)
        ci_upper = np.percentile(bootstrap_fracs, 97.5)

        return HypothesisResult(
            hypothesis="H5",
            description="Community structure creates FOMO infection clusters",
            test_statistic=float(observed_within_frac),
            p_value=float(perm_p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(perm_p_value < self.alpha and observed_within_frac > expected_within_frac),
            alpha=self.alpha,
            sample_size=total_edges,
            additional_metrics={
                'n_communities': len(communities),
                'modularity': modularity,
                'observed_within_frac': observed_within_frac,
                'expected_within_frac': expected_within_frac,
                'largest_community_size': max(community_sizes) if community_sizes else 0,
                'permutation_p_value': perm_p_value,
                'n_permutations': n_permutations,
            }
        )
    
    def generate_report(
        self,
        results: Dict[str, HypothesisResult]
    ) -> str:
        """
        Generate a formatted report of hypothesis test results.

        Args:
            results: Dict of hypothesis results

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 70)
        report.append("HYPOTHESIS TESTING REPORT")
        report.append("=" * 70)
        report.append(f"Significance level: α = {self.alpha}")

        # Filter to only HypothesisResult entries
        hr_results = {k: v for k, v in results.items() if isinstance(v, HypothesisResult)}

        # Check if correction was applied
        first_result = next(iter(hr_results.values()), None)
        if first_result and 'correction_method' in first_result.additional_metrics:
            method = first_result.additional_metrics['correction_method']
            report.append(f"Multiple testing correction: {method.upper()}")
        report.append("")

        for h_name in sorted(hr_results.keys()):
            result = hr_results[h_name]
            report.append("-" * 70)

            # Check for adjusted values
            p_adj = result.additional_metrics.get('p_value_adjusted')
            reject_adj = result.additional_metrics.get('reject_null_adjusted')

            if p_adj is not None:
                p_orig = result.additional_metrics.get('p_value_original', result.p_value)
                status = "REJECTED" if reject_adj else "NOT REJECTED"
                report.append(f"{h_name}: {status} (p_adj={p_adj:.4f}, p_orig={p_orig:.4f})")
            else:
                status = "REJECTED" if result.reject_null else "NOT REJECTED"
                report.append(f"{h_name}: {status} (p={result.p_value:.4f})")

            report.append(f"  {result.description}")
            report.append(f"  Test statistic: {result.test_statistic:.4f}")
            report.append(f"  Effect size: {result.effect_size:.4f}")
            report.append(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append("")

            # Additional metrics (excluding p-value related ones already shown)
            report.append("  Additional metrics:")
            for key, value in result.additional_metrics.items():
                if key in ['p_value_original', 'p_value_adjusted', 'reject_null_adjusted', 'reject_null_original', 'correction_method']:
                    continue
                if isinstance(value, float):
                    report.append(f"    {key}: {value:.4f}")
                elif isinstance(value, dict):
                    report.append(f"    {key}:")
                    for k2, v2 in value.items():
                        if isinstance(v2, float):
                            report.append(f"      {k2}: {v2:.4f}")
                        else:
                            report.append(f"      {k2}: {v2}")
                else:
                    report.append(f"    {key}: {value}")
            report.append("")

        # Summary
        report.append("=" * 70)
        report.append("SUMMARY")
        report.append("=" * 70)

        # Count rejections (use adjusted if available, originals for comparison)
        n_rejected_orig = sum(
            1 for r in hr_results.values()
            if r.additional_metrics.get('reject_null_original', r.reject_null)
        )
        n_rejected_adj = sum(
            1 for r in hr_results.values()
            if r.additional_metrics.get('reject_null_adjusted', r.reject_null)
        )

        if any('p_value_adjusted' in r.additional_metrics for r in hr_results.values()):
            report.append(f"Hypotheses supported (original): {n_rejected_orig}/{len(hr_results)}")
            report.append(f"Hypotheses supported (adjusted): {n_rejected_adj}/{len(hr_results)}")
        else:
            report.append(f"Hypotheses supported: {n_rejected_orig}/{len(hr_results)}")

        for h_name, result in sorted(hr_results.items()):
            reject_adj = result.additional_metrics.get('reject_null_adjusted', result.reject_null)
            status = "✓ Supported" if reject_adj else "✗ Not supported"
            p_adj = result.additional_metrics.get('p_value_adjusted', result.p_value)
            report.append(f"  {h_name}: {status} (p={p_adj:.4f})")

        return "\n".join(report)


def main():
    """Test hypothesis testing module."""
    import networkx as nx
    
    print("Testing hypothesis testing module...")
    
    # Create test data
    G = nx.barabasi_albert_graph(1000, 3, seed=42)
    
    # Mock state history
    state_history = pd.DataFrame({
        't': list(range(100)) * 10,
        'I': [np.random.poisson(50 + t) for t in range(100)] * 10
    })
    
    # Mock FGI values
    fgi_values = np.random.uniform(30, 70, 100)
    
    # Mock estimated parameters
    estimated_params = EstimationResult(
        beta=0.3, sigma=0.2, gamma=0.1,
        r_squared=0.85, loss=0.001
    )
    
    # Run tests
    tester = HypothesisTester(alpha=0.05)
    results = tester.test_all(G, state_history, fgi_values, estimated_params)
    
    # Print report
    report = tester.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
