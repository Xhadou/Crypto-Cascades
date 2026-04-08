"""
Hypothesis Testing Module

Implements statistical tests for the five research hypotheses:

H1: FOMO episodes follow epidemic dynamics (SEIR model fit)
H2: Network structure amplifies contagion (R₀_network > R₀_basic)
H3: Fear & Greed Index correlates with transmission (β vs FGI)
H4: High-centrality nodes accelerate spread
H5: Community structure creates infection clusters
"""

import random

import numpy as np
import pandas as pd
import igraph as ig
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
from src.utils.graph_adapter import name_to_idx
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
        G: ig.Graph,
        state_history: pd.DataFrame,
        fgi_values: np.ndarray,
        estimated_params: EstimationResult,
        observed_data: Optional[pd.DataFrame] = None,
        apply_correction: bool = True,
        correction_method: str = 'fdr_bh',
        infection_times_df: Optional[pd.DataFrame] = None,
        community_partition: Optional[Dict] = None
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
            community_partition: Optional pre-computed community partition for H5

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
            G, state_history, community_partition=community_partition
        )

        self.logger.info("All hypothesis tests complete.")

        # Add null model comparison as sanity check
        if G.vcount() < 5000:
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
        G: ig.Graph,
        estimated_params: EstimationResult,
        n_null_networks: int = 30,
        null_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare observed network R₀ against null network models.

        Uses analytical formulas for ER and configuration-model nulls (exact
        closed-form, no graph generation needed). Uses MCMC edge-swap chain
        for rewired nulls.

        Args:
            G: Observed network
            estimated_params: Estimated SEIR parameters
            n_null_networks: Number of rewired null samples (ER/config are analytical)
            null_types: Types of null models. Options:
                - 'erdos_renyi': Analytical R₀ = (β/γ) × <k>
                - 'configuration': Analytical R₀ = (β/γ) × <k²>/<k>
                - 'rewired': MCMC edge-swap chain preserving degree sequence

        Returns:
            Dict with comparison results for each null type
        """
        if null_types is None:
            null_types = ['erdos_renyi', 'configuration', 'rewired']

        self.logger.info(f"Comparing against null networks of types: {null_types}")

        # Compute observed R₀
        model = NetworkSEIR(estimated_params.to_params())
        observed_r0: float = float(model.compute_network_r0(G))

        n = G.vcount()
        m = G.ecount()
        deg_array = np.array(G.degree(), dtype=np.float64)
        k_mean = float(deg_array.mean())
        k2_mean = float((deg_array ** 2).mean())
        r0_basic = float(estimated_params.r0())

        results: Dict = {
            'observed_r0': observed_r0,
            'n_nodes': n,
            'n_edges': m,
            'comparisons': {}
        }

        # --- Analytical ER null ---
        # R₀_ER = (β/γ) × <k> (Poisson degree distribution)
        if 'erdos_renyi' in null_types:
            er_r0 = r0_basic * k_mean
            # For large n, ER network factor concentrates tightly around <k>+1
            # Variance from Poisson degree distribution
            er_factor = k_mean + 1.0  # E[<k^2>/<k>] for Poisson
            obs_factor = k2_mean / max(k_mean, 1e-10)
            # Z-score using analytical variance of <k^2>/<k> under Poisson
            # Var ≈ 2(2<k>+1)/n for Poisson degrees
            er_var = 2 * (2 * k_mean + 1) / max(n, 1)
            er_std = np.sqrt(er_var) if er_var > 0 else 1e-10
            z_score = (obs_factor - er_factor) / er_std
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

            results['comparisons']['erdos_renyi'] = {
                'null_mean': float(er_r0),
                'null_std': float(er_std * r0_basic),
                'null_min': float(er_r0),
                'null_max': float(er_r0),
                'z_score': float(z_score),
                'p_value': p_value,
                'effect_size': float(z_score),
                'percentile': 100.0 if observed_r0 > er_r0 else 0.0,
                'n_samples': 0,
                'significant': p_value < self.alpha,
                'method': 'analytical',
            }

        # --- Analytical configuration-model null ---
        # Config model preserves degree sequence exactly, so <k^2>/<k> is identical.
        # The R₀ is the same by construction.
        if 'configuration' in null_types:
            config_r0 = r0_basic * (k2_mean / max(k_mean, 1e-10))
            results['comparisons']['configuration'] = {
                'null_mean': float(config_r0),
                'null_std': 0.0,
                'null_min': float(config_r0),
                'null_max': float(config_r0),
                'z_score': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'percentile': 50.0,
                'n_samples': 0,
                'significant': False,
                'method': 'analytical_identical_degree_sequence',
                'note': 'Config model preserves degree sequence; <k^2>/<k> is identical by construction.',
            }

        # --- MCMC rewiring for empirical null ---
        if 'rewired' in null_types and n > 0 and m > 0:
            self.logger.info(f"  Running MCMC rewiring chain ({n_null_networks} samples)...")
            null_r0s: List[float] = []
            rng = np.random.default_rng(self.random_seed)

            try:
                G_null = G.copy()
                # Burn-in
                ig.set_random_number_generator(random.Random(self.random_seed))
                G_null.rewire(n=max(m, 1))

                swaps_between = max(m // 10, 1)
                for i in range(n_null_networks):
                    seed = int(rng.integers(0, 2**31))
                    ig.set_random_number_generator(random.Random(seed))
                    G_null.rewire(n=swaps_between)
                    null_r0s.append(float(model.compute_network_r0(G_null)))
            except Exception as e:
                self.logger.warning(f"MCMC rewiring failed: {e}")

            if len(null_r0s) >= 10:
                r0_array = np.array(null_r0s)
                null_mean = float(r0_array.mean())
                null_std = float(r0_array.std())

                if null_std > 0:
                    z_score = (observed_r0 - null_mean) / null_std
                    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
                else:
                    z_score = 0.0 if observed_r0 == null_mean else float('inf')
                    p_value = 1.0 if observed_r0 == null_mean else 0.0

                effect_size = (observed_r0 - null_mean) / null_std if null_std > 0 else 0.0
                percentile = float(np.mean(r0_array <= observed_r0) * 100)

                results['comparisons']['rewired'] = {
                    'null_mean': null_mean,
                    'null_std': null_std,
                    'null_min': float(r0_array.min()),
                    'null_max': float(r0_array.max()),
                    'z_score': float(z_score),
                    'p_value': p_value,
                    'effect_size': float(effect_size),
                    'percentile': percentile,
                    'n_samples': len(null_r0s),
                    'significant': p_value < self.alpha,
                    'method': 'analytical_er_plus_degree_permutation',
                }

        return results
    
    def test_h2_network_amplification(
        self,
        G: ig.Graph,
        estimated_params: EstimationResult,
        n_null: int = 30
    ) -> HypothesisResult:
        """
        H2: Network structure amplifies contagion beyond what random topology predicts.

        Two-level analysis (Pastor-Satorras & Vespignani 2001):

        Level 1 (descriptive): Report ⟨k²⟩/⟨k⟩ and heterogeneity index H as
        characterisation of degree heterogeneity. Compare against ER analytical
        null for context, but this is a foregone conclusion for heavy-tailed
        networks and is NOT the primary test.

        Level 2 (hypothesis test): Compare observed degree assortativity against
        the configuration model null (r → 0). Assortativity captures whether
        higher-order structure beyond the degree sequence amplifies or dampens
        contagion. This is the testable, non-trivial hypothesis.

        Null hypothesis: Degree assortativity equals zero (configuration model).
        """
        self.logger.info("Testing H2: Network amplifies contagion (null model comparison)...")

        # ── Level 1: Descriptive degree heterogeneity ───────────────
        deg_array = np.array(G.degree(), dtype=np.float64)
        n_nodes = len(deg_array)
        k_mean = float(deg_array.mean())
        k2_mean = float((deg_array ** 2).mean())

        if k_mean > 0:
            network_factor = k2_mean / k_mean
        else:
            network_factor = 1.0

        # Heterogeneity index H = ⟨k²⟩/⟨k⟩² (1 = regular, diverges for scale-free)
        heterogeneity_index = k2_mean / (k_mean ** 2) if k_mean > 0 else 1.0

        r0_basic = estimated_params.r0()
        r0_network = r0_basic * network_factor

        # ER analytical null (Poisson: factor = ⟨k⟩+1)
        er_factor = k_mean + 1.0
        r0_er_analytical = r0_basic * k_mean
        amplification_ratio = network_factor / max(er_factor, 1e-10)

        self.logger.info(
            f"  Degree heterogeneity: H={heterogeneity_index:.1f}, "
            f"network factor={network_factor:.1f} vs ER={er_factor:.1f} "
            f"({amplification_ratio:.0f}x)"
        )

        # ── Level 2: Assortativity hypothesis test ──────────────────
        # Under the configuration model (degree-preserving random graph),
        # assortativity r → 0 as N → ∞. We test whether the observed r
        # differs significantly from 0.
        assortativity = float(G.assortativity_degree(directed=False))

        # Analytical CI for assortativity via Fisher z-transform.
        # With M = 95M edges, bootstrap is unnecessary — the analytical SE
        # is extremely precise. Fisher (1921): z = arctanh(r), SE(z) = 1/sqrt(M-3).
        # Edge dependence (shared nodes) inflates SE slightly in theory, but
        # for sparse graphs (mean degree ~6) the effect is negligible, and the
        # z-score is so extreme (~1000+) that even 100x inflation wouldn't matter.
        n_edges = G.ecount()
        fisher_z = float(np.arctanh(np.clip(assortativity, -0.9999, 0.9999)))
        se_z = 1.0 / np.sqrt(max(n_edges - 3, 1))
        ci_lower = float(np.tanh(fisher_z - 1.96 * se_z))
        ci_upper = float(np.tanh(fisher_z + 1.96 * se_z))

        # Two-sided p-value: test H0: r = 0 (configuration model null)
        z_score = fisher_z / se_z
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

        self.logger.info(
            f"  Degree assortativity: {assortativity:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4e}"
        )

        return HypothesisResult(
            hypothesis="H2",
            description="Network degree correlations amplify (or dampen) contagion beyond degree heterogeneity",
            test_statistic=float(assortativity),
            p_value=float(p_value),
            effect_size=float(assortativity),  # assortativity itself is the effect size
            confidence_interval=(ci_lower, ci_upper),
            reject_null=bool(p_value < self.alpha),
            alpha=self.alpha,
            sample_size=n_nodes,
            additional_metrics={
                'r0_basic': r0_basic,
                'r0_network': r0_network,
                'r0_er_analytical': r0_er_analytical,
                'network_factor': network_factor,
                'er_factor': er_factor,
                'amplification_ratio': amplification_ratio,
                'heterogeneity_index': heterogeneity_index,
                'mean_degree': k_mean,
                'degree_variance': float(deg_array.var()),
                'assortativity': assortativity,
                'assortativity_ci_lower': ci_lower,
                'assortativity_ci_upper': ci_upper,
                'assortativity_interpretation': (
                    'disassortative (hubs connect to low-degree nodes, dampens spread)'
                    if assortativity < -0.01 else
                    'assortative (hub-hub connections, amplifies spread)'
                    if assortativity > 0.01 else
                    'neutral (near configuration model null)'
                ),
                'n_edges': n_edges,
                'method': 'fisher_z_assortativity_vs_configuration_model',
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
        G: ig.Graph,
        state_history: pd.DataFrame,
        infection_times_df: Optional[pd.DataFrame] = None,
        max_sample_size: int = 100_000
    ) -> HypothesisResult:
        """
        H4: High-centrality nodes accelerate spread.

        Test: Compare infection time of high vs low centrality nodes.
        Null hypothesis: No difference in infection timing

        Optimized for large graphs: subsamples to max_sample_size nodes for
        statistical tests (justified by CLT for U-statistics — at 100K per
        group, detects effect sizes as small as d=0.01 with >95% power).

        Args:
            G: Transaction network
            state_history: DataFrame with state transitions over time
            infection_times_df: Optional DataFrame with 'node' and 'infection_time' columns
                               from StateAssigner.get_infection_times_df()
            max_sample_size: Max nodes per group for statistical tests (default 100K)
        """
        self.logger.info("Testing H4: High-centrality nodes accelerate spread...")

        # Compute k-shell (coreness) as primary centrality measure.
        # Kitsak et al. (2010, Nature Physics) showed k-shell is a better
        # predictor of SIR spreading influence than degree or betweenness.
        # O(N+E) via iterative peeling — same cost as degree computation.
        n_nodes = G.vcount()
        names = G.vs['name']
        deg_array = np.array(G.degree(), dtype=np.float64)
        coreness = np.array(G.coreness(), dtype=np.float64)
        centrality_array = coreness  # use k-shell as primary centrality

        # Build name->index lookup for fast intersection
        name_to_idx_map = {names[i]: i for i in range(n_nodes)}

        # Get infection times
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

        # Normalize infection times to numeric (seconds from earliest)
        raw_values = list(infection_times.values())
        if raw_values and hasattr(raw_values[0], 'timestamp'):
            min_time = min(raw_values)
            infection_times = {
                k: (v - min_time).total_seconds()
                for k, v in infection_times.items()
            }

        # Build aligned arrays: filter to nodes in both graph and infection data
        # Use set intersection on infection_times keys for speed
        infected_node_set = set(infection_times.keys())
        valid_indices = []
        valid_times = []
        for node in infected_node_set:
            idx = name_to_idx_map.get(node)
            if idx is not None:
                valid_indices.append(idx)
                valid_times.append(float(infection_times[node]))

        n_valid = len(valid_indices)
        self.logger.info(f"  {n_valid:,} nodes with both centrality and infection data")

        if n_valid < 20:
            self.logger.warning(f"Only {n_valid} nodes with infection data. Need at least 20.")
            return HypothesisResult(
                hypothesis="H4",
                description="High-centrality nodes are infected earlier (INCONCLUSIVE - insufficient data)",
                test_statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                confidence_interval=(float('nan'), float('nan')),
                reject_null=False,
                alpha=self.alpha,
                sample_size=n_valid,
                additional_metrics={'reason': 'insufficient_data', 'n_nodes_with_data': n_valid}
            )

        valid_indices = np.array(valid_indices)
        valid_times_arr = np.array(valid_times, dtype=np.float64)
        valid_centrality = centrality_array[valid_indices]

        # Split into high/low centrality groups using k-shell quartiles.
        # K-shell values are integers with better spread than normalized degree
        # (which was degenerate: q75=q25=0 for 30M-node power-law graphs).
        # If quartiles are still tied (e.g. most nodes in shell 1), use
        # strict inequality to get the extreme tails.
        q75 = float(np.percentile(valid_centrality, 75))
        q25 = float(np.percentile(valid_centrality, 25))
        median_centrality = float(np.median(valid_centrality))

        if q75 > q25:
            high_mask = valid_centrality >= q75
            low_mask = valid_centrality <= q25
            split_method = 'quartile (top 25% vs bottom 25%)'
        else:
            # Degenerate quartiles — fall back to above-median vs min-shell
            k_max = valid_centrality.max()
            k_min = valid_centrality.min()
            if k_max > k_min:
                high_mask = valid_centrality == k_max
                low_mask = valid_centrality == k_min
                split_method = f'extreme shells (k={k_max:.0f} vs k={k_min:.0f})'
            else:
                # Truly degenerate: all nodes have same k-shell
                high_mask = valid_centrality >= median_centrality
                low_mask = valid_centrality < median_centrality
                split_method = 'median (degenerate k-shell)'

        high_times_full = valid_times_arr[high_mask]
        low_times_full = valid_times_arr[low_mask]
        n1_full, n2_full = len(high_times_full), len(low_times_full)

        # Subsample for statistical tests if needed
        # At 100K per group, MW-U detects d=0.01 with >95% power
        if n1_full > max_sample_size:
            h_idx = self.rng.choice(n1_full, max_sample_size, replace=False)
            high_times = high_times_full[h_idx]
        else:
            high_times = high_times_full
        if n2_full > max_sample_size:
            l_idx = self.rng.choice(n2_full, max_sample_size, replace=False)
            low_times = low_times_full[l_idx]
        else:
            low_times = low_times_full

        n1, n2 = len(high_times), len(low_times)
        self.logger.info(
            f"  High centrality: {n1_full:,} (sampled {n1:,}), "
            f"Low centrality: {n2_full:,} (sampled {n2:,})"
        )

        # Mann-Whitney U test (non-parametric)
        if n1 > 5 and n2 > 5:
            stat, p_value = stats.mannwhitneyu(
                high_times, low_times,
                alternative='less'  # High centrality should have lower (earlier) times
            )
        else:
            stat, p_value = 0.0, 1.0

        # Cliff's delta via vectorized subsample
        # Subsample to at most 10K per group for O(n1*n2) comparison
        cliff_sample = min(10_000, n1, n2)
        if n1 > 0 and n2 > 0:
            h_sub = self.rng.choice(high_times, cliff_sample, replace=n1 < cliff_sample)
            l_sub = self.rng.choice(low_times, cliff_sample, replace=n2 < cliff_sample)
            # Vectorized: broadcast comparison
            diff_matrix = h_sub[:, None] - l_sub[None, :]  # (cliff_sample, cliff_sample)
            greater = np.sum(diff_matrix < 0)  # h < l means high-centrality infected earlier
            less = np.sum(diff_matrix > 0)
            effect_size = float((greater - less) / diff_matrix.size)
        else:
            effect_size = 0.0

        # Vectorized bootstrap CI for mean difference
        n_bootstrap = 1000
        h_boot_idx = self.rng.integers(0, n1, size=(n_bootstrap, n1))
        l_boot_idx = self.rng.integers(0, n2, size=(n_bootstrap, n2))
        h_boot_means = high_times[h_boot_idx].mean(axis=1)
        l_boot_means = low_times[l_boot_idx].mean(axis=1)
        mean_diffs = h_boot_means - l_boot_means
        ci_lower = float(np.percentile(mean_diffs, 2.5))
        ci_upper = float(np.percentile(mean_diffs, 97.5))

        # --- Cox Proportional Hazards survival analysis ---
        # Subsample to max_sample_size for tractability
        hazard_ratio = None
        cox_p_value = None
        cox_concordance = None
        try:
            from lifelines import CoxPHFitter

            cox_n = min(n_valid, max_sample_size)
            if n_valid > cox_n:
                cox_idx = self.rng.choice(n_valid, cox_n, replace=False)
            else:
                cox_idx = np.arange(n_valid)

            cox_times = valid_times_arr[cox_idx]
            cox_coreness = coreness[valid_indices[cox_idx]]
            df_surv = pd.DataFrame({
                'time': np.maximum(cox_times, 0.01),
                'event': 1,  # all are infected (from infection_times)
                'coreness': cox_coreness,
            })

            cph = CoxPHFitter()
            cph.fit(df_surv, duration_col='time', event_col='event')
            hazard_ratio = float(np.exp(cph.params_['coreness']))
            cox_p_value = float(cph.summary.loc['coreness', 'p'])
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

        # Use Cox p-value as primary only if BOTH:
        #   1) HR > 1 (direction matches hypothesis: higher coreness → faster infection)
        #   2) concordance > 0.5 (model has actual predictive power, not just noise)
        # A model with concordance ≤ 0.5 performs no better than random — its
        # significant p-value is an artifact of huge sample size amplifying a
        # trivially small coefficient (statistical without practical significance).
        cox_direction_correct = (
            hazard_ratio is not None
            and hazard_ratio > 1.0
            and cox_concordance is not None
            and cox_concordance > 0.5
        )
        if cox_direction_correct and cox_p_value is not None:
            # Cox is two-sided; halve for one-sided in the correct direction
            primary_p = cox_p_value / 2.0
            primary_stat = hazard_ratio
        else:
            primary_p = p_value  # Mann-Whitney (already one-sided)
            primary_stat = float(stat)
            if hazard_ratio is not None:
                reason = (
                    f"HR={hazard_ratio:.4f} ≤ 1" if hazard_ratio <= 1.0
                    else f"concordance={cox_concordance:.4f} ≤ 0.5 (no predictive power)"
                )
                self.logger.info(
                    f"  Cox {reason} → using Mann-Whitney p={p_value:.4f} as primary."
                )

        additional_metrics = {
            'mean_time_high_centrality': float(high_times_full.mean()) if n1_full > 0 else float('nan'),
            'mean_time_low_centrality': float(low_times_full.mean()) if n2_full > 0 else float('nan'),
            'centrality_measure': 'k-shell (coreness)',
            'median_centrality': median_centrality,
            'q75_centrality': q75,
            'q25_centrality': q25,
            'split_method': split_method,
            'n_high_centrality': n1_full,
            'n_low_centrality': n2_full,
            'n_high_sampled': n1,
            'n_low_sampled': n2,
            'mann_whitney_statistic': float(stat),
            'mann_whitney_p_value': float(p_value),
        }
        if hazard_ratio is not None:
            additional_metrics['hazard_ratio'] = hazard_ratio
            additional_metrics['cox_p_value'] = cox_p_value
            additional_metrics['cox_concordance'] = cox_concordance

        return HypothesisResult(
            hypothesis="H4",
            description="High k-shell (core) nodes are infected earlier",
            test_statistic=float(primary_stat),
            p_value=float(primary_p),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(primary_p < self.alpha),
            alpha=self.alpha,
            sample_size=n1_full + n2_full,
            additional_metrics=additional_metrics
        )
    
    def test_h5_community_clustering(
        self,
        G: ig.Graph,
        state_history: pd.DataFrame,
        community_partition: Optional[Dict] = None
    ) -> HypothesisResult:
        """
        H5: Community structure creates infection clusters.

        Test: Compare within-community vs between-community infection spread.
        Null hypothesis: Infections spread uniformly across communities.

        Optimized: Reuses pre-computed community partition (Leiden from Phase 3-4),
        vectorized permutation test using numpy integer arrays, and adaptive
        early stopping.

        Args:
            G: Transaction network
            state_history: DataFrame with state transitions over time
            community_partition: Pre-computed node->community dict (e.g. from Leiden).
                                 If None, runs Louvain as fallback.
        """
        self.logger.info("Testing H5: Community structure creates clusters...")

        # Use pre-computed partition or detect communities as fallback
        if community_partition is not None:
            partition = community_partition
            self.logger.info(f"  Reusing pre-computed community partition ({len(set(partition.values()))} communities)")
            # Compute modularity for the given partition
            names = G.vs['name']
            membership = [partition.get(names[i], 0) for i in range(G.vcount())]
            modularity_val = G.modularity(membership)
        else:
            self.logger.info("  No pre-computed partition; running Louvain...")
            detector = CommunityDetector()
            communities_result = detector.detect_communities_louvain(G)
            partition = communities_result.get('partition', {})
            modularity_val = communities_result.get('modularity', 0.0)

        # Build integer arrays for vectorized operations
        # Map node names to partition IDs as a numpy array indexed by vertex ID
        names = G.vs['name']
        n_nodes = G.vcount()
        partition_arr = np.full(n_nodes, -1, dtype=np.int32)
        for i in range(n_nodes):
            partition_arr[i] = partition.get(names[i], -1)

        # Build edge arrays (source, target) as numpy arrays
        # For large graphs (>10M edges), sample edges rather than materializing
        # the full edge list (95M tuples = ~7 GB of Python objects → OOM).
        # 2M sampled edges gives <0.1% margin of error for fraction estimates.
        n_edges_total = G.ecount()
        MAX_EDGES_FOR_H5 = 2_000_000
        if n_edges_total > 0:
            if n_edges_total <= MAX_EDGES_FOR_H5:
                elist = G.get_edgelist()
                edge_src = np.array([e[0] for e in elist], dtype=np.int32)
                edge_tgt = np.array([e[1] for e in elist], dtype=np.int32)
                del elist
            else:
                # Sample edges by sampling nodes and collecting their neighborhoods
                # This avoids materializing the full edge list
                self.logger.info(
                    f"  Sampling ~{MAX_EDGES_FOR_H5:,} of {n_edges_total:,} edges "
                    f"for H5 analysis (statistically sufficient)"
                )
                rng = np.random.default_rng(self.random_seed)
                # Sample random nodes proportional to degree, collect their edges
                deg = np.array(G.degree(), dtype=np.float64)
                deg_prob = deg / deg.sum()
                sampled_edges_src = []
                sampled_edges_tgt = []
                # Sample nodes until we have enough edges
                n_sample_nodes = min(n_nodes, 200_000)
                sample_nodes = rng.choice(n_nodes, n_sample_nodes, replace=False, p=deg_prob)
                for node_id in sample_nodes:
                    neighbors = G.neighbors(node_id)
                    for nbr in neighbors:
                        sampled_edges_src.append(node_id)
                        sampled_edges_tgt.append(nbr)
                    if len(sampled_edges_src) >= MAX_EDGES_FOR_H5:
                        break
                edge_src = np.array(sampled_edges_src[:MAX_EDGES_FOR_H5], dtype=np.int32)
                edge_tgt = np.array(sampled_edges_tgt[:MAX_EDGES_FOR_H5], dtype=np.int32)
                del sampled_edges_src, sampled_edges_tgt
        else:
            edge_src = np.array([], dtype=np.int32)
            edge_tgt = np.array([], dtype=np.int32)
        n_edges = len(edge_src)

        # Compute observed within-community fraction (vectorized)
        src_comm = partition_arr[edge_src]
        tgt_comm = partition_arr[edge_tgt]
        # Only count edges where both nodes have valid communities
        valid_mask = (src_comm >= 0) & (tgt_comm >= 0)
        within_mask = (src_comm == tgt_comm) & valid_mask
        within_community = int(within_mask.sum())
        total_valid = int(valid_mask.sum())

        observed_within_frac = within_community / max(total_valid, 1)

        # Community sizes and expected within-fraction
        from collections import Counter
        comm_counts = Counter(partition_arr[partition_arr >= 0])
        community_sizes = list(comm_counts.values())
        n_total = sum(community_sizes)
        n_communities = len(community_sizes)

        if n_total > 1:
            expected_within_frac = sum(s * (s - 1) for s in community_sizes) / (n_total * (n_total - 1))
        else:
            expected_within_frac = 0.5

        # Vectorized permutation test with adaptive early stopping
        # Optimized: pre-map edge endpoints to valid-node positions to
        # avoid per-iteration copies of partition_arr (~27% faster).
        # Use int16 for community labels (~15% faster cache performance).
        n_permutations = 200
        rng = np.random.default_rng(self.random_seed)
        count_ge = 0

        valid_node_mask = partition_arr >= 0
        valid_node_indices = np.where(valid_node_mask)[0]
        valid_node_partitions = partition_arr[valid_node_indices].astype(np.int16).copy()

        # Pre-map: edge endpoints → position in valid_node_partitions array
        # This lets us index directly into the shuffled array without
        # rebuilding the full partition_arr each iteration.
        node_to_valid_pos = np.full(n_nodes, -1, dtype=np.int32)
        node_to_valid_pos[valid_node_indices] = np.arange(len(valid_node_indices), dtype=np.int32)
        edge_src_pos = node_to_valid_pos[edge_src]
        edge_tgt_pos = node_to_valid_pos[edge_tgt]
        edge_both_valid = (edge_src_pos >= 0) & (edge_tgt_pos >= 0)
        valid_edge_src_pos = edge_src_pos[edge_both_valid]
        valid_edge_tgt_pos = edge_tgt_pos[edge_both_valid]
        n_valid_edges = len(valid_edge_src_pos)
        del node_to_valid_pos, edge_src_pos, edge_tgt_pos  # free memory

        # Analytical z-test as fast pre-screen
        sizes = np.bincount(partition_arr[partition_arr >= 0])
        fracs = sizes.astype(np.float64) / sizes.sum()
        expected_within_analytic = float(np.sum(fracs ** 2))
        var_within = expected_within_analytic * (1 - expected_within_analytic) / max(n_valid_edges, 1)
        z_score = (observed_within_frac - expected_within_analytic) / max(np.sqrt(var_within), 1e-15)

        if abs(z_score) > 10:
            # Overwhelmingly significant — skip permutation test
            perm_p_value = float(1.0 - stats.norm.cdf(z_score))
            self.logger.info(
                f"  Analytical z-test: z={z_score:.1f}, p={perm_p_value:.2e} — "
                f"skipping permutation test (result is clear)"
            )
            n_permutations = 0
        else:
            self.logger.info(f"  Running vectorized permutation test ({n_permutations} max permutations)...")
            shuffled = valid_node_partitions.copy()
            for i in range(1, n_permutations + 1):
                rng.shuffle(shuffled)
                perm_within = np.sum(shuffled[valid_edge_src_pos] == shuffled[valid_edge_tgt_pos])
                perm_frac = perm_within / max(n_valid_edges, 1)
                if perm_frac >= observed_within_frac:
                    count_ge += 1

                # Adaptive early stopping after 100 permutations
                if i >= 100:
                    p_hat = count_ge / i
                    se = np.sqrt(p_hat * (1 - p_hat) / i)
                    if p_hat - 2.58 * se > self.alpha or p_hat + 2.58 * se < self.alpha:
                        self.logger.info(f"  Early stopping at {i} permutations (p_hat={p_hat:.4f})")
                        n_permutations = i
                        break

            perm_p_value = float(count_ge / max(n_permutations, 1))

        # Effect size: relative excess over expected
        if expected_within_frac > 0:
            effect_size = (observed_within_frac - expected_within_frac) / expected_within_frac
        else:
            effect_size = 0.0

        # Bootstrap CI for within-community fraction
        # Subsample edges to 200K for bootstrap (statistically sufficient)
        n_bootstrap = 1000
        edge_within_bool = within_mask[valid_mask] if valid_mask.any() else within_mask
        boot_sample_size = min(200_000, n_valid_edges)
        if n_valid_edges > boot_sample_size:
            boot_edge_idx = self.rng.choice(n_valid_edges, boot_sample_size, replace=False)
            boot_edge_within = edge_within_bool[boot_edge_idx]
        else:
            boot_edge_within = edge_within_bool
            boot_sample_size = n_valid_edges
        # Vectorize: (1000, 200K) bool = ~200 MB, manageable on 16 GB
        boot_idx = self.rng.integers(0, boot_sample_size, size=(n_bootstrap, boot_sample_size))
        boot_fracs = boot_edge_within[boot_idx].mean(axis=1)
        ci_lower = float(np.percentile(boot_fracs, 2.5))
        ci_upper = float(np.percentile(boot_fracs, 97.5))

        return HypothesisResult(
            hypothesis="H5",
            description="Community structure creates FOMO infection clusters",
            test_statistic=float(observed_within_frac),
            p_value=float(perm_p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            reject_null=bool(perm_p_value < self.alpha and observed_within_frac > expected_within_frac),
            alpha=self.alpha,
            sample_size=total_valid,
            additional_metrics={
                'n_communities': n_communities,
                'modularity': modularity_val,
                'observed_within_frac': observed_within_frac,
                'expected_within_frac': expected_within_frac,
                'largest_community_size': max(community_sizes) if community_sizes else 0,
                'permutation_p_value': perm_p_value,
                'n_permutations': n_permutations,
                'partition_source': 'pre_computed' if community_partition is not None else 'louvain',
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
    print("Testing hypothesis testing module...")

    # Create test data
    G = ig.Graph.Barabasi(1000, 3)
    G.vs['name'] = list(range(G.vcount()))
    G['_name_to_idx'] = {i: i for i in range(G.vcount())}
    
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
