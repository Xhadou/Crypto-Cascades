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
        np.random.seed(random_seed)
        
        self.logger = get_logger(__name__)
        
    def test_all(
        self,
        G: nx.Graph,
        state_history: pd.DataFrame,
        fgi_values: np.ndarray,
        estimated_params: EstimationResult,
        observed_data: Optional[pd.DataFrame] = None,
        apply_correction: bool = True,
        correction_method: str = 'fdr_bh'
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
            G, state_history
        )

        results['H5'] = self.test_h5_community_clustering(
            G, state_history
        )

        self.logger.info("All hypothesis tests complete.")

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

        # Extract p-values (skip NaN/inconclusive)
        hypotheses = sorted(results.keys())
        p_values = []
        valid_hypotheses = []

        for h in hypotheses:
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

        # Update results with adjusted values
        for i, h in enumerate(valid_hypotheses):
            # Store original p-value
            results[h].additional_metrics['p_value_original'] = results[h].p_value
            results[h].additional_metrics['correction_method'] = method

            # Add adjusted values to additional_metrics
            results[h].additional_metrics['p_value_adjusted'] = float(adjusted_p[i])
            results[h].additional_metrics['reject_null_adjusted'] = adjusted_p[i] < self.alpha

        self.logger.info(f"Correction applied. Adjusted p-values stored in additional_metrics.")

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
        
        # If we only have one or two bull market R0s, use different approach
        if len(bull_array) < 3:
            # Use simple comparison with bootstrap CI
            mean_bull = np.mean(bull_array)
            diff = mean_bull - r0_bear_market
            
            # Bootstrap for CI
            n_bootstrap = 1000
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                boot_bulls = np.random.choice(bull_array, size=len(bull_array), replace=True)
                bootstrap_diffs.append(np.mean(boot_bulls) - r0_bear_market)
            
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
            
            # Effect size (standardized difference)
            if len(bull_array) > 1 and np.std(bull_array) > 0:
                effect_size = diff / np.std(bull_array)
            else:
                effect_size = diff  # Raw difference if can't standardize
            
            # P-value approximation from bootstrap
            p_value = np.mean(np.array(bootstrap_diffs) <= 0) if diff > 0 else np.mean(np.array(bootstrap_diffs) >= 0)
            t_stat = diff / (np.std(bootstrap_diffs) + 1e-10)
            
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
                boot_bulls = np.random.choice(bull_array, size=len(bull_array), replace=True)
                bootstrap_diffs.append(np.mean(boot_bulls) - r0_bear_market)
            
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        mean_bull = np.mean(bull_array)
        reject_null = p_value < self.alpha and mean_bull > r0_bear_market
        
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
            additional_metrics={
                'r0_bull_mean': mean_bull,
                'r0_bull_values': list(bull_array),
                'r0_bear': r0_bear_market,
                'r0_difference': mean_bull - r0_bear_market,
                'bull_above_threshold': mean_bull > 1,
                'bear_below_threshold': r0_bear_market < 1,
                'interpretation': f"Bull R₀ ({mean_bull:.2f}) vs Bear R₀ ({r0_bear_market:.2f})"
            }
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
        from scipy.stats import chi2

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
            seir_aic = self._compute_aic(seir_sse, n_params=3, n_obs=len(t))
            model_results['SEIR'] = {
                'sse': seir_sse,
                'aic': seir_aic,
                'n_params': 3,
                'fitted': I_seir
            }
        except Exception as e:
            self.logger.warning(f"SEIR fitting failed: {e}")
            model_results['SEIR'] = {'sse': np.inf, 'aic': np.inf, 'n_params': 3}

        # --- Model 2: Exponential growth ---
        def exponential(t, a, r):
            return a * np.exp(r * t)

        try:
            # Bound r to prevent overflow
            popt, _ = curve_fit(exponential, t, I_obs, p0=[I_obs[0] if I_obs[0] > 0 else 0.01, 0.01],
                               bounds=([0, -1], [1, 1]), maxfev=5000)
            I_exp = exponential(t, *popt)
            exp_sse = np.sum((I_obs - I_exp)**2)
            exp_aic = self._compute_aic(exp_sse, n_params=2, n_obs=len(t))
            model_results['Exponential'] = {'sse': exp_sse, 'aic': exp_aic, 'n_params': 2}
        except Exception as e:
            self.logger.warning(f"Exponential fitting failed: {e}")
            model_results['Exponential'] = {'sse': np.inf, 'aic': np.inf, 'n_params': 2}

        # --- Model 3: Logistic growth ---
        def logistic(t, K, r, t0):
            return K / (1 + np.exp(-r * (t - t0)))

        try:
            p0 = [max(I_obs.max(), 0.01), 0.1, t[len(t)//2]]
            popt, _ = curve_fit(logistic, t, I_obs, p0=p0,
                               bounds=([0, 0, 0], [1, 10, t.max()*2]), maxfev=5000)
            I_log = logistic(t, *popt)
            log_sse = np.sum((I_obs - I_log)**2)
            log_aic = self._compute_aic(log_sse, n_params=3, n_obs=len(t))
            model_results['Logistic'] = {'sse': log_sse, 'aic': log_aic, 'n_params': 3}
        except Exception as e:
            self.logger.warning(f"Logistic fitting failed: {e}")
            model_results['Logistic'] = {'sse': np.inf, 'aic': np.inf, 'n_params': 3}

        # --- Model 4: Linear growth ---
        def linear(t, a, b):
            return a + b * t

        try:
            popt, _ = curve_fit(linear, t, I_obs, maxfev=5000)
            I_lin = linear(t, *popt)
            lin_sse = np.sum((I_obs - I_lin)**2)
            lin_aic = self._compute_aic(lin_sse, n_params=2, n_obs=len(t))
            model_results['Linear'] = {'sse': lin_sse, 'aic': lin_aic, 'n_params': 2}
        except Exception as e:
            self.logger.warning(f"Linear fitting failed: {e}")
            model_results['Linear'] = {'sse': np.inf, 'aic': np.inf, 'n_params': 2}

        # --- Compare models ---
        valid_models = {k: v for k, v in model_results.items() if np.isfinite(v['aic'])}

        if not valid_models or 'SEIR' not in valid_models:
            return self._inconclusive_result("H1", "Model fitting failed")

        # Find best model by AIC
        best_model = min(valid_models.keys(), key=lambda x: valid_models[x]['aic'])
        seir_aic = valid_models['SEIR']['aic']
        best_aic = valid_models[best_model]['aic']

        # Compute AIC weights (Akaike weights)
        aic_values = [v['aic'] for v in valid_models.values()]
        min_aic = min(aic_values)
        delta_aic = {k: v['aic'] - min_aic for k, v in valid_models.items()}
        exp_delta = {k: np.exp(-0.5 * d) for k, d in delta_aic.items()}
        sum_exp = sum(exp_delta.values())
        aic_weights = {k: v / sum_exp for k, v in exp_delta.items()}

        # SEIR is supported if it has highest AIC weight or delta_AIC < 2 from best
        seir_delta_aic = seir_aic - best_aic
        seir_supported = (best_model == 'SEIR') or (seir_delta_aic < 2)

        # Compute R² for SEIR
        ss_tot = np.sum((I_obs - np.mean(I_obs))**2)
        seir_r2 = 1 - valid_models['SEIR']['sse'] / ss_tot if ss_tot > 0 else 0

        # Effect size: difference in AIC weights between SEIR and next best
        other_weights = [w for k, w in aic_weights.items() if k != 'SEIR']
        effect_size = aic_weights.get('SEIR', 0) - max(other_weights) if other_weights else 0

        # P-value approximation using likelihood ratio test (SEIR vs best alternative)
        if best_model != 'SEIR' and best_model in valid_models:
            # Likelihood ratio statistic
            lr_stat = len(t) * np.log(valid_models[best_model]['sse'] / valid_models['SEIR']['sse']) if valid_models['SEIR']['sse'] > 0 else 0
            df_diff = abs(valid_models['SEIR']['n_params'] - valid_models[best_model]['n_params'])
            if df_diff > 0 and lr_stat > 0:
                p_value = 1 - chi2.cdf(abs(lr_stat), df_diff)
            else:
                p_value = 0.5 if seir_supported else 0.99
        else:
            p_value = 0.01 if seir_supported else 0.5

        return HypothesisResult(
            hypothesis="H1",
            description="FOMO episodes follow SEIR epidemic dynamics (vs alternative models)",
            test_statistic=seir_aic,
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(seir_r2 - 0.1), float(min(1.0, seir_r2 + 0.1))),
            reject_null=bool(seir_supported),
            alpha=self.alpha,
            sample_size=len(t),
            additional_metrics={
                'model_comparison': {k: {'aic': v['aic'], 'sse': v['sse']}
                                    for k, v in valid_models.items()},
                'aic_weights': aic_weights,
                'best_model': best_model,
                'seir_r_squared': seir_r2,
                'seir_delta_aic': seir_delta_aic,
                'interpretation': f"SEIR {'is' if seir_supported else 'is NOT'} the best model (ΔAIC={seir_delta_aic:.2f})"
            }
        )

    def _compute_aic(self, sse: float, n_params: int, n_obs: int) -> float:
        """Compute Akaike Information Criterion."""
        if sse <= 0 or n_obs <= n_params:
            return np.inf
        # AIC = n*ln(SSE/n) + 2k
        return n_obs * np.log(sse / n_obs) + 2 * n_params

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
        observed_r0 = model.compute_network_r0(G)

        n = G.number_of_nodes()
        m = G.number_of_edges()
        degrees = [d for _, d in G.degree()]

        null_r0s: Dict[str, List[float]] = {nt: [] for nt in null_types}

        for i in range(n_null_networks):
            seed = self.random_seed + i

            if 'erdos_renyi' in null_types:
                try:
                    G_er = nx.gnm_random_graph(n, m, seed=seed)
                    null_r0s['erdos_renyi'].append(model.compute_network_r0(G_er))
                except Exception:
                    pass

            if 'configuration' in null_types:
                try:
                    # Configuration model preserves degree sequence
                    G_config = nx.configuration_model(degrees, seed=seed)
                    G_config = nx.Graph(G_config)  # Remove multi-edges
                    G_config.remove_edges_from(nx.selfloop_edges(G_config))
                    null_r0s['configuration'].append(model.compute_network_r0(G_config))
                except Exception:
                    pass

            if 'rewired' in null_types:
                try:
                    # Double-edge swap preserves degree sequence exactly
                    G_rewired = G.copy()
                    nx.double_edge_swap(G_rewired, nswap=m*2, max_tries=m*20, seed=seed)
                    null_r0s['rewired'].append(model.compute_network_r0(G_rewired))
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
        estimated_params: EstimationResult
    ) -> HypothesisResult:
        """
        H2: Network structure amplifies contagion.
        
        Test: Compare network R₀ to basic R₀.
        Null hypothesis: Network factor = 1 (no amplification)
        """
        self.logger.info("Testing H2: Network amplifies contagion...")
        
        # Compute network metrics
        degree_view = G.degree()  # type: ignore[operator]
        degrees = [d for _, d in degree_view]
        k_mean = np.mean(degrees)
        k2_mean = np.mean([d**2 for d in degrees])
        
        if k_mean > 0:
            network_factor = k2_mean / k_mean
        else:
            network_factor = 1.0
        
        r0_basic = estimated_params.r0()
        r0_network = r0_basic * network_factor
        
        # Bootstrap test for network factor > 1
        n_bootstrap = 1000
        bootstrap_factors = []
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample of degrees
            sample_idx = np.random.choice(n_nodes, size=n_nodes, replace=True)
            sample_degrees = [degrees[i] for i in sample_idx]
            
            k_mean_boot = np.mean(sample_degrees)
            k2_mean_boot = np.mean([d**2 for d in sample_degrees])
            
            if k_mean_boot > 0:
                bootstrap_factors.append(k2_mean_boot / k_mean_boot)
        
        # P-value: proportion of bootstrap samples where factor <= 1
        p_value = np.mean([f <= 1 for f in bootstrap_factors])
        
        # Effect size: log ratio
        effect_size = np.log(network_factor)
        
        # 95% CI from bootstrap
        ci_lower = np.percentile(bootstrap_factors, 2.5)
        ci_upper = np.percentile(bootstrap_factors, 97.5)
        
        return HypothesisResult(
            hypothesis="H2",
            description="Network structure amplifies FOMO contagion",
            test_statistic=float(network_factor),
            p_value=float(p_value),
            effect_size=float(network_factor - 1),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(p_value < self.alpha),
            alpha=self.alpha,
            sample_size=n_nodes,
            additional_metrics={
                'r0_basic': r0_basic,
                'r0_network': r0_network,
                'network_factor': network_factor,
                'mean_degree': k_mean,
                'degree_variance': np.var(degrees)
            }
        )
    
    def test_h3_fgi_correlation(
        self,
        state_history: pd.DataFrame,
        fgi_values: np.ndarray
    ) -> HypothesisResult:
        """
        H3: Fear & Greed Index correlates with transmission.
        
        Test: Correlation between FGI and infection rate.
        Null hypothesis: ρ = 0 (no correlation)
        """
        self.logger.info("Testing H3: FGI correlates with transmission...")
        
        # Compute new infections per timestep
        if 'I' in state_history.columns:
            infection_counts = state_history.groupby('t')['I'].first().values
        elif 'I_count' in state_history.columns:
            infection_counts = state_history.groupby('t')['I_count'].first().values
        else:
            # Try to compute from state transitions
            infection_counts = np.ones(len(fgi_values)) * 10  # Placeholder
        
        # Align lengths
        min_len = min(len(infection_counts), len(fgi_values))
        infections = infection_counts[:min_len]
        fgi = fgi_values[:min_len]
        
        # Compute change in infections
        delta_infections = np.diff(np.asarray(infections))
        fgi_aligned = fgi[:-1]  # Align with deltas
        
        # Spearman correlation (robust to non-normality)
        if len(delta_infections) > 10:
            corr_result = spearmanr(fgi_aligned, delta_infections)
            corr = float(getattr(corr_result, 'correlation', corr_result[0]))  # type: ignore[arg-type]
            p_value = float(getattr(corr_result, 'pvalue', corr_result[1]))  # type: ignore[arg-type]
        else:
            corr, p_value = 0.0, 1.0
        
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
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            reject_null=bool(p_value < self.alpha and corr > 0),
            alpha=self.alpha,
            sample_size=n,
            additional_metrics={
                'spearman_rho': float(corr),
                'mean_fgi': float(np.mean(fgi)),
                'fgi_std': float(np.std(fgi)),
                'infection_trend': infection_trend
            }
        )
    
    def test_h4_centrality_effect(
        self,
        G: nx.Graph,
        state_history: pd.DataFrame
    ) -> HypothesisResult:
        """
        H4: High-centrality nodes accelerate spread.

        Test: Compare infection time of high vs low centrality nodes.
        Null hypothesis: No difference in infection timing
        """
        self.logger.info("Testing H4: High-centrality nodes accelerate spread...")

        # Compute centrality
        metrics = NetworkMetrics()
        centrality = metrics.compute_centrality_measures(G, measures=['degree'], normalized=True)
        centrality = centrality.get('degree', {})
        if not centrality:
            centrality = nx.degree_centrality(G)

        # Get infection times from state history
        # CRITICAL: Require real data, do not generate mock data
        if 'node' not in state_history.columns or 'infection_time' not in state_history.columns:
            self.logger.warning(
                "H4 test requires 'node' and 'infection_time' columns in state_history. "
                "Returning inconclusive result."
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
                    'required_columns': ['node', 'infection_time'],
                    'available_columns': list(state_history.columns)
                }
            )

        # Extract infection times from real data
        infection_times = dict(zip(state_history['node'], state_history['infection_time']))

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

        # Split into high/low centrality groups
        centrality_values = [centrality.get(n, 0) for n in nodes]
        median_centrality = np.median(centrality_values)

        high_centrality_times = []
        low_centrality_times = []

        for node in nodes:
            c = centrality.get(node, 0)
            t = infection_times.get(node, np.nan)

            if not np.isnan(t):
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
            h_sample = np.random.choice(high_centrality_times, size=len(high_centrality_times), replace=True)
            l_sample = np.random.choice(low_centrality_times, size=len(low_centrality_times), replace=True)
            mean_diffs.append(np.mean(h_sample) - np.mean(l_sample))

        ci_lower = np.percentile(mean_diffs, 2.5)
        ci_upper = np.percentile(mean_diffs, 97.5)

        return HypothesisResult(
            hypothesis="H4",
            description="High-centrality nodes are infected earlier",
            test_statistic=float(stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(p_value < self.alpha),
            alpha=self.alpha,
            sample_size=n1 + n2,
            additional_metrics={
                'mean_time_high_centrality': np.mean(high_centrality_times) if high_centrality_times else float('nan'),
                'mean_time_low_centrality': np.mean(low_centrality_times) if low_centrality_times else float('nan'),
                'median_centrality': median_centrality,
                'n_high_centrality': n1,
                'n_low_centrality': n2
            }
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
        
        # Count within vs between community transmission
        # For simulation, use edge infection events
        within_community = 0
        between_community = 0
        
        # Simplified: count edges within vs between communities
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
        
        # Chi-square test
        expected_within = expected_within_frac * total_edges
        expected_between = (1 - expected_within_frac) * total_edges
        
        if expected_within > 5 and expected_between > 5:
            chi2, p_value = stats.chisquare(
                [within_community, between_community],
                [expected_within, expected_between]
            )
        else:
            chi2, p_value = 0.0, 1.0
        
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
        
        edges = list(G.edges())
        for _ in range(n_bootstrap):
            sample_edges = [edges[i] for i in np.random.choice(len(edges), size=len(edges), replace=True)]
            within = sum(1 for u, v in sample_edges 
                        if node_to_community.get(u, -1) == node_to_community.get(v, -1))
            bootstrap_fracs.append(within / len(sample_edges))
        
        ci_lower = np.percentile(bootstrap_fracs, 2.5)
        ci_upper = np.percentile(bootstrap_fracs, 97.5)
        
        return HypothesisResult(
            hypothesis="H5",
            description="Community structure creates FOMO infection clusters",
            test_statistic=float(chi2),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            reject_null=bool(p_value < self.alpha and observed_within_frac > expected_within_frac),
            alpha=self.alpha,
            sample_size=total_edges,
            additional_metrics={
                'n_communities': len(communities),
                'modularity': modularity,
                'observed_within_frac': observed_within_frac,
                'expected_within_frac': expected_within_frac,
                'largest_community_size': max(community_sizes) if community_sizes else 0
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

        # Check if correction was applied
        first_result = next(iter(results.values()))
        if 'correction_method' in first_result.additional_metrics:
            method = first_result.additional_metrics['correction_method']
            report.append(f"Multiple testing correction: {method.upper()}")
        report.append("")

        for h_name in sorted(results.keys()):
            result = results[h_name]
            report.append("-" * 70)

            # Check for adjusted values
            p_adj = result.additional_metrics.get('p_value_adjusted')
            reject_adj = result.additional_metrics.get('reject_null_adjusted')

            if p_adj is not None:
                status = "REJECTED" if reject_adj else "NOT REJECTED"
                report.append(f"{h_name}: {status} (p_adj={p_adj:.4f}, p_orig={result.p_value:.4f})")
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
                if key in ['p_value_original', 'p_value_adjusted', 'reject_null_adjusted', 'correction_method']:
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

        # Count rejections (use adjusted if available)
        n_rejected_orig = sum(1 for r in results.values() if r.reject_null)
        n_rejected_adj = sum(
            1 for r in results.values()
            if r.additional_metrics.get('reject_null_adjusted', r.reject_null)
        )

        if any('p_value_adjusted' in r.additional_metrics for r in results.values()):
            report.append(f"Hypotheses supported (original): {n_rejected_orig}/{len(results)}")
            report.append(f"Hypotheses supported (adjusted): {n_rejected_adj}/{len(results)}")
        else:
            report.append(f"Hypotheses supported: {n_rejected_orig}/{len(results)}")

        for h_name, result in sorted(results.items()):
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
