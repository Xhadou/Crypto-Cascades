## Context

You are improving an existing research project called "Crypto Cascades" that models FOMO-driven cryptocurrency buying behavior as an epidemic using SEIR dynamics on Bitcoin transaction networks.

**The codebase is already implemented but has critical bugs, missing features, and methodology gaps that must be fixed before the research can be considered valid.**

**IMPORTANT**: Read each issue carefully. Many fixes require changes across multiple files. Test your changes mentally before implementing. Do NOT break existing functionality.

---

## Project Structure (Already Exists)

---

## Fix Implementation Order

Execute these fixes IN ORDER. Each section builds on previous fixes. After each major section, verify the code still runs.

---

# SECTION 1: CRITICAL BUG FIXES

These bugs affect the validity of research results and must be fixed first.

---

## Fix 1.1: State Transition Logic - Separate Immunity Waning Window

**File**: `src/state_engine/state_assigner.py`

**Problem**: The `recovery_window` parameter is incorrectly used for both "time to recover from infection" AND "time before immunity wanes." These are epidemiologically distinct.

**Changes Required**:

1. Add new parameter `immunity_waning_days` to `__init__`:

```python
def __init__(
    self,
    susceptible_window_days: int = 7,
    exposure_window_hours: int = 24,
    infected_threshold: float = 0.0,
    recovery_window_days: int = 3,
    immunity_waning_days: int = 30,  # NEW PARAMETER
    min_usd_value: float = 100.0
):
    """
    Initialize state assigner.
  
    Args:
        susceptible_window_days: Days without buying to be susceptible
        exposure_window_hours: Hours after contact to be exposed
        infected_threshold: Minimum net BTC to be infected (positive = buying)
        recovery_window_days: Days of dormancy after infection before recovered
        immunity_waning_days: Days in recovered state before becoming susceptible again
        min_usd_value: Minimum USD transaction value to count
    """
    self.susceptible_window = timedelta(days=susceptible_window_days)
    self.exposure_window = timedelta(hours=exposure_window_hours)
    self.infected_threshold = infected_threshold
    self.recovery_window = timedelta(days=recovery_window_days)
    self.immunity_waning_window = timedelta(days=immunity_waning_days)  # NEW
    self.min_usd_value = min_usd_value
```

2. Fix the `_compute_new_state` method RECOVERED branch:

```python
if prev_state == State.RECOVERED:
    # R -> S: Immunity wanes after immunity_waning_window (not recovery_window!)
    recovery_time = self.recovery_times.get(wallet)
    if recovery_time and (current_time - recovery_time) > self.immunity_waning_window:
        return State.SUSCEPTIBLE
    return State.RECOVERED
```

3. Update `configs/config.yaml` to include the new parameter:

```yaml
state_assignment:
  susceptible:
    no_buy_window_days: 7
  exposed:
    contact_window_hours: 24
  infected:
    net_positive_threshold: 0.0
    min_usd_value: 100
  recovered:
    dormancy_window_days: 3
    immunity_waning_days: 30  # NEW
```

---

## Fix 1.2: Add State Transition Validation

**File**: `src/state_engine/state_assigner.py`

**Problem**: No validation that state transitions follow valid SEIR paths. Invalid transitions (e.g., S→R, E→S) could occur due to bugs and go undetected.

**Add after the State enum definition** (around line 30):

```python
# Valid state transitions in SEIR model
# S can go to S (stay), E (exposure), or I (direct infection after exposure)
# E can go to E (stay) or I (become infected)
# I can go to I (stay) or R (recover)
# R can go to R (stay) or S (immunity wanes)
VALID_TRANSITIONS: Dict[State, Set[State]] = {
    State.SUSCEPTIBLE: {State.SUSCEPTIBLE, State.EXPOSED, State.INFECTED},
    State.EXPOSED: {State.EXPOSED, State.INFECTED},
    State.INFECTED: {State.INFECTED, State.RECOVERED},
    State.RECOVERED: {State.RECOVERED, State.SUSCEPTIBLE},
}


def validate_transition(from_state: State, to_state: State) -> bool:
    """
    Validate that a state transition is epidemiologically valid.
  
    Args:
        from_state: Current state
        to_state: Proposed new state
    
    Returns:
        True if transition is valid, False otherwise
    """
    return to_state in VALID_TRANSITIONS.get(from_state, set())
```

**Modify `_compute_new_state` to validate before returning** (at the end of the method):

```python
def _compute_new_state(
    self,
    wallet: int,
    prev_state: State,
    is_buying: bool,
    G: nx.Graph,
    infected_wallets: Set[int],
    current_time: datetime
) -> State:
    """Compute new state for a wallet based on transition rules."""
  
    # ... existing logic ...
  
    # Determine new_state based on existing logic
    # (keep all the existing if/elif/else blocks)
  
    # ADD THIS AT THE END, before returning:
    # Validate the transition
    if not validate_transition(prev_state, new_state):
        self.logger.warning(
            f"Invalid transition {prev_state.value} -> {new_state.value} "
            f"for wallet {wallet}. Keeping current state."
        )
        return prev_state
  
    return new_state
```

---

## Fix 1.3: Remove Circular Mock Data in H4 Test

**File**: `src/hypothesis/hypothesis_tester.py`

**Problem**: When real infection time data is missing, the H4 test generates fake data BASED ON centrality, which would obviously confirm the hypothesis. This is circular reasoning.

**Replace the mock data section in `test_h4_centrality_effect`** (around lines 280-300):

```python
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
  
    # Continue with the rest of the existing test logic...
    # (keep the median split, Mann-Whitney test, etc.)
```

**Also update `run_state_assignment` in `state_assigner.py`** to track infection times properly:

```python
def run_state_assignment(
    self,
    G: nx.Graph,
    flows: pd.DataFrame,
    initial_infected: Optional[List[int]] = None,
    initial_infected_fraction: float = 0.01
) -> pd.DataFrame:
    """
    Run state assignment over all time periods.
  
    Returns:
        DataFrame with state counts over time AND tracks infection times
        in self.infection_times dictionary
    """
    # ... existing code ...
  
    # IMPORTANT: After the main loop, create a state history DataFrame
    # that includes per-node infection times for H4 test
  
    # At the end of the method, add:
    self._node_infection_times_df = pd.DataFrame([
        {'node': node, 'infection_time': time}
        for node, time in self.infection_times.items()
    ])
  
    return result_df

def get_infection_times_df(self) -> pd.DataFrame:
    """Get DataFrame of node infection times for hypothesis testing."""
    if hasattr(self, '_node_infection_times_df'):
        return self._node_infection_times_df
    return pd.DataFrame(columns=['node', 'infection_time'])
```

---

## Fix 1.4: Improve H1 Test Methodology - Model Comparison

**File**: `src/hypothesis/hypothesis_tester.py`

**Problem**: H1 claims to test "FOMO follows epidemic dynamics" but only compares R² against random permutations. This doesn't actually validate SEIR dynamics - need to compare against alternative models.

**Replace the entire `test_h1_epidemic_dynamics` method**:

```python
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
        t = observed_data['t'].values.astype(float)
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
        popt, _ = curve_fit(exponential, t, I_obs, p0=[I_obs[0], 0.01], 
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
        p0 = [I_obs.max(), 0.1, t[len(t)//2]]
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
        lr_stat = len(t) * np.log(valid_models[best_model]['sse'] / valid_models['SEIR']['sse'])
        df_diff = abs(valid_models['SEIR']['n_params'] - valid_models[best_model]['n_params'])
        if df_diff > 0:
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
```

---

# SECTION 2: MISSING STATISTICAL FEATURES

---

## Fix 2.1: Add Multiple Testing Correction

**File**: `src/hypothesis/hypothesis_tester.py`

**Problem**: Running 6 hypothesis tests inflates Type I error rate. Need Bonferroni or FDR correction.

**Add these methods to the `HypothesisTester` class**:

```python
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
    
        # We can't modify the dataclass directly, so we'll add to additional_metrics
        results[h].additional_metrics['p_value_adjusted'] = float(adjusted_p[i])
        results[h].additional_metrics['reject_null_adjusted'] = adjusted_p[i] < self.alpha
  
    self.logger.info(f"Correction applied. Adjusted p-values stored in additional_metrics.")
  
    return results
```

**Update `test_all` to optionally apply correction**:

```python
def test_all(
    self,
    G: nx.Graph,
    state_history: pd.DataFrame,
    fgi_values: np.ndarray,
    estimated_params: EstimationResult,
    observed_data: Optional[pd.DataFrame] = None,
    apply_correction: bool = True,  # NEW PARAMETER
    correction_method: str = 'fdr_bh'  # NEW PARAMETER
) -> Dict[str, HypothesisResult]:
    """
    Run all hypothesis tests.
  
    Args:
        ...existing args...
        apply_correction: Whether to apply multiple testing correction
        correction_method: Correction method ('bonferroni', 'holm', 'fdr_bh')
    """
    # ... existing test calls ...
  
    # Apply multiple testing correction
    if apply_correction:
        results = self.apply_multiple_testing_correction(results, method=correction_method)
  
    return results
```

**Update `generate_report` to show both original and adjusted p-values**:

```python
def generate_report(self, results: Dict[str, HypothesisResult]) -> str:
    """Generate a formatted report of hypothesis test results."""
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
  
    return "\n".join(report)
```

---

## Fix 2.2: Add Null Model Comparison for Network Effects

**File**: `src/hypothesis/hypothesis_tester.py`

**Add this new method**:

```python
def compare_against_null_networks(
    self,
    G: nx.Graph,
    estimated_params: EstimationResult,
    n_null_networks: int = 100,
    null_types: Optional[List[str]] = None
) -> Dict[str, Dict]:
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
  
    null_r0s = {nt: [] for nt in null_types}
  
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
    results = {
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
```

---

## Fix 2.3: Add Time-Varying R₀ Estimation

**File**: `src/epidemic_model/network_seir.py`

**Add this method to the `NetworkSEIR` class**:

```python
def compute_time_varying_r0(
    self,
    state_df: pd.DataFrame,
    window_size: int = 7,
    method: str = 'wallinga_teunis'
) -> pd.DataFrame:
    """
    Estimate time-varying reproduction number R(t).
  
    Args:
        state_df: DataFrame with SEIR state counts over time
        window_size: Rolling window size for estimation
        method: Estimation method:
            - 'ratio': Simple ratio method (new infections / current infections)
            - 'wallinga_teunis': Wallinga-Teunis method (requires serial interval)
        
    Returns:
        DataFrame with columns [t, R_t, R_t_lower, R_t_upper]
    """
    self.logger.info(f"Computing time-varying R₀ using {method} method...")
  
    results = []
  
    # Extract data
    t = state_df['t'].values if 't' in state_df.columns else np.arange(len(state_df))
  
    if 'I' in state_df.columns:
        I = state_df['I'].values.astype(float)
        S = state_df['S'].values.astype(float)
        E = state_df['E'].values.astype(float)
    elif 'I_frac' in state_df.columns:
        N = 10000  # Assumed population
        I = (state_df['I_frac'].values * N).astype(float)
        S = (state_df['S_frac'].values * N).astype(float)
        E = (state_df['E_frac'].values * N).astype(float)
    else:
        self.logger.error("Cannot find infection data in state_df")
        return pd.DataFrame()
  
    # Compute new infections (incidence)
    new_infections = np.diff(E + I)
    new_infections = np.maximum(new_infections, 0)  # Can't be negative
  
    if method == 'ratio':
        # Simple ratio method
        for i in range(window_size, len(state_df)):
            window_start = i - window_size
        
            # Current infected in window
            I_window = I[window_start:i]
            I_mean = np.mean(I_window)
        
            # New infections in window
            if i < len(new_infections):
                new_inf_window = new_infections[window_start:i]
                new_inf_sum = np.sum(new_inf_window)
            else:
                new_inf_sum = 0
        
            # Susceptible fraction for adjustment
            S_frac = S[i] / (S[i] + E[i] + I[i] + 1e-10)
            N_total = S[i] + E[i] + I[i]
        
            if I_mean > 0 and S_frac > 0:
                # R_t ≈ (new infections per time) / (gamma * I) * (N / S)
                # Simplified: R_t ≈ (new_inf / I) * (1 / S_frac)
                R_t = (new_inf_sum / window_size) / (self.params.gamma * I_mean)
                # Adjust for susceptible depletion
                R_t_adjusted = R_t / S_frac if S_frac > 0.1 else R_t
            else:
                R_t = np.nan
                R_t_adjusted = np.nan
        
            # Bootstrap CI
            if I_mean > 0:
                R_t_samples = []
                for _ in range(100):
                    boot_idx = np.random.choice(window_size, window_size, replace=True)
                    boot_I = I_window[boot_idx].mean()
                    boot_new = new_inf_window[boot_idx].sum() if i < len(new_infections) else 0
                    if boot_I > 0:
                        R_t_samples.append((boot_new / window_size) / (self.params.gamma * boot_I))
            
                if R_t_samples:
                    R_t_lower = np.percentile(R_t_samples, 2.5)
                    R_t_upper = np.percentile(R_t_samples, 97.5)
                else:
                    R_t_lower = R_t_upper = R_t
            else:
                R_t_lower = R_t_upper = np.nan
        
            results.append({
                't': t[i],
                'R_t': R_t,
                'R_t_adjusted': R_t_adjusted,
                'R_t_lower': R_t_lower,
                'R_t_upper': R_t_upper,
                'I': I[i],
                'S_frac': S_frac,
                'new_infections': new_inf_sum / window_size
            })
  
    df = pd.DataFrame(results)
  
    if len(df) > 0:
        self.logger.info(
            f"R(t) range: [{df['R_t'].min():.2f}, {df['R_t'].max():.2f}], "
            f"mean: {df['R_t'].mean():.2f}"
        )
  
    return df
```

---

# SECTION 3: STOCHASTIC SIMULATION IMPROVEMENT

---

## Fix 3.1: Implement Gillespie Algorithm

**File**: `src/epidemic_model/network_seir.py`

**Add this method to the `NetworkSEIR` class** (this is a proper continuous-time stochastic simulation):

```python
def simulate_gillespie(
    self,
    G: nx.Graph,
    initial_infected: List[int],
    t_max: float,
    fgi_values: Optional[np.ndarray] = None,
    record_interval: float = 1.0
) -> pd.DataFrame:
    """
    Run continuous-time stochastic simulation using Gillespie algorithm.
  
    This is more accurate than discrete-time simulation for capturing
    proper epidemic dynamics.
  
    Args:
        G: NetworkX graph
        initial_infected: List of initially infected node IDs
        t_max: Maximum simulation time
        fgi_values: Optional FGI values (indexed by integer time)
        record_interval: Time interval for recording state counts
    
    Returns:
        DataFrame with state counts over time
    """
    self.logger.info(
        f"Running Gillespie simulation (N={G.number_of_nodes():,}, T={t_max})"
    )
  
    # Initialize node states
    node_states = {node: State.SUSCEPTIBLE for node in G.nodes()}
    for node in initial_infected:
        if node in node_states:
            node_states[node] = State.INFECTED
  
    # Create efficient neighbor lookup
    neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
  
    # Track nodes in each state for efficient rate calculation
    S_nodes = set(n for n, s in node_states.items() if s == State.SUSCEPTIBLE)
    E_nodes = set(n for n, s in node_states.items() if s == State.EXPOSED)
    I_nodes = set(n for n, s in node_states.items() if s == State.INFECTED)
    R_nodes = set(n for n, s in node_states.items() if s == State.RECOVERED)
  
    # Results storage
    results = []
    t = 0.0
    last_record_time = 0.0
  
    # Record initial state
    results.append({
        't': 0.0,
        'S': len(S_nodes),
        'E': len(E_nodes),
        'I': len(I_nodes),
        'R': len(R_nodes)
    })
  
    iteration = 0
    max_iterations = int(t_max * G.number_of_nodes() * 10)  # Safety limit
  
    while t < t_max and iteration < max_iterations:
        iteration += 1
    
        # Get effective beta based on current FGI
        if fgi_values is not None and int(t) < len(fgi_values):
            beta_eff = self.params.effective_beta(fgi_values[int(t)])
        else:
            beta_eff = self.params.beta
    
        # Calculate total event rates
        # S -> E: For each S node with infected neighbor
        exposure_rate = 0.0
        exposable_nodes = []
        for s_node in S_nodes:
            infected_neighbors = sum(1 for n in neighbors[s_node] if n in I_nodes)
            if infected_neighbors > 0:
                node_rate = beta_eff * infected_neighbors
                exposure_rate += node_rate
                exposable_nodes.append((s_node, node_rate))
    
        # E -> I: All exposed nodes
        infection_rate = self.params.sigma * len(E_nodes)
    
        # I -> R: All infected nodes
        recovery_rate = self.params.gamma * len(I_nodes)
    
        # R -> S: All recovered nodes (immunity waning)
        waning_rate = self.params.omega * len(R_nodes)
    
        total_rate = exposure_rate + infection_rate + recovery_rate + waning_rate
    
        if total_rate == 0:
            # No more events possible
            break
    
        # Time to next event (exponential distribution)
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
    
        if t > t_max:
            break
    
        # Choose which event occurs
        rand = np.random.random() * total_rate
    
        if rand < exposure_rate:
            # Exposure event: choose which S node
            cumsum = 0
            for node, rate in exposable_nodes:
                cumsum += rate
                if cumsum >= rand:
                    # S -> E
                    S_nodes.remove(node)
                    E_nodes.add(node)
                    node_states[node] = State.EXPOSED
                    break
                
        elif rand < exposure_rate + infection_rate:
            # Infection event: random E node becomes I
            if E_nodes:
                node = np.random.choice(list(E_nodes))
                E_nodes.remove(node)
                I_nodes.add(node)
                node_states[node] = State.INFECTED
            
        elif rand < exposure_rate + infection_rate + recovery_rate:
            # Recovery event: random I node becomes R
            if I_nodes:
                node = np.random.choice(list(I_nodes))
                I_nodes.remove(node)
                R_nodes.add(node)
                node_states[node] = State.RECOVERED
            
        else:
            # Immunity waning: random R node becomes S
            if R_nodes:
                node = np.random.choice(list(R_nodes))
                R_nodes.remove(node)
                S_nodes.add(node)
                node_states[node] = State.SUSCEPTIBLE
    
        # Record at intervals
        if t - last_record_time >= record_interval:
            results.append({
                't': t,
                'S': len(S_nodes),
                'E': len(E_nodes),
                'I': len(I_nodes),
                'R': len(R_nodes)
            })
            last_record_time = t
  
    # Final state
    if len(results) == 0 or results[-1]['t'] < t:
        results.append({
            't': min(t, t_max),
            'S': len(S_nodes),
            'E': len(E_nodes),
            'I': len(I_nodes),
            'R': len(R_nodes)
        })
  
    df = pd.DataFrame(results)
  
    # Add fractions
    N = G.number_of_nodes()
    for col in ['S', 'E', 'I', 'R']:
        df[f'{col}_frac'] = df[col] / N
  
    self.logger.info(
        f"Gillespie simulation complete: {iteration} events, "
        f"final t={df['t'].iloc[-1]:.1f}"
    )
  
    return df
```

---

# SECTION 4: DATA VALIDATION & ERROR HANDLING

---

## Fix 4.1: Add Comprehensive Data Validation

**File**: `src/preprocessing/orbitaal_parser.py`

**Add this method to the `OrbitaalParser` class**:

```python
def validate_transactions(
    self,
    df: pd.DataFrame,
    strict: bool = False
) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate transaction data integrity.
  
    Args:
        df: Transaction DataFrame to validate
        strict: If True, return False for any issues. If False, attempt to fix.
    
    Returns:
        Tuple of (is_valid, list_of_issues, cleaned_dataframe)
    """
    issues = []
    df_clean = df.copy()
  
    # 1. Check required columns
    required_cols = {'source_id', 'target_id'}
    optional_cols = {'btc_value', 'usd_value', 'timestamp', 'datetime'}
  
    missing_required = required_cols - set(df.columns)
    if missing_required:
        issues.append(f"CRITICAL: Missing required columns: {missing_required}")
        return False, issues, df_clean
  
    present_optional = optional_cols & set(df.columns)
    if not present_optional:
        issues.append("WARNING: No value or timestamp columns found")
  
    # 2. Check for null values in required columns
    for col in required_cols:
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            issues.append(f"Found {null_count} null values in {col}")
            if not strict:
                df_clean = df_clean.dropna(subset=[col])
                issues.append(f"  -> Removed {null_count} rows with null {col}")
  
    # 3. Check for self-loops
    self_loops = df_clean['source_id'] == df_clean['target_id']
    self_loop_count = self_loops.sum()
    if self_loop_count > 0:
        issues.append(f"Found {self_loop_count} self-loop transactions")
        if not strict:
            df_clean = df_clean[~self_loops]
            issues.append(f"  -> Removed {self_loop_count} self-loops")
  
    # 4. Check for negative values
    if 'btc_value' in df_clean.columns:
        neg_btc = (df_clean['btc_value'] < 0).sum()
        if neg_btc > 0:
            issues.append(f"Found {neg_btc} negative BTC values")
            if not strict:
                df_clean = df_clean[df_clean['btc_value'] >= 0]
  
    if 'usd_value' in df_clean.columns:
        neg_usd = (df_clean['usd_value'] < 0).sum()
        if neg_usd > 0:
            issues.append(f"Found {neg_usd} negative USD values")
            if not strict:
                df_clean = df_clean[df_clean['usd_value'] >= 0]
  
    # 5. Check for extreme outliers
    if 'btc_value' in df_clean.columns:
        btc_max = df_clean['btc_value'].max()
        if btc_max > 1_000_000:  # More than 1M BTC is suspicious
            issues.append(f"WARNING: Extremely large BTC value: {btc_max}")
  
    # 6. Check timestamp validity
    if 'timestamp' in df_clean.columns:
        # Bitcoin launched Jan 3, 2009
        min_valid_ts = 1230940800  # 2009-01-03
        max_valid_ts = 1640000000  # ~2021-12
    
        invalid_ts = (
            (df_clean['timestamp'] < min_valid_ts) | 
            (df_clean['timestamp'] > max_valid_ts)
        ).sum()
    
        if invalid_ts > 0:
            issues.append(f"Found {invalid_ts} transactions with invalid timestamps")
  
    # 7. Check for duplicates
    dupes = df_clean.duplicated().sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate rows")
        if not strict:
            df_clean = df_clean.drop_duplicates()
            issues.append(f"  -> Removed {dupes} duplicates")
  
    # 8. Summary statistics
    n_removed = len(df) - len(df_clean)
    if n_removed > 0:
        pct_removed = 100 * n_removed / len(df)
        issues.append(f"SUMMARY: Removed {n_removed} rows ({pct_removed:.2f}%)")
  
    is_valid = len(issues) == 0 or (not strict and len(df_clean) > 0)
  
    # Log issues
    for issue in issues:
        if issue.startswith("CRITICAL"):
            self.logger.error(issue)
        elif issue.startswith("WARNING"):
            self.logger.warning(issue)
        else:
            self.logger.info(issue)
  
    return is_valid, issues, df_clean
```

**Update `load_snapshot` and `load_stream` to use validation**:

```python
def load_snapshot(
    self,
    filepath: Union[str, Path],
    min_usd_value: float = 0.0,
    min_btc_value: float = 0.0,
    validate: bool = True  # NEW PARAMETER
) -> pd.DataFrame:
    """Load and preprocess a snapshot file."""
    # ... existing loading code ...
  
    if validate:
        is_valid, issues, df = self.validate_transactions(df, strict=False)
        if not is_valid:
            self.logger.error(f"Validation failed for {filepath}")
            return pd.DataFrame()
  
    # ... rest of existing code ...
```

---

## Fix 4.2: Improve Error Handling Pattern

**File**: Create new file `src/utils/exceptions.py`

```python
"""
Custom Exceptions for Crypto Cascades

Provides specific exception types for better error handling and debugging.
"""


class CryptoCascadesError(Exception):
    """Base exception for all Crypto Cascades errors."""
    pass


class DataLoadError(CryptoCascadesError):
    """Error loading or parsing data files."""
    def __init__(self, filepath: str, reason: str):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to load {filepath}: {reason}")


class DataValidationError(CryptoCascadesError):
    """Error validating data integrity."""
    def __init__(self, issues: list):
        self.issues = issues
        super().__init__(f"Data validation failed: {', '.join(issues[:3])}")


class InsufficientDataError(CryptoCascadesError):
    """Not enough data for analysis."""
    def __init__(self, required: int, available: int, data_type: str = "records"):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data: need {required} {data_type}, have {available}"
        )


class ModelFittingError(CryptoCascadesError):
    """Error fitting model parameters."""
    def __init__(self, model: str, reason: str):
        self.model = model
        self.reason = reason
        super().__init__(f"Failed to fit {model}: {reason}")


class HypothesisTestError(CryptoCascadesError):
    """Error running hypothesis test."""
    def __init__(self, hypothesis: str, reason: str):
        self.hypothesis = hypothesis
        self.reason = reason
        super().__init__(f"Hypothesis {hypothesis} test failed: {reason}")


class ConfigurationError(CryptoCascadesError):
    """Error in configuration."""
    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(f"Configuration error for '{key}': {reason}")
```

**Update `src/utils/__init__.py`**:

```python
"""Utility modules for logging, configuration, and caching."""

from src.utils.exceptions import (
    CryptoCascadesError,
    DataLoadError,
    DataValidationError,
    InsufficientDataError,
    ModelFittingError,
    HypothesisTestError,
    ConfigurationError,
)

__all__ = [
    'CryptoCascadesError',
    'DataLoadError',
    'DataValidationError',
    'InsufficientDataError',
    'ModelFittingError',
    'HypothesisTestError',
    'ConfigurationError',
]
```

---

## Fix 4.3: Centralize Magic Numbers

**File**: `configs/config.yaml`

**Add new section for thresholds**:

```yaml
# Computational thresholds
thresholds:
  # Graph size thresholds
  large_graph_nodes: 10000          # Skip expensive metrics above this
  very_large_graph_nodes: 50000     # Use heavy sampling above this
  max_nodes_for_centrality: 5000    # Max nodes for closeness centrality
  
  # Sampling parameters
  betweenness_sample_size: 500      # k parameter for betweenness approximation
  clustering_sample_size: 10000     # Nodes to sample for clustering
  
  # Statistical thresholds
  min_degrees_for_powerlaw: 50      # Minimum non-zero degrees for power-law fit
  min_nodes_for_hypothesis: 20      # Minimum nodes for hypothesis tests
  min_time_points: 10               # Minimum time points for model fitting
  
  # Monte Carlo parameters
  default_bootstrap_samples: 100
  default_null_networks: 100
  max_gillespie_iterations_factor: 10  # max_iter = t_max * N * this_factor
```

**Create new file `src/utils/constants.py`**:

```python
"""
Constants and Thresholds

Centralized location for all magic numbers and thresholds.
These can be overridden by config.yaml values.
"""

from src.utils.config_manager import get_config


def get_threshold(key: str, default: float) -> float:
    """Get a threshold value from config or use default."""
    config = get_config()
    return config.get(f'thresholds.{key}', default)


# Graph size thresholds
LARGE_GRAPH_NODES = lambda: get_threshold('large_graph_nodes', 10000)
VERY_LARGE_GRAPH_NODES = lambda: get_threshold('very_large_graph_nodes', 50000)
MAX_NODES_FOR_CENTRALITY = lambda: get_threshold('max_nodes_for_centrality', 5000)

# Sampling parameters
BETWEENNESS_SAMPLE_SIZE = lambda: int(get_threshold('betweenness_sample_size', 500))
CLUSTERING_SAMPLE_SIZE = lambda: int(get_threshold('clustering_sample_size', 10000))

# Statistical thresholds
MIN_DEGREES_FOR_POWERLAW = lambda: int(get_threshold('min_degrees_for_powerlaw', 50))
MIN_NODES_FOR_HYPOTHESIS = lambda: int(get_threshold('min_nodes_for_hypothesis', 20))
MIN_TIME_POINTS = lambda: int(get_threshold('min_time_points', 10))

# Monte Carlo
DEFAULT_BOOTSTRAP_SAMPLES = lambda: int(get_threshold('default_bootstrap_samples', 100))
DEFAULT_NULL_NETWORKS = lambda: int(get_threshold('default_null_networks', 100))
```

**Update `src/network_analysis/metrics.py` to use constants** (example):

```python
from src.utils.constants import (
    LARGE_GRAPH_NODES, 
    MAX_NODES_FOR_CENTRALITY,
    BETWEENNESS_SAMPLE_SIZE,
    MIN_DEGREES_FOR_POWERLAW
)

# In compute_centrality_measures:
if n_nodes > LARGE_GRAPH_NODES() and sample_size is None:
    self.logger.warning(
        f"Betweenness centrality on {n_nodes:,} nodes is expensive. "
        f"Using k={BETWEENNESS_SAMPLE_SIZE()} sample approximation."
    )
    sample_size = min(BETWEENNESS_SAMPLE_SIZE(), n_nodes)

# In fit_power_law:
if len(degrees) < MIN_DEGREES_FOR_POWERLAW():
    self.logger.warning("Not enough non-zero degrees for power-law fit")
    return {'error': 'insufficient data'}
```

---

# SECTION 5: ADDITIONAL TESTS

---

## Fix 5.1: Add Missing Test Coverage

**File**: `tests/test_state_assigner.py` (NEW FILE)

```python
"""
Unit Tests for State Assignment Engine

Tests SEIR state assignment logic including:
- State transition validation
- Immunity waning separate from recovery
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

from src.state_engine.state_assigner import (
    StateAssigner, State, validate_transition, VALID_TRANSITIONS
)


class TestValidTransitions:
    """Tests for state transition validation."""
  
    def test_valid_s_to_e(self):
        assert validate_transition(State.SUSCEPTIBLE, State.EXPOSED) == True
  
    def test_valid_s_to_i(self):
        assert validate_transition(State.SUSCEPTIBLE, State.INFECTED) == True
  
    def test_valid_e_to_i(self):
        assert validate_transition(State.EXPOSED, State.INFECTED) == True
  
    def test_valid_i_to_r(self):
        assert validate_transition(State.INFECTED, State.RECOVERED) == True
  
    def test_valid_r_to_s(self):
        assert validate_transition(State.RECOVERED, State.SUSCEPTIBLE) == True
  
    def test_invalid_s_to_r(self):
        """S cannot directly become R without going through E/I."""
        assert validate_transition(State.SUSCEPTIBLE, State.RECOVERED) == False
  
    def test_invalid_e_to_s(self):
        """E cannot go back to S."""
        assert validate_transition(State.EXPOSED, State.SUSCEPTIBLE) == False
  
    def test_invalid_e_to_r(self):
        """E cannot skip I and go directly to R."""
        assert validate_transition(State.EXPOSED, State.RECOVERED) == False
  
    def test_invalid_i_to_s(self):
        """I cannot go directly back to S."""
        assert validate_transition(State.INFECTED, State.SUSCEPTIBLE) == False
  
    def test_invalid_i_to_e(self):
        """I cannot go back to E."""
        assert validate_transition(State.INFECTED, State.EXPOSED) == False
  
    def test_invalid_r_to_e(self):
        """R cannot go to E."""
        assert validate_transition(State.RECOVERED, State.EXPOSED) == False
  
    def test_invalid_r_to_i(self):
        """R cannot go directly to I."""
        assert validate_transition(State.RECOVERED, State.INFECTED) == False


class TestStateAssignerParameters:
    """Test StateAssigner initialization and parameters."""
  
    def test_default_parameters(self):
        assigner = StateAssigner()
        assert assigner.susceptible_window == timedelta(days=7)
        assert assigner.recovery_window == timedelta(days=3)
        assert assigner.immunity_waning_window == timedelta(days=30)
  
    def test_custom_parameters(self):
        assigner = StateAssigner(
            susceptible_window_days=14,
            recovery_window_days=5,
            immunity_waning_days=60
        )
        assert assigner.susceptible_window == timedelta(days=14)
        assert assigner.recovery_window == timedelta(days=5)
        assert assigner.immunity_waning_window == timedelta(days=60)
  
    def test_immunity_waning_separate_from_recovery(self):
        """Verify immunity waning and recovery are tracked separately."""
        assigner = StateAssigner(
            recovery_window_days=3,
            immunity_waning_days=30
        )
        # These should be different
        assert assigner.recovery_window != assigner.immunity_waning_window


class TestStateTransitions:
    """Test actual state transition logic."""
  
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        return G
  
    @pytest.fixture
    def assigner(self):
        return StateAssigner(
            susceptible_window_days=7,
            recovery_window_days=3,
            immunity_waning_days=30,
            min_usd_value=0
        )
  
    def test_susceptible_stays_susceptible_without_infected_neighbor(self, assigner, simple_graph):
        """S node with no infected neighbors stays S."""
        infected_wallets = set()  # No one infected
    
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.SUSCEPTIBLE,
            is_buying=False,
            G=simple_graph,
            infected_wallets=infected_wallets,
            current_time=datetime.now()
        )
    
        assert new_state == State.SUSCEPTIBLE
  
    def test_susceptible_becomes_exposed_with_infected_neighbor(self, assigner, simple_graph):
        """S node with infected neighbor becomes E (if not buying)."""
        infected_wallets = {1}  # Node 1 is infected, neighbor of node 0
    
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.SUSCEPTIBLE,
            is_buying=False,
            G=simple_graph,
            infected_wallets=infected_wallets,
            current_time=datetime.now()
        )
    
        assert new_state == State.EXPOSED
  
    def test_susceptible_becomes_infected_if_buying_with_infected_neighbor(self, assigner, simple_graph):
        """S node with infected neighbor and buying becomes I directly."""
        infected_wallets = {1}
    
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.SUSCEPTIBLE,
            is_buying=True,
            G=simple_graph,
            infected_wallets=infected_wallets,
            current_time=datetime.now()
        )
    
        assert new_state == State.INFECTED


class TestImmunityWaning:
    """Test immunity waning logic."""
  
    def test_recovered_stays_recovered_before_waning(self):
        """R stays R before immunity_waning_window passes."""
        assigner = StateAssigner(immunity_waning_days=30)
    
        G = nx.Graph()
        G.add_node(0)
    
        # Set recovery time to 10 days ago
        recovery_time = datetime.now() - timedelta(days=10)
        assigner.recovery_times[0] = recovery_time
    
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.RECOVERED,
            is_buying=False,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )
    
        assert new_state == State.RECOVERED
  
    def test_recovered_becomes_susceptible_after_waning(self):
        """R becomes S after immunity_waning_window passes."""
        assigner = StateAssigner(immunity_waning_days=30)
    
        G = nx.Graph()
        G.add_node(0)
    
        # Set recovery time to 35 days ago
        recovery_time = datetime.now() - timedelta(days=35)
        assigner.recovery_times[0] = recovery_time
    
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.RECOVERED,
            is_buying=False,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )
    
        assert new_state == State.SUSCEPTIBLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

# SECTION 6: FINAL VALIDATION

After implementing all fixes, run these validation checks:

```bash
# 1. Run all tests
pytest tests/ -v --tb=short

# 2. Check for import errors
python -c "from src.main import CryptoCascadesPipeline; print('Imports OK')"

# 3. Run quick validation on sample data
python -m src.main --phase preprocess --verbose --dry-run

# 4. Type check (if mypy available)
mypy src/ --ignore-missing-imports

# 5. Verify new features
python -c "
from src.hypothesis.hypothesis_tester import HypothesisTester
t = HypothesisTester()
print('Multiple testing correction:', hasattr(t, 'apply_multiple_testing_correction'))
print('Null model comparison:', hasattr(t, 'compare_against_null_networks'))

from src.epidemic_model.network_seir import NetworkSEIR
m = NetworkSEIR()
print('Gillespie simulation:', hasattr(m, 'simulate_gillespie'))
print('Time-varying R0:', hasattr(m, 'compute_time_varying_r0'))

from src.state_engine.state_assigner import StateAssigner, validate_transition
print('Transition validation:', validate_transition is not None)
"
```

---

## Deliverables Checklist

When complete, verify ALL of the following:

### Critical Bug Fixes

- [ ] `immunity_waning_days` parameter added and separate from `recovery_window_days`
- [ ] `validate_transition()` function added and used in `_compute_new_state`
- [ ] H4 test no longer generates mock data - returns inconclusive when data missing
- [ ] H1 test compares SEIR against exponential, logistic, linear models using AIC

### Statistical Features

- [ ] `apply_multiple_testing_correction()` method added with Bonferroni, Holm, FDR options
- [ ] `compare_against_null_networks()` method added
- [ ] `compute_time_varying_r0()` method added

### Stochastic Simulation

- [ ] `simulate_gillespie()` method added for continuous-time simulation

### Data Quality

- [ ] `validate_transactions()` method added
- [ ] Custom exceptions in `src/utils/exceptions.py`
- [ ] Magic numbers centralized in `config.yaml` and `constants.py`

### Tests

- [ ] `tests/test_state_assigner.py` created with transition validation tests
- [ ] All existing tests still pass
- [ ] New features have test coverage

---

## Execution Notes

1. **Make changes incrementally** - complete one section at a time and verify tests pass
2. **Do not remove existing functionality** - only add/modify
3. **Maintain backward compatibility** - new parameters should have defaults
4. **Run tests after each major change**
5. **Update docstrings when modifying methods**
