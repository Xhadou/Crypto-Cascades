"""
Network SEIR Epidemic Model

Implements SEIR epidemic dynamics on networks with FOMO amplification
based on the Fear & Greed Index.

Mathematical Model:
    dS/dt = -β_eff × S × I / N + ω × R  (immunity waning)
    dE/dt = β_eff × S × I / N - σ × E
    dI/dt = σ × E - γ × I
    dR/dt = γ × I - ω × R

Where:
    β_eff = min(β × (1 + α × expit(k × (FGI - 50) / 50)), 0.99)  (sigmoidal FOMO)
    β = transmission rate
    σ = incubation rate (1/latent period)
    γ = recovery rate
    ω = immunity waning rate
    FGI = Fear & Greed Index (0-100)
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import solve_ivp
from scipy.special import expit
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Literal, overload
from dataclasses import dataclass
from enum import Enum
import logging

from src.state_engine.state_assigner import State
from src.utils.logger import get_logger


@dataclass
class SEIRParameters:
    """Parameters for the SEIR model."""
    beta: float = 0.3      # Transmission rate
    sigma: float = 0.2     # Incubation rate (1/latent period)
    gamma: float = 0.1     # Recovery rate
    omega: float = 0.01    # Immunity waning rate
    
    # FOMO amplification
    fomo_alpha: float = 1.0   # FOMO sensitivity
    fomo_k: float = 2.0       # Sigmoid steepness parameter
    fomo_enabled: bool = True
    
    def __post_init__(self):
        """Validate parameters."""
        assert 0 < self.beta <= 1, "beta must be in (0, 1]"
        assert 0 < self.sigma <= 1, "sigma must be in (0, 1]"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 <= self.omega <= 1, "omega must be in [0, 1]"
        
    def r0(self) -> float:
        """Compute basic reproduction number R₀ = β/γ."""
        return self.beta / self.gamma
    
    def effective_beta(self, fgi_value: float) -> float:
        """
        Compute effective transmission rate with sigmoidal FOMO amplification.

        Uses sigmoid (logistic) function for bounded, saturating response:
            β_eff = min(β × (1 + α × expit(k × (FGI - 50) / 50)), 0.99)

        The sigmoid provides:
        - Natural saturation at extreme FGI values (no unbounded growth)
        - β_eff is always in (0, 0.99] (valid probability)
        - Smooth, differentiable transition centered at FGI=50

        Args:
            fgi_value: Fear & Greed Index (0-100)

        Returns:
            Effective β with FOMO amplification, clamped to [0, 0.99]
        """
        if not self.fomo_enabled:
            return self.beta

        fomo_factor = 1.0 + self.fomo_alpha * expit(self.fomo_k * (fgi_value - 50) / 50)
        return min(self.beta * fomo_factor, 0.99)


class NetworkSEIR:
    """
    SEIR epidemic model on networks.
    
    Supports both:
    - Mean-field (deterministic ODE) simulation
    - Network-based (stochastic) simulation
    """
    
    def __init__(
        self,
        params: Optional[SEIRParameters] = None,
        random_seed: int = 42
    ):
        """
        Initialize the SEIR model.
        
        Args:
            params: Model parameters (uses defaults if None)
            random_seed: Random seed for reproducibility
        """
        self.params = params or SEIRParameters()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.logger = get_logger(__name__)

    def __getstate__(self):
        """Exclude logger from pickling (for ProcessPoolExecutor compatibility)."""
        state = self.__dict__.copy()
        state.pop('logger', None)
        return state

    def __setstate__(self, state):
        """Restore logger after unpickling."""
        self.__dict__.update(state)
        self.logger = get_logger(__name__)
        
    def _seir_ode(
        self,
        t: float,
        y: np.ndarray,
        N: int,
        beta: float,
        sigma: float,
        gamma: float,
        omega: float
    ) -> List[float]:
        """
        SEIR ODE system for scipy.integrate.solve_ivp.

        Args:
            t: Time
            y: State vector [S, E, I, R]
            N: Total population
            beta: Effective transmission rate
            sigma: Incubation rate
            gamma: Recovery rate
            omega: Immunity waning rate

        Returns:
            Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        
        # Force of infection
        lambda_t = beta * I / N
        
        dSdt = -lambda_t * S + omega * R
        dEdt = lambda_t * S - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I - omega * R
        
        return [dSdt, dEdt, dIdt, dRdt]
    
    def simulate_meanfield(
        self,
        N: int,
        initial_infected: int,
        t_max: int,
        fgi_values: Optional[np.ndarray] = None,
        dt: float = 1.0
    ) -> pd.DataFrame:
        """
        Run mean-field (ODE) simulation of SEIR dynamics.
        
        Args:
            N: Total population
            initial_infected: Initial number of infected
            t_max: Maximum time steps
            fgi_values: Fear & Greed Index values (one per timestep)
            dt: Time step
            
        Returns:
            DataFrame with columns [t, S, E, I, R] and fractions
        """
        self.logger.info(f"Running mean-field SEIR simulation (N={N:,}, T={t_max})")
        
        # Initial conditions
        I0 = initial_infected
        E0 = 0
        R0 = 0
        S0 = N - I0 - E0 - R0
        
        # Time points
        t = np.arange(0, t_max, dt)

        y0 = [S0, E0, I0, R0]

        # Build ODE function with time-varying beta via closure
        if fgi_values is not None:
            def ode_func(t_val, y_val):
                idx = min(int(t_val), len(fgi_values) - 1)
                beta_eff = self.params.effective_beta(fgi_values[idx])
                return self._seir_ode(
                    t_val, y_val, N, beta_eff,
                    self.params.sigma, self.params.gamma, self.params.omega
                )
        else:
            def ode_func(t_val, y_val):
                return self._seir_ode(
                    t_val, y_val, N, self.params.beta,
                    self.params.sigma, self.params.gamma, self.params.omega
                )

        # Solve the full ODE system in one call with RK45
        sol = solve_ivp(
            ode_func,
            t_span=[t[0], t[-1]],
            y0=y0,
            method='RK45',
            t_eval=t,
            rtol=1e-8,
            atol=1e-8
        )

        # Build results from sol.y (shape: 4 x len(t))
        results = []
        for i in range(len(sol.t)):
            S, E, I, R = sol.y[:, i]
            # Clamp to non-negative (numerical noise can push slightly below 0)
            S, E, I, R = max(S, 0), max(E, 0), max(I, 0), max(R, 0)

            if fgi_values is not None:
                idx = min(int(sol.t[i]), len(fgi_values) - 1)
                beta_eff = self.params.effective_beta(fgi_values[idx])
            else:
                beta_eff = self.params.beta

            results.append({
                't': sol.t[i],
                'S': S,
                'E': E,
                'I': I,
                'R': R,
                'S_frac': S / N,
                'E_frac': E / N,
                'I_frac': I / N,
                'R_frac': R / N,
                'beta_eff': beta_eff,
            })

        return pd.DataFrame(results)
    
    def simulate_network_stochastic(
        self,
        G: nx.Graph,
        initial_infected: List[int],
        t_max: int,
        fgi_values: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Run stochastic network simulation using discrete-time approach.
        
        Args:
            G: NetworkX graph
            initial_infected: List of initially infected node IDs
            t_max: Maximum time steps
            fgi_values: Fear & Greed Index values (one per timestep)
            
        Returns:
            DataFrame with state counts over time
        """
        self.logger.info(
            f"Running stochastic network SEIR simulation "
            f"(N={G.number_of_nodes():,}, T={t_max})"
        )
        
        # Initialize node states
        node_states = {node: State.SUSCEPTIBLE for node in G.nodes()}
        for node in initial_infected:
            if node in node_states:
                node_states[node] = State.INFECTED
        
        # Track time in each state for transitions
        time_in_state = {node: 0 for node in G.nodes()}
        
        results = []
        
        for t in range(t_max):
            # Get effective beta
            if fgi_values is not None and t < len(fgi_values):
                beta_eff = self.params.effective_beta(fgi_values[t])
            else:
                beta_eff = self.params.beta
            
            # Count current states
            counts: Dict[str, Any] = self._count_states(node_states)
            counts['t'] = t
            counts['beta_eff'] = beta_eff
            results.append(counts)
            
            # Compute transitions
            new_states = node_states.copy()
            
            for node in G.nodes():
                current_state = node_states[node]
                new_state = self._node_transition(
                    node, current_state, G, node_states, 
                    beta_eff, time_in_state[node]
                )
                
                if new_state != current_state:
                    new_states[node] = new_state
                    time_in_state[node] = 0
                else:
                    time_in_state[node] += 1
            
            node_states = new_states
            
        df = pd.DataFrame(results)
        
        # Add fractions
        N = G.number_of_nodes()
        for col in ['S', 'E', 'I', 'R']:
            df[f'{col}_frac'] = df[col] / N
            
        return df
    
    def _node_transition(
        self,
        node: int,
        current_state: State,
        G: nx.Graph,
        node_states: Dict[int, State],
        beta_eff: float,
        time_in_state: int
    ) -> State:
        """Compute state transition for a single node."""
        
        if current_state == State.SUSCEPTIBLE:
            # S -> E: Contact with infected neighbor
            infected_neighbors = sum(
                1 for n in G.neighbors(node) 
                if node_states.get(n) == State.INFECTED
            )
            degree = G.degree(node)  # type: ignore[misc]
            
            if degree > 0 and infected_neighbors > 0:
                # Probability based on fraction of infected neighbors
                p_expose = 1 - (1 - beta_eff) ** infected_neighbors
                if self.rng.random() < p_expose:
                    return State.EXPOSED

        elif current_state == State.EXPOSED:
            # E -> I: Incubation ends
            if self.rng.random() < self.params.sigma:
                return State.INFECTED

        elif current_state == State.INFECTED:
            # I -> R: Recovery
            if self.rng.random() < self.params.gamma:
                return State.RECOVERED

        elif current_state == State.RECOVERED:
            # R -> S: Immunity waning
            if self.rng.random() < self.params.omega:
                return State.SUSCEPTIBLE
                
        return current_state
    
    def _count_states(self, node_states: Dict[int, State]) -> Dict[str, int]:
        """Count nodes in each state."""
        counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for state in node_states.values():
            counts[state.value] += 1
        return counts
    
    @overload
    def compute_network_r0(
        self, G: nx.Graph, n_bootstrap: int = ..., *, return_ci: Literal[False] = ...
    ) -> float: ...

    @overload
    def compute_network_r0(
        self, G: nx.Graph, n_bootstrap: int = ..., *, return_ci: Literal[True] = ...
    ) -> Tuple[float, Tuple[float, float]]: ...

    def compute_network_r0(
        self,
        G: nx.Graph,
        n_bootstrap: int = 0,
        return_ci: bool = False
    ) -> Union[float, Tuple[float, Tuple[float, float]]]:
        """
        Compute network-adjusted R₀ with optional bootstrap CI.

        For networks: R₀_network = (β/γ) × <k²>/<k>

        Where <k²>/<k> is the variance-to-mean ratio of degree.

        Args:
            G: NetworkX graph
            n_bootstrap: Number of bootstrap samples for CI (0 = no CI)
            return_ci: Whether to return confidence interval tuple

        Returns:
            If return_ci=False: Network-adjusted R₀ (float)
            If return_ci=True: Tuple of (R₀, (lower_CI, upper_CI))
        """
        degrees = [d for _, d in G.degree()]  # type: ignore[misc]

        if len(degrees) == 0:
            return (0.0, (0.0, 0.0)) if return_ci else 0.0

        k_mean = np.mean(degrees)
        k2_mean = np.mean([d**2 for d in degrees])

        if k_mean == 0:
            return (0.0, (0.0, 0.0)) if return_ci else 0.0

        r0_basic = self.params.beta / self.params.gamma

        def compute_r0_from_degrees(deg_list: List[int]) -> float:
            km = np.mean(deg_list)
            k2m = np.mean([d**2 for d in deg_list])
            return float(r0_basic * (k2m / km)) if km > 0 else 0.0

        network_factor = k2_mean / k_mean
        r0_network = r0_basic * network_factor

        self.logger.info(
            f"R₀_basic = {r0_basic:.3f}, "
            f"Network factor = {network_factor:.3f}, "
            f"R₀_network = {r0_network:.3f}"
        )

        if not return_ci or n_bootstrap == 0:
            return float(r0_network)

        # Bootstrap CI
        r0_samples = []
        for _ in range(n_bootstrap):
            boot_degrees = self.rng.choice(degrees, len(degrees), replace=True).tolist()
            r0_samples.append(compute_r0_from_degrees(boot_degrees))

        ci = (float(np.percentile(r0_samples, 2.5)),
              float(np.percentile(r0_samples, 97.5)))

        self.logger.info(f"R₀_network 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

        return float(r0_network), ci

    def compute_time_varying_r0(
        self,
        state_df: pd.DataFrame,
        window_size: int = 7,
        method: str = 'ratio',
        serial_interval_mean: float = 7.0,
        serial_interval_std: float = 3.5
    ) -> pd.DataFrame:
        """
        Estimate time-varying reproduction number R(t).

        Args:
            state_df: DataFrame with SEIR state counts over time
            window_size: Rolling window size for estimation
            method: Estimation method:
                - 'ratio': Simple ratio method (new infections / current infections)
                - 'cori': Cori et al. (2013) method via epyestim (falls back to ratio)
            serial_interval_mean: Mean serial interval for Cori method (days)
            serial_interval_std: Std dev of serial interval for Cori method (days)

        Returns:
            DataFrame with columns [t, R_t, R_t_lower, R_t_upper]
        """
        self.logger.info(f"Computing time-varying R(t) using {method} method...")

        if method == 'cori':
            cori_result = self._compute_rt_cori(
                state_df, serial_interval_mean, serial_interval_std
            )
            if cori_result is not None:
                return cori_result
            self.logger.info("Cori method unavailable, falling back to ratio method")

        return self._compute_rt_ratio(state_df, window_size)

    def _compute_rt_cori(
        self,
        state_df: pd.DataFrame,
        serial_interval_mean: float = 7.0,
        serial_interval_std: float = 3.5
    ) -> Optional[pd.DataFrame]:
        """
        Estimate R(t) using Cori et al. (2013) method via epyestim.

        Args:
            state_df: DataFrame with SEIR state counts over time
            serial_interval_mean: Mean serial interval (days)
            serial_interval_std: Std dev of serial interval (days)

        Returns:
            DataFrame with R(t) estimates, or None if epyestim is unavailable
        """
        try:
            from epyestim import estimate_r
        except ImportError:
            self.logger.warning(
                "epyestim not installed; Cori R(t) method unavailable. "
                "Install with: pip install epyestim"
            )
            return None

        # Extract incidence (new infections per time step)
        if 'I' in state_df.columns:
            I = np.asarray(state_df['I'].values, dtype=float)
        elif 'I_frac' in state_df.columns:
            N_assumed = 10000
            I = np.asarray(state_df['I_frac'].values, dtype=float) * N_assumed
        else:
            self.logger.error("Cannot find infection data in state_df")
            return None

        # Compute incidence as change in infected count, clipped to non-negative
        incidence = np.diff(I)
        incidence = np.maximum(incidence, 0)

        if incidence.sum() <= 0:
            self.logger.warning("No incidence detected; cannot estimate R(t) via Cori method")
            return None

        # epyestim expects a pandas Series with DatetimeIndex
        t_col = np.asarray(state_df['t'].values) if 't' in state_df.columns else np.arange(len(state_df))
        dates = pd.date_range(start='2020-01-01', periods=len(incidence), freq='D')
        incidence_series = pd.Series(incidence, index=dates, name='cases')

        try:
            rt_result = estimate_r(
                incidence_series,
                si_distr=None,
                mean_si=serial_interval_mean,
                std_si=serial_interval_std,
            )

            # Build output DataFrame consistent with ratio method format
            result_df = pd.DataFrame({
                't': t_col[1:len(rt_result) + 1] if len(t_col) > len(rt_result) else np.arange(len(rt_result)),
                'R_t': rt_result['R_mean'].values if 'R_mean' in rt_result.columns else rt_result.iloc[:, 0].values,
            })

            # Add CI columns if available
            if 'R_q0.025' in rt_result.columns:
                result_df['R_t_lower'] = rt_result['R_q0.025'].values
            elif 'Q0.025' in rt_result.columns:
                result_df['R_t_lower'] = rt_result['Q0.025'].values

            if 'R_q0.975' in rt_result.columns:
                result_df['R_t_upper'] = rt_result['R_q0.975'].values
            elif 'Q0.975' in rt_result.columns:
                result_df['R_t_upper'] = rt_result['Q0.975'].values

            valid_rt = result_df['R_t'].dropna()
            if len(valid_rt) > 0:
                self.logger.info(
                    f"Cori R(t) range: [{valid_rt.min():.2f}, {valid_rt.max():.2f}], "
                    f"mean: {valid_rt.mean():.2f}"
                )

            return result_df

        except Exception as e:
            self.logger.warning(f"Cori R(t) estimation failed: {e}")
            return None

    def _compute_rt_ratio(
        self,
        state_df: pd.DataFrame,
        window_size: int = 7
    ) -> pd.DataFrame:
        """
        Estimate R(t) using simple ratio method.

        Args:
            state_df: DataFrame with SEIR state counts over time
            window_size: Rolling window size for estimation

        Returns:
            DataFrame with columns [t, R_t, R_t_adjusted, R_t_lower, R_t_upper, ...]
        """
        results = []

        # Extract data
        t = np.asarray(state_df['t'].values) if 't' in state_df.columns else np.arange(len(state_df))

        if 'I' in state_df.columns:
            I = np.asarray(state_df['I'].values, dtype=float)
            S = np.asarray(state_df['S'].values, dtype=float)
            E = np.asarray(state_df['E'].values, dtype=float)
        elif 'I_frac' in state_df.columns:
            N = 10000  # Assumed population
            I = np.asarray(state_df['I_frac'].values, dtype=float) * N
            S = np.asarray(state_df['S_frac'].values, dtype=float) * N
            E = np.asarray(state_df['E_frac'].values, dtype=float) * N
        else:
            self.logger.error("Cannot find infection data in state_df")
            return pd.DataFrame()

        # Compute new infections (incidence)
        new_infections = np.diff(E + I)
        new_infections = np.maximum(new_infections, 0)  # Can't be negative

        # Simple ratio method
        for i in range(window_size, len(state_df)):
            window_start = i - window_size

            # Current infected in window
            I_window = I[window_start:i]
            I_mean = np.mean(I_window)

            # New infections in window
            if i <= len(new_infections):
                new_inf_window = new_infections[window_start:min(i, len(new_infections))]
                new_inf_sum = np.sum(new_inf_window)
            else:
                new_inf_sum = 0

            # Susceptible fraction for adjustment
            S_frac = S[i] / (S[i] + E[i] + I[i] + 1e-10)

            if I_mean > 0 and S_frac > 0:
                # R_t ~ (new infections per time) / (gamma * I) * (N / S)
                # Simplified: R_t ~ (new_inf / I) * (1 / S_frac)
                R_t = (new_inf_sum / window_size) / (self.params.gamma * I_mean)
                # Adjust for susceptible depletion
                R_t_adjusted = R_t / S_frac if S_frac > 0.1 else R_t
            else:
                R_t = np.nan
                R_t_adjusted = np.nan

            # Bootstrap CI - align arrays before bootstrapping
            if I_mean > 0 and len(I_window) > 0 and len(new_inf_window) > 0:
                # Align arrays to the same length to avoid index bias
                min_len = min(len(I_window), len(new_inf_window))
                if min_len < 2:
                    R_t_lower = R_t_upper = R_t
                else:
                    I_aligned = I_window[:min_len]
                    new_inf_aligned = new_inf_window[:min_len]

                    R_t_samples = []
                    for _ in range(100):
                        boot_idx = self.rng.choice(min_len, min_len, replace=True)
                        boot_I = np.mean(I_aligned[boot_idx])
                        boot_new = np.sum(new_inf_aligned[boot_idx])
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
            valid_r_t = df['R_t'].dropna()
            if len(valid_r_t) > 0:
                self.logger.info(
                    f"R(t) range: [{valid_r_t.min():.2f}, {valid_r_t.max():.2f}], "
                    f"mean: {valid_r_t.mean():.2f}"
                )

        return df
    
    def simulate_gillespie(
        self,
        G: nx.Graph,
        initial_infected: List[int],
        t_max: float,
        fgi_values: Optional[np.ndarray] = None,
        record_interval: float = 1.0,
        early_termination: bool = True,
        min_infected_for_continuation: int = 0
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
            early_termination: Whether to stop when epidemic dies out
            min_infected_for_continuation: Min infected+exposed to continue

        Returns:
            DataFrame with state counts over time
        """
        self.logger.info(
            f"Running Gillespie simulation (N={G.number_of_nodes():,}, T={t_max})"
        )

        # Validate graph connectivity
        undirected_G = G.to_undirected() if G.is_directed() else G
        if not nx.is_connected(undirected_G):
            n_components = nx.number_connected_components(undirected_G)
            self.logger.warning(
                f"Graph is disconnected ({n_components} components). "
                "Simulation may not reach all nodes."
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

            # Early termination check
            if early_termination:
                if len(I_nodes) == 0 and len(E_nodes) == 0:
                    self.logger.info(f"Epidemic died out at t={t:.2f}")
                    break
                if len(I_nodes) <= min_infected_for_continuation and len(E_nodes) == 0:
                    self.logger.info(f"Below continuation threshold at t={t:.2f}")
                    break

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
            dt = self.rng.exponential(1.0 / total_rate)
            t += dt

            if t > t_max:
                break

            # Choose which event occurs
            rand = self.rng.random() * total_rate

            if rand < exposure_rate:
                # Exposure event: choose which S node
                cumsum = 0.0
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
                    node = self.rng.choice(list(E_nodes))
                    E_nodes.remove(node)
                    I_nodes.add(node)
                    node_states[node] = State.INFECTED

            elif rand < exposure_rate + infection_rate + recovery_rate:
                # Recovery event: random I node becomes R
                if I_nodes:
                    node = self.rng.choice(list(I_nodes))
                    I_nodes.remove(node)
                    R_nodes.add(node)
                    node_states[node] = State.RECOVERED

            else:
                # Immunity waning: random R node becomes S
                if R_nodes:
                    node = self.rng.choice(list(R_nodes))
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

        # Pad results to t_max if terminated early
        if results and results[-1]['t'] < t_max:
            final_state = results[-1].copy()
            pad_time = final_state['t'] + record_interval
            while pad_time <= t_max:
                padded = final_state.copy()
                padded['t'] = pad_time
                results.append(padded)
                pad_time += record_interval

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

    def run_monte_carlo(
        self,
        G: nx.Graph,
        initial_infected_count: int,
        t_max: int,
        n_simulations: int = 100,
        fgi_values: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run multiple stochastic simulations for uncertainty quantification.
        
        Args:
            G: NetworkX graph
            initial_infected_count: Number of initial infected
            t_max: Maximum time
            n_simulations: Number of simulation runs
            fgi_values: Fear & Greed Index values
            
        Returns:
            Dict with mean, std, and percentiles for each state
        """
        self.logger.info(f"Running {n_simulations} Monte Carlo simulations...")

        all_results = []
        nodes = list(G.nodes())

        # Spawn independent child seeds for each run
        from numpy.random import SeedSequence
        ss = SeedSequence(self.random_seed)
        child_seeds = ss.spawn(n_simulations)

        for i in range(n_simulations):
            # Each run gets its own RNG from a spawned seed
            run_rng = np.random.default_rng(child_seeds[i])
            initial_infected = run_rng.choice(
                nodes,
                size=min(initial_infected_count, len(nodes)),
                replace=False
            ).tolist()

            # Temporarily swap in per-run RNG for stochastic simulation
            saved_rng = self.rng
            self.rng = run_rng

            # Run simulation
            df = self.simulate_network_stochastic(
                G, initial_infected, t_max, fgi_values
            )
            df['run'] = i
            all_results.append(df)

            # Restore original RNG
            self.rng = saved_rng
            
        # Combine results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Compute statistics
        stats = {}
        for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
            grouped = combined.groupby('t')[col]
            stats[col] = {
                'mean': grouped.mean().values,
                'std': grouped.std().values,
                'q05': grouped.quantile(0.05).values,
                'q25': grouped.quantile(0.25).values,
                'q50': grouped.quantile(0.50).values,
                'q75': grouped.quantile(0.75).values,
                'q95': grouped.quantile(0.95).values,
            }
        
        stats['t'] = np.arange(t_max)
        stats['n_simulations'] = n_simulations
        
        return stats

    def _single_mc_simulation(
        self,
        G: nx.Graph,
        initial_infected_count: int,
        t_max: int,
        fgi_values: Optional[np.ndarray],
        run_idx: int
    ) -> pd.DataFrame:
        """
        Execute a single Monte Carlo simulation (picklable for multiprocessing).

        Args:
            G: NetworkX graph
            initial_infected_count: Number of initial infected
            t_max: Maximum time
            fgi_values: Fear & Greed Index values
            run_idx: Index for random seed offset

        Returns:
            DataFrame with simulation results for this run
        """
        from numpy.random import SeedSequence
        nodes = list(G.nodes())
        # Create a deterministic child seed for this run
        ss = SeedSequence(self.random_seed)
        child_seed = ss.spawn(run_idx + 1)[run_idx]
        run_rng = np.random.default_rng(child_seed)

        initial_infected = run_rng.choice(
            nodes,
            size=min(initial_infected_count, len(nodes)),
            replace=False
        ).tolist()

        # Use per-run RNG for stochastic simulation
        saved_rng = self.rng
        self.rng = run_rng

        df = self.simulate_network_stochastic(
            G, initial_infected, t_max, fgi_values
        )
        df['run'] = run_idx

        self.rng = saved_rng
        return df

    def run_monte_carlo_parallel(
        self,
        G: nx.Graph,
        initial_infected_count: int,
        t_max: int,
        n_simulations: int = 100,
        fgi_values: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None
    ) -> Dict:
        """
        Run Monte Carlo simulations using parallel workers.
        
        Falls back to sequential execution if n_workers=1 or
        if the concurrent.futures import fails.
        
        Args:
            G: NetworkX graph
            initial_infected_count: Number of initial infected
            t_max: Maximum time
            n_simulations: Number of simulation runs
            fgi_values: Fear & Greed Index values
            n_workers: Number of parallel workers (None = cpu_count)
            
        Returns:
            Dict with mean, std, and percentiles for each state
        """
        import os as _os
        if n_workers is None:
            n_workers = min(_os.cpu_count() or 4, n_simulations)
        
        if n_workers <= 1:
            return self.run_monte_carlo(
                G, initial_infected_count, t_max, n_simulations, fgi_values
            )
        
        self.logger.info(
            f"Running {n_simulations} Monte Carlo simulations "
            f"with {n_workers} workers..."
        )
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_results: List[pd.DataFrame] = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    self._single_mc_simulation,
                    G, initial_infected_count, t_max, fgi_values, i
                ): i
                for i in range(n_simulations)
            }
            for future in as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    run_i = futures[future]
                    self.logger.warning(
                        f"Monte Carlo run {run_i} failed: {e}"
                    )
        
        if not all_results:
            self.logger.error("All Monte Carlo simulations failed")
            return {}
        
        combined = pd.concat(all_results, ignore_index=True)
        
        stats = {}
        for col in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
            grouped = combined.groupby('t')[col]
            stats[col] = {
                'mean': grouped.mean().values,
                'std': grouped.std().values,
                'q05': grouped.quantile(0.05).values,
                'q25': grouped.quantile(0.25).values,
                'q50': grouped.quantile(0.50).values,
                'q75': grouped.quantile(0.75).values,
                'q95': grouped.quantile(0.95).values,
            }
        
        stats['t'] = np.arange(t_max)
        stats['n_simulations'] = len(all_results)
        
        return stats
    
    def fit_to_observed(
        self,
        observed_df: pd.DataFrame,
        N: int,
        fgi_values: Optional[np.ndarray] = None,
        loss_type: str = 'mse'
    ) -> Callable:
        """
        Create loss function for parameter fitting.
        
        Args:
            observed_df: DataFrame with columns S, E, I, R (or fractions)
            N: Total population
            fgi_values: Fear & Greed Index values
            loss_type: 'mse', 'mae', or 'nll' (negative log-likelihood)
            
        Returns:
            Loss function that takes parameters [beta, sigma, gamma] and returns loss
        """
        # Normalize observed data if needed
        if observed_df['S'].max() > 1:
            S_obs = np.asarray(observed_df['S'].values) / N
            E_obs = np.asarray(observed_df['E'].values) / N
            I_obs = np.asarray(observed_df['I'].values) / N
            R_obs = np.asarray(observed_df['R'].values) / N
        else:
            S_obs = np.asarray(observed_df['S_frac'].values)
            E_obs = np.asarray(observed_df['E_frac'].values)
            I_obs = np.asarray(observed_df['I_frac'].values)
            R_obs = np.asarray(observed_df['R_frac'].values)
        
        t_max = len(observed_df)
        
        def loss_function(params: np.ndarray) -> float:
            beta, sigma, gamma = params
            
            # Update model parameters
            self.params.beta = beta
            self.params.sigma = sigma
            self.params.gamma = gamma
            
            # Run simulation
            I0 = max(1, int(I_obs[0] * N))
            sim_df = self.simulate_meanfield(N, I0, t_max, fgi_values)
            
            # Compute loss
            if loss_type == 'mse':
                loss = (
                    np.mean((np.asarray(sim_df['S_frac'].values) - S_obs) ** 2) +
                    np.mean((np.asarray(sim_df['E_frac'].values) - E_obs) ** 2) +
                    np.mean((np.asarray(sim_df['I_frac'].values) - I_obs) ** 2) +
                    np.mean((np.asarray(sim_df['R_frac'].values) - R_obs) ** 2)
                )
            elif loss_type == 'mae':
                loss = (
                    np.mean(np.abs(np.asarray(sim_df['S_frac'].values) - S_obs)) +
                    np.mean(np.abs(np.asarray(sim_df['E_frac'].values) - E_obs)) +
                    np.mean(np.abs(np.asarray(sim_df['I_frac'].values) - I_obs)) +
                    np.mean(np.abs(np.asarray(sim_df['R_frac'].values) - R_obs))
                )
            else:  # Negative log-likelihood (Poisson)
                eps = 1e-10
                sim_counts = sim_df[['S', 'E', 'I', 'R']].values
                obs_counts = np.column_stack([S_obs, E_obs, I_obs, R_obs]) * N
                
                loss = -np.sum(
                    obs_counts * np.log(sim_counts + eps) - sim_counts
                )
            
            return float(loss)
        
        return loss_function


def main():
    """Test the SEIR model."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    from src.preprocessing.graph_builder import GraphBuilder
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    
    # Create simple test graph
    print("Testing mean-field simulation...")
    params = SEIRParameters(beta=0.3, sigma=0.2, gamma=0.1, omega=0.01)
    model = NetworkSEIR(params)
    
    # Mean-field simulation
    results = model.simulate_meanfield(N=10000, initial_infected=10, t_max=100)
    print(f"\nMean-field results (T=100):")
    print(f"  Final S: {results['S'].iloc[-1]:.0f}")
    print(f"  Final E: {results['E'].iloc[-1]:.0f}")
    print(f"  Final I: {results['I'].iloc[-1]:.0f}")
    print(f"  Final R: {results['R'].iloc[-1]:.0f}")
    print(f"  Peak I: {results['I'].max():.0f} at t={results['I'].idxmax()}")
    
    # Test with FOMO factor
    print("\nTesting with FOMO factor...")
    fgi_values = np.linspace(30, 80, 100)  # Rising sentiment
    results_fomo = model.simulate_meanfield(
        N=10000, initial_infected=10, t_max=100, fgi_values=fgi_values
    )
    print(f"  Peak I with FOMO: {results_fomo['I'].max():.0f} at t={results_fomo['I'].idxmax()}")
    
    # Test network-based simulation
    print("\nTesting network simulation...")
    G = nx.barabasi_albert_graph(1000, 3, seed=42)
    
    results_network = model.simulate_network_stochastic(
        G, initial_infected=[0, 1, 2], t_max=50
    )
    print(f"  Network final I: {results_network['I'].iloc[-1]}")
    
    # Network R0
    r0_network = model.compute_network_r0(G)
    print(f"  Network R₀: {r0_network:.3f}")


if __name__ == "__main__":
    main()
