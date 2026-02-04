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
    β_eff = β × (1 + α × (FGI - 50) / 50)  (FOMO factor)
    β = transmission rate
    σ = incubation rate (1/latent period)
    γ = recovery rate
    ω = immunity waning rate
    FGI = Fear & Greed Index (0-100)
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import odeint
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
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
        Compute effective transmission rate with FOMO factor.
        
        Args:
            fgi_value: Fear & Greed Index (0-100)
            
        Returns:
            Effective β with FOMO amplification
        """
        if not self.fomo_enabled:
            return self.beta
            
        fomo_factor = 1.0 + self.fomo_alpha * (fgi_value - 50) / 50
        return self.beta * max(0.1, fomo_factor)  # Ensure positive


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
        np.random.seed(random_seed)
        
        self.logger = get_logger(__name__)
        
    def _seir_ode(
        self,
        y: np.ndarray,
        t: float,
        N: int,
        beta: float,
        sigma: float,
        gamma: float,
        omega: float
    ) -> List[float]:
        """
        SEIR ODE system for scipy.integrate.odeint.
        
        Args:
            y: State vector [S, E, I, R]
            t: Time
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
        
        results = []
        y = [S0, E0, I0, R0]
        
        for i, ti in enumerate(t):
            # Get effective beta
            if fgi_values is not None and i < len(fgi_values):
                beta_eff = self.params.effective_beta(fgi_values[i])
            else:
                beta_eff = self.params.beta
                
            # Store current state
            S, E, I, R = y
            results.append({
                't': ti,
                'S': S,
                'E': E,
                'I': I,
                'R': R,
                'S_frac': S / N,
                'E_frac': E / N,
                'I_frac': I / N,
                'R_frac': R / N,
                'beta_eff': beta_eff
            })
            
            # Integrate one step
            if i < len(t) - 1:
                t_span = [ti, t[i + 1]]
                sol = odeint(
                    self._seir_ode,
                    y,
                    t_span,
                    args=(N, beta_eff, self.params.sigma, self.params.gamma, self.params.omega)
                )
                y = sol[-1]
        
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
                if np.random.random() < p_expose:
                    return State.EXPOSED
                    
        elif current_state == State.EXPOSED:
            # E -> I: Incubation ends
            if np.random.random() < self.params.sigma:
                return State.INFECTED
                
        elif current_state == State.INFECTED:
            # I -> R: Recovery
            if np.random.random() < self.params.gamma:
                return State.RECOVERED
                
        elif current_state == State.RECOVERED:
            # R -> S: Immunity waning
            if np.random.random() < self.params.omega:
                return State.SUSCEPTIBLE
                
        return current_state
    
    def _count_states(self, node_states: Dict[int, State]) -> Dict[str, int]:
        """Count nodes in each state."""
        counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for state in node_states.values():
            counts[state.value] += 1
        return counts
    
    def compute_network_r0(self, G: nx.Graph) -> float:
        """
        Compute network-adjusted R₀.
        
        For networks: R₀_network = (β/γ) × <k²>/<k>
        
        Where <k²>/<k> is the variance-to-mean ratio of degree.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Network-adjusted R₀
        """
        degrees = [d for _, d in G.degree()]  # type: ignore[misc]
        
        if len(degrees) == 0:
            return 0.0
            
        k_mean = np.mean(degrees)
        k2_mean = np.mean([d**2 for d in degrees])
        
        if k_mean == 0:
            return 0.0
            
        r0_basic = self.params.beta / self.params.gamma
        network_factor = k2_mean / k_mean
        
        r0_network = r0_basic * network_factor
        
        self.logger.info(
            f"R₀_basic = {r0_basic:.3f}, "
            f"Network factor = {network_factor:.3f}, "
            f"R₀_network = {r0_network:.3f}"
        )
        
        return float(r0_network)
    
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
        
        for i in range(n_simulations):
            # Random initial infected
            np.random.seed(self.random_seed + i)
            initial_infected = np.random.choice(
                nodes, 
                size=min(initial_infected_count, len(nodes)),
                replace=False
            ).tolist()
            
            # Run simulation
            df = self.simulate_network_stochastic(
                G, initial_infected, t_max, fgi_values
            )
            df['run'] = i
            all_results.append(df)
            
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
