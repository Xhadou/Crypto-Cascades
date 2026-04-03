"""
State Assignment Engine

Assigns SEIR (Susceptible-Exposed-Infected-Recovered) states to wallets
based on their transaction behavior and network position.

This module implements behavioral state assignment using real UNIX timestamps
from the ORBITAAL dataset.
"""

import pandas as pd
import numpy as np
import igraph as ig
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.graph_adapter import name_to_idx


class State(Enum):
    """SEIR compartment states for FOMO epidemic model."""
    SUSCEPTIBLE = 'S'  # Not yet exposed to FOMO
    EXPOSED = 'E'      # Connected to infected, may become infected
    INFECTED = 'I'     # Actively buying (exhibiting FOMO behavior)
    RECOVERED = 'R'    # Was infected, now dormant

    def __str__(self) -> str:
        return self.value


# Valid state transitions in SEIR model
# S can go to S (stay), E (exposure), or I (direct infection after exposure / spontaneous)
# E can go to E (stay), I (become infected), or S (exposure timeout)
# I can go to I (stay) or R (recover)
# R can go to R (stay) or S (immunity wanes)
VALID_TRANSITIONS: Dict[State, Set[State]] = {
    State.SUSCEPTIBLE: {State.SUSCEPTIBLE, State.EXPOSED, State.INFECTED},
    State.EXPOSED: {State.EXPOSED, State.INFECTED, State.SUSCEPTIBLE},
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


class StateAssigner:
    """
    Assign behavioral states to wallets based on transaction patterns.
    
    State definitions:
    - SUSCEPTIBLE: No buying activity in past N days
    - EXPOSED: Connected to an infected wallet within exposure window
    - INFECTED: Actively buying (net positive BTC flow above threshold)
    - RECOVERED: Was infected but dormant for M days
    """
    
    def __init__(
        self,
        susceptible_window_days: int = 7,
        exposure_window_hours: int = 24,
        infected_threshold: float = 0.0,
        recovery_window_days: int = 3,
        immunity_waning_days: int = 30,
        min_usd_value: float = 100.0,
        infected_z_threshold: float = 1.5,
        exposure_timeout_days: int = 14,
        spontaneous_infection_rate: float = 0.001,
        random_seed: int = 42
    ):
        """
        Initialize state assigner.

        Args:
            susceptible_window_days: Days without buying to be susceptible
            exposure_window_hours: Hours after contact to be exposed
            infected_threshold: Minimum net BTC to be infected (positive = buying).
                Used as fallback when wallet history is unavailable or has zero variance.
            recovery_window_days: Days of dormancy after infection before recovered
            immunity_waning_days: Days in recovered state before becoming susceptible again
            min_usd_value: Minimum USD transaction value to count
            infected_z_threshold: Z-score threshold for infection classification.
                A wallet is classified as infected when its net flow z-score
                exceeds this value relative to its own transaction history.
            exposure_timeout_days: Days in EXPOSED state before reverting to
                SUSCEPTIBLE if no infection occurs. Models finite exposure
                periods where the contagion opportunity expires.
            spontaneous_infection_rate: Probability per timestep that a
                SUSCEPTIBLE wallet buying BTC becomes INFECTED without
                contact with an infected neighbor (importation from outside
                the observed network). Range [0, 1].
            random_seed: Random seed for reproducibility of stochastic processes.
        """
        self.susceptible_window = timedelta(days=susceptible_window_days)
        self.exposure_window = timedelta(hours=exposure_window_hours)
        self.infected_threshold = infected_threshold
        self.recovery_window = timedelta(days=recovery_window_days)
        self.immunity_waning_window = timedelta(days=immunity_waning_days)
        self.min_usd_value = min_usd_value
        self.infected_z_threshold = infected_z_threshold
        self.exposure_timeout_days = exposure_timeout_days
        self.spontaneous_infection_rate = spontaneous_infection_rate

        # RNG for stochastic processes (spontaneous infection)
        self.rng = np.random.default_rng(random_seed)

        # State tracking
        self.wallet_states: Dict[int, State] = {}
        self.state_history: Dict[int, List[Tuple[datetime, State]]] = defaultdict(list)
        self.infection_times: Dict[int, datetime] = {}
        self.recovery_times: Dict[int, datetime] = {}
        self.last_buying_activity: Dict[int, datetime] = {}
        self.exposure_start_times: Dict[int, datetime] = {}
        self._node_infection_times_df: pd.DataFrame = pd.DataFrame(
            columns=['node', 'infection_time']
        )

        self.logger = get_logger(__name__)
        
    def reset(self) -> None:
        """Reset all state tracking."""
        self.wallet_states = {}
        self.state_history = defaultdict(list)
        self.infection_times = {}
        self.recovery_times = {}
        self.last_buying_activity = {}
        self.exposure_start_times = {}
        self._node_infection_times_df = pd.DataFrame(
            columns=['node', 'infection_time']
        )
        
    def compute_wallet_flows(
        self,
        df: pd.DataFrame,
        time_column: str = 'datetime'
    ) -> pd.DataFrame:
        """
        Compute time-windowed BTC flows for each wallet by date.
        
        Args:
            df: Transaction DataFrame with source_id, target_id, btc_value, datetime
            time_column: Column containing timestamps
            
        Returns:
            DataFrame with wallet flows per day
        """
        if time_column not in df.columns:
            raise ValueError(
                f"Column '{time_column}' not found in transaction DataFrame. "
                f"Available columns: {list(df.columns)}. "
                f"Ensure data was preprocessed with daily snapshots "
                f"(python -m src.main --phase download && "
                f"python -m src.main --phase preprocess)."
            )
            
        self.logger.info("Computing wallet flows...")
            
        # Filter by minimum value
        df_filtered = df.copy()
        if 'usd_value' in df.columns:
            df_filtered = df_filtered[df_filtered['usd_value'] >= self.min_usd_value]
            
        # Create daily aggregation
        df_filtered['date'] = pd.to_datetime(df_filtered[time_column]).dt.date
        
        # Outgoing (selling/spending)
        outgoing = df_filtered.groupby(['source_id', 'date']).agg({
            'btc_value': 'sum'
        }).reset_index()
        outgoing.columns = ['wallet_id', 'date', 'btc_out']
        
        # Incoming (buying/receiving)
        incoming = df_filtered.groupby(['target_id', 'date']).agg({
            'btc_value': 'sum'
        }).reset_index()
        incoming.columns = ['wallet_id', 'date', 'btc_in']
        
        # Merge
        flows = pd.merge(outgoing, incoming, on=['wallet_id', 'date'], how='outer')
        flows = flows.fillna(0)
        flows['net_btc'] = flows['btc_in'] - flows['btc_out']
        
        self.logger.info(
            f"Computed flows for {flows['wallet_id'].nunique():,} wallets "
            f"across {flows['date'].nunique()} dates"
        )
        
        return flows
    
    def _get_wallet_net_flow_in_window(
        self,
        flows: pd.DataFrame,
        wallet_id: int,
        current_date,
        window_days: int
    ) -> float:
        """Get net BTC flow for a wallet within a time window."""
        window_start = current_date - timedelta(days=window_days)
        
        wallet_flows = flows[
            (flows['wallet_id'] == wallet_id) &
            (flows['date'] >= window_start) &
            (flows['date'] <= current_date)
        ]
        
        return wallet_flows['net_btc'].sum()
    
    def _has_recent_buying(
        self,
        flows: pd.DataFrame,
        wallet_id: int,
        current_date,
        window_days: int
    ) -> bool:
        """Check if wallet has positive net flow in the window."""
        net_flow = self._get_wallet_net_flow_in_window(
            flows, wallet_id, current_date, window_days
        )
        return net_flow > self.infected_threshold

    def _is_buying_zscore(
        self,
        net_flow: float,
        wallet_mean: float,
        wallet_std: float
    ) -> bool:
        """
        Determine if buying behavior is unusually high using z-score.

        Compares the current net flow against the wallet's own transaction
        history. If the z-score exceeds infected_z_threshold, the wallet
        is classified as exhibiting abnormal buying (FOMO) behavior.

        Falls back to the simple infected_threshold when the wallet has
        no history or constant flow (zero/near-zero standard deviation).

        Args:
            net_flow: Current period net BTC flow for the wallet.
            wallet_mean: Historical mean net BTC flow for the wallet.
            wallet_std: Historical standard deviation of net BTC flow.

        Returns:
            True if buying behavior is unusually high, False otherwise.
        """
        if wallet_std < 1e-10:
            # No variance in history — fall back to simple threshold
            return net_flow > self.infected_threshold
        z = (net_flow - wallet_mean) / wallet_std
        return z > self.infected_z_threshold
        
    def assign_states_at_time(
        self,
        g: ig.Graph,
        flows: pd.DataFrame,
        current_time: datetime,
        previous_states: Optional[Dict[int, State]] = None
    ) -> Dict[int, State]:
        """
        Assign states to all wallets at a specific time.

        Args:
            g: Transaction graph (igraph, for neighbor lookup)
            flows: Wallet flow DataFrame
            current_time: Current timestamp
            previous_states: States from previous timestep

        Returns:
            Dict mapping wallet_id to State
        """
        if previous_states is None:
            previous_states = {}

        current_date = current_time.date() if isinstance(current_time, datetime) else current_time
        states = {}

        # Precompute flows for efficiency
        window_start = current_date - timedelta(days=self.susceptible_window.days)
        recent_flows = flows[
            (flows['date'] >= window_start) &
            (flows['date'] <= current_date)
        ]

        # Aggregate net flow per wallet in window
        wallet_net_flow = recent_flows.groupby('wallet_id')['net_btc'].sum().to_dict()

        # Precompute per-wallet historical mean/std for z-score classification
        historical_flows = flows[flows['date'] <= current_date]
        wallet_stats = historical_flows.groupby('wallet_id')['net_btc'].agg(['mean', 'std']).fillna(0)
        wallet_mean_dict = wallet_stats['mean'].to_dict()
        wallet_std_dict = wallet_stats['std'].to_dict()

        # Get all wallets (wallet IDs stored in vs['name'])
        all_wallets = set(g.vs['name'])

        # Build name-to-index mapping once for neighbor lookups
        n2i = name_to_idx(g)
        names = g.vs['name']

        # Identify currently infected wallets for exposure check
        infected_wallets = {
            w for w, s in previous_states.items()
            if s == State.INFECTED
        }

        for wallet in all_wallets:
            prev_state = previous_states.get(wallet, State.SUSCEPTIBLE)
            net_flow = wallet_net_flow.get(wallet, 0)
            wallet_mean = wallet_mean_dict.get(wallet, 0.0)
            wallet_std = wallet_std_dict.get(wallet, 0.0)
            is_buying = self._is_buying_zscore(net_flow, wallet_mean, wallet_std)

            # State transition logic
            new_state = self._compute_new_state(
                wallet, prev_state, is_buying, g,
                infected_wallets, current_time,
                n2i=n2i, names=names
            )

            states[wallet] = new_state

            # Record transition if state changed
            if wallet not in self.state_history or \
               (self.state_history[wallet] and self.state_history[wallet][-1][1] != new_state):
                self.state_history[wallet].append((current_time, new_state))

        return states
    
    def _compute_new_state(
        self,
        wallet: int,
        prev_state: State,
        is_buying: bool,
        g: ig.Graph,
        infected_wallets: Set[int],
        current_time: datetime,
        n2i: Optional[Dict[int, int]] = None,
        names: Optional[list] = None,
    ) -> State:
        """Compute new state for a wallet based on transition rules."""

        new_state = prev_state  # Default: stay in current state

        if prev_state == State.RECOVERED:
            # R -> S: Immunity wanes after immunity_waning_window (not recovery_window!)
            recovery_time = self.recovery_times.get(wallet)
            if recovery_time and (current_time - recovery_time) > self.immunity_waning_window:
                new_state = State.SUSCEPTIBLE
            else:
                new_state = State.RECOVERED

        elif prev_state == State.INFECTED:
            # I -> R: Stop buying for recovery window
            if is_buying:
                self.last_buying_activity[wallet] = current_time
                new_state = State.INFECTED
            else:
                last_active = self.last_buying_activity.get(wallet)
                if last_active:
                    days_dormant = (current_time - last_active).days
                    if days_dormant >= self.recovery_window.days:
                        self.recovery_times[wallet] = current_time
                        new_state = State.RECOVERED
                    else:
                        new_state = State.INFECTED
                else:
                    new_state = State.INFECTED

        elif prev_state == State.EXPOSED:
            # E -> I: Start buying
            # E -> S: Exposure timeout (no infection within timeout window)
            if is_buying:
                self.infection_times[wallet] = current_time
                self.last_buying_activity[wallet] = current_time
                new_state = State.INFECTED
            else:
                exposure_start = self.exposure_start_times.get(wallet)
                if (
                    exposure_start
                    and current_time
                    and (current_time - exposure_start).days > self.exposure_timeout_days
                ):
                    new_state = State.SUSCEPTIBLE
                    # Clear expired exposure tracking
                    del self.exposure_start_times[wallet]
                else:
                    new_state = State.EXPOSED

        else:  # SUSCEPTIBLE
            # S -> I: Spontaneous infection (importation from outside network)
            # S -> E: Contact with infected neighbor
            # S -> I: Direct infection (start buying after contact)

            # Check for spontaneous infection first (external importation)
            if is_buying and self.rng.random() < self.spontaneous_infection_rate:
                self.infection_times[wallet] = current_time
                self.last_buying_activity[wallet] = current_time
                new_state = State.INFECTED
            else:
                has_infected_neighbor = False
                try:
                    # Build mapping lazily if not provided
                    if n2i is None:
                        n2i = name_to_idx(g)
                    if names is None:
                        names = g.vs['name']

                    wallet_idx = n2i[wallet]
                    neighbor_indices = g.neighbors(wallet_idx)
                    if g.is_directed():
                        # For directed graph, also check predecessors (incoming edges)
                        neighbor_indices = list(set(neighbor_indices) | set(g.neighbors(wallet_idx, mode='in')))
                    neighbor_names = {names[i] for i in neighbor_indices}
                    has_infected_neighbor = bool(neighbor_names & infected_wallets)
                except (ig.InternalError, KeyError, ValueError):
                    pass

                if has_infected_neighbor:
                    if is_buying:
                        self.infection_times[wallet] = current_time
                        self.last_buying_activity[wallet] = current_time
                        new_state = State.INFECTED
                    else:
                        self.exposure_start_times[wallet] = current_time
                        new_state = State.EXPOSED
                else:
                    new_state = State.SUSCEPTIBLE

        # Validate the transition
        if not validate_transition(prev_state, new_state):
            self.logger.warning(
                f"Invalid transition {prev_state.value} -> {new_state.value} "
                f"for wallet {wallet}. Keeping current state."
            )
            return prev_state

        return new_state
        
    def run_state_assignment(
        self,
        g: ig.Graph,
        flows: pd.DataFrame,
        initial_infected: Optional[List[int]] = None,
        initial_infected_fraction: float = 0.01
    ) -> pd.DataFrame:
        """
        Run state assignment over all time periods.

        Memory-optimised: pre-groups flows by date, pre-computes wallet
        stats, and only processes active wallets + their neighbors each
        day instead of all 30M wallets.

        Args:
            g: Transaction graph (igraph)
            flows: Wallet flow DataFrame with date column
            initial_infected: List of initially infected wallets
            initial_infected_fraction: Fraction of wallets to initially infect if not specified

        Returns:
            DataFrame with state counts over time
        """
        self.reset()

        dates = sorted(flows['date'].unique())
        self.logger.info(f"Running state assignment over {len(dates)} time periods...")

        all_wallets = set(g.vs['name'])
        n2i = name_to_idx(g)
        names = g.vs['name']

        if initial_infected is None:
            total_buying = flows.groupby('wallet_id')['net_btc'].sum()
            n_initial = max(1, int(len(all_wallets) * initial_infected_fraction))
            top_buyers = total_buying.nlargest(n_initial)
            initial_infected = list(top_buyers.index)

        self.logger.info(f"Initial infected: {len(initial_infected)} wallets")

        # Use defaultdict — only store non-SUSCEPTIBLE states.
        # This keeps the dict small (only active wallets) instead of 30M entries.
        current_states: Dict[int, State] = {}
        n_susceptible = len(all_wallets)

        initial_time = datetime.combine(dates[0], datetime.min.time())
        for w in initial_infected:
            if w in all_wallets:
                current_states[w] = State.INFECTED
                n_susceptible -= 1
                self.infection_times[w] = initial_time
                self.last_buying_activity[w] = initial_time
                self.state_history[w].append((initial_time, State.INFECTED))

        # Pre-group flows by date to avoid filtering 103M rows each iteration
        self.logger.info("Pre-grouping flows by date...")
        flows_by_date = {
            date: group for date, group in flows.groupby('date')
        }

        # Pre-compute cumulative wallet stats for z-score classification
        self.logger.info("Pre-computing wallet statistics...")
        wallet_cum_stats = flows.groupby('wallet_id')['net_btc'].agg(
            ['mean', 'std']
        ).fillna(0)
        wallet_mean_dict = wallet_cum_stats['mean'].to_dict()
        wallet_std_dict = wallet_cum_stats['std'].to_dict()
        del wallet_cum_stats

        state_counts = []

        for date in tqdm(dates, desc="Assigning states"):
            current_time = datetime.combine(date, datetime.min.time())

            # Get flows for this date and a rolling window
            current_date = date
            day_flows = flows_by_date.get(current_date, pd.DataFrame())

            if day_flows.empty:
                # No activity this day — just count states
                n_e = sum(1 for s in current_states.values() if s == State.EXPOSED)
                n_i = sum(1 for s in current_states.values() if s == State.INFECTED)
                n_r = sum(1 for s in current_states.values() if s == State.RECOVERED)
                state_counts.append({
                    'date': date, 'datetime': current_time,
                    'S': len(all_wallets) - n_e - n_i - n_r,
                    'E': n_e, 'I': n_i, 'R': n_r,
                    'total': len(all_wallets)
                })
                continue

            # Aggregate net flow per wallet for this window
            window_dates = [d for d in dates if d >= current_date - timedelta(days=self.susceptible_window.days) and d <= current_date]
            window_flows = pd.concat(
                [flows_by_date[d] for d in window_dates if d in flows_by_date],
                ignore_index=True
            )
            wallet_net_flow = window_flows.groupby('wallet_id')['net_btc'].sum().to_dict()
            del window_flows

            # Only process wallets that are active today + infected + their neighbors
            infected_wallets = {w for w, s in current_states.items() if s == State.INFECTED}
            active_wallets = set(day_flows['wallet_id'].unique()) if not day_flows.empty else set()

            # Add neighbors of infected wallets (they might become exposed)
            wallets_to_process = active_wallets.copy()
            for w in infected_wallets:
                if w in n2i:
                    for ni in g.neighbors(n2i[w]):
                        wallets_to_process.add(names[ni])

            # Also process wallets in non-susceptible states (they might transition)
            wallets_to_process.update(current_states.keys())

            new_states = {}
            for wallet in wallets_to_process:
                if wallet not in all_wallets:
                    continue
                prev_state = current_states.get(wallet, State.SUSCEPTIBLE)
                net_flow = wallet_net_flow.get(wallet, 0)
                wallet_mean = wallet_mean_dict.get(wallet, 0.0)
                wallet_std = wallet_std_dict.get(wallet, 0.0)
                is_buying = self._is_buying_zscore(net_flow, wallet_mean, wallet_std)

                new_state = self._compute_new_state(
                    wallet, prev_state, is_buying, g,
                    infected_wallets, current_time,
                    n2i=n2i, names=names
                )

                if new_state != State.SUSCEPTIBLE:
                    new_states[wallet] = new_state
                elif wallet in current_states:
                    # Wallet returned to susceptible — remove from active dict
                    pass

                if wallet not in self.state_history or \
                   (self.state_history[wallet] and self.state_history[wallet][-1][1] != new_state):
                    self.state_history[wallet].append((current_time, new_state))

            # Update current_states: remove wallets that returned to S,
            # add/update wallets that changed
            current_states = {w: s for w, s in current_states.items() if w in wallets_to_process}
            current_states.update(new_states)
            
            # Count states
            counts = {s: 0 for s in State}
            for state in current_states.values():
                counts[state] += 1
                
            state_counts.append({
                'date': date,
                'datetime': current_time,
                'S': counts[State.SUSCEPTIBLE],
                'E': counts[State.EXPOSED],
                'I': counts[State.INFECTED],
                'R': counts[State.RECOVERED],
                'total': len(current_states)
            })
            
        self.wallet_states = current_states

        if state_counts:
            result_df = pd.DataFrame(state_counts)
        else:
            result_df = pd.DataFrame(columns=['date', 'datetime', 'S', 'E', 'I', 'R', 'total'])
            self.logger.warning("State assignment produced no results (empty state_counts)")

        # Create a state history DataFrame that includes per-node infection times for H4 test
        self._node_infection_times_df = pd.DataFrame([
            {'node': node, 'infection_time': time}
            for node, time in self.infection_times.items()
        ])

        if len(result_df) > 0 and 'S' in result_df.columns:
            self.logger.info(
                f"State assignment complete. Final: S={result_df['S'].iloc[-1]}, "
                f"E={result_df['E'].iloc[-1]}, I={result_df['I'].iloc[-1]}, "
                f"R={result_df['R'].iloc[-1]}"
            )
        else:
            self.logger.warning("State assignment completed but no time steps recorded")

        return result_df

    def get_infection_times_df(self) -> pd.DataFrame:
        """Get DataFrame of node infection times for hypothesis testing."""
        if hasattr(self, '_node_infection_times_df'):
            return self._node_infection_times_df
        return pd.DataFrame(columns=['node', 'infection_time'])
        
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Compute state transition matrix from history.
        
        Returns:
            DataFrame with transition counts
        """
        transitions = defaultdict(lambda: defaultdict(int))
        
        for wallet, history in self.state_history.items():
            for i in range(len(history) - 1):
                from_state = history[i][1]
                to_state = history[i + 1][1]
                transitions[from_state.value][to_state.value] += 1
                
        # Convert to DataFrame
        all_states = [s.value for s in State]
        matrix = pd.DataFrame(
            [[transitions[f][t] for t in all_states] for f in all_states],
            index=all_states,
            columns=all_states
        )
        
        return matrix
    
    def get_state_history_df(self) -> pd.DataFrame:
        """
        Convert state history to DataFrame.
        
        Returns:
            DataFrame with columns [wallet_id, datetime, state]
        """
        records = []
        for wallet, history in self.state_history.items():
            for time, state in history:
                records.append({
                    'wallet_id': wallet,
                    'datetime': time,
                    'state': state.value
                })
        
        return pd.DataFrame(records)
    
    def compute_individual_r(
        self,
        g: ig.Graph
    ) -> Dict[int, int]:
        """
        Compute individual reproduction number (secondary infections caused).

        For each infected node, count how many of its neighbors
        became infected after it.

        Args:
            g: Transaction graph (igraph)

        Returns:
            Dict mapping wallet_id to count of secondary infections
        """
        individual_r = defaultdict(int)

        # Build infection timeline
        infection_order = []
        for wallet, history in self.state_history.items():
            for time, state in history:
                if state == State.INFECTED:
                    infection_order.append((time, wallet))
                    break

        infection_order.sort(key=lambda x: x[0])
        infection_times_sorted = {w: t for t, w in infection_order}

        n2i = name_to_idx(g)
        names = g.vs['name']

        # For each infected wallet, count neighbors infected after
        for wallet, infection_time in infection_times_sorted.items():
            try:
                wallet_idx = n2i[wallet]
                neighbor_indices = g.neighbors(wallet_idx)
                if g.is_directed():
                    neighbor_indices = list(set(neighbor_indices) | set(g.neighbors(wallet_idx, mode='in')))
                neighbor_names = [names[i] for i in neighbor_indices]
            except (ig.InternalError, KeyError, ValueError):
                continue

            for neighbor in neighbor_names:
                if neighbor in infection_times_sorted:
                    neighbor_time = infection_times_sorted[neighbor]
                    if neighbor_time > infection_time:
                        individual_r[wallet] += 1

        return dict(individual_r)
    
    def get_normalized_state_counts(
        self,
        state_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize state counts to fractions.
        
        Args:
            state_df: DataFrame from run_state_assignment
            
        Returns:
            DataFrame with state fractions
        """
        df = state_df.copy()
        total = df['total']
        
        for col in ['S', 'E', 'I', 'R']:
            df[f'{col}_frac'] = df[col] / total
            
        return df


def main():
    """Test state assignment."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    from src.preprocessing.graph_builder import GraphBuilder
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    assigner = StateAssigner(
        susceptible_window_days=7,
        recovery_window_days=3,
        min_usd_value=50.0
    )
    
    # Load sample stream data (has timestamps)
    sample_path = "data/raw/orbitaal/orbitaal-stream_graph-2016_07_08.csv"
    
    import os
    if os.path.exists(sample_path):
        df = parser.load_stream(sample_path)
        print(f"Loaded {len(df):,} transactions")
        
        # Build graph
        G = builder.build_transaction_graph(df)
        print(f"Graph: {G.vcount():,} nodes, {G.ecount():,} edges")
        
        # Compute flows
        flows = assigner.compute_wallet_flows(df)
        print(f"\nComputed flows for {flows['wallet_id'].nunique():,} wallets")
        
        # Run state assignment
        state_counts = assigner.run_state_assignment(G, flows)
        
        print("\nState counts over time:")
        print(state_counts)
        
        # Normalized
        normalized = assigner.get_normalized_state_counts(state_counts)
        print("\nNormalized state fractions:")
        print(normalized[['date', 'S_frac', 'E_frac', 'I_frac', 'R_frac']])
        
        # Transition matrix
        trans_matrix = assigner.get_transition_matrix()
        print("\nTransition matrix:")
        print(trans_matrix)
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
