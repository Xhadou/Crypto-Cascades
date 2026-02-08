"""
State Assignment Engine

Assigns SEIR (Susceptible-Exposed-Infected-Recovered) states to wallets
based on their transaction behavior and network position.

This module implements behavioral state assignment using real UNIX timestamps
from the ORBITAAL dataset.
"""

import pandas as pd
import numpy as np
import networkx as nx
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from tqdm import tqdm

from src.utils.logger import get_logger


class State(Enum):
    """SEIR compartment states for FOMO epidemic model."""
    SUSCEPTIBLE = 'S'  # Not yet exposed to FOMO
    EXPOSED = 'E'      # Connected to infected, may become infected
    INFECTED = 'I'     # Actively buying (exhibiting FOMO behavior)
    RECOVERED = 'R'    # Was infected, now dormant

    def __str__(self) -> str:
        return self.value


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
        self.immunity_waning_window = timedelta(days=immunity_waning_days)
        self.min_usd_value = min_usd_value
        
        # State tracking
        self.wallet_states: Dict[int, State] = {}
        self.state_history: Dict[int, List[Tuple[datetime, State]]] = defaultdict(list)
        self.infection_times: Dict[int, datetime] = {}
        self.recovery_times: Dict[int, datetime] = {}
        self.last_buying_activity: Dict[int, datetime] = {}
        
        self.logger = get_logger(__name__)
        
    def reset(self) -> None:
        """Reset all state tracking."""
        self.wallet_states = {}
        self.state_history = defaultdict(list)
        self.infection_times = {}
        self.recovery_times = {}
        self.last_buying_activity = {}
        
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
            raise ValueError(f"Column {time_column} not found")
            
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
        
    def assign_states_at_time(
        self,
        G: nx.Graph,
        flows: pd.DataFrame,
        current_time: datetime,
        previous_states: Optional[Dict[int, State]] = None
    ) -> Dict[int, State]:
        """
        Assign states to all wallets at a specific time.
        
        Args:
            G: Transaction graph (for neighbor lookup)
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
        
        # Get all wallets
        all_wallets = set(G.nodes())
        
        # Identify currently infected wallets for exposure check
        infected_wallets = {
            w for w, s in previous_states.items() 
            if s == State.INFECTED
        }
        
        for wallet in all_wallets:
            prev_state = previous_states.get(wallet, State.SUSCEPTIBLE)
            net_flow = wallet_net_flow.get(wallet, 0)
            is_buying = net_flow > self.infected_threshold
            
            # State transition logic
            new_state = self._compute_new_state(
                wallet, prev_state, is_buying, G, 
                infected_wallets, current_time
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
        G: nx.Graph,
        infected_wallets: Set[int],
        current_time: datetime
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
            if is_buying:
                self.infection_times[wallet] = current_time
                self.last_buying_activity[wallet] = current_time
                new_state = State.INFECTED
            else:
                new_state = State.EXPOSED

        else:  # SUSCEPTIBLE
            # S -> E: Contact with infected neighbor
            # S -> I: Direct infection (start buying after contact)

            has_infected_neighbor = False
            try:
                neighbors = set(G.neighbors(wallet))
                if G.is_directed():
                    # For directed graph, also check predecessors (incoming edges)
                    neighbors.update(G.predecessors(wallet))  # type: ignore[union-attr]
                has_infected_neighbor = bool(neighbors & infected_wallets)
            except:
                pass

            if has_infected_neighbor:
                if is_buying:
                    self.infection_times[wallet] = current_time
                    self.last_buying_activity[wallet] = current_time
                    new_state = State.INFECTED
                else:
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
        G: nx.Graph,
        flows: pd.DataFrame,
        initial_infected: Optional[List[int]] = None,
        initial_infected_fraction: float = 0.01
    ) -> pd.DataFrame:
        """
        Run state assignment over all time periods.
        
        Args:
            G: Transaction graph
            flows: Wallet flow DataFrame with date column
            initial_infected: List of initially infected wallets
            initial_infected_fraction: Fraction of wallets to initially infect if not specified
            
        Returns:
            DataFrame with state counts over time
        """
        self.reset()
        
        # Get unique dates
        dates = sorted(flows['date'].unique())
        self.logger.info(f"Running state assignment over {len(dates)} time periods...")
        
        # Initialize states
        all_wallets = list(G.nodes())
        
        if initial_infected is None:
            # Select top buyers as initial infected
            total_buying = flows.groupby('wallet_id')['net_btc'].sum()
            n_initial = max(1, int(len(all_wallets) * initial_infected_fraction))
            top_buyers = total_buying.nlargest(n_initial)
            initial_infected = list(top_buyers.index)
            
        self.logger.info(f"Initial infected: {len(initial_infected)} wallets")
            
        # Initialize all wallets as susceptible
        current_states = {w: State.SUSCEPTIBLE for w in all_wallets}
        
        # Set initial infected
        initial_time = datetime.combine(dates[0], datetime.min.time())
        for w in initial_infected:
            if w in current_states:
                current_states[w] = State.INFECTED
                self.infection_times[w] = initial_time
                self.last_buying_activity[w] = initial_time
                self.state_history[w].append((initial_time, State.INFECTED))
                
        # Track state counts
        state_counts = []
        
        for date in tqdm(dates, desc="Assigning states"):
            current_time = datetime.combine(date, datetime.min.time())
            
            # Assign states
            current_states = self.assign_states_at_time(
                G, flows, current_time, current_states
            )
            
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

        result_df = pd.DataFrame(state_counts)

        # Create a state history DataFrame that includes per-node infection times for H4 test
        self._node_infection_times_df = pd.DataFrame([
            {'node': node, 'infection_time': time}
            for node, time in self.infection_times.items()
        ])

        self.logger.info(
            f"State assignment complete. Final: S={result_df['S'].iloc[-1]}, "
            f"E={result_df['E'].iloc[-1]}, I={result_df['I'].iloc[-1]}, "
            f"R={result_df['R'].iloc[-1]}"
        )

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
        G: nx.Graph
    ) -> Dict[int, int]:
        """
        Compute individual reproduction number (secondary infections caused).
        
        For each infected node, count how many of its neighbors
        became infected after it.
        
        Args:
            G: Transaction graph
            
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
        
        # For each infected wallet, count neighbors infected after
        for wallet, infection_time in infection_times_sorted.items():
            try:
                neighbors = set(G.neighbors(wallet))
                if G.is_directed():
                    neighbors.update(G.predecessors(wallet))  # type: ignore[union-attr]
            except:
                continue
                
            for neighbor in neighbors:
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
        print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
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
