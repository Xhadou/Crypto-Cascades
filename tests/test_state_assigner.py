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

    def test_valid_s_to_s(self):
        """S can stay S."""
        assert validate_transition(State.SUSCEPTIBLE, State.SUSCEPTIBLE) == True

    def test_valid_s_to_e(self):
        """S can become E (exposed)."""
        assert validate_transition(State.SUSCEPTIBLE, State.EXPOSED) == True

    def test_valid_s_to_i(self):
        """S can become I (direct infection after exposure contact)."""
        assert validate_transition(State.SUSCEPTIBLE, State.INFECTED) == True

    def test_valid_e_to_e(self):
        """E can stay E."""
        assert validate_transition(State.EXPOSED, State.EXPOSED) == True

    def test_valid_e_to_i(self):
        """E can become I."""
        assert validate_transition(State.EXPOSED, State.INFECTED) == True

    def test_valid_i_to_i(self):
        """I can stay I."""
        assert validate_transition(State.INFECTED, State.INFECTED) == True

    def test_valid_i_to_r(self):
        """I can become R."""
        assert validate_transition(State.INFECTED, State.RECOVERED) == True

    def test_valid_r_to_r(self):
        """R can stay R."""
        assert validate_transition(State.RECOVERED, State.RECOVERED) == True

    def test_valid_r_to_s(self):
        """R can become S (immunity wanes)."""
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


class TestValidTransitionsDict:
    """Test the VALID_TRANSITIONS dictionary structure."""

    def test_all_states_have_transitions(self):
        """All states should be in the VALID_TRANSITIONS dict."""
        for state in State:
            assert state in VALID_TRANSITIONS

    def test_all_states_can_stay(self):
        """All states should be able to stay in current state."""
        for state in State:
            assert state in VALID_TRANSITIONS[state]


class TestStateAssignerParameters:
    """Test StateAssigner initialization and parameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        assigner = StateAssigner()
        assert assigner.susceptible_window == timedelta(days=7)
        assert assigner.recovery_window == timedelta(days=3)
        assert assigner.immunity_waning_window == timedelta(days=30)

    def test_custom_parameters(self):
        """Test custom parameter values."""
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
        assert assigner.recovery_window.days == 3
        assert assigner.immunity_waning_window.days == 30

    def test_exposure_window_in_hours(self):
        """Test exposure window is in hours."""
        assigner = StateAssigner(exposure_window_hours=48)
        assert assigner.exposure_window == timedelta(hours=48)


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
        """Create a StateAssigner with test parameters."""
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

    def test_exposed_stays_exposed_if_not_buying(self, assigner, simple_graph):
        """E node stays E if not buying."""
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.EXPOSED,
            is_buying=False,
            G=simple_graph,
            infected_wallets=set(),
            current_time=datetime.now()
        )

        assert new_state == State.EXPOSED

    def test_exposed_becomes_infected_if_buying(self, assigner, simple_graph):
        """E node becomes I if buying."""
        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.EXPOSED,
            is_buying=True,
            G=simple_graph,
            infected_wallets=set(),
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

    def test_recovered_stays_at_boundary(self):
        """R stays R exactly at immunity_waning_window boundary."""
        assigner = StateAssigner(immunity_waning_days=30)

        G = nx.Graph()
        G.add_node(0)

        # Set recovery time to exactly 30 days ago
        recovery_time = datetime.now() - timedelta(days=30)
        assigner.recovery_times[0] = recovery_time

        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.RECOVERED,
            is_buying=False,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )

        # At exactly the boundary, should still be RECOVERED (> not >=)
        assert new_state == State.RECOVERED


class TestRecovery:
    """Test recovery logic separate from immunity waning."""

    def test_infected_stays_infected_if_buying(self):
        """I stays I if still buying."""
        assigner = StateAssigner(recovery_window_days=3)

        G = nx.Graph()
        G.add_node(0)

        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.INFECTED,
            is_buying=True,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )

        assert new_state == State.INFECTED

    def test_infected_stays_infected_during_recovery_window(self):
        """I stays I during recovery window after stopping buying."""
        assigner = StateAssigner(recovery_window_days=3)

        G = nx.Graph()
        G.add_node(0)

        # Last buying was 2 days ago (within recovery window)
        assigner.last_buying_activity[0] = datetime.now() - timedelta(days=2)

        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.INFECTED,
            is_buying=False,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )

        assert new_state == State.INFECTED

    def test_infected_becomes_recovered_after_dormancy(self):
        """I becomes R after recovery_window without buying."""
        assigner = StateAssigner(recovery_window_days=3)

        G = nx.Graph()
        G.add_node(0)

        # Last buying was 5 days ago (past recovery window)
        assigner.last_buying_activity[0] = datetime.now() - timedelta(days=5)

        new_state = assigner._compute_new_state(
            wallet=0,
            prev_state=State.INFECTED,
            is_buying=False,
            G=G,
            infected_wallets=set(),
            current_time=datetime.now()
        )

        assert new_state == State.RECOVERED


class TestGetInfectionTimesDf:
    """Test the get_infection_times_df method."""

    def test_returns_dataframe(self):
        """get_infection_times_df returns a DataFrame."""
        assigner = StateAssigner()
        df = assigner.get_infection_times_df()
        assert isinstance(df, pd.DataFrame)

    def test_empty_before_assignment(self):
        """Returns empty DataFrame before running assignment."""
        assigner = StateAssigner()
        df = assigner.get_infection_times_df()
        assert len(df) == 0
        assert 'node' in df.columns
        assert 'infection_time' in df.columns


class TestReset:
    """Test state reset functionality."""

    def test_reset_clears_all_tracking(self):
        """Reset should clear all state tracking."""
        assigner = StateAssigner()

        # Add some state
        assigner.wallet_states[0] = State.INFECTED
        assigner.infection_times[0] = datetime.now()
        assigner.recovery_times[1] = datetime.now()
        assigner.last_buying_activity[0] = datetime.now()
        assigner.state_history[0].append((datetime.now(), State.INFECTED))

        # Reset
        assigner.reset()

        # Verify all cleared
        assert len(assigner.wallet_states) == 0
        assert len(assigner.infection_times) == 0
        assert len(assigner.recovery_times) == 0
        assert len(assigner.last_buying_activity) == 0
        assert len(assigner.state_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
