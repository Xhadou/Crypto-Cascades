"""
Epidemic Early Warning Indicators

Implements early warning signals for epidemic onset based on
critical slowing down theory. Increasing variance, autocorrelation,
and skewness of the infected fraction signal an approaching tipping point.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional

from src.utils.logger import get_logger


class EpidemicEarlyWarning:
    """Early warning indicators for epidemic onset."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def compute_ews_indicators(
        self,
        state_history: pd.DataFrame,
        window_size: int = 7,
        alarm_autocorr_threshold: float = 0.7,
        alarm_variance_factor: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute early warning signals based on critical slowing down theory.

        Indicators computed:
        - **Variance**: Rising variance suggests the system is losing resilience.
        - **Autocorrelation (lag-1)**: Increasing autocorrelation indicates
          slower recovery from perturbations.
        - **Skewness**: Positive skewness signals an asymmetric distribution
          of fluctuations, often preceding a transition.
        - **Alarm**: A simple combined threshold on autocorrelation and variance.

        Args:
            state_history: DataFrame with at least an ``I_frac`` column.
            window_size: Rolling window length (days / time-steps).
            alarm_autocorr_threshold: Autocorrelation above which the alarm
                component for lag-1 fires.
            alarm_variance_factor: Multiplier of the running mean variance
                used for the alarm component.

        Returns:
            DataFrame with columns ``t``, ``variance``, ``autocorrelation``,
            ``skewness``, and ``alarm``.
        """
        if 'I_frac' not in state_history.columns:
            self.logger.warning(
                "state_history has no 'I_frac' column — "
                "returning empty indicators"
            )
            return pd.DataFrame(
                columns=['t', 'variance', 'autocorrelation', 'skewness', 'alarm']
            )

        I_values: np.ndarray = np.asarray(state_history['I_frac'].values, dtype=np.float64)

        if len(I_values) <= window_size:
            self.logger.warning(
                f"Time series length ({len(I_values)}) <= window_size "
                f"({window_size}). Returning empty indicators."
            )
            return pd.DataFrame(
                columns=['t', 'variance', 'autocorrelation', 'skewness', 'alarm']
            )

        indicators: List[Dict] = []

        for t in range(window_size, len(I_values)):
            window: np.ndarray = I_values[t - window_size:t]

            # Variance increase (critical slowing down)
            variance = float(np.var(window))

            # Autocorrelation at lag-1
            if len(window) > 1:
                autocorr = float(np.corrcoef(window[:-1], window[1:])[0, 1])
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0

            # Skewness
            skewness = float(stats.skew(window)) if len(window) > 2 else 0.0

            # Simple alarm rule
            mean_I_so_far = float(np.mean(I_values[:t])) if t > 0 else 0.0
            alarm = bool(
                autocorr > alarm_autocorr_threshold
                and variance > mean_I_so_far * alarm_variance_factor
            )

            indicators.append({
                't': t,
                'variance': variance,
                'autocorrelation': autocorr,
                'skewness': skewness,
                'alarm': alarm,
            })

        return pd.DataFrame(indicators)

    def detect_transition_point(
        self,
        ews_df: pd.DataFrame,
        consecutive_alarms: int = 3
    ) -> Optional[int]:
        """
        Detect the earliest time-step with *consecutive_alarms* consecutive
        alarm flags, suggesting the start of an epidemic transition.

        Args:
            ews_df: Output of :meth:`compute_ews_indicators`.
            consecutive_alarms: Number of consecutive alarm flags required.

        Returns:
            Time-step of first sustained alarm, or ``None``.
        """
        if ews_df.empty or 'alarm' not in ews_df.columns:
            return None

        streak = 0
        for _, row in ews_df.iterrows():
            if row['alarm']:
                streak += 1
                if streak >= consecutive_alarms:
                    return int(row['t']) - consecutive_alarms + 1
            else:
                streak = 0

        return None
