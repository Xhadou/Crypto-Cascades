"""Rolling-window R0 estimation for strengthening the three-period design.

Instead of estimating a single R0 per period, this module fits SEIR
parameters in overlapping windows across the time span and returns
a time-indexed DataFrame showing how R0 evolves within each period.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.estimation.estimator import ParameterEstimator
from src.utils.exceptions import ConfigurationError, InsufficientDataError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_rolling_r0(
    state_data: pd.DataFrame,
    window_days: int = 30,
    step_days: int = 7,
    N: int = 5000,
    fgi_values: Optional[np.ndarray] = None,
    method: str = 'lsq',
    n_bootstrap: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Estimate R0 in overlapping windows across the full time span.

    Slides a window of *window_days* rows across *state_data*, advancing
    by *step_days* each iteration.  In each window the full SEIR parameter
    set is re-estimated and R0 = beta / gamma is recorded.

    Args:
        state_data: DataFrame with columns ``date``, ``S_frac``, ``E_frac``,
            ``I_frac``, ``R_frac``.  If ``date`` is absent the DataFrame
            index is used for window start/end labels.
        window_days: Number of rows (days) per estimation window.
        step_days: Number of rows to advance between consecutive windows.
        N: Population size passed to :class:`ParameterEstimator`.
        fgi_values: Optional Fear & Greed Index array aligned with
            *state_data* rows.
        method: Estimation method (``'lsq'`` or ``'mle'``).
        n_bootstrap: Bootstrap resamples inside each window estimation
            (set to 0 for speed).
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``window_start``, ``window_end``, ``R0``,
        ``beta``, ``sigma``, ``gamma``, ``omega``.  Each row corresponds
        to one successfully estimated window.
    """
    if window_days < 10:
        raise ConfigurationError(
            key="window_days",
            reason=(
                f"{window_days} is too small; need at least 10 "
                "data points for a meaningful SEIR fit."
            ),
        )

    if len(state_data) < window_days:
        raise InsufficientDataError(
            required=window_days,
            available=len(state_data),
            data_type="rows",
        )

    estimator = ParameterEstimator(
        method=method, n_bootstrap=n_bootstrap, random_seed=random_seed
    )

    has_date_col = 'date' in state_data.columns
    dates = state_data['date'] if has_date_col else state_data.index

    results = []

    for start_idx in range(0, len(state_data) - window_days + 1, step_days):
        end_idx = start_idx + window_days
        window = state_data.iloc[start_idx:end_idx].reset_index(drop=True)
        fgi_window = (
            fgi_values[start_idx:end_idx] if fgi_values is not None else None
        )

        try:
            est = estimator.estimate(
                window, N=N, fgi_values=fgi_window
            )
            if est.success:
                r0 = est.beta / est.gamma if est.gamma > 0 else np.inf
                results.append({
                    'window_start': dates.iloc[start_idx],
                    'window_end': dates.iloc[end_idx - 1],
                    'R0': r0,
                    'beta': est.beta,
                    'sigma': est.sigma,
                    'gamma': est.gamma,
                    'omega': est.omega,
                })
        except Exception as e:
            logger.debug(f"Window {start_idx}-{end_idx} failed: {e}")
            continue

    if not results:
        logger.warning("No windows produced successful estimates.")

    return pd.DataFrame(results)
