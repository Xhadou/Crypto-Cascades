"""
Model Comparison Framework

Compare different epidemic models (SIR, SEIR, SEIRS) using
information criteria (AIC, BIC) and goodness-of-fit metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.utils.logger import get_logger


class ModelComparison:
    """Compare different epidemic models using information criteria."""

    MODELS: Dict[str, Dict] = {
        'SIR':  {'n_params': 2, 'states': ['S', 'I', 'R']},
        'SEIR': {'n_params': 3, 'states': ['S', 'E', 'I', 'R']},
        'SEIRS': {'n_params': 4, 'states': ['S', 'E', 'I', 'R']},  # with waning
    }

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def compute_information_criteria(
        self,
        observed: pd.DataFrame,
        predicted: pd.DataFrame,
        n_params: int
    ) -> Dict[str, object]:
        """
        Compute AIC, BIC, and adjusted R² under a Gaussian likelihood.

        Args:
            observed: DataFrame with ``I_frac`` column (ground truth).
            predicted: DataFrame with ``I_frac`` column (model output).
            n_params: Number of free parameters in the model.

        Returns:
            Dict with ``aic``, ``bic``, ``r_squared``, ``r_squared_adj``,
            ``log_likelihood``, ``n_params``, ``n_obs``.
        """
        n = len(observed)
        obs_arr = np.asarray(observed['I_frac'].values, dtype=float)
        pred_arr = np.asarray(predicted['I_frac'].values, dtype=float)
        residuals = obs_arr - pred_arr
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((obs_arr - float(np.mean(obs_arr))) ** 2))

        # Maximum-likelihood variance
        sigma2 = ss_res / n if n > 0 else 1e-10
        sigma2 = max(sigma2, 1e-30)  # avoid log(0)

        # Log-likelihood (Gaussian)
        ll = -n / 2 * (1 + np.log(2 * np.pi * sigma2))

        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n) - 2 * ll

        # R² and adjusted R²
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if n > n_params + 1:
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
        else:
            r2_adj = r2

        return {
            'aic': float(aic),
            'bic': float(bic),
            'r_squared': float(r2),
            'r_squared_adj': float(r2_adj),
            'log_likelihood': float(ll),
            'n_params': n_params,
            'n_obs': n,
        }

    def compare_models(
        self,
        observed: pd.DataFrame,
        predictions: Dict[str, pd.DataFrame],
        n_params_map: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model predictions against observed data.

        Args:
            observed: Observed data with ``I_frac``.
            predictions: ``{model_name: predicted_df, ...}``.
            n_params_map: Optional ``{model_name: n_params}``.
                Falls back to :attr:`MODELS` definitions.

        Returns:
            DataFrame sorted by AIC with one row per model.
        """
        rows: List[Dict] = []
        for name, pred_df in predictions.items():
            n_p: int = (n_params_map or {}).get(
                name, self.MODELS.get(name, {}).get('n_params', 3)
            ) or 3
            metrics = self.compute_information_criteria(
                observed, pred_df, n_p
            )
            metrics['model'] = name
            rows.append(metrics)

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values('aic').reset_index(drop=True)
            result['delta_aic'] = result['aic'] - result['aic'].iloc[0]
        return result

    def select_best_model(
        self,
        comparison_df: pd.DataFrame,
        criterion: str = 'aic'
    ) -> str:
        """
        Return the name of the model with the lowest information criterion.

        Args:
            comparison_df: Output of :meth:`compare_models`.
            criterion: ``'aic'`` or ``'bic'``.

        Returns:
            Model name string.
        """
        if comparison_df.empty:
            return ''
        return str(comparison_df.sort_values(criterion).iloc[0]['model'])
