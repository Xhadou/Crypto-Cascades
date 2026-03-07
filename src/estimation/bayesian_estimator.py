"""Bayesian SEIR parameter estimation using NumPyro (optional).

This module provides a Bayesian alternative to the frequentist estimators
in ``estimator.py``.  It uses NumPyro's NUTS (No-U-Turn Sampler) to draw
posterior samples for the SEIR parameters (beta, sigma, gamma, omega) and
returns an ``EstimationResult`` with posterior medians and 95% credible
intervals.

NumPyro and JAX are **optional** dependencies.  If they are not installed,
``HAS_NUMPYRO`` is ``False`` and instantiating ``BayesianEstimator`` raises
a clear ``ImportError`` with installation instructions.
"""

import numpy as np
from src.estimation.estimator import EstimationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    HAS_NUMPYRO = True

    # Auto-detect GPU; fall back to CPU silently
    try:
        _gpu_available = len(jax.devices("gpu")) > 0
    except RuntimeError:
        _gpu_available = False

    if _gpu_available:
        numpyro.set_platform("gpu")
        logger.info("JAX GPU backend detected — using GPU for MCMC")
    else:
        numpyro.set_platform("cpu")
        logger.debug("No GPU detected — using CPU for MCMC")
except ImportError:
    HAS_NUMPYRO = False


class BayesianEstimator:
    """Bayesian SEIR parameter estimation using the NUTS sampler.

    Fits the four SEIR parameters (beta, sigma, gamma, omega) by running
    Hamiltonian Monte Carlo via NumPyro.  The forward model is a simple
    Euler discretisation of the SEIR ODEs (JAX-compatible so that NUTS
    can compute gradients).  Observations at each time step are modelled
    as Dirichlet-distributed compartment fractions.

    Parameters
    ----------
    num_warmup : int
        Number of MCMC warm-up (adaptation) steps.
    num_samples : int
        Number of posterior samples to draw after warm-up.
    num_chains : int
        Number of independent MCMC chains.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 2000,
        num_chains: int = 1,
        random_seed: int = 42,
    ):
        if not HAS_NUMPYRO:
            raise ImportError(
                "NumPyro and JAX are required for Bayesian estimation. "
                "Install with: pip install numpyro jax jaxlib"
            )
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.random_seed = random_seed
        logger.info(
            "BayesianEstimator initialised "
            "(warmup=%d, samples=%d, chains=%d, seed=%d)",
            num_warmup,
            num_samples,
            num_chains,
            random_seed,
        )

    # ------------------------------------------------------------------
    # JAX-compatible SEIR forward model
    # ------------------------------------------------------------------

    @staticmethod
    def _seir_step(state, beta, sigma, gamma, omega, N):
        """Single Euler step for the SEIR ODEs (JAX-traceable).

        Parameters
        ----------
        state : jnp.ndarray
            Current [S, E, I, R] counts.
        beta, sigma, gamma, omega : float
            SEIR rate parameters.
        N : float
            Total population (kept constant).

        Returns
        -------
        jnp.ndarray
            Updated [S, E, I, R] counts after one time unit.
        """
        S, E, I, R = state
        dS = -beta * S * I / N + omega * R
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I - omega * R
        return jnp.array([S + dS, E + dE, I + dI, R + dR])

    @staticmethod
    def _solve_seir_jax(beta, sigma, gamma, omega, y0, N, T):
        """Forward-solve the SEIR system for *T* time steps.

        Uses simple Euler integration so the entire computation stays
        inside JAX and is differentiable for NUTS.

        Parameters
        ----------
        beta, sigma, gamma, omega : float
            SEIR rate parameters.
        y0 : jnp.ndarray
            Initial [S, E, I, R] fractions (summing to 1).
        N : float
            Total population.
        T : int
            Number of time steps (including the initial state).

        Returns
        -------
        jnp.ndarray
            Array of shape ``(T, 4)`` with compartment fractions at each
            time step.
        """
        state = y0 * N  # work in counts for numerical stability

        states = [y0]
        for _ in range(T - 1):
            state = BayesianEstimator._seir_step(
                state, beta, sigma, gamma, omega, N
            )
            state = jnp.clip(state, 0.0, N)
            states.append(state / N)

        return jnp.stack(states)

    # ------------------------------------------------------------------
    # NumPyro probabilistic model
    # ------------------------------------------------------------------

    @staticmethod
    def _model(obs_matrix, N):
        """NumPyro probabilistic model for SEIR estimation.

        Priors
        ------
        * beta  ~ Beta(2, 5)        — transmission rate, mode around 0.2
        * sigma ~ Gamma(2, 10)      — incubation rate
        * gamma ~ Gamma(1, 10)      — recovery rate
        * omega ~ Beta(1, 50)       — waning-immunity rate (small)

        Likelihood
        ----------
        Observed compartment fractions at each time step follow a
        Dirichlet distribution whose mean is the simulated fraction
        vector and whose concentration controls observation noise.

        Parameters
        ----------
        obs_matrix : jnp.ndarray
            Array of shape ``(T, 4)`` with observed [S, E, I, R]
            fractions at each time step.
        N : float
            Total population size.
        """
        beta = numpyro.sample("beta", dist.Beta(2, 5))
        sigma = numpyro.sample("sigma", dist.Gamma(2, 10))
        gamma = numpyro.sample("gamma", dist.Gamma(1, 10))
        omega = numpyro.sample("omega", dist.Beta(1, 50))

        T = obs_matrix.shape[0]

        # Use the first observed row as initial condition
        y0 = obs_matrix[0]
        predicted = BayesianEstimator._solve_seir_jax(
            beta, sigma, gamma, omega, y0, N, T
        )

        # Observation noise concentration
        concentration = numpyro.sample("concentration", dist.Gamma(10, 0.1))

        # Dirichlet likelihood for each time step
        for t in range(T):
            alpha_t = predicted[t] * concentration + 1e-6
            numpyro.sample(f"obs_{t}", dist.Dirichlet(alpha_t), obs=obs_matrix[t])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, obs_fracs, N=5000, fgi_values=None):
        """Run MCMC estimation and return an ``EstimationResult``.

        Parameters
        ----------
        obs_fracs : pandas.DataFrame
            DataFrame with columns ``S_frac``, ``E_frac``, ``I_frac``,
            ``R_frac`` giving compartment fractions at each time step.
        N : int
            Total population size.
        fgi_values : array-like or None
            Fear & Greed Index values (currently unused; reserved for
            future FOMO-coupling extensions).

        Returns
        -------
        EstimationResult
            Result with posterior medians as point estimates, 95%
            credible intervals, and the raw posterior samples attached
            as ``result.posterior_samples``.
        """
        # Build observation matrix (T x 4) as a JAX array
        frac_cols = ["S_frac", "E_frac", "I_frac", "R_frac"]
        obs_matrix = jnp.array(obs_fracs[frac_cols].values, dtype=jnp.float32)

        logger.info(
            "Starting MCMC: %d warmup + %d samples, %d chain(s), "
            "T=%d time steps, N=%d",
            self.num_warmup,
            self.num_samples,
            self.num_chains,
            obs_matrix.shape[0],
            N,
        )

        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(self.random_seed),
            obs_matrix=obs_matrix,
            N=float(N),
        )
        samples = mcmc.get_samples()

        # Posterior summaries
        def _median(key):
            return float(np.median(np.asarray(samples[key])))

        def _ci(key, lo=2.5, hi=97.5):
            arr = np.asarray(samples[key])
            return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

        result = EstimationResult(
            beta=_median("beta"),
            sigma=_median("sigma"),
            gamma=_median("gamma"),
            omega=_median("omega"),
            beta_ci=_ci("beta"),
            sigma_ci=_ci("sigma"),
            gamma_ci=_ci("gamma"),
            omega_ci=_ci("omega"),
            success=True,
            message="Bayesian estimation via NumPyro NUTS",
        )

        # Attach raw posterior samples for downstream diagnostics
        result.posterior_samples = {
            k: np.asarray(v) for k, v in samples.items()
        }

        logger.info(
            "MCMC complete: beta=%.4f [%.4f, %.4f], sigma=%.4f, "
            "gamma=%.4f, omega=%.4f, R0=%.3f",
            result.beta,
            result.beta_ci[0],
            result.beta_ci[1],
            result.sigma,
            result.gamma,
            result.omega,
            result.r0(),
        )

        return result
