"""Rademacher Anti-Serum: a distribution-free haircut for multiple testing.

Implements *The Elements of Quantitative Investing* Ch 8. When ``N`` strategies are
searched simultaneously, in-sample Sharpes/ICs are inflated. The Rademacher
complexity measures how much the *strategy set* can fit random noise (duplicates do
not inflate it; independent strategies do), giving a finite-sample, distribution-free
lower bound on true performance.

Inputs are ``(T, N)`` matrices: column ``n`` is strategy ``n``'s per-period
standardized returns (or ICs), ``T`` rows of (assumed iid) observations.
"""

from __future__ import annotations

import numpy as np


def rademacher_complexity(returns_matrix, n_draws: int = 1000, seed: int | None = None) -> float:
    """Empirical Rademacher complexity ``E[ max_n (eps' x_n) / T ]`` (EQI Ch 8).

    Averaged over random ``+/-1`` sign vectors ``eps``; depends on the *effective*
    number of strategies, not the nominal ``N``.

    Parameters
    ----------
    returns_matrix : array_like, shape (T, N)
        Standardized per-period strategy returns (or ICs).
    n_draws : int, default 1000
        Number of random sign vectors to average over.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        The empirical Rademacher complexity.
    """
    x = np.atleast_2d(np.asarray(returns_matrix, float))
    n_obs = x.shape[0]
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_draws, n_obs))
    sup = (signs @ x / n_obs).max(axis=1)
    return float(sup.mean())


def massart_bound(n_strategies: int, n_obs: int) -> float:
    """Massart's worst-case bound on the Rademacher complexity, ``sqrt(2 log N) / T``.

    Parameters
    ----------
    n_strategies : int
        Number of strategies ``N``.
    n_obs : int
        Number of observations ``T``.

    Returns
    -------
    float
        The upper bound on the Rademacher complexity.
    """
    return np.sqrt(2.0 * np.log(n_strategies)) / n_obs


def rademacher_haircut_sharpe(
    sharpe_hat: float,
    returns_matrix,
    delta: float = 0.05,
    n_draws: int = 1000,
    seed: int | None = None,
) -> float:
    """Lower bound on true per-period Sharpe after the data-snooping haircut (EQI Ch 8).

    The bound is ``theta >= theta_hat - 2*R_hat - est_term - snoop_term`` with
    ``est_term = 3*sqrt(2 log(2/delta)/T)`` and ``snoop_term = sqrt(2 log(2N/delta)/T)``,
    holding simultaneously over all strategies with probability ``>= 1 - delta`` for
    sub-Gaussian standardized returns.

    Parameters
    ----------
    sharpe_hat : float
        In-sample per-period Sharpe of the selected strategy.
    returns_matrix : array_like, shape (T, N)
        Standardized per-period returns of all searched strategies.
    delta : float, default 0.05
        Failure probability (the bound holds with probability ``1 - delta``).
    n_draws : int, default 1000
        Random draws for the Rademacher complexity.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        The haircut lower bound on the true per-period Sharpe.
    """
    x = np.atleast_2d(np.asarray(returns_matrix, float))
    n_obs, n_strat = x.shape
    r_hat = rademacher_complexity(x, n_draws, seed)
    est_term = 3.0 * np.sqrt(2.0 * np.log(2.0 / delta) / n_obs)
    snoop_term = np.sqrt(2.0 * np.log(2.0 * n_strat / delta) / n_obs)
    return sharpe_hat - 2.0 * r_hat - est_term - snoop_term


def rademacher_haircut_signal(
    ic_hat: float, ic_matrix, delta: float = 0.05, n_draws: int = 1000, seed: int | None = None
) -> float:
    """Lower bound on true IC for bounded signals ``|IC| <= 1`` (EQI Ch 8).

    ``theta_hat - 2*R_hat - 2*sqrt(log(2/delta)/T)``.

    Parameters
    ----------
    ic_hat : float
        In-sample information coefficient of the selected signal.
    ic_matrix : array_like, shape (T, N)
        Per-period ICs of all searched signals.
    delta : float, default 0.05
        Failure probability.
    n_draws : int, default 1000
        Random draws for the Rademacher complexity.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        The haircut lower bound on the true IC.
    """
    x = np.atleast_2d(np.asarray(ic_matrix, float))
    n_obs = x.shape[0]
    r_hat = rademacher_complexity(x, n_draws, seed)
    return ic_hat - 2.0 * r_hat - 2.0 * np.sqrt(np.log(2.0 / delta) / n_obs)
