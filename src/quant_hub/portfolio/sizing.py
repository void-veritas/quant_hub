"""Position sizing heuristics and Kelly capital allocation.

Implements the sizing material of *Advanced Portfolio Management* Ch 6 and *The
Elements of Quantitative Investing* Ch 13: the sizing rules from APM's horse race
(proportional wins under estimation error), volatility targeting, and the Kelly
family (continuous, binary, fractional, growth rate, uncertainty-adjusted).

Sizing rules return market values / weights scaled to a target gross exposure
``gross`` unless noted; the Kelly helpers return a capital fraction.
"""

from __future__ import annotations

import numpy as np


def _scale_to_gross(raw, gross):
    """Scale a raw size vector so that ``sum(|size|) == gross``."""
    total = np.sum(np.abs(raw))
    return gross * raw / total if total > 0 else raw


def proportional_size(alpha, gross: float = 1.0):
    """Size proportional to expected return -- APM's empirical winner.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns / convictions.
    gross : float, default 1.0
        Target gross exposure.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sizes scaled to ``gross``.
    """
    return _scale_to_gross(np.asarray(alpha, float), gross)


def risk_parity_size(alpha, idio_vol, gross: float = 1.0):
    """Size proportional to ``alpha / sigma`` (equal dollar idio vol per conviction).

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    idio_vol : array_like, shape (n,)
        Per-asset idiosyncratic volatilities.
    gross : float, default 1.0
        Target gross exposure.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sizes scaled to ``gross``.
    """
    return _scale_to_gross(np.asarray(alpha, float) / np.asarray(idio_vol, float), gross)


def mean_variance_size(alpha, idio_vol, gross: float = 1.0):
    """Size proportional to ``alpha / sigma^2`` (naive MV).

    Loses the most Sharpe under estimation error (APM Ch 6), because inverse-variance
    scaling concentrates risk in low-vol names where alpha error is costliest.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    idio_vol : array_like, shape (n,)
        Per-asset idiosyncratic volatilities.
    gross : float, default 1.0
        Target gross exposure.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sizes scaled to ``gross``.
    """
    return _scale_to_gross(np.asarray(alpha, float) / np.asarray(idio_vol, float) ** 2, gross)


def shrunk_mv_size(alpha, idio_vol, sector_vol, shrink: float, gross: float = 1.0):
    """Shrunk mean-variance size ``alpha / (p sigma^2 + (1-p) sigma_sector^2)`` (APM Ch 6).

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    idio_vol : array_like, shape (n,)
        Per-asset idiosyncratic volatilities.
    sector_vol : array_like, shape (n,)
        Sector (group) volatilities used as the shrinkage target.
    shrink : float
        Shrinkage weight ``p`` in ``[0, 1]`` on the idiosyncratic variance.
    gross : float, default 1.0
        Target gross exposure.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sizes scaled to ``gross``.
    """
    denom = (
        shrink * np.asarray(idio_vol, float) ** 2
        + (1.0 - shrink) * np.asarray(sector_vol, float) ** 2
    )
    return _scale_to_gross(np.asarray(alpha, float) / denom, gross)


def vol_target_scale(weights, cov, target_vol: float):
    """Rescale weights so predicted portfolio volatility equals ``target_vol`` (APM Ch 6.6).

    Parameters
    ----------
    weights : array_like, shape (n,)
        Portfolio weights.
    cov : array_like, shape (n, n)
        Asset covariance.
    target_vol : float
        Desired portfolio volatility.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The rescaled weights.
    """
    w = np.asarray(weights, float)
    current = np.sqrt(w @ np.asarray(cov, float) @ w)
    return w * target_vol / current if current > 0 else w


def kelly_fraction(mu: float, sigma: float) -> float:
    """Growth-optimal capital fraction ``mu / (mu^2 + sigma^2)`` (EQI 13.9).

    Parameters
    ----------
    mu : float
        Expected per-period return.
    sigma : float
        Per-period return volatility.

    Returns
    -------
    float
        The full-Kelly fraction of capital to deploy.
    """
    return mu / (mu**2 + sigma**2)


def kelly_fraction_highvol(sharpe: float, sigma: float) -> float:
    """Kelly fraction in the ``mu << sigma`` regime, ``SR / sigma`` (EQI 13.10).

    Parameters
    ----------
    sharpe : float
        Per-period Sharpe ratio.
    sigma : float
        Per-period volatility.

    Returns
    -------
    float
        The approximate Kelly fraction.
    """
    return sharpe / sigma


def kelly_binary(prob_win: float, payoff_win: float, payoff_loss: float) -> float:
    """Kelly stake for a binary bet, ``p / loss - q / win`` (EQI Example 13.3).

    Parameters
    ----------
    prob_win : float
        Probability of winning ``p`` (loss probability is ``1 - p``).
    payoff_win : float
        Gain per unit staked on a win.
    payoff_loss : float
        Loss per unit staked on a loss (a positive number).

    Returns
    -------
    float
        The optimal stake as a fraction of capital; ``<= 0`` means do not bet.
    """
    return prob_win / payoff_loss - (1.0 - prob_win) / payoff_win


def fractional_kelly(mu: float, sigma: float, fraction: float) -> float:
    """A fixed fraction (< 1) of full Kelly (EQI Ch 13).

    Equivalent to higher risk aversion; trades growth for lower volatility of wealth.

    Parameters
    ----------
    mu : float
        Expected per-period return.
    sigma : float
        Per-period volatility.
    fraction : float
        Fraction of full Kelly to deploy (e.g. 0.5 for "half Kelly").

    Returns
    -------
    float
        The fractional-Kelly capital fraction.
    """
    return fraction * kelly_fraction(mu, sigma)


def kelly_growth_rate(sharpe: float) -> float:
    """Expected log-growth at full Kelly, ``0.5 SR^2 / (SR^2 + 1)`` (EQI 13.9).

    Parameters
    ----------
    sharpe : float
        Per-period Sharpe ratio.

    Returns
    -------
    float
        The expected per-period log-growth rate.
    """
    return 0.5 * sharpe**2 / (sharpe**2 + 1.0)


def kelly_with_uncertainty(mu: float, sigma: float, param_variance: float) -> float:
    """Kelly under parameter uncertainty, ``mu / (mu^2 + sigma^2 + tau^2)`` (EQI Ch 13).

    Estimation variance ``param_variance`` adds to return variance in the
    denominator -- the same shrinkage as robust MVO (Thorp).

    Parameters
    ----------
    mu : float
        Expected per-period return.
    sigma : float
        Per-period volatility.
    param_variance : float
        Variance ``tau^2`` of the return-parameter estimate.

    Returns
    -------
    float
        The uncertainty-adjusted Kelly fraction (always <= full Kelly).
    """
    return mu / (mu**2 + sigma**2 + param_variance)
