"""Hedging: remove unrewarded risk to raise the Sharpe ratio.

Implements *The Elements of Quantitative Investing* Ch 12 and *Advanced Portfolio
Management* Ch 11: the minimum-variance hedge ratio, estimation-error shrinkage of
the hedge, and factor hedging via factor-mimicking portfolios. Recurring
conclusion: always hedge somewhat less than the naive beta when betas are estimated.
"""

from __future__ import annotations

import numpy as np


def hedge_ratio(cov_core_hedge: float, var_hedge: float) -> float:
    """Minimum-variance hedge size ``-cov(r_core, r_hedge) / var(r_hedge)`` (EQI Ch 12).

    Equals ``-beta`` from the time-series regression of core on hedge returns.

    Parameters
    ----------
    cov_core_hedge : float
        Covariance between core and hedge returns.
    var_hedge : float
        Variance of the hedge return.

    Returns
    -------
    float
        The optimal hedge size (units of hedge per unit of core).
    """
    return -cov_core_hedge / var_hedge


def hedged_sharpe(sharpe_core: float, correlation: float) -> float:
    """Sharpe after hedging out an alpha-free instrument, ``SR_core / sqrt(1 - rho^2)``.

    Parameters
    ----------
    sharpe_core : float
        Sharpe ratio of the unhedged core position.
    correlation : float
        Correlation ``rho`` between core and hedge returns.

    Returns
    -------
    float
        The hedged Sharpe ratio (strictly higher for ``|rho| > 0``).
    """
    return sharpe_core / np.sqrt(1.0 - correlation**2)


def shrunk_hedge_ratio(weights, beta_hat, beta_error_cov) -> float:
    """Estimation-error shrinkage factor for the aggregate hedge (EQI Ch 12).

    ``gamma* = 1 - w' Omega_eta w / (w' beta_hat)^2``; hedge
    ``-gamma* (w' beta_hat)`` units instead of the naive amount. Over-hedging from
    noisy betas is corrected by shrinking toward zero.

    Parameters
    ----------
    weights : array_like, shape (n,)
        Portfolio weights.
    beta_hat : array_like, shape (n,)
        Estimated per-asset betas to the hedge instrument.
    beta_error_cov : array_like, shape (n, n)
        Covariance ``Omega_eta`` of the beta estimation errors.

    Returns
    -------
    float
        The shrinkage factor ``gamma*`` applied to the naive aggregate hedge.
    """
    w = np.asarray(weights, float)
    beta = np.asarray(beta_hat, float)
    omega_eta = np.atleast_2d(np.asarray(beta_error_cov, float))
    aggregate_beta = w @ beta
    if aggregate_beta == 0:
        return 0.0
    return 1.0 - (w @ omega_eta @ w) / aggregate_beta**2


def factor_hedge_overlay(factor_model, weights):
    """Overlay ``-P b`` that neutralizes all factor exposures (EQI Ch 12).

    Leaves pure idiosyncratic risk by trading the factor-mimicking portfolios.

    Parameters
    ----------
    factor_model : quant_hub.risk.factor_model.FactorModel
        The risk model providing FMPs and exposures.
    weights : array_like, shape (n,)
        The portfolio to hedge.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Hedge weights to add to the portfolio (``-FMP @ exposures``).
    """
    b = factor_model.exposures(weights)
    p = factor_model.factor_mimicking_portfolios()
    return -(p @ b)
