"""Mean-variance portfolio construction.

Implements the construction material of *The Elements of Quantitative Investing*
Ch 9-10 and *Advanced Portfolio Management* Ch 11: the mean-variance-optimal
solution in its equivalent forms, the volatility/correlation form ``v* ~ C^-1 s``,
minimum-variance and robust (covariance-inflation) variants, and factor-neutral
proportional weights.

Functions take an asset covariance ``cov`` (e.g. from
:meth:`~quant_hub.risk.factor_model.FactorModel.asset_covariance`) and solve with
``numpy.linalg.solve`` rather than explicit inverses.
"""

from __future__ import annotations

import numpy as np


def max_sharpe(alpha, cov) -> float:
    """Ex-ante Sharpe ratio of the MVO portfolio, ``sqrt(alpha' Omega^-1 alpha)``.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    cov : array_like, shape (n, n)
        Asset covariance ``Omega``.

    Returns
    -------
    float
        The maximum attainable Sharpe ratio.
    """
    a = np.asarray(alpha, float)
    return float(np.sqrt(a @ np.linalg.solve(np.asarray(cov, float), a)))


def mean_variance_weights(
    alpha, cov, vol_target: float | None = None, risk_aversion: float | None = None
):
    """Mean-variance-optimal weights ``w*`` (EQI Ch 9).

    With ``vol_target`` the solution is
    ``w* = (sigma / sqrt(alpha' Omega^-1 alpha)) Omega^-1 alpha`` (only relative
    alphas matter). With ``risk_aversion`` ``rho`` it is ``Omega^-1 alpha / rho``.
    With neither, the unnormalized direction ``Omega^-1 alpha``.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    cov : array_like, shape (n, n)
        Asset covariance ``Omega``.
    vol_target : float, optional
        Target portfolio volatility ``sigma``.
    risk_aversion : float, optional
        Risk-aversion ``rho`` (ignored when ``vol_target`` is given).

    Returns
    -------
    numpy.ndarray, shape (n,)
        The optimal weights.
    """
    a = np.asarray(alpha, float)
    omega_inv_a = np.linalg.solve(np.asarray(cov, float), a)
    if vol_target is not None:
        return vol_target / np.sqrt(a @ omega_inv_a) * omega_inv_a
    if risk_aversion is not None:
        return omega_inv_a / risk_aversion
    return omega_inv_a


def mvo_vol_correlation_form(sharpes, correlation):
    """Optimal dollar volatilities ``v* ~ C^-1 s`` from asset Sharpes (EQI Ch 9).

    Uncorrelated assets give ``v_i ~ s_i`` (Sharpes add in squares); a
    positive-Sharpe asset can be shorted when ``s_i / s_j < rho`` (hedging role).

    Parameters
    ----------
    sharpes : array_like, shape (n,)
        Per-asset Sharpe ratios ``s``.
    correlation : array_like, shape (n, n)
        Asset return correlation matrix ``C``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Optimal dollar volatilities (unnormalized).
    """
    s = np.asarray(sharpes, float)
    return np.linalg.solve(np.asarray(correlation, float), s)


def min_variance_weights(cov, expected=None):
    """Minimum-variance portfolio under a single linear constraint ``b' w = 1``.

    Parameters
    ----------
    cov : array_like, shape (n, n)
        Asset covariance ``Omega``.
    expected : array_like, shape (n,), optional
        The constraint vector ``b``. Defaults to all-ones (the classic global
        minimum-variance portfolio); pass expected returns for the min-variance
        unit-expected-return portfolio.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The portfolio weights, normalized so ``b' w = 1``.
    """
    omega = np.asarray(cov, float)
    b = np.ones(omega.shape[0]) if expected is None else np.asarray(expected, float)
    x = np.linalg.solve(omega, b)
    return x / (b @ x)


def robust_mvo_weights(
    alpha, cov, alpha_uncertainty, vol_target: float | None = None, risk_aversion: float = 1.0
):
    """Robust MVO that inflates the covariance by alpha uncertainty (EQI Ch 10).

    Reduces to plain MVO as the uncertainty goes to zero and toward equal weight as
    it grows -- the "uncertainty inflates covariance" shrinkage.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns.
    cov : array_like, shape (n, n)
        Asset covariance ``Omega``.
    alpha_uncertainty : float or array_like, shape (n, n)
        Alpha covariance ``Omega_alpha``; a scalar ``tau^2`` means ``tau^2 I``.
    vol_target : float, optional
        Target portfolio volatility.
    risk_aversion : float, default 1.0
        Risk-aversion used when ``vol_target`` is not given.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The robust optimal weights.
    """
    a = np.asarray(alpha, float)
    omega = np.asarray(cov, float)
    if np.isscalar(alpha_uncertainty):
        omega_alpha = float(alpha_uncertainty) * np.eye(len(a))
    else:
        omega_alpha = np.asarray(alpha_uncertainty, float)
    inflated = omega + omega_alpha
    if vol_target is not None:
        return mean_variance_weights(a, inflated, vol_target=vol_target)
    return mean_variance_weights(a, inflated, risk_aversion=risk_aversion)


def factor_neutral_weights(alpha, loadings, gross: float = 1.0):
    """Proportional weights with zero factor exposure (APM Procedure 6.3 / 11.4).

    Regresses alpha on the loadings, keeps the residual
    ``(I - B(B'B)^-1 B') alpha``, and normalizes to the target gross exposure --
    minimal distortion of convictions subject to ``B' w = 0``.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Expected returns / convictions.
    loadings : array_like, shape (n, m)
        Factor loadings ``B``.
    gross : float, default 1.0
        Target gross exposure ``sum(|w|)``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Factor-neutral weights scaled to ``gross``.
    """
    a = np.asarray(alpha, float)
    b = np.atleast_2d(np.asarray(loadings, float))
    residual = a - b @ np.linalg.lstsq(b, a, rcond=None)[0]
    denom = np.sum(np.abs(residual))
    return gross * residual / denom if denom > 0 else residual
