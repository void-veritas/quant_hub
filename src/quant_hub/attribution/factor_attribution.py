"""Factor vs idiosyncratic PnL attribution with error bars.

Implements *The Elements of Quantitative Investing* Ch 14: splitting realized PnL
into factor and idiosyncratic components and attaching the confidence intervals
that arise because factor-mimicking-portfolio factor returns are *estimates* -- the
book's insistence that attribution without standard errors misleads.

Shape conventions: ``n`` assets, ``m`` factors, ``T`` periods.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def factor_idio_pnl(weights_ts, factor_returns_ts, idio_returns_ts, loadings):
    """Factor and idiosyncratic PnL summed over time (EQI 14.1).

    Parameters
    ----------
    weights_ts : array_like, shape (T, n)
        Portfolio weights each period.
    factor_returns_ts : array_like, shape (T, m)
        Factor returns each period.
    idio_returns_ts : array_like, shape (T, n)
        Idiosyncratic returns each period.
    loadings : array_like, shape (n, m)
        Factor loadings ``B`` (assumed constant).

    Returns
    -------
    factor_pnl : float
        Total factor PnL.
    idio_pnl : float
        Total idiosyncratic PnL.
    per_factor_pnl : numpy.ndarray, shape (m,)
        Factor PnL attributed to each factor (sums to ``factor_pnl``).
    """
    w = np.atleast_2d(np.asarray(weights_ts, float))
    f = np.atleast_2d(np.asarray(factor_returns_ts, float))
    eps = np.atleast_2d(np.asarray(idio_returns_ts, float))
    b = np.asarray(loadings, float)
    exposures = w @ b  # T x m
    per_factor = np.sum(exposures * f, axis=0)  # m
    factor_pnl = float(per_factor.sum())
    idio_pnl = float(np.sum(eps * w))
    return factor_pnl, idio_pnl, per_factor


def attribution_confidence_interval(
    weights_ts, loadings, factor_cov_noise, confidence: float = 0.95
):
    """Standard error and confidence half-width of the factor (and idio) PnL (EQI Ch 14).

    The attribution variance is ``sum_t b_t' (B'Omega_e^-1 B)^-1 b_t`` with
    ``b_t = B' w_t``; the same magnitude applies to idiosyncratic PnL (they are
    negatively correlated). Check whether zero lies inside the interval before
    claiming skill.

    Parameters
    ----------
    weights_ts : array_like, shape (T, n)
        Portfolio weights each period.
    loadings : array_like, shape (n, m)
        Factor loadings ``B``.
    factor_cov_noise : array_like, shape (m, m)
        FMP estimation-noise covariance (see
        :meth:`~quant_hub.risk.factor_model.FactorModel.factor_covariance_noise`).
    confidence : float, default 0.95
        Coverage probability of the interval.

    Returns
    -------
    std_error : float
        Standard error of the attributed PnL.
    half_width : float
        Half-width of the confidence interval at ``confidence``.
    """
    w = np.atleast_2d(np.asarray(weights_ts, float))
    b = np.asarray(loadings, float)
    noise = np.asarray(factor_cov_noise, float)
    exposures = w @ b  # T x m
    variance = float(np.sum(exposures * (exposures @ noise)))
    std_error = np.sqrt(variance)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    return std_error, z * std_error


def maximal_attribution(factor_model, subset_idx, factor_returns):
    """Maximal PnL attribution to a factor subset ``S`` (EQI Procedure 14.1).

    Pushes the ``S``-explainable part of the complement's exposures into ``S`` via
    the adjusted betas ``A = Omega_{U,S} Omega_{S,S}^-1``, giving the unique
    representation-independent attribution to the chosen factors.

    Parameters
    ----------
    factor_model : quant_hub.risk.factor_model.FactorModel
        The risk model (supplies the factor covariance).
    subset_idx : sequence of int
        Indices of the factors in the subset ``S``.
    factor_returns : array_like, shape (m,) or (m, T)
        Factor returns to attribute.

    Returns
    -------
    numpy.ndarray
        The maximal attribution to each factor in ``S`` (shape ``(|S|,)`` for a
        single period, ``(|S|, T)`` for a panel).
    """
    fcov = factor_model.factor_cov
    s = list(subset_idx)
    omega_ss = fcov[np.ix_(s, s)]
    omega_us = fcov[:, s]  # (m x |S|)
    adj = omega_us @ np.linalg.inv(omega_ss)  # A: maps all factors onto S
    f = np.asarray(factor_returns, float)
    return (adj.T @ f)[: len(s)] if f.ndim == 1 else adj.T @ f
