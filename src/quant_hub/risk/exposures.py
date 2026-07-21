"""Risk decomposition and exposure reporting.

Implements *Advanced Portfolio Management* Ch 3/7: the risk-decomposition report
(%Var, $Vol, MCFR per factor) a PM watches, portfolio beta, and the dollar-beta
arithmetic. Operates on a :class:`~quant_hub.risk.factor_model.FactorModel` and a
weight / net-market-value vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_beta(nmv, betas) -> float:
    """Portfolio dollar beta ``sum(NMV_i * beta_i)`` -- additive across positions (APM Ch 3).

    Parameters
    ----------
    nmv : array_like, shape (n,)
        Net market value of each position.
    betas : array_like, shape (n,)
        Per-position beta to the market.

    Returns
    -------
    float
        The portfolio dollar beta.
    """
    return float(np.sum(np.asarray(nmv, float) * np.asarray(betas, float)))


def percent_beta(nmv, betas) -> float:
    """Dollar beta divided by net market value (APM Ch 3).

    Parameters
    ----------
    nmv : array_like, shape (n,)
        Net market value of each position.
    betas : array_like, shape (n,)
        Per-position beta.

    Returns
    -------
    float
        The percentage beta, or ``nan`` if the book is dollar-neutral.
    """
    net = np.sum(np.asarray(nmv, float))
    return dollar_beta(nmv, betas) / net if net != 0 else np.nan


def portfolio_beta(weights, hedge_weights, cov) -> float:
    """Beta of portfolio ``w`` to instrument ``v``: ``w' Omega v / v' Omega v`` (APM Ch 11).

    Parameters
    ----------
    weights : array_like, shape (n,)
        Portfolio weights ``w``.
    hedge_weights : array_like, shape (n,)
        The reference instrument ``v`` (e.g. an FMP or index replication).
    cov : array_like, shape (n, n)
        Asset covariance ``Omega``.

    Returns
    -------
    float
        The regression beta of ``w`` on ``v``.
    """
    w = np.asarray(weights, float)
    v = np.asarray(hedge_weights, float)
    omega = np.asarray(cov, float)
    denom = v @ omega @ v
    return float(w @ omega @ v / denom) if denom > 0 else np.nan


def risk_decomposition(
    factor_model, weights, factor_names=None, annualization: float = 1.0
) -> pd.DataFrame:
    """Per-factor risk-decomposition report (APM Ch 7.1).

    Parameters
    ----------
    factor_model : quant_hub.risk.factor_model.FactorModel
        The risk model.
    weights : array_like, shape (n,)
        Portfolio weights.
    factor_names : sequence of str, optional
        Names for the factor rows; defaults to ``f0, f1, ...``.
    annualization : float, default 1.0
        Periods per year; volatilities are scaled by its square root.

    Returns
    -------
    pandas.DataFrame
        One row per factor with columns ``exposure`` (``b = B'w``), ``factor_vol``,
        ``dollar_vol`` (exposure * factor_vol), ``pct_var`` (share of total variance),
        and ``mcfr`` (marginal contribution to factor risk). A trailing ``idio`` row
        carries the idiosyncratic variance share.
    """
    b = factor_model.exposures(weights)
    fcov = factor_model.factor_cov
    total_var = factor_model.portfolio_variance(weights)
    mcfr = factor_model.marginal_contribution_to_factor_risk(weights)
    factor_vols = np.sqrt(np.diag(fcov))
    per_factor_var = b * (fcov @ b)  # b_i * (Omega_f b)_i sums to the factor variance
    names = factor_names if factor_names is not None else [f"f{i}" for i in range(len(b))]
    rows = {
        "exposure": b,
        "factor_vol": factor_vols * np.sqrt(annualization),
        "dollar_vol": b * factor_vols * np.sqrt(annualization),
        "pct_var": per_factor_var / total_var if total_var > 0 else np.zeros_like(b),
        "mcfr": mcfr,
    }
    df = pd.DataFrame(rows, index=names)
    df.loc["idio", "pct_var"] = factor_model.pct_idio_variance(weights)
    return df
