"""Leverage bounds and strategic risk limits.

Implements *Advanced Portfolio Management* Ch 7/10: the leverage floor/ceiling and
the feasibility check ("if floor > ceiling the fund does not work"), the
Sharpe-maximizing market-exposure share, and the single-stock and single-factor
caps. The minimal model has ``n`` equal idiosyncratic-vol positions with per-name
Sharpe ``s``, so portfolio vol / GMV = ``sigma / sqrt(n)``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def leverage_floor(min_return: float, n_positions: int, idio_vol: float, sharpe: float) -> float:
    """Minimum leverage needed to hit a return target (APM Ch 10).

    ``L >= (sqrt(n) / sigma) * min_return / s``.

    Parameters
    ----------
    min_return : float
        Required expected return on AUM.
    n_positions : int
        Number of (equal-vol) positions ``n``.
    idio_vol : float
        Per-position idiosyncratic volatility ``sigma``.
    sharpe : float
        Per-position Sharpe ratio ``s``.

    Returns
    -------
    float
        The minimum feasible leverage.
    """
    return np.sqrt(n_positions) / idio_vol * min_return / sharpe


def leverage_ceiling(
    max_drawdown: float,
    n_positions: int,
    idio_vol: float,
    sharpe: float,
    loss_prob: float = 0.025,
    vol_shock: float = 2.0,
) -> float:
    """Maximum leverage consistent with a drawdown limit (APM Ch 10).

    ``L <= (sqrt(n) / sigma) * max_drawdown / (-kappa * z(loss_prob) - s)``, where
    ``z`` is the normal quantile and ``kappa`` the vol-shock multiplier (realized vol
    in a drawdown exceeds predicted).

    Parameters
    ----------
    max_drawdown : float
        Maximum tolerable drawdown (a positive fraction).
    n_positions : int
        Number of positions ``n``.
    idio_vol : float
        Per-position idiosyncratic volatility ``sigma``.
    sharpe : float
        Per-position Sharpe ratio ``s``.
    loss_prob : float, default 0.025
        Tail probability at which the drawdown limit must hold.
    vol_shock : float, default 2.0
        Multiplier ``kappa`` on predicted vol during stress.

    Returns
    -------
    float
        The maximum feasible leverage; ``inf`` if the denominator is non-positive.
    """
    z = stats.norm.ppf(loss_prob)
    denom = -vol_shock * z - sharpe
    if denom <= 0:
        return np.inf
    return np.sqrt(n_positions) / idio_vol * max_drawdown / denom


def leverage_band(
    min_return: float,
    max_drawdown: float,
    n_positions: int,
    idio_vol: float,
    sharpe: float,
    loss_prob: float = 0.025,
    vol_shock: float = 2.0,
):
    """Leverage floor, ceiling, and feasibility (APM Ch 10).

    Parameters
    ----------
    min_return, max_drawdown, n_positions, idio_vol, sharpe : float
        See :func:`leverage_floor` and :func:`leverage_ceiling`.
    loss_prob : float, default 0.025
        Tail probability for the ceiling.
    vol_shock : float, default 2.0
        Vol-shock multiplier for the ceiling.

    Returns
    -------
    tuple
        ``(floor, ceiling, feasible)`` where ``feasible`` is False when the floor
        exceeds the ceiling (the strategy is not sustainable at these parameters).
    """
    floor = leverage_floor(min_return, n_positions, idio_vol, sharpe)
    ceiling = leverage_ceiling(max_drawdown, n_positions, idio_vol, sharpe, loss_prob, vol_shock)
    return floor, ceiling, floor <= ceiling


def market_variance_share(sharpe_portfolio: float, sharpe_market: float) -> float:
    """Sharpe-optimal share of variance from market exposure (APM Ch 7.2.2).

    ``1 / (1 + (SR_portfolio / SR_market)^2)``.

    Parameters
    ----------
    sharpe_portfolio : float
        Sharpe ratio of the market-neutral book.
    sharpe_market : float
        Sharpe ratio of the market.

    Returns
    -------
    float
        The optimal fraction of total variance to source from market exposure.
    """
    return 1.0 / (1.0 + (sharpe_portfolio / sharpe_market) ** 2)


def market_gmv_share(
    sharpe_portfolio: float, sharpe_market: float, vol_portfolio: float, vol_market: float
) -> float:
    """GMV share to allocate to the market (APM Ch 7.2.2).

    ``1 / (1 + sigma_market * SR_portfolio / (sigma_portfolio * SR_market))``.

    Parameters
    ----------
    sharpe_portfolio : float
        Sharpe ratio of the market-neutral book.
    sharpe_market : float
        Sharpe ratio of the market.
    vol_portfolio : float
        Volatility of the market-neutral book.
    vol_market : float
        Volatility of the market.

    Returns
    -------
    float
        The optimal share of gross market value in the market instrument.
    """
    ratio = vol_market * sharpe_portfolio / (vol_portfolio * sharpe_market)
    return 1.0 / (1.0 + ratio)


def single_stock_cap(conviction_profitability_ratio: float, n_stocks: int) -> float:
    """Maximum single-name weight ``CPR^2 / n_stocks`` (APM Ch 7.2.3).

    Parameters
    ----------
    conviction_profitability_ratio : float
        Realized alpha of high- vs low-conviction ideas (``<= 2`` even for the best).
    n_stocks : int
        Number of names in the book.

    Returns
    -------
    float
        The maximum sensible weight for a single position.
    """
    return conviction_profitability_ratio**2 / n_stocks


def single_factor_cap(worst_case_factor_return: float, tolerable_loss: float) -> float:
    """Maximum factor exposure / GMV from a worst-case factor return (APM Ch 7.2.4).

    ``tolerable_loss / |worst_case_factor_return|`` -- a static backstop for
    low-predicted-vol, high-tail-risk factors whose vol models break exactly when
    they matter.

    Parameters
    ----------
    worst_case_factor_return : float
        Subjective worst-case factor return (e.g. -0.05 per month).
    tolerable_loss : float
        Maximum portfolio loss attributable to this factor.

    Returns
    -------
    float
        The maximum exposure as a fraction of GMV.
    """
    return tolerable_loss / abs(worst_case_factor_return)
