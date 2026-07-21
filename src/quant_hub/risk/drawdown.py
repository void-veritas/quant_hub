"""Drawdown measurement and drawdown-aware sizing.

Implements *The Elements of Quantitative Investing* Ch 13 and *Advanced Portfolio
Management* Ch 9: high-watermark drawdowns, the Grossman-Zhou drawdown-constrained
Kelly fraction (which guarantees a maximum drawdown ex ante), and a simple
stop-loss policy.
"""

from __future__ import annotations

import numpy as np


def high_watermark(equity):
    """Running maximum of an equity curve.

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.

    Returns
    -------
    numpy.ndarray, shape (T,)
        The high-watermark path.
    """
    return np.maximum.accumulate(np.asarray(equity, float))


def drawdown_series(equity):
    """Drawdown path ``d_t = 1 - W_t / max_{s<=t} W_s`` (values >= 0).

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.

    Returns
    -------
    numpy.ndarray, shape (T,)
        The drawdown from the high watermark at each point.
    """
    e = np.asarray(equity, float)
    return 1.0 - e / np.maximum.accumulate(e)


def max_drawdown(equity) -> float:
    """Largest drawdown over an equity curve.

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.

    Returns
    -------
    float
        The maximum drawdown (a positive fraction).
    """
    return float(drawdown_series(equity).max())


def grossman_zhou_fraction(
    mu: float, sigma: float, max_drawdown_limit: float, current_drawdown: float
) -> float:
    """Grossman-Zhou (1993) drawdown-constrained capital fraction (EQI 13.13).

    ``f = (mu / sigma^2) * (1 - (1 - D) / (1 - d))``: invests ``x* * D`` at the
    watermark, scales down roughly linearly with drawdown, and liquidates as
    ``d -> D``. Guarantees the drawdown limit ``D`` with probability one.

    Parameters
    ----------
    mu : float
        Expected per-period return.
    sigma : float
        Per-period volatility.
    max_drawdown_limit : float
        The maximum allowed drawdown ``D`` (a positive fraction).
    current_drawdown : float
        The current drawdown ``d`` from the high watermark.

    Returns
    -------
    float
        The fraction of capital to deploy (clamped at 0).
    """
    base = mu / sigma**2
    factor = 1.0 - (1.0 - max_drawdown_limit) / (1.0 - current_drawdown)
    return max(base * factor, 0.0)


def stop_loss_triggered(equity, threshold: float) -> bool:
    """Whether drawdown from the high watermark has breached a stop-loss threshold.

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.
    threshold : float
        Drawdown threshold (a positive fraction); set around 1.5-2.5x annualized
        volatility. A stop-loss buys firm survival, not performance (APM Ch 9).

    Returns
    -------
    bool
        True once the maximum drawdown reaches or exceeds ``threshold``.
    """
    return bool(max_drawdown(equity) >= threshold)
