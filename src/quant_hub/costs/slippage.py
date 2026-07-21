"""Market-impact / slippage models.

Implements *The Elements of Quantitative Investing* Ch 11 and *Advanced Portfolio
Management* Ch 8: the named temporary-impact laws and the rule-of-thumb optimal
event-trade size. ``participation`` is the trade quantity ``Q`` as a fraction of
interval volume ``V``; costs are per-unit unless stated otherwise.
"""

from __future__ import annotations

import numpy as np


def square_root_impact(sigma: float, participation: float, coefficient: float = 1.0) -> float:
    """Universal square-root impact ``coefficient * sigma * sqrt(Q/V)`` (EQI Ch 11).

    The dimensional-analysis exponent ``beta = 1/2`` that holds across asset classes.

    Parameters
    ----------
    sigma : float
        Volatility over the trading interval.
    participation : float
        Trade quantity as a fraction of interval volume, ``Q / V``.
    coefficient : float, default 1.0
        Market-specific impact constant.

    Returns
    -------
    float
        The per-unit impact cost.
    """
    return coefficient * sigma * np.sqrt(participation)


def almgren_impact(
    sigma: float, participation: float, coefficient: float = 1.0, exponent: float = 0.6
) -> float:
    """Almgren constant-rate impact ``coefficient * sigma * (Q/V)^exponent`` (EQI Ch 11).

    Parameters
    ----------
    sigma : float
        Interval volatility.
    participation : float
        Trade quantity as a fraction of interval volume.
    coefficient : float, default 1.0
        Market-specific impact constant.
    exponent : float, default 0.6
        Impact exponent ``beta`` (Almgren's empirical estimate).

    Returns
    -------
    float
        The per-unit impact cost.
    """
    return coefficient * sigma * participation**exponent


def obizhaeva_wang_impact(
    participation: float, horizon: float, decay_time: float, coefficient: float = 1.0
) -> float:
    """Obizhaeva-Wang transient impact with exponential book resilience (EQI Ch 11).

    Interpolates between slow execution (``decay_time << horizon``:
    ``~ coefficient * decay_time * Q/V``) and fast execution
    (``~ coefficient * decay_time * (horizon/2) * Q/V``).

    Parameters
    ----------
    participation : float
        Trade quantity as a fraction of interval volume.
    horizon : float
        Execution horizon.
    decay_time : float
        Order-book resilience time constant ``tau``.
    coefficient : float, default 1.0
        Market-specific impact constant.

    Returns
    -------
    float
        The per-unit impact cost.
    """
    slow = coefficient * decay_time * participation
    fast = coefficient * decay_time * (horizon / 2.0) * participation
    weight = np.exp(-horizon / decay_time)  # ->1 fast (horizon << tau), ->0 slow
    return weight * fast + (1.0 - weight) * slow


def gatheral_impact(
    sigma: float, participation: float, horizon: float, coefficient: float = 1.0
) -> float:
    """Gatheral square-root propagator impact ``(4/3) c sigma sqrt(Q T / V)`` (EQI Ch 11).

    Parameters
    ----------
    sigma : float
        Interval volatility.
    participation : float
        Trade quantity as a fraction of interval volume.
    horizon : float
        Execution horizon ``T``.
    coefficient : float, default 1.0
        Market-specific impact constant.

    Returns
    -------
    float
        The per-unit impact cost.
    """
    return (4.0 / 3.0) * coefficient * sigma * np.sqrt(participation * horizon)


def optimal_event_trade_size(
    alpha: float,
    dollar_volume: float,
    days_to_event: float,
    daily_vol: float,
    coefficient: float = 1.0,
) -> float:
    """Rule-of-thumb event-trade size under quadratic impact (APM Ch 8.3).

    ``size = coefficient * alpha * V * T / (2 * sigma)``: trade at constant
    participation into the event, liquidate at the same rate after (discount for
    concentration).

    Parameters
    ----------
    alpha : float
        Expected event return.
    dollar_volume : float
        Daily dollar volume ``V``.
    days_to_event : float
        Days until the event ``T``.
    daily_vol : float
        Daily volatility ``sigma``.
    coefficient : float, default 1.0
        Market-wide impact constant.

    Returns
    -------
    float
        The recommended position size.
    """
    return coefficient * alpha * dollar_volume * days_to_event / (2.0 * daily_vol)
