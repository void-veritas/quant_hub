"""Perpetual-futures funding signals -- the crypto-native carry alphas.

Funding is the perp analogue of the financing / dividend leg (EQI Ch 2): a long
perp pays funding when it is positive. These helpers build carry, funding-momentum,
and cross-venue funding-spread signals from venue-native funding rates. Rates are at
each venue's native cadence (Hyperliquid hourly, Binance 8h/4h), so pass the correct
``interval_hours`` when annualizing.
"""

from __future__ import annotations

import numpy as np

HOURS_PER_YEAR = 24 * 365


def annualize_funding(rate, interval_hours: float):
    """Convert a per-interval funding rate to an annualized rate.

    Parameters
    ----------
    rate : float or array_like
        Funding rate per settlement interval.
    interval_hours : float
        Length of the settlement interval in hours (1 on Hyperliquid, 8 or 4 on
        Binance).

    Returns
    -------
    float or numpy.ndarray
        The annualized funding rate.
    """
    return rate * (HOURS_PER_YEAR / interval_hours)


def funding_carry(funding_rate, interval_hours: float):
    """Carry signal -- annualized funding earned by a short perp position.

    A long perp pays funding when it is positive, so a positive signal favors the
    short side.

    Parameters
    ----------
    funding_rate : float or array_like
        Per-interval funding rate.
    interval_hours : float
        Settlement interval length in hours.

    Returns
    -------
    float or numpy.ndarray
        The annualized carry (positive favors the short side).
    """
    return annualize_funding(np.asarray(funding_rate, float), interval_hours)


def funding_momentum(funding_rates, lookback: int):
    """Trailing mean funding over ``lookback`` intervals -- persistence of the carry.

    Parameters
    ----------
    funding_rates : pandas.Series or pandas.DataFrame
        Funding rates indexed by time.
    lookback : int
        Number of intervals in the rolling window.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The rolling-mean funding signal.
    """
    return funding_rates.rolling(lookback).mean()


def cross_venue_funding_spread(funding_a, interval_a: float, funding_b, interval_b: float):
    """Annualized funding-spread signal between two venues for the same asset.

    A positive spread means venue A's funding is richer than venue B's (long B /
    short A collects the difference).

    Parameters
    ----------
    funding_a : float or array_like
        Venue A per-interval funding rate.
    interval_a : float
        Venue A settlement interval in hours.
    funding_b : float or array_like
        Venue B per-interval funding rate (aligned to A).
    interval_b : float
        Venue B settlement interval in hours.

    Returns
    -------
    float or numpy.ndarray
        The annualized funding spread ``A - B``.
    """
    return annualize_funding(funding_a, interval_a) - annualize_funding(funding_b, interval_b)


def funding_zscore(funding_rates, lookback: int):
    """Rolling z-score of funding -- extreme funding as a mean-reversion trigger.

    Parameters
    ----------
    funding_rates : pandas.Series or pandas.DataFrame
        Funding rates indexed by time.
    lookback : int
        Rolling window length in intervals.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The rolling z-score of funding.
    """
    mean = funding_rates.rolling(lookback).mean()
    std = funding_rates.rolling(lookback).std()
    return (funding_rates - mean) / std
