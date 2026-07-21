"""Momentum and reversal signals.

Implements the momentum material of *Advanced Portfolio Management* Ch 5 (with the
Novy-Marx term structure): time-series and cross-sectional momentum with the
standard skip window (the most recent period reverses, so it is excluded) and
short-term reversal.

Prices are a wide :class:`~pandas.DataFrame` (index = time, columns = assets) or a
single :class:`~pandas.Series`; the signal shares that shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(prices, lookback: int, skip: int = 1):
    """Trailing return over ``[t - lookback, t - skip]``, skipping recent bars.

    The ``skip`` window removes short-term reversal; a ``lookback`` of roughly
    3-12 months (in bars) captures continuation.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Price levels (one column per asset for a DataFrame).
    lookback : int
        Formation window length in bars.
    skip : int, default 1
        Number of most-recent bars to exclude.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The momentum signal, aligned to ``prices``.
    """
    past = prices.shift(skip)
    reference = prices.shift(lookback)
    return past / reference - 1.0


def short_term_reversal(prices, lookback: int = 1):
    """Negative of the most-recent return -- the ``<= 1 month`` reversal effect (APM Ch 5).

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Price levels.
    lookback : int, default 1
        Window over which the reversal is measured.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The reversal signal (positive when the recent return was negative).
    """
    return -(prices / prices.shift(lookback) - 1.0)


def cross_sectional_momentum(prices, lookback: int, skip: int = 1):
    """Cross-sectionally demeaned momentum -- each asset's momentum minus the date mean.

    Positive values indicate outperformance versus peers.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Price levels.
    lookback : int
        Formation window length in bars.
    skip : int, default 1
        Number of most-recent bars to exclude.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The cross-sectional momentum signal.
    """
    raw = momentum(prices, lookback, skip)
    if isinstance(raw, pd.DataFrame):
        return raw.sub(raw.mean(axis=1), axis=0)
    return raw - raw.mean()


def time_series_momentum(prices, lookback: int, skip: int = 1):
    """Sign of the trailing return -- absolute (trend-following) momentum in ``{-1, 0, 1}``.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Price levels.
    lookback : int
        Formation window length in bars.
    skip : int, default 1
        Number of most-recent bars to exclude.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The sign of trailing momentum.
    """
    return np.sign(momentum(prices, lookback, skip))
