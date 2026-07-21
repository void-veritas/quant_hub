"""Feature transforms for alpha / alt-data construction.

Implements the feature-engineering recipe of *Advanced Portfolio Management*
Ch 8.4 / 11.5: normalization and shape transforms for turning raw characteristics
into well-behaved loadings. Compose these, then z-score and orthogonalize against
the risk model (see :func:`quant_hub.alpha.signals.orthogonalize`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_transform(x, offset: float = 0.0):
    """Signed log transform ``log(1 + |x| + offset) * sign(x)`` for heavy-tailed features.

    Parameters
    ----------
    x : array_like or pandas.Series
        Feature values.
    offset : float, default 0.0
        Constant added inside the log to control behaviour near zero.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The transformed feature, preserving the sign of ``x``.
    """
    a = np.asarray(x, float)
    out = np.log1p(np.abs(a) + offset) * np.sign(a)
    return pd.Series(out, index=x.index) if isinstance(x, pd.Series) else out


def rank_normalize(x):
    """Map a feature to uniform ranks in ``(0, 1)``.

    Parameters
    ----------
    x : array_like or pandas.Series
        Feature values.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The rank-normalized feature.
    """
    s = pd.Series(np.asarray(x, float))
    ranks = s.rank() / (len(s) + 1)
    return (
        pd.Series(ranks.to_numpy(), index=x.index) if isinstance(x, pd.Series) else ranks.to_numpy()
    )


def arctan_transform(x, scale: float = 1.0):
    """Bounded squashing to ``(-1, 1)`` via ``(2/pi) arctan(x / scale)``.

    Parameters
    ----------
    x : array_like or pandas.Series
        Feature values.
    scale : float, default 1.0
        Scale controlling how quickly the transform saturates.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The squashed feature.
    """
    a = np.asarray(x, float)
    out = (2.0 / np.pi) * np.arctan(a / scale)
    return pd.Series(out, index=x.index) if isinstance(x, pd.Series) else out


def normalize_by(x, scale, rolling: int | None = None):
    """Divide a feature by a scale (market cap, volume, EV), optionally rolling-averaged.

    Parameters
    ----------
    x : array_like or pandas.Series
        Feature values (numerator).
    scale : array_like
        Denominator series (e.g. dollar volume).
    rolling : int, optional
        If given, use the trailing rolling mean of ``scale`` over this window.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The normalized feature.
    """
    denom = (
        pd.Series(np.asarray(scale, float)).rolling(rolling).mean().to_numpy()
        if rolling
        else np.asarray(scale, float)
    )
    out = np.asarray(x, float) / denom
    return pd.Series(out, index=x.index) if isinstance(x, pd.Series) else out


def term_structure(series, short: int, long: int):
    """Short-window mean minus long-window mean -- a feature's slope over time.

    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame
        Time-indexed feature.
    short : int
        Short rolling window length.
    long : int
        Long rolling window length.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The difference of the two rolling means.
    """
    return series.rolling(short).mean() - series.rolling(long).mean()


def change_vs_average(series, window: int):
    """Current value minus its trailing average over ``window``.

    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame
        Time-indexed feature.
    window : int
        Rolling window length.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The deviation from the trailing average.
    """
    return series - series.rolling(window).mean()
