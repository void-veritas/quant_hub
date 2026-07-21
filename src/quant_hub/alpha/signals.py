"""Signal construction and the fundamental law of active management.

Implements the signal material of *The Elements of Quantitative Investing* Ch 9 and
*Advanced Portfolio Management* Ch 6/8: cross-sectional normalization, the
information coefficient, IC-to-Sharpe/IR conversions, orthogonalization of a signal
against risk-model loadings (keeping only the part that is not already a factor),
and centralized signal combination.

A "signal" here is a cross-section of per-asset scores at one date; panel callers
apply these per date (e.g. via ``DataFrame.apply`` along rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(signal):
    """Standardize a signal to mean 0 and unit standard deviation.

    Parameters
    ----------
    signal : array_like or pandas.Series, shape (n,)
        Cross-section of scores.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (n,)
        The standardized signal (zeros if the input has no dispersion).
    """
    x = np.asarray(signal, float)
    sd = x.std(ddof=1)
    z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    return pd.Series(z, index=signal.index) if isinstance(signal, pd.Series) else z


def rank_transform(signal):
    """Map a signal to evenly spaced ranks on ``(-1, 1)`` -- robust to outliers.

    Parameters
    ----------
    signal : array_like or pandas.Series, shape (n,)
        Cross-section of scores.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (n,)
        The rank-transformed signal.
    """
    x = np.asarray(signal, float)
    order = x.argsort().argsort().astype(float)
    scaled = 2.0 * order / (len(x) - 1) - 1.0 if len(x) > 1 else np.zeros_like(x)
    return pd.Series(scaled, index=signal.index) if isinstance(signal, pd.Series) else scaled


def winsorize(signal, limit: float = 3.0):
    """Clip a signal to ``+/- limit`` standard deviations after standardizing.

    Parameters
    ----------
    signal : array_like or pandas.Series, shape (n,)
        Cross-section of scores.
    limit : float, default 3.0
        Clipping bound in standard deviations.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (n,)
        The winsorized, standardized signal.
    """
    z = np.asarray(zscore(signal), float)
    clipped = np.clip(z, -limit, limit)
    return pd.Series(clipped, index=signal.index) if isinstance(signal, pd.Series) else clipped


def information_coefficient(signal, forward_returns) -> float:
    """Cross-sectional correlation between a signal and next-period returns (EQI Ch 9).

    Parameters
    ----------
    signal : array_like, shape (n,)
        Cross-section of scores at the decision time.
    forward_returns : array_like, shape (n,)
        Realized next-period returns for the same assets.

    Returns
    -------
    float
        The information coefficient (0 if either series has no dispersion).
    """
    s = np.asarray(signal, float)
    r = np.asarray(forward_returns, float)
    if s.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(s, r)[0, 1])


def ic_to_sharpe(ic: float, breadth: int) -> float:
    """Fundamental law: ``SR = IC * sqrt(n)`` (EQI Ch 9).

    Parameters
    ----------
    ic : float
        Information coefficient.
    breadth : int
        Number of independent bets ``n``.

    Returns
    -------
    float
        The implied Sharpe ratio.
    """
    return ic * np.sqrt(breadth)


def ic_to_information_ratio(ic: float, breadth: int, periods_per_year: float) -> float:
    """Annualized information ratio ``IC * sqrt(n * T)`` (EQI Ch 9).

    Parameters
    ----------
    ic : float
        Information coefficient.
    breadth : int
        Number of independent bets ``n``.
    periods_per_year : float
        Number of independent forecasts per year ``T``.

    Returns
    -------
    float
        The annualized information ratio.
    """
    return ic * np.sqrt(breadth * periods_per_year)


def orthogonalize(signal, loadings, weights=None):
    """Residualize a signal against factor loadings (EQI Ch 9 / APM 6.3, 11.5).

    Returns the component of the signal not spanned by the loadings,
    ``c - B (B'WB)^-1 B'W c`` with ``W = diag(1/weights)``. This is the "keep only
    what is not already a factor" step for custom-factor and factor-neutral
    construction.

    Parameters
    ----------
    signal : array_like or pandas.Series, shape (n,)
        Cross-section of scores ``c``.
    loadings : array_like, shape (n, m)
        Factor loadings ``B``.
    weights : array_like, shape (n,), optional
        Regression weights (typically idiosyncratic variances); ``None`` gives OLS.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (n,)
        The orthogonalized (residual) signal.
    """
    c = np.asarray(signal, float)
    b = np.atleast_2d(np.asarray(loadings, float))
    w = np.ones(len(c)) if weights is None else 1.0 / np.asarray(weights, float)
    bt_w = b.T * w
    projection = b @ np.linalg.pinv(bt_w @ b) @ (bt_w @ c)
    residual = c - projection
    return pd.Series(residual, index=signal.index) if isinstance(signal, pd.Series) else residual


def combine_signals(signals, weights=None):
    """Weighted combination of standardized signals (centralized aggregation, EQI Ch 9).

    Each signal is z-scored before combining so their scales are comparable.

    Parameters
    ----------
    signals : sequence of array_like
        Aligned per-asset score vectors, each of shape ``(n,)``.
    weights : array_like, optional
        Combination weights, one per signal; defaults to equal weights.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The combined signal.
    """
    mat = np.column_stack([np.asarray(zscore(s), float) for s in signals])
    w = np.ones(mat.shape[1]) if weights is None else np.asarray(weights, float)
    return mat @ w
