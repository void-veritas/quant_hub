"""Univariate volatility estimation and state-space vol models.

Implements the univariate volatility material of *The Elements of Quantitative
Investing* Ch 2: exponentially weighted (RiskMetrics) variance, GARCH(1,1),
realized variance, the Roll microstructure model, and the simplified
Harvey-Shephard stochastic-volatility filter.

Inputs are return series (log returns recommended). Recursions preserve a pandas
index when given a :class:`pandas.Series`; annualization helpers are at the end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def halflife_to_decay(halflife: float) -> float:
    """Convert an EWMA half-life to its decay factor ``K = 2 ** (-1 / halflife)``.

    Parameters
    ----------
    halflife : float
        Half-life in periods (the horizon over which a weight halves).

    Returns
    -------
    float
        The decay factor ``K`` in ``(0, 1)`` applied to the previous estimate.
    """
    return 0.5 ** (1.0 / halflife)


def decay_to_halflife(decay: float) -> float:
    """Convert an EWMA decay factor to its half-life ``-log 2 / log K``.

    Parameters
    ----------
    decay : float
        The decay factor ``K`` in ``(0, 1)``.

    Returns
    -------
    float
        The equivalent half-life in periods.
    """
    return -np.log(2.0) / np.log(decay)


def _wrap_like(values: np.ndarray, template):
    """Return `values` as a Series sharing `template`'s index when it is a Series."""
    return pd.Series(values, index=template.index) if isinstance(template, pd.Series) else values


def ewma_variance(returns, halflife: float, init: float | None = None):
    """Filtered exponentially weighted variance of a return series.

    Recursion ``sigma_t^2 = (1 - K) r_t^2 + K sigma_{t-1}^2`` with ``K`` set by
    `halflife` (EQI Ch 2). This is the filtered (in-sample) variance using ``r_t``;
    shift by one period for a one-step-ahead forecast.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Return series.
    halflife : float
        EWMA half-life in periods.
    init : float, optional
        Seed variance for the recursion; defaults to the sample variance.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (T,)
        The variance path (a Series when `returns` is a Series).
    """
    r = np.asarray(returns, float)
    decay = halflife_to_decay(halflife)
    out = np.empty(len(r))
    var = float(np.nanvar(r)) if init is None else init
    for t in range(len(r)):
        var = (1.0 - decay) * r[t] ** 2 + decay * var
        out[t] = var
    return _wrap_like(out, returns)


def ewma_volatility(returns, halflife: float, init: float | None = None):
    """Exponentially weighted volatility -- the square root of :func:`ewma_variance`.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Return series.
    halflife : float
        EWMA half-life in periods.
    init : float, optional
        Seed variance; defaults to the sample variance.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (T,)
        The volatility path.
    """
    return np.sqrt(ewma_variance(returns, halflife, init))


def realized_variance(intraday_returns) -> float:
    """Uncentered realized variance ``sum(r^2)`` over one window (EQI Ch 2).

    Parameters
    ----------
    intraday_returns : array_like
        Higher-frequency returns within the window (e.g. 5-minute returns for a day).

    Returns
    -------
    float
        The realized variance.
    """
    r = np.asarray(intraday_returns, float)
    return float(np.sum(r**2))


def realized_volatility(intraday_returns) -> float:
    """Realized volatility -- the square root of :func:`realized_variance`.

    Parameters
    ----------
    intraday_returns : array_like
        Higher-frequency returns within the window.

    Returns
    -------
    float
        The realized volatility.
    """
    return float(np.sqrt(realized_variance(intraday_returns)))


def realized_variance_by_day(returns: pd.Series) -> pd.Series:
    """Aggregate a higher-frequency return series into daily realized variance.

    Parameters
    ----------
    returns : pandas.Series
        Returns indexed by a UTC :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    pandas.Series
        Realized variance ``sum(r^2)`` per calendar day.
    """
    squared = pd.Series(np.asarray(returns, float) ** 2, index=returns.index)
    return squared.groupby(returns.index.normalize()).sum()


def roll_effective_spread(prices) -> float:
    """Roll (1984) implied effective spread from return autocovariance.

    The observed price is the efficient price plus a bid-ask bounce, so successive
    price changes carry autocovariance ``E(dP_t dP_{t-1}) = -c^2`` and the spread is
    ``2c`` (EQI Ch 2). Returns 0 when the empirical lag-1 autocovariance is
    non-negative (the model then implies no measurable spread).

    Parameters
    ----------
    prices : array_like, shape (T,)
        Observed transaction prices.

    Returns
    -------
    float
        The implied effective (full) spread in price units.
    """
    dp = np.diff(np.asarray(prices, float))
    if len(dp) < 2:
        return 0.0
    autocov = np.cov(dp[:-1], dp[1:])[0, 1]
    return 2.0 * np.sqrt(-autocov) if autocov < 0 else 0.0


def harvey_shephard_volatility(returns, halflife: float):
    """Simplified Harvey-Shephard stochastic-volatility filter (EQI Ch 2).

    The log-variance state follows an EWMA of the noisy observation
    ``log(r^2) - gamma`` with ``gamma = E[log chi^2_1] ~ -1.27``; volatility is
    ``exp(state / 2)``, positive by construction.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Return series.
    halflife : float
        EWMA half-life in periods for the log-variance state.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (T,)
        The volatility path (a Series when `returns` is a Series).
    """
    r = np.asarray(returns, float)
    decay = halflife_to_decay(halflife)
    gamma = -1.27
    out = np.empty(len(r))
    state = np.log(max(np.nanvar(r), 1e-300))
    for t in range(len(r)):
        log_r2 = np.log(max(r[t] ** 2, 1e-300))
        state = (1.0 - decay) * state + decay * (log_r2 - gamma)
        out[t] = np.exp(state / 2.0)
    return _wrap_like(out, returns)


def annualize_volatility(vol, periods_per_year: float):
    """Scale a per-period volatility to annual by ``vol * sqrt(periods_per_year)``.

    Parameters
    ----------
    vol : float or array_like
        Per-period volatility.
    periods_per_year : float
        Number of periods in a year (e.g. 252 for daily, 35040 for 15-minute bars).

    Returns
    -------
    float or numpy.ndarray
        The annualized volatility.
    """
    return vol * np.sqrt(periods_per_year)


class GARCH11:
    """Gaussian GARCH(1,1) conditional-variance model.

    Fits ``h_t^2 = omega + alpha * r_{t-1}^2 + beta * h_{t-1}^2`` by maximum
    likelihood (minimizing ``sum(log h_t^2 + r_t^2 / h_t^2)``). Returns are
    demeaned before fitting.

    Attributes
    ----------
    omega, alpha, beta : float
        Fitted parameters (available after :meth:`fit`).

    Notes
    -----
    GARCH MLE is sensitive to starting values; the persistence ``alpha + beta`` is
    typically recovered well, the split between the two less so.
    """

    def __init__(self):
        self.omega = self.alpha = self.beta = None
        self._last_r2 = self._last_h2 = None

    def _recursion(self, r, omega, alpha, beta):
        """Return the conditional-variance path for the given parameters."""
        h2 = np.empty(len(r))
        h2[0] = np.var(r)
        for t in range(1, len(r)):
            h2[t] = omega + alpha * r[t - 1] ** 2 + beta * h2[t - 1]
        return h2

    def _neg_loglik(self, params, r):
        """Negative Gaussian log-likelihood; large penalty outside the valid region."""
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e12
        h2 = self._recursion(r, omega, alpha, beta)
        return float(np.sum(np.log(h2) + r**2 / h2))

    def fit(self, returns):
        """Estimate ``(omega, alpha, beta)`` by constrained maximum likelihood.

        Parameters
        ----------
        returns : array_like, shape (T,)
            Return series (demeaned internally).

        Returns
        -------
        GARCH11
            ``self``, with parameters set, to allow chaining.
        """
        r = np.asarray(returns, float)
        r = r - r.mean()
        var = np.var(r)
        x0 = [var * 0.05, 0.05, 0.90]
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
        constraint = {"type": "ineq", "fun": lambda p: 1.0 - p[1] - p[2] - 1e-6}
        result = minimize(
            self._neg_loglik, x0, args=(r,), bounds=bounds, constraints=constraint, method="SLSQP"
        )
        self.omega, self.alpha, self.beta = result.x
        h2 = self._recursion(r, *result.x)
        self._last_r2, self._last_h2 = r[-1] ** 2, h2[-1]
        return self

    def conditional_variance(self, returns):
        """Return the in-sample conditional-variance path for the fitted parameters.

        Parameters
        ----------
        returns : array_like, shape (T,)
            Return series (demeaned internally).

        Returns
        -------
        numpy.ndarray or pandas.Series, shape (T,)
            The ``h_t^2`` path.
        """
        r = np.asarray(returns, float)
        r = r - r.mean()
        return _wrap_like(self._recursion(r, self.omega, self.alpha, self.beta), returns)

    def unconditional_variance(self) -> float:
        """Long-run variance ``omega / (1 - alpha - beta)``.

        Returns
        -------
        float
            The stationary (unconditional) variance.
        """
        return self.omega / (1.0 - self.alpha - self.beta)

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast conditional variance 1..`horizon` steps ahead.

        Forecasts mean-revert geometrically toward the unconditional variance at
        rate ``alpha + beta``.

        Parameters
        ----------
        horizon : int, default 1
            Number of steps to forecast.

        Returns
        -------
        numpy.ndarray, shape (horizon,)
            The ``h^2`` forecasts.
        """
        uncond = self.unconditional_variance()
        persistence = self.alpha + self.beta
        step1 = self.omega + self.alpha * self._last_r2 + self.beta * self._last_h2
        out, current = [], step1
        for _ in range(horizon):
            out.append(current)
            current = uncond + persistence * (current - uncond)
        return np.array(out)

    def kurtosis(self) -> float:
        """Unconditional kurtosis of returns under the fitted model.

        Returns
        -------
        float
            The kurtosis (> 3, leptokurtic); ``inf`` when the fourth moment diverges.
        """
        a, b = self.alpha, self.beta
        denom = 1.0 - (a + b) ** 2 - 2.0 * a**2
        return np.inf if denom <= 0 else 3.0 * (1.0 - (a + b) ** 2) / denom
