"""Performance and forecast-evaluation metrics.

Implements the performance material of *The Elements of Quantitative Investing*
Ch 3/5 and *Advanced Portfolio Management* Ch 6: Sharpe/information ratios with the
Lo (2002) standard error and confidence intervals, the Cantelli distribution-free
loss bound, drawdowns, probabilistic-forecast losses (Brier, log-loss,
calibration), and robust volatility-forecast losses (QLIKE, MSE).

The probabilistic-forecast helpers double as the scoring layer for the Polymarket
persistence models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns, periods_per_year: float | None = None, risk_free: float = 0.0) -> float:
    """Sharpe ratio ``mean / std`` of returns, optionally annualized.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Periodic returns.
    periods_per_year : float, optional
        If given, annualize by multiplying by ``sqrt(periods_per_year)``.
    risk_free : float, default 0.0
        Per-period risk-free rate subtracted from returns.

    Returns
    -------
    float
        The (annualized) Sharpe ratio.
    """
    r = np.asarray(returns, float) - risk_free
    sr = r.mean() / r.std(ddof=1)
    return sr * np.sqrt(periods_per_year) if periods_per_year else sr


def information_ratio(residual_returns, periods_per_year: float | None = None) -> float:
    """Information ratio -- the Sharpe ratio of residual (idiosyncratic) returns.

    Parameters
    ----------
    residual_returns : array_like, shape (T,)
        Idiosyncratic (factor-neutral) returns.
    periods_per_year : float, optional
        If given, annualize by ``sqrt(periods_per_year)``.

    Returns
    -------
    float
        The (annualized) information ratio.
    """
    return sharpe_ratio(residual_returns, periods_per_year)


def sharpe_standard_error(sharpe: float, n_obs: int) -> float:
    """Lo (2002) iid standard error of a Sharpe estimate: ``sqrt((1 + SR^2/2) / T)``.

    Parameters
    ----------
    sharpe : float
        The Sharpe estimate (per-period or annualized).
    n_obs : int
        Number of observations ``T`` (in the same frequency as ``sharpe``).

    Returns
    -------
    float
        The standard error, in the same units as ``sharpe``.
    """
    return np.sqrt((1.0 + 0.5 * sharpe**2) / n_obs)


def sharpe_confidence_interval(sharpe: float, n_obs: int, confidence: float = 0.95):
    """Two-sided confidence interval for a Sharpe ratio (Lo standard error).

    Parameters
    ----------
    sharpe : float
        The Sharpe estimate.
    n_obs : int
        Number of observations ``T``.
    confidence : float, default 0.95
        Coverage probability of the interval.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` bounds of the interval.
    """
    se = sharpe_standard_error(sharpe, n_obs)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    return sharpe - z * se, sharpe + z * se


def t_statistic(returns) -> float:
    """t-statistic of the mean return (equals per-period Sharpe times ``sqrt(T)``).

    Parameters
    ----------
    returns : array_like, shape (T,)
        Periodic returns.

    Returns
    -------
    float
        The t-statistic of the mean.
    """
    r = np.asarray(returns, float)
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))


def cantelli_loss_bound(loss_sigmas: float, sharpe: float) -> float:
    """Distribution-free tail bound ``P(loss > L*sigma) <= 1 / (1 + (L + SR)^2)``.

    The one-sided Cantelli inequality (EQI Ch 3); far looser than the Gaussian tail
    but assumption-free.

    Parameters
    ----------
    loss_sigmas : float
        Loss threshold ``L`` in units of volatility.
    sharpe : float
        The strategy's (per-period) Sharpe ratio.

    Returns
    -------
    float
        An upper bound on the probability of a loss exceeding ``L`` sigma.
    """
    return 1.0 / (1.0 + (loss_sigmas + sharpe) ** 2)


def equity_from_returns(returns):
    """Cumulative equity curve ``prod(1 + r)`` from a return series.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Periodic returns.

    Returns
    -------
    numpy.ndarray, shape (T,)
        The compounded equity curve.
    """
    return np.cumprod(1.0 + np.asarray(returns, float))


def drawdown_series(equity):
    """Drawdown path ``equity / running_peak - 1`` (values <= 0).

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.

    Returns
    -------
    numpy.ndarray, shape (T,)
        The drawdown at each point.
    """
    e = np.asarray(equity, float)
    return e / np.maximum.accumulate(e) - 1.0


def max_drawdown(equity) -> float:
    """Worst (most negative) drawdown over an equity curve.

    Parameters
    ----------
    equity : array_like, shape (T,)
        Equity / cumulative-return curve.

    Returns
    -------
    float
        The minimum of the drawdown series.
    """
    return float(drawdown_series(equity).min())


def hit_rate(returns) -> float:
    """Fraction of strictly positive returns.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Periodic returns.

    Returns
    -------
    float
        The share of returns greater than zero.
    """
    r = np.asarray(returns, float)
    return float(np.mean(r > 0))


def brier_score(prob, outcome) -> float:
    """Mean squared error of probabilistic forecasts, ``mean((p - y)^2)``.

    Parameters
    ----------
    prob : array_like, shape (T,)
        Forecast probabilities in ``[0, 1]``.
    outcome : array_like, shape (T,)
        Realized binary outcomes in ``{0, 1}``.

    Returns
    -------
    float
        The Brier score (lower is better).
    """
    p = np.asarray(prob, float)
    y = np.asarray(outcome, float)
    return float(np.mean((p - y) ** 2))


def log_loss(prob, outcome, eps: float = 1e-15) -> float:
    """Binary cross-entropy (log loss) of probabilistic forecasts.

    Parameters
    ----------
    prob : array_like, shape (T,)
        Forecast probabilities in ``[0, 1]``.
    outcome : array_like, shape (T,)
        Realized binary outcomes in ``{0, 1}``.
    eps : float, default 1e-15
        Clipping bound to keep the log finite.

    Returns
    -------
    float
        The mean log loss (lower is better).
    """
    p = np.clip(np.asarray(prob, float), eps, 1.0 - eps)
    y = np.asarray(outcome, float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_curve(prob, outcome, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table of mean predicted vs observed frequency per probability bin.

    Parameters
    ----------
    prob : array_like, shape (T,)
        Forecast probabilities in ``[0, 1]``.
    outcome : array_like, shape (T,)
        Realized binary outcomes in ``{0, 1}``.
    n_bins : int, default 10
        Number of equal-width probability bins.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_mid``, ``mean_pred``, ``mean_obs``, ``count`` (one row per
        non-empty bin). A well-calibrated forecast has ``mean_pred ~ mean_obs``.
    """
    p = np.asarray(prob, float)
    y = np.asarray(outcome, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append(
                (0.5 * (edges[b] + edges[b + 1]), p[mask].mean(), y[mask].mean(), int(mask.sum()))
            )
    return pd.DataFrame(rows, columns=["bin_mid", "mean_pred", "mean_obs", "count"])


def qlike(realized_var, forecast_var) -> float:
    """QLIKE robust volatility-forecast loss ``mean(rv/fv - log(rv/fv) - 1)`` (EQI Ch 5).

    Rank-robust under an unbiased variance proxy and asymmetric (penalizes
    under-forecasting variance more heavily).

    Parameters
    ----------
    realized_var : array_like, shape (T,)
        Realized-variance proxy (e.g. squared returns or realized variance).
    forecast_var : array_like, shape (T,)
        Forecast variance.

    Returns
    -------
    float
        The mean QLIKE loss (lower is better).
    """
    rv = np.asarray(realized_var, float)
    fv = np.asarray(forecast_var, float)
    ratio = rv / fv
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_variance(realized_var, forecast_var) -> float:
    """Mean squared error between realized and forecast variance (EQI Ch 5).

    Parameters
    ----------
    realized_var : array_like, shape (T,)
        Realized-variance proxy.
    forecast_var : array_like, shape (T,)
        Forecast variance.

    Returns
    -------
    float
        The mean squared error (lower is better).
    """
    return float(np.mean((np.asarray(realized_var, float) - np.asarray(forecast_var, float)) ** 2))
