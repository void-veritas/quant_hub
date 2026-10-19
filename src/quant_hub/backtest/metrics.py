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


# ------------------------------------------------------------- non-normal Sharpe inference
def return_moments(returns) -> dict:
    """Skewness and kurtosis as moment ratios (Bouchaud): ``m3/σ³`` and ``m4/σ⁴``.

    Returns a dict with ``skew``, ``kurtosis`` (non-excess; 3 for a normal),
    ``excess_kurtosis`` and ``n``.
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    m = x - x.mean()
    s = np.sqrt((m**2).mean())
    return {
        "skew": float((m**3).mean() / s**3),
        "kurtosis": float((m**4).mean() / s**4),
        "excess_kurtosis": float((m**4).mean() / s**4 - 3.0),
        "n": int(len(x)),
    }


def sharpe_standard_error_mertens(returns, periods_per_year: float | None = None) -> float:
    """Standard error of the per-period Sharpe allowing for skew and kurtosis (Mertens 2002).

    ``Var(SR) = (1 + SR²/2 − γ₃·SR + (γ₄−3)/4·SR²) / T``; with ``periods_per_year``
    the result is annualised by ``√periods``. Lo's iid-normal formula is the special
    case γ₃ = 0, γ₄ = 3.
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    sr = x.mean() / x.std(ddof=1)
    mo = return_moments(x)
    var = (1 + 0.5 * sr**2 - mo["skew"] * sr + (mo["kurtosis"] - 3) / 4 * sr**2) / len(x)
    se = float(np.sqrt(max(var, 0.0)))
    return se * np.sqrt(periods_per_year) if periods_per_year else se


def probabilistic_sharpe_ratio(returns, benchmark_sharpe: float = 0.0) -> float:
    """PSR (Bailey & López de Prado): ``P[true SR > benchmark]`` given the sample SR.

    ``benchmark_sharpe`` is per period (not annualised). Uses the Mertens standard
    error, so fat tails and negative skew lower the PSR.
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    sr = x.mean() / x.std(ddof=1)
    se = sharpe_standard_error_mertens(x)
    return float(stats.norm.cdf((sr - benchmark_sharpe) / se))


def deflated_sharpe_ratio(returns, n_trials: int, trial_sharpe_variance: float) -> dict:
    """DSR: PSR against the Sharpe expected from the best of ``n_trials`` null strategies.

    ``trial_sharpe_variance`` is the variance of the per-period Sharpe ratios across
    the trials that were run (how dispersed the search was). The benchmark is
    ``√V · ((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e)))`` with Euler's γ (Bailey & López de
    Prado 2014). Returns ``{"sr0": benchmark per period, "dsr": probability}``.
    """
    if n_trials < 2:
        return {"sr0": 0.0, "dsr": probabilistic_sharpe_ratio(returns, 0.0)}
    gamma = 0.5772156649
    z = stats.norm.ppf
    sr0 = np.sqrt(trial_sharpe_variance) * (
        (1 - gamma) * z(1 - 1 / n_trials) + gamma * z(1 - 1 / (n_trials * np.e))
    )
    return {"sr0": float(sr0), "dsr": probabilistic_sharpe_ratio(returns, float(sr0))}


def block_bootstrap_sharpe_ci(
    returns,
    block: int = 20,
    n_boot: int = 2000,
    confidence: float = 0.95,
    periods_per_year: float = 365.0,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of the annualised Sharpe from a circular block bootstrap.

    Blocks of ``block`` consecutive observations preserve autocorrelation and
    volatility clustering that the iid formulas ignore.
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n // block + 1))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samples = x[idx]
    sr = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    lo, hi = np.percentile(sr, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(lo), float(hi)
