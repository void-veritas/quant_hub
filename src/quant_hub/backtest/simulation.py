"""Resampled paths for a strategy's outcome distribution (PCRM Ch 3, simplified).

History is one draw. These tools turn the realised daily PnL (or its components) into
many plausible one-year paths, unconditionally (circular block bootstrap) or
conditioned on a state variable (time- and state-dependent resampling: blocks are
started at days whose state resembles the one asked for, with an exponential decay
in time). Nothing here re-runs the strategy; it resamples what the strategy did, so
it answers "what range of years does this behaviour imply", not "what would the
weights have been".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_hub.backtest import metrics


def block_paths(
    R, horizon: int = 365, n_paths: int = 5000, block: int = 20, p=None, seed: int = 0
) -> np.ndarray:
    """Paths (n_paths, horizon, K) by concatenating blocks whose starts are drawn with ``p``.

    ``R`` is (T, K) daily PnL or components; a block starting at ``t`` covers
    ``t..t+block-1`` circularly. With ``p=None`` the starts are uniform (plain
    circular block bootstrap).
    """
    x = np.asarray(R, float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = horizon // block + 1
    starts = rng.choice(n, size=(n_paths, n_blocks), p=p)
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[
        :, :horizon
    ] % n
    return x[idx]


def state_probabilities(
    state, target: float, bandwidth: float | None = None, half_life: int | None = None
) -> np.ndarray:
    """Probabilities over history concentrating on days whose ``state`` is near ``target``.

    Gaussian kernel in the state (``bandwidth`` defaults to the state's std / 2)
    multiplied by an exponential decay in time when ``half_life`` is given; this is
    the crisp/kernel step of time- and state-dependent resampling.
    """
    s = np.asarray(state, float)
    bw = bandwidth if bandwidth is not None else np.nanstd(s) / 2.0
    k = np.exp(-0.5 * ((s - target) / bw) ** 2)
    k = np.where(np.isfinite(k), k, 0.0)
    if half_life:
        t = np.arange(len(s))
        k = k * 0.5 ** ((len(s) - 1 - t) / half_life)
    return k / k.sum()


def path_statistics(paths: np.ndarray, periods_per_year: int = 365) -> pd.DataFrame:
    """Per path: total return (sum), annualised vol and Sharpe, max drawdown, worst day."""
    r = paths.sum(axis=2) if paths.ndim == 3 else paths
    mu, sd = r.mean(axis=1), r.std(axis=1, ddof=1)
    eq = np.cumsum(r, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq - peak).min(axis=1)
    return pd.DataFrame(
        {
            "total_return": r.sum(axis=1),
            "vol_ann": sd * np.sqrt(periods_per_year),
            "sharpe": np.where(sd > 0, mu / sd * np.sqrt(periods_per_year), np.nan),
            "max_drawdown": dd,
            "worst_day": r.min(axis=1),
        }
    )


def summarise_paths(
    stats_df: pd.DataFrame, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)
) -> pd.DataFrame:
    """Quantiles of the path statistics plus the probability of a losing horizon."""
    q = stats_df.quantile(list(quantiles)).T
    q.columns = [f"p{int(c * 100)}" for c in quantiles]
    q["mean"] = stats_df.mean()
    q.loc["total_return", "P(loss)"] = float((stats_df["total_return"] < 0).mean())
    return q


def effective_history(p) -> float:
    """``exp(entropy)`` of the start-probabilities: how many days the paths really draw from."""
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def moments_check(paths: np.ndarray, history) -> pd.DataFrame:
    """Compare mean/vol/skew/kurtosis of the simulated days with the historical days."""
    sim = paths.sum(axis=2).reshape(-1) if paths.ndim == 3 else paths.reshape(-1)
    hist = np.asarray(history, float)
    rows = {}
    for name, x in (("history", hist), ("simulated", sim)):
        m = metrics.return_moments(x)
        rows[name] = {
            "mean": x.mean(),
            "vol": x.std(ddof=1),
            "skew": m["skew"],
            "kurtosis": m["kurtosis"],
        }
    return pd.DataFrame(rows).T
