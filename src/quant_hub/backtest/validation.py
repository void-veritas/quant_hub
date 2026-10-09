"""Validation splits and out-of-sample reporting for daily strategy returns.

Three ways to cut a daily return series, all point-in-time by construction:

- `walk_forward(dates, train, test)`: rolling train/test windows.
- `purged_kfold(dates, k, purge, embargo)`: López de Prado's purged k-fold, test
  fold = one contiguous block, training rows within `purge` days before and
  `embargo` days after the block are dropped.
- `cpcv(dates, n_groups, n_test, purge, embargo)`: combinatorial purged CV. Every
  combination of `n_test` of `n_groups` contiguous groups is a test set; the
  test folds are then stitched into `C(n_groups, n_test) * n_test / n_groups`
  full backtest paths, giving a distribution of out-of-sample Sharpes instead
  of a single number.

For strategies without fitted parameters (fixed rules, as in the YOLO variants)
the train sets are unused and CPCV reduces to "Sharpe on every stitched path".
`compare_variants` reports, per variant, the mean / worst path Sharpe and the
Rademacher haircut over the set of variants (multiple_testing module), which is
what a one-change-at-a-time experiment table should show.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from quant_hub.backtest import metrics
from quant_hub.backtest.multiple_testing import rademacher_haircut_sharpe


def _as_index(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates))


def walk_forward(
    dates, train: int, test: int, step: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling (train_idx, test_idx) pairs in row positions; `step` defaults to `test`."""
    idx = _as_index(dates)
    n = len(idx)
    step = step or test
    out = []
    start = 0
    while start + train + test <= n:
        tr = np.arange(start, start + train)
        te = np.arange(start + train, start + train + test)
        out.append((tr, te))
        start += step
    return out


def _contiguous_groups(n: int, k: int) -> list[np.ndarray]:
    return [np.asarray(g) for g in np.array_split(np.arange(n), k)]


def _purge(train: np.ndarray, test: np.ndarray, purge: int, embargo: int) -> np.ndarray:
    lo, hi = test.min(), test.max()
    keep = (train < lo - purge) | (train > hi + embargo)
    return train[keep]


def purged_kfold(
    dates, k: int = 5, purge: int = 0, embargo: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K contiguous test blocks; training rows near each block are purged / embargoed."""
    n = len(_as_index(dates))
    groups = _contiguous_groups(n, k)
    out = []
    for g in groups:
        tr = np.setdiff1d(np.arange(n), g)
        out.append((_purge(tr, g, purge, embargo), g))
    return out


def cpcv(dates, n_groups: int = 6, n_test: int = 2, purge: int = 0, embargo: int = 0) -> dict:
    """Combinatorial purged CV: splits plus the stitched paths (list of row-index arrays)."""
    n = len(_as_index(dates))
    groups = _contiguous_groups(n, n_groups)
    splits = []
    for combo in combinations(range(n_groups), n_test):
        te = np.concatenate([groups[i] for i in combo])
        tr = np.setdiff1d(np.arange(n), te)
        for i in combo:  # purge around every test group separately
            tr = _purge(tr, groups[i], purge, embargo)
        splits.append({"test_groups": combo, "train": tr, "test": te})
    # path assembly (de Prado AFML 12.4): each group appears in n_test * C(n-1, n_test-1) splits;
    # assign each (split, group) pair to a path so that every path holds every group exactly once
    n_paths = len(splits) * n_test // n_groups
    paths = [[None] * n_groups for _ in range(n_paths)]
    used = {g: 0 for g in range(n_groups)}
    for s_i, s in enumerate(splits):
        for g in s["test_groups"]:
            p = used[g]
            paths[p][g] = s_i
            used[g] += 1
    return {"groups": groups, "splits": splits, "paths": paths, "n_paths": n_paths}


def path_returns(returns: pd.Series, cv: dict) -> list[pd.Series]:
    """Stitch one full-length return series per CPCV path (same rows, different split provenance).

    For parameter-free strategies every path is the same series; the function
    exists so that a fitted strategy can supply per-split out-of-sample returns
    through `returns_by_split` instead.
    """
    return [returns.iloc[np.concatenate(cv["groups"])] for _ in range(cv["n_paths"])]


def group_sharpes(returns: pd.Series, cv: dict, periods_per_year: int = 365) -> pd.Series:
    """Sharpe of a fixed-rule strategy on each contiguous CPCV group (sub-period distribution)."""
    out = {}
    for i, g in enumerate(cv["groups"]):
        r = returns.iloc[g]
        out[f"g{i}:{r.index[0].date()}..{r.index[-1].date()}"] = metrics.sharpe_ratio(
            r.to_numpy(), periods_per_year
        )
    return pd.Series(out, name="sharpe")


def compare_variants(
    returns: pd.DataFrame,
    n_groups: int = 6,
    periods_per_year: int = 365,
    n_draws: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Variant table: full-sample Sharpe, per-group min / mean / share positive, Rademacher haircut.

    `returns` is (T, N): one daily return column per variant, same dates. The
    haircut treats the N columns as the set of strategies that was searched.
    """
    cv = cpcv(returns.index, n_groups=n_groups, n_test=1)
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        gs = group_sharpes(returns[col], cv, periods_per_year)
        rows[col] = {
            "sharpe": metrics.sharpe_ratio(r.to_numpy(), periods_per_year),
            "groups_min": gs.min(),
            "groups_mean": gs.mean(),
            "groups_pos_%": 100 * (gs > 0).mean(),
            "n_days": len(r),
        }
    table = pd.DataFrame(rows).T
    X = returns.fillna(0.0).to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    daily_sr = table["sharpe"] / np.sqrt(periods_per_year)
    table["sharpe_after_haircut"] = [
        rademacher_haircut_sharpe(float(sr), X, n_draws=n_draws, seed=seed)
        * np.sqrt(periods_per_year)
        for sr in daily_sr
    ]
    return table


def regime_table(
    returns: pd.Series, benchmark: pd.Series, window: int = 90, periods_per_year: int = 365
) -> pd.DataFrame:
    """Return / Sharpe in bull vs bear regimes (sign of trailing `window`-day benchmark return)."""
    bench = benchmark.reindex(returns.index).fillna(0.0)
    trailing = (1 + bench).rolling(window).apply(np.prod, raw=True) - 1
    regime = pd.Series(np.where(trailing.shift(1) >= 0, "bull", "bear"), index=returns.index)
    regime[trailing.shift(1).isna()] = "warmup"
    rows = {}
    for name, grp in returns.groupby(regime):
        if name == "warmup":
            continue
        rows[name] = {
            "days": len(grp),
            "ann_return_%": 100 * grp.mean() * periods_per_year,
            "sharpe": metrics.sharpe_ratio(grp.to_numpy(), periods_per_year),
            "benchmark_ann_%": 100 * bench[grp.index].mean() * periods_per_year,
        }
    return pd.DataFrame(rows).T
