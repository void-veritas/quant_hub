"""Data quality checks: gap detection and bar sanity."""

from __future__ import annotations

import pandas as pd


def find_gaps(ts: pd.Series, freq: str) -> pd.DataFrame:
    """Missing regular timestamps between min(ts) and max(ts).

    Returns one row per contiguous gap: gap_start, gap_end (both missing
    timestamps, inclusive) and n_missing. Legitimate holes (delistings,
    venue outages) show up here too — this reports, it doesn't judge.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values().unique()
    if len(ts) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing"])
    expected = pd.date_range(ts[0], ts[-1], freq=freq, tz="UTC")
    missing = expected.difference(ts)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing"])
    step = pd.tseries.frequencies.to_offset(freq)
    breaks = missing.to_series().diff() != step
    group = breaks.cumsum()
    out = missing.to_series().groupby(group).agg(["min", "max", "size"])
    out.columns = ["gap_start", "gap_end", "n_missing"]
    return out.reset_index(drop=True)


def check_bars(df: pd.DataFrame) -> list[str]:
    """Sanity issues in an OHLCV frame; returns human-readable problem strings."""
    problems = []
    if df["ts"].duplicated().any():
        problems.append(f"{df['ts'].duplicated().sum()} duplicate timestamps")
    if not df["ts"].is_monotonic_increasing:
        problems.append("timestamps not sorted")
    for col in ("open", "high", "low", "close"):
        bad = (df[col] <= 0).sum()
        if bad:
            problems.append(f"{bad} non-positive values in {col}")
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        problems.append(f"{bad_hl} bars with high < low")
    body_out = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
    if body_out:
        problems.append(f"{body_out} bars with close outside [low, high]")
    return problems
