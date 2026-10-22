"""Curated copper layer: contract calendar, volume-based roll, continuous series, carry.

Built from `raw/futures_1d` (one asset per contract). Nothing here is bought or
fetched; everything is a deterministic function of the raw bars, and the rules are
written down so a backtest can cite them:

- **Calendar.** Contracts are keyed by `HGZ2026`-style ids (raw symbols are reused
  every decade; `databento.contract_id`). Expiry is approximated by the third-last
  business day of the delivery month (COMEX copper rule); the empirical last bar is
  kept next to it.
- **Sessions.** Databento's `ohlcv-1d` is a UTC day; the Sunday-evening Globex
  opening becomes a tiny Sunday bar. Sundays are dropped (`daily_panel`), so one row
  is one weekday.
- **Front and second.** Per weekday the front is the unexpired outright with the
  largest 5-day average volume, with hysteresis: the front never rolls back to an
  earlier expiry, and it must leave a contract at least `ROLL_GUARD` business days
  before its expiry. The second is the next unexpired contract by expiry whose 5-day
  volume is at least `SECOND_MIN_SHARE` of the front's (thin serial months are
  skipped), else the loudest of the next `SECOND_SPAN` expiries. This is the "roll
  by volume" rule of the continuous series.
- **Returns.** The continuous front return on day t is earned on the contract that
  was front at t-1 (`ret_front`), so a roll is a trade, not a price jump. Same for
  the second (`ret_second`); `ret_spread = ret_front - ret_second` is the
  dollar-neutral calendar-spread return (long front, short second).
- **Carry.** `carry_ann = (p_front / p_second - 1) * 365 / days(front -> second expiry)`:
  positive in backwardation (the long front earns roll yield), negative in contango.

Outputs (`build_copper`): a `curve` frame (one row per weekday) and the
`contract_table`; `engine_inputs` turns the raw bars into the wide, forward-filled
contract prices the daily-rebalance engine needs.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from quant_hub.commodities import storage
from quant_hub.commodities.connectors.databento import (
    expiry_code,
    is_spread,
    resolve_year,
    root_of,
)

MONTHS = "FGHJKMNQUVXZ"
ROLL_GUARD = 3  # business days before expiry by which the front must have rolled
VOL_WINDOW = 5  # days of volume averaged for the roll decision
SECOND_MIN_SHARE = 0.05  # the second needs at least this share of the front's 5-day volume
SECOND_SPAN = 4  # ... among the next four expiries; otherwise the loudest of those four


def load_contract_bars(root: str = "HG", exchange: str = "cme") -> pd.DataFrame:
    """Raw daily bars of every outright contract of `root` (spreads excluded), long form."""
    glob = storage.dataset_glob("futures_1d")
    df = duckdb.sql(
        f"SELECT asset, ts, open, high, low, close, volume FROM read_parquet('{glob}', "
        f"hive_partitioning = true) WHERE exchange = '{exchange}'"
    ).df()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    keep = df["asset"].map(lambda a: root_of(a) == root and not is_spread(a))
    return df[keep].sort_values(["asset", "ts"]).reset_index(drop=True)


def _third_last_business_day(year: int, month: int) -> pd.Timestamp:
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    days = pd.bdate_range(end - pd.Timedelta(days=10), end)
    return days[-3]


def resolve_expiry_month(symbol: str, first_bar: pd.Timestamp) -> pd.Timestamp:
    """Delivery month of a contract id (`HGZ2026`), or of a one-digit symbol via its first bar."""
    code = expiry_code(symbol)
    if code is None:
        raise ValueError(f"{symbol} is not an outright contract symbol")
    return pd.Timestamp(year=resolve_year(code, first_bar), month=MONTHS.index(code[0]) + 1, day=1)


def contract_table(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per contract: expiry month, approximate expiry date, first/last bar, bars, volume."""
    g = bars.groupby("asset")
    t = pd.DataFrame(
        {
            "first_bar": g["ts"].min(),
            "last_bar": g["ts"].max(),
            "bars": g.size(),
            "volume": g["volume"].sum(),
        }
    )
    t["expiry_month"] = [
        resolve_expiry_month(s, fb) for s, fb in zip(t.index, t["first_bar"], strict=True)
    ]
    t["expiry"] = [_third_last_business_day(m.year, m.month) for m in t["expiry_month"]]
    t["expiry"] = t["expiry"].dt.tz_localize("UTC")
    return t.sort_values(["expiry_month", "first_bar"]).rename_axis("contract")


def daily_panel(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide close (forward-filled inside each contract's life) and volume (0 = no bar), weekdays."""
    b = bars[bars["ts"].dt.dayofweek < 5]
    close = b.pivot(index="ts", columns="asset", values="close")
    volume = b.pivot(index="ts", columns="asset", values="volume").fillna(0.0)
    return close.ffill().where(close.bfill().notna()), volume


def select_front_second(
    close: pd.DataFrame, volume: pd.DataFrame, table: pd.DataFrame
) -> pd.DataFrame:
    """Per weekday: front and second contract under the volume rule with hysteresis."""
    order = list(table.index)  # by expiry
    pos = {c: i for i, c in enumerate(order)}
    guard = {c: table.loc[c, "expiry"] - pd.offsets.BDay(ROLL_GUARD) for c in order}
    expiry = table["expiry"]
    v5 = volume.rolling(VOL_WINDOW, min_periods=1).mean()
    rows = []
    front = None
    for ts in close.index:
        live = [c for c in order if ts < guard[c] and not np.isnan(close.at[ts, c])]
        if not live:
            rows.append(dict(ts=ts, front=None, second=None))
            continue
        best = max(live, key=lambda c: (v5.at[ts, c], -pos[c]))
        if front is None or front not in live or pos[best] > pos[front]:
            front = best  # roll forward only; a front past its guard is replaced
        after = [c for c in live if pos[c] > pos[front] and volume.at[ts, c] > 0][:SECOND_SPAN]
        liquid = [c for c in after if v5.at[ts, c] >= SECOND_MIN_SHARE * v5.at[ts, front]]
        second = (
            liquid[0] if liquid else (max(after, key=lambda c: v5.at[ts, c]) if after else None)
        )
        rows.append(dict(ts=ts, front=front, second=second))
    out = pd.DataFrame(rows).set_index("ts")
    out["front_expiry"] = out["front"].map(expiry)
    out["second_expiry"] = out["second"].map(expiry)
    return out


def build_curve(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curve frame (prices, volumes, continuous returns, spread return, carry) + contract table."""
    table = contract_table(bars)
    close, volume = daily_panel(bars)
    sel = select_front_second(close, volume, table)
    idx = close.index
    cmap = {c: i for i, c in enumerate(close.columns)}
    C, V = close.to_numpy(), volume.to_numpy()

    def pick(names, P):
        out = np.full(len(idx), np.nan)
        for i, n in enumerate(names):
            if n is not None and n in cmap:
                out[i] = P[i, cmap[n]]
        return out

    f, s = sel["front"].tolist(), sel["second"].tolist()
    curve = pd.DataFrame(index=idx)
    curve["front"], curve["second"] = f, s
    curve["p_front"], curve["p_second"] = pick(f, C), pick(s, C)
    curve["v_front"], curve["v_second"] = pick(f, V), pick(s, V)
    # returns earned on the contracts held at t-1
    prev_f, prev_s = [None, *f[:-1]], [None, *s[:-1]]
    curve["ret_front"] = pick(prev_f, C) / pick(prev_f, np.vstack([C[:1], C[:-1]])) - 1
    curve["ret_second"] = pick(prev_s, C) / pick(prev_s, np.vstack([C[:1], C[:-1]])) - 1
    curve.loc[idx[0], ["ret_front", "ret_second"]] = np.nan
    curve["ret_spread"] = curve["ret_front"] - curve["ret_second"]
    days = (sel["second_expiry"] - sel["front_expiry"]).dt.days.clip(lower=1)
    curve["days_to_second"] = days
    curve["carry_ann"] = (curve["p_front"] / curve["p_second"] - 1) * 365 / days
    prev = curve["front"].shift(1)
    curve["roll"] = (curve["front"] != prev) & prev.notna()
    return curve, table


def engine_inputs(bars: pd.DataFrame) -> pd.DataFrame:
    """Wide contract closes (weekdays, forward-filled within a contract's life) for the engine."""
    close, _ = daily_panel(bars)
    return close


def build_copper(root: str = "HG") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw copper bars and build the curve and the contract table."""
    bars = load_contract_bars(root)
    return build_curve(bars)


def write_curated(curve: pd.DataFrame, table: pd.DataFrame, root: str = "HG") -> None:
    """Store the curve and the contract table in the curated layer (synced to the bucket)."""
    curve.reset_index().to_parquet(
        storage.layer_path("curated", root, "curve.parquet"), index=False
    )
    table.reset_index().to_csv(storage.layer_path("curated", root, "contracts.csv"), index=False)


def load_curated(root: str = "HG") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored curve (indexed by `ts`) and contract table."""
    curve = pd.read_parquet(storage.layer_path("curated", root, "curve.parquet"))
    curve["ts"] = pd.to_datetime(curve["ts"], utc=True)
    table = pd.read_csv(storage.layer_path("curated", root, "contracts.csv"), index_col="contract")
    return curve.set_index("ts"), table
