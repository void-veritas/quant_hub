"""Point-in-time guarantees of the panel tools moved from yolo-microstructure."""

import numpy as np
import pandas as pd

from quant_hub.portfolio.neutral import neutralise, rolling_betas
from quant_hub.portfolio.pit_construction import build
from quant_hub.universe import cap_universe

ASSETS = [f"A{i}" for i in range(8)]


def synthetic_panel(seed: int = 0, days: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for a in ASSETS:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
        rows.append(
            pd.DataFrame(
                {"date": dates, "asset": a, "close": px, "funding": rng.normal(0, 3e-4, days)}
            )
        )
    df = pd.concat(rows, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)
    df["price_returns"] = df.groupby("asset")["close"].pct_change()
    df["in_universe"] = True
    df["weight"] = rng.normal(0, 0.05, len(df))
    df["scaled_weight"] = df["weight"]
    return df


def test_betas_and_neutralisation_are_point_in_time():
    df = synthetic_panel()
    last = df["date"].max()
    df["beta"] = rolling_betas(df, window=60)
    shocked = df.copy()
    shocked.loc[shocked["date"] == last, "price_returns"] *= 5
    assert np.allclose(
        df.loc[df["date"] == last, "beta"],
        rolling_betas(shocked, window=60)[df["date"] == last],
        equal_nan=True,
    )
    w_d = neutralise(df["scaled_weight"], df["date"], "dollar")
    assert w_d.groupby(df["date"]).sum().abs().max() < 1e-6
    w_b = neutralise(df["scaled_weight"], df["date"], "beta", beta=df["beta"])
    ok = df["beta"].notna()
    pb = (w_b * df["beta"]).where(ok).groupby(df["date"]).sum()
    assert pb[ok.groupby(df["date"]).all()].abs().max() < 1e-9


def test_construction_covariance_is_point_in_time():
    df = synthetic_panel()
    last = df["date"].max()
    shocked = df.copy()
    shocked.loc[shocked["date"] == last, "price_returns"] *= 10
    for m in ("vol_target", "mv", "hrp"):
        a = build(df, m, df["scaled_weight"], window=120)
        b = build(shocked, m, shocked["scaled_weight"], window=120)
        sel = df["date"] == last
        assert np.allclose(a[sel], b[sel], equal_nan=True), m


def test_cap_universe_uses_yesterdays_rank():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    tickers = [f"C{i}" for i in range(15)]
    rows = [
        (t, d, 1e9 * (i + 1) * (1 + 0.01 * np.sin(k / 7 + i)))
        for i, t in enumerate(tickers)
        for k, d in enumerate(dates)
    ]
    mc = pd.DataFrame(rows, columns=["Ticker", "Date", "MarketCapUSD"])
    base = cap_universe(mc, n=5, smooth=1, min_history=1)
    shocked = mc.copy()
    last = dates[-1]
    shocked.loc[shocked["Date"] == last, "MarketCapUSD"] = shocked.loc[
        shocked["Date"] == last, "MarketCapUSD"
    ].to_numpy()[::-1]
    out = cap_universe(shocked, n=5, smooth=1, min_history=1)
    assert (
        base[base["date"] == last]
        .set_index("asset")["in_universe"]
        .equals(out[out["date"] == last].set_index("asset")["in_universe"])
    )
