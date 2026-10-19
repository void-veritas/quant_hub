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


def test_ewma_covariance_reacts_faster_and_risk_share_cap_binds():
    import numpy as np
    import pandas as pd

    from quant_hub.portfolio.pit_construction import cap_risk_share, pit_covariance

    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    vol = np.where(np.arange(200) < 170, 0.01, 0.05)  # regime switch 30 days before the end
    ret = pd.DataFrame(rng.normal(0, vol[:, None], (200, 5)), index=dates, columns=list("ABCDE"))
    d = dates[-1] + pd.Timedelta(days=1)
    lw = pit_covariance(ret, d, 120, "lw")
    ew = pit_covariance(ret, d, 120, "ewma", halflife=15)
    assert np.sqrt(np.diag(ew)).mean() > 1.4 * np.sqrt(np.diag(lw)).mean()
    cov = np.diag([0.04, 0.01, 0.01, 0.01, 0.01])
    w = np.array([0.5, 0.125, 0.125, 0.125, 0.125])
    capped = cap_risk_share(w, cov, 0.3)
    share = capped * (cov @ capped) / (capped @ cov @ capped)
    assert share.max() <= 0.3 + 1e-6 and np.sign(capped).tolist() == np.sign(w).tolist()


def test_project_out_zeroes_style_loadings():
    import numpy as np
    import pandas as pd

    from quant_hub.portfolio.neutral import project_out

    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range("2024-01-01", periods=3, freq="D"):
        for a in range(20):
            rows.append(
                dict(
                    date=d,
                    asset=f"A{a}",
                    size=rng.normal(),
                    vol=rng.normal(),
                    w=rng.normal(0, 0.05),
                )
            )
    df = pd.DataFrame(rows)
    out = project_out(
        df["w"], df["date"], df["asset"], df[["date", "asset", "size", "vol"]], ("size", "vol")
    )
    df["w2"] = out
    for _, g in df.groupby("date"):
        assert abs((g["w2"] * g["size"]).sum()) < 1e-9 and abs((g["w2"] * g["vol"]).sum()) < 1e-9
    assert out.abs().max() <= 0.25 + 1e-12
