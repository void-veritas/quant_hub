"""Panel risk model on a synthetic panel with a known factor structure."""

import numpy as np
import pandas as pd

from quant_hub.risk import panel_risk_model as prm


def synthetic(n_assets=100, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    size_raw = rng.normal(size=n_assets)
    f_mkt = rng.normal(0, 0.02, n_days)
    f_size = rng.normal(0, 0.015, n_days)
    f_vol = rng.normal(0, 0.015, n_days)
    rows = []
    vol_level = np.exp(rng.normal(np.log(0.04), 0.3, n_assets))
    price = np.full(n_assets, 100.0)
    for t, d in enumerate(dates):
        z_size = (size_raw - size_raw.mean()) / size_raw.std()
        z_vol = (np.log(vol_level) - np.log(vol_level).mean()) / np.log(vol_level).std()
        eps = rng.normal(0, vol_level)
        # return earned from decision t-1 to decision t, stamped at t
        r = f_mkt[t] + z_size * f_size[t] + z_vol * f_vol[t] + eps
        price = price * (1 + r)
        for i, a in enumerate(assets):
            rows.append(
                dict(
                    date=d,
                    asset=a,
                    close=price[i],
                    total_returns=r[i],
                    ewma_vol=vol_level[i],
                    carry_funding=0.0,
                    in_universe=True,
                )
            )
    panel = pd.DataFrame(rows)
    caps = pd.DataFrame(
        np.tile(np.exp(size_raw * 1.0 + 20), (n_days, 1)), index=dates, columns=assets
    )
    truth = pd.DataFrame({"market": f_mkt, "size": f_size, "vol": f_vol}, index=dates)
    return panel, caps, truth


def test_recovers_factor_returns_and_exposures():
    panel, caps, truth = synthetic()
    expo = prm.build_exposures(panel, caps=caps, momentum_days=5, funding_days=5)
    model = prm.fit(panel, expo, factors=("market", "size", "vol"))
    # factor return for date d is earned d -> d+1, truth is stamped at d+1
    for k in ("market", "size", "vol"):
        est = model.factor_returns[k]
        c = np.corrcoef(est.to_numpy(), truth[k].shift(-1).reindex(est.index).to_numpy())[0, 1]
        assert c > 0.95, (k, c)
    assert model.r2.mean() > 0.1  # weighted R² with idio 4% daily vs market 2%
    assert (model.n == 100).all()


def test_forecast_is_calibrated_and_attribution_adds_up():
    panel, caps, _ = synthetic(n_days=600, seed=1)
    expo = prm.build_exposures(panel, caps=caps, momentum_days=5, funding_days=5)
    # no winsorisation here so that factor + idio PnL reproduces the realised return exactly
    model = prm.fit(panel, expo, factors=("market", "size", "vol"), winsor_sd=1e6)
    rng = np.random.default_rng(2)
    # a long-short book tilted on size, rebalanced daily
    w = expo[["date", "asset", "size"]].copy()
    w["weight"] = (w["size"] + rng.normal(0, 0.5, len(w))) / 100.0
    w = w[["date", "asset", "weight"]]
    pred = prm.predicted_vol(model, w, min_obs=60)
    real = (
        panel.assign(r_next=panel.groupby("asset")["total_returns"].shift(-1))
        .merge(w, on=["date", "asset"])
        .assign(p=lambda x: x["weight"] * x["r_next"])
        .groupby("date")["p"]
        .sum()
    )
    bias = prm.bias_statistic(pred["total"], real)
    assert 0.8 < bias["bias"] < 1.25, bias["bias"]
    assert 0.03 < pred["idio_share"].mean() < 0.98  # 100 names at 1/100: idio diversifies away
    pnl = prm.attribute(model, w)
    # attribution reproduces the realised return of the book (returns are not winsorised here)
    diff = (pnl["total"] - real.reindex(pnl.index)).abs().max()
    assert diff < 1e-10
    summ = prm.summarise_attribution(pnl)
    assert abs(summ["share"].drop("total").sum() - 1) < 1e-9
