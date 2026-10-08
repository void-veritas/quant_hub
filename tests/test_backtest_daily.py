"""Tests for the daily rebalance engine against rsims reference output."""

import numpy as np
import pandas as pd
import pytest

from quant_hub.backtest.engines.daily_rebalance import positions_from_buffer, run_backtest

# crypto-pod trading/yolo/yolo_simulation.ipynb, first simulated day 2020-02-02:
# scaled target weights (cell 15) and the resulting rsims ledger (cell 25) with
# initial_cash 100000, trade_buffer 0.1, commission 0.05%, margin 10%.
PRICES = {
    "BCH": 379.920,
    "BTC": 9377.010,
    "ETC": 11.565,
    "ETH": 187.480,
    "LINK": 2.826,
    "LTC": 70.750,
}
WEIGHTS = {
    "BCH": -0.035413670,
    "BTC": -0.069545871,
    "ETC": 0.010408448,
    "ETH": -0.009550267,
    "LINK": -0.106200685,
    "LTC": 0.073186888,
}
REF_POS = {
    "BCH": 0.0,
    "BTC": -0.2084446,
    "ETC": 0.0,
    "ETH": 0.0,
    "LINK": -1988.7007988,
    "LTC": 32.7729866,
}
REF_VAL = {"BCH": 0.0, "BTC": -1954.587, "ETC": 0.0, "ETH": 0.0, "LINK": -5620.068, "LTC": 2318.689}
REF_COM = {
    "BCH": 0.0,
    "BTC": 0.9772936,
    "ETC": 0.0,
    "ETH": 0.0,
    "LINK": 2.8100342,
    "LTC": 1.1593444,
}


def _frames(rows):
    idx = pd.to_datetime(["2020-02-02", "2020-02-03"][: len(rows)])
    cols = list(PRICES)
    px = pd.DataFrame([[PRICES[c] for c in cols]] * len(rows), index=idx, columns=cols)
    w = pd.DataFrame([[WEIGHTS[c] for c in cols]] * len(rows), index=idx, columns=cols)
    return px, w


def test_rsims_first_day_edge_of_band_trades():
    """Outside the band trade to its edge; inside it (band clipped at 0) stay flat."""
    px, w = _frames([0])
    res = run_backtest(
        px, w, trade_buffer=0.1, initial_cash=100_000, margin=0.1, commission_pct=0.0005
    )
    for c in PRICES:
        # reference weights are printed to 9 digits, hence the loose tolerance
        assert res.positions.iloc[0][c] == pytest.approx(REF_POS[c], abs=1e-3), c
        assert res.position_value.iloc[0][c] == pytest.approx(REF_VAL[c], abs=1e-2), c
        assert res.commissions.iloc[0][c] == pytest.approx(REF_COM[c], abs=1e-5), c
    gross = sum(abs(v) for v in REF_VAL.values())
    assert res.margin.iloc[0] == pytest.approx(0.1 * gross, abs=1e-2)
    assert res.equity.iloc[0] == pytest.approx(100_000 - sum(REF_COM.values()), abs=1e-3)


def test_no_buffer_hits_target_and_second_day_no_trade():
    px, w = _frames([0, 1])
    res = run_backtest(px, w, trade_buffer=0.0, initial_cash=100_000, margin=0.1)
    for c in PRICES:
        assert res.position_value.iloc[0][c] == pytest.approx(WEIGHTS[c] * 100_000, rel=1e-9), c
    assert np.allclose(res.trades.iloc[1].to_numpy(), 0.0)  # same prices, same weights


def test_relative_buffer_excel_mode():
    cur = np.array([100.0])
    px = np.array([10.0])
    # current weight 0.10, target 0.104: 4% deviation < 5% buffer -> no trade
    assert positions_from_buffer(cur, px, np.array([0.104]), 10_000, 0.05, "relative")[0] == 100.0
    # 6% deviation -> trade all the way to the target
    assert positions_from_buffer(cur, px, np.array([0.106]), 10_000, 0.05, "relative")[
        0
    ] == pytest.approx(106.0)
    # zero target closes regardless of buffer
    assert positions_from_buffer(cur, px, np.array([0.0]), 10_000, 0.05, "relative")[0] == 0.0


def test_funding_and_pnl_accrue_on_previous_positions():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    px = pd.DataFrame({"X": [100.0, 110.0]}, index=idx)
    w = pd.DataFrame({"X": [0.5, 0.5]}, index=idx)
    f = pd.DataFrame({"X": [0.01, 0.002]}, index=idx)  # paid to longs
    res = run_backtest(px, w, f, initial_cash=1_000, margin=0.1)
    # day 1: no position yet -> no funding; buy 5 units
    assert res.funding.iloc[0]["X"] == 0.0
    assert res.positions.iloc[0]["X"] == pytest.approx(5.0)
    # day 2: pnl 5*(110-100)=50, funding 5*110*0.002=1.1
    assert res.period_pnl.iloc[1]["X"] == pytest.approx(51.1)
    assert res.funding.iloc[1]["X"] == pytest.approx(1.1)
    assert res.equity.iloc[1] == pytest.approx(1_051.1)
    # capitalise_profits=False: sizing capital stays at min(initial, equity) = 1000 -> 500/110 units
    assert res.positions.iloc[1]["X"] == pytest.approx(500 / 110)
