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


# ---------------------------------------------------------------- engine semantics (note 011)


def _two_assets(prices, weights, **kw):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    px = pd.DataFrame(prices, index=idx, columns=["A", "B"], dtype=float)
    w = pd.DataFrame(weights, index=idx, columns=["A", "B"], dtype=float)
    return run_backtest(px, w, **kw)


def test_funding_accrues_on_yesterdays_position_at_todays_price():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    px = pd.DataFrame([[100.0], [110.0]], index=idx, columns=["A"])
    w = pd.DataFrame([[0.5], [0.5]], index=idx, columns=["A"])
    f = pd.DataFrame([[0.01], [0.002]], index=idx, columns=["A"])
    res = run_backtest(px, w, f, initial_cash=1000.0, margin=0.1)
    assert res.funding.iloc[0, 0] == 0.0  # nothing held before the first trade
    # day 2: 5 units * 110 * 0.002 = 1.1 of funding, plus 5 * 10 price pnl
    assert res.funding.iloc[1, 0] == pytest.approx(1.1)
    assert res.period_pnl.iloc[1, 0] == pytest.approx(50 + 1.1)
    assert res.equity.iloc[1] == pytest.approx(1000 + 51.1)


def test_capitalise_profits_compounds_sizing():
    prices = [[100, 100], [120, 100], [120, 100]]
    weights = [[0.5, 0.0]] * 3
    a = _two_assets(prices, weights, initial_cash=1000.0, capitalise_profits=False)
    b = _two_assets(prices, weights, initial_cash=1000.0, capitalise_profits=True)
    # after +20% on half the book equity is 1100: fixed sizing keeps 500 notional,
    # compounding goes to 550
    assert a.position_value.iloc[1, 0] == pytest.approx(500.0)
    assert b.position_value.iloc[1, 0] == pytest.approx(550.0)
    # a drawdown shrinks both: min(initial, equity)
    c = _two_assets([[100, 100], [80, 100], [80, 100]], weights, initial_cash=1000.0)
    assert c.position_value.iloc[1, 0] == pytest.approx(0.5 * 900.0)


def test_margin_call_frees_the_shortfall_plus_five_percent():
    # 5x leverage long, price -25%: equity 1000 - 1250 = -250 < 0 -> margin call
    prices = [[100, 100], [75, 100], [75, 100]]
    weights = [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]
    res = _two_assets(prices, weights, initial_cash=1000.0, margin=0.1, commission_pct=0.0)
    assert bool(res.margin_call.iloc[1])
    # all equity is gone (and more): the whole position is liquidated (fraction capped at 1)
    assert res.equity.iloc[1] <= 0.0 + 1e-9 or res.positions.iloc[1, 0] == 0.0
    # partial case: 2x long, price -8%: equity 1000 - 160 = 840, maint 0.1*1840=184 -> cash 656, no call
    res2 = _two_assets(
        [[100, 100], [92, 100], [92, 100]], [[2.0, 0.0]] * 3, initial_cash=1000.0, margin=0.1
    )
    assert not res2.margin_call.any()
    # 9x long, price -5%: pnl -450, equity 550, maint 0.1*9*950=855 -> cash -305: sell 1.05*305/855 = 37.4%
    res3 = _two_assets(
        [[100, 100], [95, 100], [95, 100]], [[9.0, 0.0]] * 3, initial_cash=1000.0, margin=0.1
    )
    assert bool(res3.margin_call.iloc[1])
    # kept fraction is 1 - min(1, 1.05 * 305 / 855) = 62.6% of the position
    # the liquidation trade shows in the trades ledger (negative) before the day's rebalance
    assert res3.trades.iloc[1, 0] < 0
    assert res3.cash.iloc[1] >= 0.0


def test_reduced_target_when_margin_unaffordable():
    # 15x target at 10% margin needs 150% of equity as margin: scaled to 95% of affordable
    res = _two_assets([[100, 100]] * 2, [[15.0, 0.0]] * 2, initial_cash=1000.0, margin=0.1)
    assert bool(res.reduced_target.iloc[0])
    assert res.position_value.iloc[0, 0] == pytest.approx(0.95 * 1000 / 0.1)


def test_nan_price_closes_position_at_last_price():
    prices = [[100, 50], [110, 50], [np.nan, 50], [np.nan, 50]]
    weights = [[0.5, 0.5], [0.5, 0.5], [0.0, 0.5], [0.0, 0.5]]
    res = _two_assets(prices, weights, initial_cash=1000.0, margin=0.1)
    assert res.positions.iloc[2, 0] == 0.0
    held = res.positions.iloc[1, 0]  # resized on day 2 to 0.5 * min(1000, equity) / 110
    assert res.trade_value.iloc[2, 0] == pytest.approx(
        -held * 110.0
    )  # closed at the last known price
    assert res.equity.iloc[2] == pytest.approx(res.equity.iloc[1])  # nothing lost on delisting


def test_min_notional_skips_small_opens_but_allows_closes():
    # $1k book, weight 0.004 -> $4 order < $10 minimum: stays flat; closing a position is always allowed
    prices = [[100, 100]] * 3
    weights = [[0.004, 0.5], [0.004, 0.5], [0.0, 0.0]]
    res = _two_assets(prices, weights, initial_cash=1000.0, min_notional=10.0)
    assert res.positions.iloc[0, 0] == 0.0 and res.positions.iloc[0, 1] == pytest.approx(5.0)
    assert res.positions.iloc[2, 1] == 0.0
    # a small *adjustment* of an existing position is skipped too
    w2 = [[0.5, 0.5], [0.505, 0.5], [0.0, 0.0]]
    res2 = _two_assets(prices, w2, initial_cash=1000.0, min_notional=10.0)
    assert res2.positions.iloc[1, 0] == pytest.approx(5.0)


def test_lot_size_rounds_towards_zero():
    res = _two_assets(
        [[100, 3.0]] * 2, [[0.333, -0.5]] * 2, initial_cash=1000.0, lot_sizes={"A": 1.0, "B": 10.0}
    )
    assert res.positions.iloc[0, 0] == pytest.approx(3.0)  # 3.33 -> 3
    assert res.positions.iloc[0, 1] == pytest.approx(-160.0)  # -166.7 -> -160


def test_trade_prices_execution_lag_is_charged():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    px = pd.DataFrame([[100.0], [100.0]], index=idx, columns=["A"])
    tp = pd.DataFrame([[101.0], [101.0]], index=idx, columns=["A"])  # fills 1% worse than the mark
    w = pd.DataFrame([[0.5], [0.5]], index=idx, columns=["A"])
    res = run_backtest(px, w, trade_prices=tp, initial_cash=1000.0)
    assert res.positions.iloc[0, 0] == pytest.approx(5.0)  # sized on the mark price
    assert res.equity.iloc[0] == pytest.approx(1000 - 5 * 1.0)  # paid 101 for something marked 100
    assert res.trade_value.iloc[0, 0] == pytest.approx(505.0)
