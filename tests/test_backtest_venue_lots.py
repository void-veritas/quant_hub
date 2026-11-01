"""Lot rounding of the daily-rebalance backtester survives float noise (2026-11-01)."""

import numpy as np

from quant_hub.backtest.engines.daily_rebalance import apply_venue_constraints


def test_lot_rounding_keeps_exact_multiples():
    # 0.29 / 0.01 = 28.999999999999996 in floats: 29 lots, not 28
    out = apply_venue_constraints(
        np.array([0.0]), np.array([0.29]), np.array([1000.0]), 0.0, np.array([0.01])
    )
    assert out[0] == 0.29
    # a size between two lots still rounds towards zero
    out = apply_venue_constraints(
        np.array([0.0]), np.array([0.2999]), np.array([1000.0]), 0.0, np.array([0.01])
    )
    assert np.isclose(out[0], 0.29)
