"""Daily target-weight backtester for perpetual futures with funding.

A faithful port of rsims::fixed_commission_backtest_with_funding (Robot Wealth),
the engine behind the YOLO simulations, plus the no-trade-buffer variant the
community's Excel helper actually trades. Semantics per period t:

1. Funding accrues on positions held from t-1 at price_t: pos * price_t * rate_t.
   `funding_rates` are "paid to longs" per period (sign already flipped).
2. Period PnL = pos * (price_t - price_{t-1}) + funding, settled into cash
   (futures-style); margin is `margin` x gross notional, moved between cash and
   a margin account for accounting only.
3. Sizing capital: equity if `capitalise_profits` else min(initial_cash, equity)
   (profits are not reinvested, drawdowns shrink the book).
4. Targets from the no-trade buffer (see `positions_from_buffer`), then trades at
   price_t, commission = commission_pct x |traded notional|.
5. If post-trade cash would fall below maintenance margin, all targets are
   scaled down proportionally (0.95 x the affordable notional), as in rsims.

Inputs are wide DataFrames indexed by date with identical columns (one per
asset): prices, target_weights (date-aligned with prices: the weight you trade
into at price_t), funding_rates. NaN prices with non-zero weights are an error;
NaN weights/funding count as 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def positions_from_buffer(
    current_positions: np.ndarray,
    prices: np.ndarray,
    target_weights: np.ndarray,
    cap_equity: float,
    trade_buffer: float,
    mode: str = "absolute",
    trade_to: str = "edge",
) -> np.ndarray:
    """Target positions (units) after applying a no-trade buffer.

    mode="absolute" (rsims): the band is target ± buffer/2 in weight units, clipped
    at zero so a long target never yields a short position; outside the band you
    trade to the band's edge (`trade_to="edge"`) or to the target.
    mode="relative" (RW Excel helper rev 2.07): no trade while
    |current - target| < buffer x |target|, otherwise trade to the target.
    A zero or NaN target always closes the position, in both modes.
    """
    pos = current_positions.copy()
    cur_w = np.where(prices > 0, current_positions * prices / cap_equity, 0.0)
    for j in range(len(pos)):
        tw = target_weights[j]
        if np.isnan(tw) or tw == 0.0:
            pos[j] = 0.0
            continue
        if trade_buffer == 0.0:
            pos[j] = tw * cap_equity / prices[j]
            continue
        if mode == "absolute":
            lo, hi = tw - trade_buffer / 2, tw + trade_buffer / 2
            if tw > 0:
                lo = max(0.0, lo)
            else:
                hi = min(0.0, hi)
            if cur_w[j] < lo:
                pos[j] = (lo if trade_to == "edge" else tw) * cap_equity / prices[j]
            elif cur_w[j] > hi:
                pos[j] = (hi if trade_to == "edge" else tw) * cap_equity / prices[j]
        elif mode == "relative":
            if abs(cur_w[j] - tw) >= trade_buffer * abs(tw):
                pos[j] = tw * cap_equity / prices[j]
        else:
            raise ValueError(mode)
    return pos


@dataclass
class BacktestResult:
    """Per-period, per-asset ledgers (wide frames) plus cash/equity series."""

    positions: pd.DataFrame
    position_value: pd.DataFrame
    trades: pd.DataFrame
    trade_value: pd.DataFrame
    commissions: pd.DataFrame
    funding: pd.DataFrame
    period_pnl: pd.DataFrame
    cash: pd.Series
    margin: pd.Series
    equity: pd.Series
    margin_call: pd.Series
    reduced_target: pd.Series

    def returns(self, on: str = "equity") -> pd.Series:
        """Daily returns on equity (compounding) or on initial equity (constant allocation)."""
        if on == "equity":
            return self.equity.pct_change().dropna()
        return (self.equity.diff() / self.equity.iloc[0]).dropna()

    def to_long(self) -> pd.DataFrame:
        """rsims-style long frame: one row per (date, asset) with all ledgers."""
        parts = {
            "position": self.positions,
            "value": self.position_value,
            "trades": self.trades,
            "trade_value": self.trade_value,
            "commission": self.commissions,
            "funding": self.funding,
            "period_pnl": self.period_pnl,
        }
        out = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1)
        out.index.names = ["date", "asset"]
        return out.reset_index()


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    trade_buffer: float = 0.0,
    buffer_mode: str = "absolute",
    trade_to: str = "edge",
    initial_cash: float = 10_000.0,
    margin: float = 0.05,
    commission_pct: float = 0.0,
    capitalise_profits: bool = False,
) -> BacktestResult:
    """Run the daily rebalance simulation. See module docstring for semantics."""
    if trade_buffer < 0:
        raise ValueError("trade_buffer must be >= 0")
    if not prices.index.equals(target_weights.index) or list(prices.columns) != list(
        target_weights.columns
    ):
        raise ValueError("prices and target_weights must share index and columns")
    if funding_rates is None:
        funding_rates = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if not prices.index.equals(funding_rates.index) or list(prices.columns) != list(
        funding_rates.columns
    ):
        raise ValueError("prices and funding_rates must share index and columns")

    P = prices.to_numpy(dtype=float)
    W = target_weights.to_numpy(dtype=float)
    F = np.nan_to_num(funding_rates.to_numpy(dtype=float))
    bad = np.isnan(P) & (np.nan_to_num(W) != 0)
    if bad.any():
        rows = prices.index[bad.any(axis=1)]
        raise ValueError(f"NaN prices where target weight is non-zero at {list(rows[:5])}")

    T, N = P.shape
    pos = np.zeros(N)
    prev_px = np.full(N, np.nan)
    cash = float(initial_cash)
    maint = 0.0

    out = {k: np.zeros((T, N)) for k in ("pos", "val", "trd", "trdv", "com", "fund", "pnl")}
    cash_s, margin_s, eq_s = np.zeros(T), np.zeros(T), np.zeros(T)
    mc_s, rt_s = np.zeros(T, bool), np.zeros(T, bool)

    for t in range(T):
        px = P[t]
        px0 = np.where(np.isnan(px), 0.0, px)  # NaN price: position must be 0 anyway
        fund = np.nan_to_num(pos * px0 * F[t])
        pnl = np.nan_to_num(pos * (px - prev_px)) + fund
        cash = cash + pnl.sum() + maint - margin * np.abs(pos * px0).sum()
        maint = margin * np.abs(pos * px0).sum()

        margin_call = False
        liq = np.zeros(N)
        liq_val = np.zeros(N)
        liq_com = np.zeros(N)
        if cash < 0:  # rsims: cash + maint < maint
            margin_call = True
            factor = 1.05 * (maint + cash) / maint if maint > 0 else 1.0
            liq = np.sign(pos) * np.minimum(np.abs(pos), factor * np.abs(pos))
            liq = np.where(np.abs(liq) > np.abs(pos), pos, liq)
            liq_val = liq * px0
            liq_com = np.abs(liq_val) * commission_pct
            pos = pos - liq
            cash = cash + maint - margin * np.abs(pos * px0).sum() - liq_com.sum()
            maint = margin * np.abs(pos * px0).sum()

        equity = cash + maint
        cap_equity = equity if capitalise_profits else min(initial_cash, equity)

        target = positions_from_buffer(
            pos, px0, W[t], cap_equity, trade_buffer, buffer_mode, trade_to
        )
        trades = target - pos
        trade_value = trades * px0
        com = np.abs(trade_value) * commission_pct
        post_cash = cash + maint - margin * np.abs(target * px0).sum() - com.sum()

        reduced = False
        if post_cash < maint:
            reduced = True
            max_notional = 0.95 * (cash + maint - com.sum()) / margin
            gross = np.abs(target * px0).sum()
            scale = max_notional / gross if gross > 0 else 0.0
            target = target * max(scale, 0.0)
            trades = target - pos
            trade_value = trades * px0
            com = np.abs(trade_value) * commission_pct
            post_cash = cash + maint - margin * np.abs(target * px0).sum() - com.sum()

        pos = target
        cash = post_cash
        maint = margin * np.abs(pos * px0).sum()

        out["pos"][t] = pos
        out["val"][t] = pos * px0
        out["trd"][t] = trades - liq
        out["trdv"][t] = trade_value - liq_val
        out["com"][t] = com + liq_com
        out["fund"][t] = fund
        out["pnl"][t] = pnl
        cash_s[t], margin_s[t], eq_s[t] = cash, maint, cash + maint
        mc_s[t], rt_s[t] = margin_call, reduced
        prev_px = px

    idx, cols = prices.index, prices.columns
    wide = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in out.items()}
    return BacktestResult(
        positions=wide["pos"],
        position_value=wide["val"],
        trades=wide["trd"],
        trade_value=wide["trdv"],
        commissions=wide["com"],
        funding=wide["fund"],
        period_pnl=wide["pnl"],
        cash=pd.Series(cash_s, index=idx, name="cash"),
        margin=pd.Series(margin_s, index=idx, name="margin"),
        equity=pd.Series(eq_s, index=idx, name="equity"),
        margin_call=pd.Series(mc_s, index=idx, name="margin_call"),
        reduced_target=pd.Series(rt_s, index=idx, name="reduced_target"),
    )
