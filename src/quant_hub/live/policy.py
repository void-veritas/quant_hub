"""Execution policy for a daily rebalance window (STATUS M7, "Исполнение").

At the window start every delta is quoted passively at the mid with add-liquidity-only
orders. Resting orders are re-quoted when the mid drifts beyond a tolerance. If the
basket's filled net exposure runs ahead of plan by more than the guard (all bids filled
in a market-wide drop, asks left hanging), the lagging side is crossed. After
``cross_after_minutes`` whatever remains is crossed; at the window end everything is
cancelled. The policy only proposes; the risk controller gates every order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_hub.live.contracts import Book
from quant_hub.live.oms import OMS, Order
from quant_hub.live.rounding import round_price, round_size, tick


@dataclass(frozen=True)
class Quote:
    """Best bid/ask at a moment; ``mid`` is what plan-vs-fact is measured against."""

    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True)
class Intent:
    coin: str
    side: str
    size: float
    price: float
    tif: str  # Alo | Ioc
    reduce_only: bool = False  # a full close below the venue minimum (exempt when reduce-only)


@dataclass(frozen=True)
class Plan:
    """What the policy wants this tick: orders to cancel and intents to send."""

    cancel: list[Order]
    send: list[Intent]
    phase: str


def passive_price(q: Quote, side: str, sz_decimals: int) -> float:
    """Mid rounded away from the touch; never equal to the far side (ALO-safe)."""
    px = round_price(q.mid, sz_decimals, side, passive=True)
    t = tick(px, sz_decimals)
    if side == "buy" and px >= q.ask:
        px = round_price(q.ask - t, sz_decimals, "buy", passive=True)
    if side == "sell" and px <= q.bid:
        px = round_price(q.bid + t, sz_decimals, "sell", passive=True)
    return px


def aggressive_price(q: Quote, side: str, sz_decimals: int, slippage_bps: float = 20.0) -> float:
    """Limit price that crosses the spread with a small cushion (IOC)."""
    px = q.ask * (1 + slippage_bps / 1e4) if side == "buy" else q.bid * (1 - slippage_bps / 1e4)
    return round_price(px, sz_decimals, side, passive=False)


class RebalancePolicy:
    def __init__(self, book: Book, sz_decimals: dict[str, int]):
        self.book = book
        self.sz_decimals = sz_decimals

    def _dec(self, coin: str) -> int:
        return self.sz_decimals.get(coin, 0)

    def phase(self, now: datetime, start: datetime, end: datetime) -> str:
        if now >= end:
            return "closed"
        if (now - start).total_seconds() >= self.book.cross_after_minutes * 60:
            return "aggressive"
        return "passive"

    def step(
        self,
        now: datetime,
        start: datetime,
        end: datetime,
        targets: dict[str, float],
        positions: dict[str, float],
        quotes: dict[str, Quote],
        oms: OMS,
        last_quote_ts: dict[str, datetime],
        planned_net_notional: float,
        planned_gross_notional: float,
    ) -> Plan:
        """One tick of the policy. ``positions`` are venue positions (signed sizes)."""
        phase = self.phase(now, start, end)
        if phase == "closed":
            return Plan(cancel=oms.open_orders(), send=[], phase=phase)

        cancel: list[Order] = []
        send: list[Intent] = []
        cross_coins = self._imbalance_cross(
            targets, positions, quotes, oms, planned_net_notional, planned_gross_notional
        )

        for coin, target in targets.items():
            q = quotes.get(coin)
            if q is None or q.bid <= 0 or q.ask <= 0:
                continue  # no market data: do nothing for this coin (never trade blind)
            dec = self._dec(coin)
            pos = positions.get(coin, 0.0)
            open_orders = oms.open_orders(coin)
            gap = target - pos  # signed size still to trade, before resting orders
            side = "buy" if gap > 0 else "sell"
            want_aggressive = phase == "aggressive" or coin in cross_coins

            # 1. cancel resting orders: wrong side, oversize, stale, or they must go aggressive
            keep_resting = 0.0
            for o in open_orders:
                wrong_side = (o.side == "buy") != (gap > 0)
                drift = abs(o.price / q.mid - 1) * 1e4 > self.book.reprice_tolerance_bps
                stale = (
                    now - last_quote_ts.get(o.cloid, now)
                ).total_seconds() >= self.book.reprice_seconds
                oversize = o.remaining > abs(gap) * (1 + 1e-6)
                if (
                    wrong_side
                    or oversize
                    or want_aggressive
                    or (stale and drift)
                    or abs(gap) < 1e-12
                ):
                    cancel.append(o)
                else:
                    keep_resting += o.remaining
            if abs(gap) < 1e-12:
                continue
            residual = round_size(abs(gap) - keep_resting, dec)
            if residual <= 0:
                continue
            if want_aggressive:
                px = aggressive_price(q, side, dec)
            else:
                px = passive_price(q, side, dec)
            tif = "Ioc" if want_aggressive else "Alo"
            if residual * px < self.book.min_notional:
                # a full close may go below the minimum as reduce-only (the backtest assumes the
                # same, note 011); anything else below it is left as dust
                if target == 0 and pos != 0 and keep_resting == 0 and residual >= abs(pos) - 1e-12:
                    send.append(Intent(coin, side, residual, px, tif, reduce_only=True))
                continue
            send.append(Intent(coin, side, residual, px, tif))
        return Plan(cancel=cancel, send=send, phase=phase)

    def _imbalance_cross(
        self,
        targets: dict[str, float],
        positions: dict[str, float],
        quotes: dict[str, Quote],
        oms: OMS,
        planned_net: float,
        planned_gross: float,
    ) -> set[str]:
        """Coins whose remaining delta should be crossed to restore the basket's balance.

        Compares the net notional filled so far with the planned net notional; when the
        realised net runs ahead of plan by more than ``imbalance_guard`` of the planned
        gross, the names on the lagging side are crossed.
        """
        if planned_gross <= 0:
            return set()
        filled_net = sum(oms.filled_notional().values())
        excess = filled_net - planned_net * min(
            1.0, self._progress(targets, positions, quotes, planned_gross)
        )
        if abs(excess) <= self.book.imbalance_guard * planned_gross:
            return set()
        lagging_side = "sell" if excess > 0 else "buy"
        out = set()
        for coin, target in targets.items():
            gap = target - positions.get(coin, 0.0)
            if (gap < 0 and lagging_side == "sell") or (gap > 0 and lagging_side == "buy"):
                out.add(coin)
        return out

    @staticmethod
    def _progress(targets, positions, quotes, planned_gross) -> float:
        remaining = 0.0
        for coin, target in targets.items():
            q = quotes.get(coin)
            if q is None:
                continue
            remaining += abs(target - positions.get(coin, 0.0)) * q.mid
        return 1.0 - remaining / planned_gross if planned_gross > 0 else 1.0
