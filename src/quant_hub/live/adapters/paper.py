"""Paper exchange: fills passive orders when the mid trades through them, IOC at the touch.

Used both for paper trading (prices from a live feed pushed via ``set_quotes``) and as
the base of the test exchange. Deterministic: a resting buy fills when ``mid <= price``
after ``passive_latency_s`` of resting, a resting sell when ``mid >= price``; an IOC fills
immediately at the far touch when its limit allows. Fees follow the book.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from quant_hub.live.adapters.base import AccountSnapshot, PlaceResult
from quant_hub.live.clock import Clock
from quant_hub.live.events import Event, Fill, OrderUpdate
from quant_hub.live.oms import Order
from quant_hub.live.policy import Quote


@dataclass
class _Resting:
    order: Order
    oid: str
    placed_at: datetime
    remaining: float
    filled: float = 0.0


@dataclass
class PaperExchange:
    name: str = "paper"
    equity: float = 1_000.0
    maker_fee: float = 0.00015
    taker_fee: float = 0.00045
    passive_latency_s: float = 0.0
    passive_fill_fraction: float = 1.0  # fraction of remaining filled per crossing tick
    decimals: dict[str, int] = field(default_factory=dict)
    clock: Clock = field(default_factory=Clock)
    positions: dict[str, float] = field(default_factory=dict)
    _quotes: dict[str, Quote] = field(default_factory=dict)
    _resting: dict[str, _Resting] = field(default_factory=dict)
    _events: list[Event] = field(default_factory=list)
    _all_fills: list[Fill] = field(default_factory=list)
    _oid: itertools.count = field(default_factory=lambda: itertools.count(1))
    _fid: itertools.count = field(default_factory=lambda: itertools.count(1))
    connected: bool = False
    cash_pnl: float = 0.0

    # --- lifecycle ------------------------------------------------------------------
    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    def sz_decimals(self) -> dict[str, int]:
        return dict(self.decimals)

    # --- market data ----------------------------------------------------------------
    def set_quotes(self, quotes: dict[str, Quote]) -> None:
        """Update the market; resting orders are matched against the new mids."""
        self._quotes.update(quotes)
        self._match()

    async def quotes(self, coins: list[str]) -> dict[str, Quote]:
        return {c: self._quotes[c] for c in coins if c in self._quotes}

    # --- account --------------------------------------------------------------------
    async def snapshot(self) -> AccountSnapshot:
        mark = sum(self.positions.get(c, 0.0) * q.mid for c, q in self._quotes.items())
        _ = mark
        return AccountSnapshot(
            equity=self.equity + self.cash_pnl + self._unrealised(),
            positions={c: s for c, s in self.positions.items() if abs(s) > 0},
            open_orders=[
                {
                    "coin": r.order.coin,
                    "cloid": r.order.cloid,
                    "oid": r.oid,
                    "side": r.order.side,
                    "size": r.order.size,
                    "remaining": r.remaining,
                    "price": r.order.price,
                }
                for r in self._resting.values()
            ],
            ts=self.clock.now().isoformat(),
        )

    _cost_basis: dict[str, float] = field(default_factory=dict)

    def _unrealised(self) -> float:
        out = 0.0
        for c, s in self.positions.items():
            q = self._quotes.get(c)
            if q is not None and s:
                out += s * q.mid - self._cost_basis.get(c, 0.0)
        return out

    # --- orders ---------------------------------------------------------------------
    async def place(self, order: Order) -> PlaceResult:
        if not self.connected:
            return PlaceResult(False, error="disconnected")
        q = self._quotes.get(order.coin)
        if q is None:
            return PlaceResult(False, error="no market")
        if order.cloid in self._resting or any(f.cloid == order.cloid for f in self._all_fills):
            return PlaceResult(False, error="duplicate cloid")
        oid = str(next(self._oid))
        now = self.clock.now()
        if order.tif == "Alo":
            crosses = (order.side == "buy" and order.price >= q.ask) or (
                order.side == "sell" and order.price <= q.bid
            )
            if crosses:
                return PlaceResult(False, error="alo would cross")
            self._resting[order.cloid] = _Resting(order, oid, now, order.size)
            self._events.append(
                OrderUpdate(now, order.coin, order.cloid, oid, "open", 0.0, order.size)
            )
            return PlaceResult(True, oid=oid)
        # IOC: fill what the limit allows at the touch, cancel the rest
        can = (order.side == "buy" and order.price >= q.ask) or (
            order.side == "sell" and order.price <= q.bid
        )
        if not can:
            self._events.append(
                OrderUpdate(
                    now, order.coin, order.cloid, oid, "canceled", 0.0, order.size, "ioc no fill"
                )
            )
            return PlaceResult(True, oid=oid)
        px = q.ask if order.side == "buy" else q.bid
        self._fill(order, oid, order.size, px, "taker", now)
        return PlaceResult(True, oid=oid, filled_size=order.size)

    async def cancel(self, order: Order) -> bool:
        if not self.connected:
            return False
        r = self._resting.pop(order.cloid, None)
        if r is None:
            return False
        self._events.append(
            OrderUpdate(
                self.clock.now(), order.coin, order.cloid, r.oid, "canceled", r.filled, r.remaining
            )
        )
        return True

    async def fills_since(self, start_ms: int) -> list[Event]:
        return [f for f in self._all_fills if int(f.ts.timestamp() * 1000) >= start_ms]

    def drain(self) -> list[Event]:
        out, self._events = self._events, []
        return out

    # --- matching -------------------------------------------------------------------
    def _match(self) -> None:
        now = self.clock.now()
        for cloid, r in list(self._resting.items()):
            q = self._quotes.get(r.order.coin)
            if q is None or (now - r.placed_at).total_seconds() < self.passive_latency_s:
                continue
            hit = (r.order.side == "buy" and q.mid <= r.order.price) or (
                r.order.side == "sell" and q.mid >= r.order.price
            )
            if not hit:
                continue
            size = (
                min(r.remaining, round(r.order.size * self.passive_fill_fraction, 12))
                or r.remaining
            )
            self._fill(r.order, r.oid, size, r.order.price, "maker", now)
            r.remaining -= size
            r.filled += size
            if r.remaining <= 1e-12:
                del self._resting[cloid]

    def _fill(
        self, order: Order, oid: str, size: float, px: float, liquidity: str, now: datetime
    ) -> None:
        fee_rate = self.maker_fee if liquidity == "maker" else self.taker_fee
        fee = size * px * fee_rate
        signed = size if order.side == "buy" else -size
        old = self.positions.get(order.coin, 0.0)
        new = old + signed
        # realised pnl when reducing; cost basis tracks the open position
        basis = self._cost_basis.get(order.coin, 0.0)
        if old == 0 or (old > 0) == (signed > 0):
            self._cost_basis[order.coin] = basis + signed * px
        else:
            avg = basis / old if old else px
            closed = min(abs(signed), abs(old)) * (1 if old > 0 else -1)
            self.cash_pnl += closed * (px - avg)
            self._cost_basis[order.coin] = basis - closed * avg + (signed + closed) * px
        self.cash_pnl -= fee
        self.positions[order.coin] = new
        if abs(new) < 1e-12:
            self.positions.pop(order.coin, None)
            self._cost_basis.pop(order.coin, None)
        fill = Fill(
            now,
            order.coin,
            order.cloid,
            oid,
            f"f{next(self._fid)}",
            order.side,
            size,
            px,
            fee,
            liquidity,
        )
        self._all_fills.append(fill)
        self._events.append(fill)
        status = "filled" if liquidity == "taker" or size >= order.size else "partial"
        self._events.append(
            OrderUpdate(
                now, order.coin, order.cloid, oid, status, size, max(order.size - size, 0.0)
            )
        )
