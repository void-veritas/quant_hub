"""Paper trading against live Hyperliquid prices: real quotes, simulated fills.

The read-only HL adapter supplies quotes; the paper exchange matches our orders
against them. The account lives in the store (``paper/<book>/account.json``) so a
paper book persists across sessions, with the same event flow as live.

Like a real account, the paper equity is marked to market (held coins are quoted on
connect, before the first snapshot) and pays or earns Hyperliquid's hourly funding on
its positions between connects (``fundingHistory``, notional at the current mid; a long
pays a positive rate). The first connect of an account starts the funding clock.
"""

from __future__ import annotations

import asyncio
import json
import time

from quant_hub.live.adapters.base import AccountSnapshot, PlaceResult
from quant_hub.live.adapters.hyperliquid import HyperliquidAdapter
from quant_hub.live.adapters.paper import PaperExchange
from quant_hub.live.contracts import fills_key
from quant_hub.live.events import Event, OrderUpdate
from quant_hub.live.oms import Order
from quant_hub.live.policy import Quote
from quant_hub.live.store import Store


class PaperOnHyperliquid:
    name = "paper_hl"

    def __init__(
        self,
        book_id: str,
        store: Store,
        equity: float,
        address: str | None = None,
        writer: bool = True,
        copy_prices: dict[str, float] | None = None,
    ):
        self.book_id = book_id
        self.store = store
        self.key = f"paper/{book_id}/account.json"
        self.feed = HyperliquidAdapter(address=address or "0x" + "0" * 40, paper=True)
        self.paper = PaperExchange(equity=equity)
        self.connected = False
        self.funding_until_ms = 0  # funding settled up to here (ms)
        self.funding_paid = 0.0  # cumulative, positive = paid by the account
        # only the session writes the account: riskwatch (hourly, also at 09:30 inside the window)
        # reads it, marks it and accrues funding in memory, and never overwrites the session's fills
        self.writer = writer
        # cascade: coin -> fill price of an upstream book today; orders on these coins fill at
        # once, in full, at that price (maker fee), whatever their limit
        self.copy_prices = dict(copy_prices or {})

    async def connect(self) -> None:
        await self.feed.connect()
        self.paper.decimals = self.feed.sz_decimals()
        raw = self.store.get(self.key)
        if raw:
            acc = json.loads(raw)
            self.paper.equity = float(acc["equity"])
            self.paper.cash_pnl = float(acc.get("cash_pnl", 0.0))
            self.paper.positions = {k: float(v) for k, v in acc.get("positions", {}).items()}
            self.paper._cost_basis = {k: float(v) for k, v in acc.get("cost_basis", {}).items()}
            self.funding_until_ms = int(acc.get("funding_until_ms") or 0)
            self.funding_paid = float(acc.get("funding_paid", 0.0))
        held = [c for c, q in self.paper.positions.items() if q]
        if held:  # mark to market before anyone reads the equity
            self.paper.set_quotes(await self.feed.quotes(held))
        await self._accrue_funding(held)
        await self.paper.connect()
        self.connected = True

    async def _accrue_funding(self, held: list[str], now_ms: int | None = None) -> None:
        now_ms = now_ms or int(time.time() * 1000)
        since = self.funding_until_ms or now_ms
        paid = 0.0
        for coin in held:
            q = self.paper._quotes.get(coin)
            if q is None or since >= now_ms:
                continue
            rows = await asyncio.to_thread(self.feed.info.funding_history, coin, since + 1, now_ms)
            rate = sum(float(r.get("fundingRate", 0.0)) for r in rows or [])
            paid += self.paper.positions[coin] * q.mid * rate
        self.paper.cash_pnl -= paid
        self.funding_paid += paid
        self.funding_until_ms = now_ms
        self._save()

    async def close(self) -> None:
        self._save()
        await self.paper.close()
        await self.feed.close()
        self.connected = False

    def _save(self) -> None:
        if not self.writer:
            return
        self.store.put(
            self.key,
            json.dumps(
                {
                    "equity": self.paper.equity,
                    "cash_pnl": self.paper.cash_pnl,
                    "positions": self.paper.positions,
                    "cost_basis": self.paper._cost_basis,
                    "funding_until_ms": self.funding_until_ms,
                    "funding_paid": self.funding_paid,
                },
                sort_keys=True,
            ),
        )

    def sz_decimals(self) -> dict[str, int]:
        return self.paper.sz_decimals()

    async def snapshot(self) -> AccountSnapshot:
        return await self.paper.snapshot()

    async def quotes(self, coins: list[str]) -> dict[str, Quote]:
        q = await self.feed.quotes(coins)
        self.paper.set_quotes(q)  # matching happens on every quote refresh
        return q

    async def place(self, order: Order) -> PlaceResult:
        px = self.copy_prices.get(order.coin)
        if px is not None and self.paper.connected:
            oid = str(next(self.paper._oid))
            now = self.paper.clock.now()
            self.paper._events.append(
                OrderUpdate(now, order.coin, order.cloid, oid, "open", 0.0, order.size)
            )
            self.paper._fill(order, oid, order.size, px, "maker", now)
            self._save()
            return PlaceResult(True, oid=oid, filled_size=order.size)
        res = await self.paper.place(order)
        self._save()
        return res

    async def cancel(self, order: Order) -> bool:
        return await self.paper.cancel(order)

    async def fills_since(self, start_ms: int) -> list[Event]:
        return await self.paper.fills_since(start_ms)

    def drain(self) -> list[Event]:
        return self.paper.drain()


def upstream_prices(store: Store, book_ids: list[str], as_of: str) -> dict[str, float]:
    """Volume-weighted fill price per coin, from the first book in ``book_ids`` that traded it."""
    import io

    import pandas as pd

    out: dict[str, float] = {}
    for b in book_ids:
        raw = store.get(fills_key(b, as_of))
        if not raw:
            continue
        f = pd.read_parquet(io.BytesIO(raw))
        if f.empty:
            continue
        f = f.assign(notional=f["size"] * f["price"])
        vwap = f.groupby("coin")["notional"].sum() / f.groupby("coin")["size"].sum()
        for coin, px in vwap.items():
            out.setdefault(str(coin), float(px))
    return out
