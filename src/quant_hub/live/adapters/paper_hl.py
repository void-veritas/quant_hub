"""Paper trading against live Hyperliquid prices: real quotes, simulated fills.

The read-only HL adapter supplies quotes; the paper exchange matches our orders
against them. Positions live in the store (``paper/<book>/positions.json``) so a
paper book persists across sessions, with the same event flow as live.
"""

from __future__ import annotations

import json

from quant_hub.live.adapters.base import AccountSnapshot, PlaceResult
from quant_hub.live.adapters.hyperliquid import HyperliquidAdapter
from quant_hub.live.adapters.paper import PaperExchange
from quant_hub.live.events import Event
from quant_hub.live.oms import Order
from quant_hub.live.policy import Quote
from quant_hub.live.store import Store


class PaperOnHyperliquid:
    name = "paper_hl"

    def __init__(self, book_id: str, store: Store, equity: float, address: str | None = None):
        self.book_id = book_id
        self.store = store
        self.key = f"paper/{book_id}/account.json"
        self.feed = HyperliquidAdapter(address=address or "0x" + "0" * 40, paper=True)
        self.paper = PaperExchange(equity=equity)
        self.connected = False

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
        await self.paper.connect()
        self.connected = True

    async def close(self) -> None:
        self._save()
        await self.paper.close()
        await self.feed.close()
        self.connected = False

    def _save(self) -> None:
        self.store.put(
            self.key,
            json.dumps(
                {
                    "equity": self.paper.equity,
                    "cash_pnl": self.paper.cash_pnl,
                    "positions": self.paper.positions,
                    "cost_basis": self.paper._cost_basis,
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
        res = await self.paper.place(order)
        self._save()
        return res

    async def cancel(self, order: Order) -> bool:
        return await self.paper.cancel(order)

    async def fills_since(self, start_ms: int) -> list[Event]:
        return await self.paper.fills_since(start_ms)

    def drain(self) -> list[Event]:
        return self.paper.drain()
