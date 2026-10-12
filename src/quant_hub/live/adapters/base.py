"""Adapter interface: target positions in, orders out, events back.

The engine never touches venue specifics. REST-style calls (``snapshot``, ``quotes``)
are the source of truth; ``drain`` yields streamed events since the last call and may
deliver duplicates, reorderings or nothing at all after a disconnect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from quant_hub.live.events import Event
from quant_hub.live.oms import Order
from quant_hub.live.policy import Quote


@dataclass
class AccountSnapshot:
    equity: float
    positions: dict[str, float] = field(default_factory=dict)  # signed sizes
    open_orders: list[dict] = field(
        default_factory=list
    )  # {coin, cloid, oid, side, size, remaining, price}
    ts: str = ""


@dataclass(frozen=True)
class PlaceResult:
    ok: bool
    oid: str = ""
    filled_size: float = 0.0  # immediate fill (IOC or crossing)
    error: str = ""


class ExchangeAdapter(Protocol):
    name: str

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    def sz_decimals(self) -> dict[str, int]: ...

    async def snapshot(self) -> AccountSnapshot: ...

    async def quotes(self, coins: list[str]) -> dict[str, Quote]: ...

    async def place(self, order: Order) -> PlaceResult: ...

    async def cancel(self, order: Order) -> bool: ...

    async def fills_since(self, start_ms: int) -> list[Event]: ...

    def drain(self) -> list[Event]: ...
