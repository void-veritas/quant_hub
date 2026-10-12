"""Test exchange: the paper exchange plus scripted misbehaviour.

Scenarios the engine must survive (STATUS M7): duplicated and reordered messages,
messages for coins we never traded, lost stream messages (REST must catch up), rejects,
partial fills, a disconnect in the middle of the session and a restart from state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

from quant_hub.live.adapters.base import PlaceResult
from quant_hub.live.adapters.paper import PaperExchange
from quant_hub.live.events import Disconnected, Event, Fill, OrderUpdate, Reconnected
from quant_hub.live.oms import Order


@dataclass
class FakeExchange(PaperExchange):
    name: str = "fake"
    duplicate_events: bool = False
    shuffle_events: bool = False
    drop_stream_fraction: float = 0.0  # fraction of streamed events never delivered
    reject_cloids: set[str] = field(default_factory=set)
    reject_next: int = 0  # reject the next n placements
    inject: list[Event] = field(default_factory=list)  # foreign/garbage events to deliver
    seed: int = 7
    _rng: random.Random = field(default_factory=lambda: random.Random(7))
    placed: list[Order] = field(default_factory=list)
    cancelled: list[Order] = field(default_factory=list)

    async def place(self, order: Order) -> PlaceResult:
        self.placed.append(order)
        if self.reject_next > 0 or order.cloid in self.reject_cloids:
            self.reject_next = max(self.reject_next - 1, 0)
            self._events.append(
                OrderUpdate(
                    self.clock.now(),
                    order.coin,
                    order.cloid,
                    "",
                    "rejected",
                    0.0,
                    order.size,
                    "scripted reject",
                )
            )
            return PlaceResult(False, error="scripted reject")
        return await super().place(order)

    async def cancel(self, order: Order) -> bool:
        self.cancelled.append(order)
        return await super().cancel(order)

    def drain(self) -> list[Event]:
        out = super().drain()
        if self.drop_stream_fraction:
            out = [e for e in out if self._rng.random() >= self.drop_stream_fraction]
        if self.duplicate_events:
            out = out + list(out)
        if self.inject:
            out = out + self.inject
            self.inject = []
        if self.shuffle_events:
            self._rng.shuffle(out)
        return out

    def disconnect(self, reason: str = "scripted") -> None:
        self.connected = False
        self._events.append(Disconnected(self.clock.now(), reason))

    def reconnect(self) -> None:
        self.connected = True
        self._events.append(Reconnected(self.clock.now()))

    def foreign_fill(self, coin: str, cloid: str, now: datetime | None = None) -> Fill:
        """A fill for an order this book never placed (another book or a bug upstream)."""
        return Fill(
            now or self.clock.now(), coin, cloid, "999", "foreign-1", "buy", 1.0, 1.0, 0.0, "maker"
        )
