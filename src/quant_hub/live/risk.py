"""Risk controller: every order and every session passes through here.

Remote commands only move the system toward safety: the kill flag and the pause flag
stop trading; nothing in the store can make the executor trade more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quant_hub.live.contracts import KILL_KEY, Book, pause_key
from quant_hub.live.store import Store


@dataclass
class RiskState:
    start_equity: float
    halted: bool = False
    halt_reason: str = ""
    rejected: list[str] = field(default_factory=list)


class RiskController:
    """Session-level guards (halt) and per-order guards (reject)."""

    def __init__(self, book: Book, store: Store, start_equity: float):
        self.book = book
        self.store = store
        self.state = RiskState(start_equity=start_equity)

    # --- session guards ---------------------------------------------------------------
    def halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halted = True
            self.state.halt_reason = reason

    def check_flags(self) -> str | None:
        """Kill (all books) or pause (this book) flags in the store."""
        if self.store.exists(KILL_KEY):
            return "kill flag set"
        if self.store.exists(pause_key(self.book.book_id)):
            return f"pause flag set for {self.book.book_id}"
        return None

    def check_equity(self, equity: float) -> str | None:
        if self.state.start_equity <= 0:
            return "non-positive start equity"
        dd = equity / self.state.start_equity - 1.0
        if dd < -self.book.daily_loss_limit:
            return f"session loss {dd:.2%} beyond limit {self.book.daily_loss_limit:.2%}"
        return None

    def in_window(self, now: datetime, as_of: str) -> bool:
        start, end = self.book.window(as_of)
        return start <= now < end

    def session_check(self, now: datetime, as_of: str, equity: float) -> str | None:
        """Reason to stop trading right now, or None."""
        if self.state.halted:
            return self.state.halt_reason
        if (why := self.check_flags()) is not None:
            self.halt(why)
            return why
        if (why := self.check_equity(equity)) is not None:
            self.halt(why)
            return why
        if not self.in_window(now, as_of):
            return "outside execution window"
        return None

    # --- per-order guards ---------------------------------------------------------------
    def check_order(
        self,
        coin: str,
        side: str,
        size: float,
        price: float,
        equity: float,
        position: float,
        target: float,
        reduce_only: bool = False,
    ) -> str | None:
        """Reject reason for an order, or None if it may be sent.

        An order may only move the position toward its target and never past it, its
        notional is capped, and the resulting |position| stays within the book's clamp.
        """
        if size <= 0 or price <= 0:
            return "non-positive size or price"
        notional = size * price
        if notional > self.book.max_order_notional:
            return f"order notional {notional:.0f} > {self.book.max_order_notional:.0f}"
        full_close = reduce_only and target == 0 and abs(size - abs(position)) <= 1e-12
        if notional < self.book.min_notional * (1 - 1e-9) and not full_close:
            return f"order notional {notional:.2f} < venue minimum {self.book.min_notional}"
        signed = size if side == "buy" else -size
        gap = target - position
        if gap * signed <= 0:
            return "order moves position away from target"
        if abs(signed) > abs(gap) * (1 + 1e-6) + 1e-12:
            return "order overshoots target"
        # the clamp is on weights: allow 2% for the gap between the sizing mid and the order price
        cap = self.book.max_abs_weight * self.book.capital_fraction * equity / price
        if abs(position + signed) > cap * 1.02 + 1e-12:
            return f"resulting |position| {abs(position + signed):.4f} > clamp {cap:.4f}"
        return None
