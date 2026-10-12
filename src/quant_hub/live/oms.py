"""Order management: the only writer of order state.

State is keyed by client order id; every incoming message is checked against the
record it claims to belong to (same coin, known cloid) before being applied. Fills are
de-duplicated by ``fill_id``. Everything is serialisable so a restarted executor can
resume mid-session from ``state/<book>/<as_of>.json``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime

from quant_hub.live.events import Fill, OrderUpdate

OPEN_STATUSES = {"new", "open", "partial"}
TERMINAL = {"filled", "canceled", "rejected"}


def make_cloid(book_id: str, as_of: str, coin: str, seq: int) -> str:
    """Deterministic 16-byte client order id (``0x`` + 32 hex), HL-compatible.

    The same (book, date, coin, seq) always maps to the same id, so a restarted
    executor recognises its own resting orders instead of duplicating them.
    """
    h = hashlib.blake2b(f"{book_id}|{as_of}|{coin}|{seq}".encode(), digest_size=16)
    return "0x" + h.hexdigest()


@dataclass
class Order:
    cloid: str
    coin: str
    side: str  # buy | sell
    size: float
    price: float
    tif: str  # Alo | Ioc
    created_at: str
    status: str = "new"
    oid: str = ""
    filled_size: float = 0.0
    notional_filled: float = 0.0
    fees: float = 0.0
    reason: str = ""

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled_size, 0.0)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def avg_price(self) -> float:
        return self.notional_filled / self.filled_size if self.filled_size else 0.0


@dataclass
class OMS:
    book_id: str
    as_of: str
    orders: dict[str, Order] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    seen_fill_ids: set[str] = field(default_factory=set)
    seq: dict[str, int] = field(default_factory=dict)  # per coin, for cloid generation
    foreign: int = 0  # messages that did not match our records (logged, ignored)

    # --- creation -----------------------------------------------------------------
    def new_order(
        self, coin: str, side: str, size: float, price: float, tif: str, now: datetime
    ) -> Order:
        n = self.seq.get(coin, 0) + 1
        self.seq[coin] = n
        o = Order(
            make_cloid(self.book_id, self.as_of, coin, n),
            coin,
            side,
            size,
            price,
            tif,
            now.isoformat(),
        )
        self.orders[o.cloid] = o
        return o

    def open_orders(self, coin: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open and (coin is None or o.coin == coin)]

    # --- message application -------------------------------------------------------
    def _owned(self, coin: str, cloid: str) -> Order | None:
        o = self.orders.get(cloid)
        if o is None or o.coin != coin:
            self.foreign += 1
            return None
        return o

    def on_update(self, ev: OrderUpdate) -> Order | None:
        o = self._owned(ev.coin, ev.cloid)
        if o is None:
            return None
        if ev.oid:
            o.oid = ev.oid
        if o.status in TERMINAL and ev.status not in TERMINAL:
            return o  # stale/out-of-order message after a terminal state: ignore
        if ev.status in ("open", "partial", "filled", "canceled", "rejected"):
            o.status = ev.status
        if ev.filled_size > o.filled_size + 1e-12 and not self._fills_cover(o, ev.filled_size):
            # the venue says more is filled than our fills show: trust the venue total,
            # the reconciliation pass will fetch the missing fill records
            o.filled_size = ev.filled_size
        if ev.reason:
            o.reason = ev.reason
        return o

    def _fills_cover(self, o: Order, filled: float) -> bool:
        return o.filled_size + 1e-9 >= filled

    def on_fill(self, ev: Fill) -> Order | None:
        """Apply a fill once; returns the order, or None if foreign/duplicate."""
        if ev.fill_id in self.seen_fill_ids:
            return None
        o = self._owned(ev.coin, ev.cloid)
        if o is None:
            return None
        if ev.side != o.side:
            self.foreign += 1
            return None
        self.seen_fill_ids.add(ev.fill_id)
        self.fills.append(ev)
        o.filled_size = min(o.size, o.filled_size + ev.size)
        o.notional_filled += ev.size * ev.price
        o.fees += ev.fee
        if ev.oid:
            o.oid = ev.oid
        if o.filled_size >= o.size - 1e-12:
            o.status = "filled"
        elif o.status != "canceled":
            o.status = "partial"
        return o

    def mark(self, cloid: str, status: str, oid: str = "", reason: str = "") -> None:
        o = self.orders[cloid]
        o.status = status
        if oid:
            o.oid = oid
        if reason:
            o.reason = reason

    # --- aggregates ----------------------------------------------------------------
    def filled_position(self) -> dict[str, float]:
        """Signed size filled this session, per coin."""
        pos: dict[str, float] = {}
        for f in self.fills:
            pos[f.coin] = pos.get(f.coin, 0.0) + (f.size if f.side == "buy" else -f.size)
        return pos

    def filled_notional(self) -> dict[str, float]:
        """Signed notional filled this session, per coin (at fill prices)."""
        out: dict[str, float] = {}
        for f in self.fills:
            out[f.coin] = out.get(f.coin, 0.0) + (1 if f.side == "buy" else -1) * f.size * f.price
        return out

    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)

    # --- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "as_of": self.as_of,
            "orders": {k: asdict(v) for k, v in self.orders.items()},
            "fills": [{**asdict(f), "ts": f.ts.isoformat()} for f in self.fills],
            "seq": self.seq,
            "foreign": self.foreign,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OMS:
        oms = cls(d["book_id"], d["as_of"], seq=dict(d.get("seq", {})), foreign=d.get("foreign", 0))
        oms.orders = {k: Order(**v) for k, v in d["orders"].items()}
        for f in d.get("fills", []):
            f = dict(f)
            f["ts"] = datetime.fromisoformat(f["ts"])
            oms.fills.append(Fill(**f))
        oms.seen_fill_ids = {f.fill_id for f in oms.fills}
        return oms
