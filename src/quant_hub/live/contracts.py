"""Contracts between the analytics and execution layers.

A *book* is one strategy with its own capital share, universe, rebalance window and
risk limits. Analytics produces a :class:`WeightsFile` per book and day; execution
consumes it only after :func:`validate_weights` passes. Every check here is also a
runtime guard, not just a test.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Book:
    """Strategy passport shared by backtest, analytics and execution."""

    book_id: str
    venue: str = "hyperliquid"
    rebalance_time: str = "09:00"  # UTC, start of the execution window
    window_minutes: int = 60  # cancel everything after this
    capital_fraction: float = 1.0  # share of account equity run by this book
    max_abs_weight: float = 0.25  # |w| per name (rule 4a)
    max_gross: float = 1.5
    max_net: float = 0.5  # |sum w| guard (neutral books should be far below)
    max_weight_change: float = 0.5  # max |w_t - w_{t-1}| per name (fat-finger guard)
    max_names: int = 100
    min_notional: float = 10.0  # venue minimum order value (HL: $10)
    universe: list[str] | None = None  # whitelist; None = any coin the venue lists
    weights_max_age_minutes: int = 30  # produced_at must be within this of the window start
    daily_loss_limit: float = 0.03  # equity drawdown within the session that halts the book
    # book drawdown from its equity peak, in units of the starting capital (riskwatch, hourly):
    # an alert at review_drawdown, the pause flag at pause_drawdown; None = not checked
    review_drawdown: float | None = None
    pause_drawdown: float | None = None
    max_order_notional: float = 2_000.0  # single order cap in quote currency
    imbalance_guard: float = (
        0.10  # cross the lagging side when |net filled - net planned| > this * gross
    )
    cross_after_minutes: int = 30  # remaining quantity goes IOC after this
    reprice_seconds: int = 60
    trade_buffer: float = 0.0  # relative no-trade zone: skip when |target-pos| <= buffer*|target|
    reprice_tolerance_bps: float = 5.0  # re-quote when the resting price drifts from mid by more
    maker_fee: float = 0.00015
    taker_fee: float = 0.00045
    tick_seconds: float = 1.0
    # paper only, the cascade (Egor, 2026-10-28): wait for these books' sessions of the day (in
    # order, until window start + price_from_wait_minutes), then fill every coin they traded at
    # their volume-weighted fill price (first book wins); the other coins trade on paper as usual.
    # The live book has the venue's request budget to itself while it trades.
    price_from: list[str] = field(default_factory=list)
    price_from_wait_minutes: int = 40
    spec: dict[str, Any] = field(default_factory=dict)  # strategy config (e.g. RunConfig kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Book:
        """Load a book passport from YAML (unknown keys are rejected)."""
        raw = yaml.safe_load(Path(path).read_text())
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown book fields: {sorted(unknown)}")
        return cls(**raw)

    def window(self, as_of: str) -> tuple[datetime, datetime]:
        """(start, end) of the execution window for a trading date, UTC."""
        hh, mm = (int(x) for x in self.rebalance_time.split(":"))
        start = datetime.combine(datetime.fromisoformat(as_of).date(), time(hh, mm), tzinfo=UTC)
        return start, start + timedelta(minutes=self.window_minutes)


@dataclass
class WeightsFile:
    """Target weights for one book and trading date, as written by analytics."""

    book_id: str
    as_of: str  # trading date YYYY-MM-DD
    produced_at: str  # ISO timestamp UTC
    weights: dict[str, float]
    prices: dict[str, float] = field(default_factory=dict)  # decision prices (for plan-vs-fact)
    meta: dict[str, Any] = field(default_factory=dict)  # git sha, spec version, data coverage
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        """Serialise deterministically (sorted keys) with a content hash in ``meta``."""
        body = asdict(self)
        body["meta"] = {k: v for k, v in body["meta"].items() if k != "sha256"}
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        body["meta"]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        return json.dumps(body, sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> WeightsFile:
        """Parse and verify the embedded content hash."""
        body = json.loads(text)
        meta = dict(body.get("meta", {}))
        sha = meta.pop("sha256", None)
        payload = json.dumps({**body, "meta": meta}, sort_keys=True, separators=(",", ":"))
        if sha is not None and sha != hashlib.sha256(payload.encode()).hexdigest():
            raise ValueError("weights file hash mismatch: content altered after production")
        return cls(**body)

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())


def validate_weights(
    wf: WeightsFile,
    book: Book,
    now: datetime,
    previous: WeightsFile | None = None,
    listed: set[str] | None = None,
) -> list[str]:
    """Return the list of violations (empty = the file may be traded).

    Checks: schema, book id, date = today, freshness, finite values, per-name clamp,
    gross/net, name count, whitelist and listing, and the change from the previous file.
    """
    problems: list[str] = []
    if wf.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {wf.schema_version} != {SCHEMA_VERSION}")
    if wf.book_id != book.book_id:
        problems.append(f"book_id {wf.book_id!r} != {book.book_id!r}")
    start, end = book.window(wf.as_of)
    if not (start - timedelta(minutes=book.weights_max_age_minutes) <= now <= end):
        problems.append(f"as_of {wf.as_of} is not today's window ({start:%F %H:%M} .. {end:%H:%M})")
    produced = datetime.fromisoformat(wf.produced_at)
    if produced.tzinfo is None:
        problems.append("produced_at is naive (must be UTC)")
    else:
        age = now - produced
        if age > timedelta(minutes=book.weights_max_age_minutes) or age < timedelta(minutes=-5):
            problems.append(f"produced_at {wf.produced_at} is stale ({age} old)")
    if not wf.weights:
        problems.append("no weights")
    for coin, w in wf.weights.items():
        if not isinstance(w, int | float) or not math.isfinite(w):
            problems.append(f"{coin}: non-finite weight {w!r}")
        elif abs(w) > book.max_abs_weight + 1e-12:
            problems.append(f"{coin}: |w|={abs(w):.4f} > {book.max_abs_weight}")
        if book.universe is not None and coin not in book.universe:
            problems.append(f"{coin}: not in book universe")
        if listed is not None and coin not in listed:
            problems.append(f"{coin}: not listed on {book.venue}")
    if wf.gross > book.max_gross + 1e-9:
        problems.append(f"gross {wf.gross:.3f} > {book.max_gross}")
    if abs(wf.net) > book.max_net + 1e-9:
        problems.append(f"|net| {abs(wf.net):.3f} > {book.max_net}")
    if len(wf.weights) > book.max_names:
        problems.append(f"{len(wf.weights)} names > {book.max_names}")
    if previous is not None:
        for coin in set(wf.weights) | set(previous.weights):
            d = abs(wf.weights.get(coin, 0.0) - previous.weights.get(coin, 0.0))
            if d > book.max_weight_change + 1e-12:
                problems.append(
                    f"{coin}: change {d:.3f} vs {previous.as_of} > {book.max_weight_change}"
                )
    return problems


FILL_COLUMNS = [
    "book_id",
    "as_of",
    "ts",
    "coin",
    "cloid",
    "oid",
    "fill_id",
    "side",
    "size",
    "price",
    "fee",
    "liquidity",
    "decision_price",
]


@dataclass(frozen=True)
class FillRecord:
    """One execution, in the shape written to ``fills/<book>/<as_of>.parquet``."""

    book_id: str
    as_of: str
    ts: str
    coin: str
    cloid: str
    oid: str
    fill_id: str
    side: str  # "buy" | "sell"
    size: float
    price: float
    fee: float
    liquidity: str  # "maker" | "taker"
    decision_price: float


def weights_key(book_id: str, as_of: str) -> str:
    return f"weights/{book_id}/{as_of}.json"


def fills_key(book_id: str, as_of: str) -> str:
    return f"fills/{book_id}/{as_of}.parquet"


def state_key(book_id: str, as_of: str) -> str:
    return f"state/{book_id}/{as_of}.json"


def pause_key(book_id: str) -> str:
    return f"flags/{book_id}/pause"


KILL_KEY = "flags/kill"
