"""Hourly risk check between sessions: alerts only, never trades.

Inside the 09:00-10:00 window the session itself guards the book (daily loss limit,
kill and pause flags). The other 23 hours the book holds its positions with nobody
looking; this check reads the account once an hour and tells the owner when:

- equity has fallen 2%, 3% or 5% since the first check of the UTC day (one message
  per level per day, so a slow bleed does not flood the chat);
- a coin's weight has drifted past 1.5x the passport clamp (a price move made it big);
- gross exposure is above 1.2x the passport's max_gross;
- the book's drawdown from its equity peak, in units of the starting capital, reaches the
  passport's ``review_drawdown`` (an alert) or ``pause_drawdown`` (an alert and the pause
  flag: no session runs until ``resume``). The start is the equity at the first check in
  that mode, so paper and live are measured separately and going live starts from zero.

Deposits, withdrawals and transfers (live: the venue's ledger) are capital, not
performance: each one moves the start, the peak and the day baseline by its amount, is
recorded in the capital file and reported once, so performance = equity - start.

Anything below these levels belongs in the daily report, not in an alert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

DAY_LEVELS = (0.02, 0.03, 0.05)
WEIGHT_X = 1.5  # x max_abs_weight
GROSS_X = 1.2  # x max_gross


def risk_key(book_id: str, as_of: str) -> str:
    return f"state/{book_id}/risk_{as_of}.json"


@dataclass
class DayRisk:
    baseline: float
    alerted: set[str] = field(default_factory=set)

    def to_json(self) -> str:
        return json.dumps({"baseline": self.baseline, "alerted": sorted(self.alerted)})

    @classmethod
    def from_json(cls, raw: bytes | str) -> DayRisk:
        d = json.loads(raw)
        return cls(float(d["baseline"]), set(d.get("alerted", [])))


def evaluate(
    equity: float,
    positions: dict[str, float],
    mids: dict[str, float],
    day: DayRisk,
    max_abs_weight: float,
    max_gross: float,
) -> list[str]:
    """New alert lines (each condition once per day); records them in ``day.alerted``."""
    out: list[str] = []
    if day.baseline > 0:
        dd = equity / day.baseline - 1
        hit = [lvl for lvl in DAY_LEVELS if dd <= -lvl and f"day{lvl}" not in day.alerted]
        if hit:
            out.append(f"equity {dd:+.1%} since 00:00 UTC ({day.baseline:.0f} -> {equity:.0f})")
            day.alerted.update(f"day{lvl}" for lvl in DAY_LEVELS if dd <= -lvl)
    if equity <= 0:
        return out
    gross = sum(abs(q * mids[c]) for c, q in positions.items() if c in mids) / equity
    if gross > GROSS_X * max_gross and "gross" not in day.alerted:
        out.append(f"gross {gross:.2f} > {GROSS_X:g} x max_gross {max_gross:g}")
        day.alerted.add("gross")
    for coin, q in sorted(positions.items()):
        if coin not in mids:
            continue
        w = q * mids[coin] / equity
        if abs(w) > WEIGHT_X * max_abs_weight and f"w:{coin}" not in day.alerted:
            out.append(f"{coin} weight {w:+.2f} > {WEIGHT_X:g} x clamp {max_abs_weight:g}")
            day.alerted.add(f"w:{coin}")
    return out


def capital_key(book_id: str, mode: str) -> str:
    return f"state/{book_id}/capital_{mode}.json"


@dataclass
class BookRisk:
    start: float  # equity at the first check in this mode, plus every external flow since
    peak: float
    alerted: set[str] = field(default_factory=set)
    since_ms: int = 0  # ledger read up to here
    flows: list = field(default_factory=list)  # [time_ms, usd, type] deposits, withdrawals

    def to_json(self) -> str:
        return json.dumps(
            {
                "start": self.start,
                "peak": self.peak,
                "alerted": sorted(self.alerted),
                "since_ms": self.since_ms,
                "flows": self.flows,
            }
        )

    @classmethod
    def from_json(cls, raw: bytes | str) -> BookRisk:
        d = json.loads(raw)
        return cls(
            float(d["start"]),
            float(d["peak"]),
            set(d.get("alerted", [])),
            int(d.get("since_ms", 0)),
            [list(f) for f in d.get("flows", [])],
        )

    def apply_flows(self, flows: list[tuple[int, float, str]], day: DayRisk) -> list[str]:
        """Book each new flow as capital; returns one info line per flow."""
        out = []
        for t, usd, kind in flows:
            if t <= self.since_ms:
                continue  # read already (the ledger query includes its start time)
            self.start += usd
            self.peak += usd
            day.baseline += usd
            self.flows.append([t, usd, kind])
            self.since_ms = t
            out.append(
                f"{kind} {usd:+.2f} USD counted as capital, not performance:"
                f" start {self.start - usd:.0f} -> {self.start:.0f}"
            )
        return out


def evaluate_book(
    equity: float, book: BookRisk, review: float | None, pause: float | None
) -> tuple[list[str], bool]:
    """(alert lines, set the pause flag). Each level fires once until a new equity peak."""
    if equity >= book.peak:
        book.peak = equity
        book.alerted.clear()
    if book.start <= 0:
        return [], False
    dd = (equity - book.peak) / book.start
    out, stop = [], False
    where = (
        f"drawdown {dd:+.1%} of starting capital {book.start:.0f}"
        f" (peak {book.peak:.0f}, now {equity:.0f})"
    )
    if pause is not None and dd <= -pause and "pause" not in book.alerted:
        out.append(
            f"{where}: past the {pause:.0%} pause level, pause flag set, no session until resume"
        )
        book.alerted.update({"pause", "review"})
        stop = True
    elif review is not None and dd <= -review and "review" not in book.alerted:
        out.append(f"{where}: past the {review:.0%} review level, look at it")
        book.alerted.add("review")
    return out, stop
