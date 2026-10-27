"""Hourly risk check between sessions: alerts only, never trades.

Inside the 09:00-10:00 window the session itself guards the book (daily loss limit,
kill and pause flags). The other 23 hours the book holds its positions with nobody
looking; this check reads the account once an hour and tells the owner when:

- equity has fallen 2%, 3% or 5% since the first check of the UTC day (one message
  per level per day, so a slow bleed does not flood the chat);
- a coin's weight has drifted past 1.5x the passport clamp (a price move made it big);
- gross exposure is above 1.2x the passport's max_gross.

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
