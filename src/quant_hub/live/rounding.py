"""Venue rounding rules (Hyperliquid perps).

Prices carry at most 5 significant figures and at most ``6 - sz_decimals`` decimals;
sizes are rounded to ``sz_decimals``. Passive prices round away from the touch so an
add-liquidity-only order is never rejected for crossing.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_UP, Decimal

MAX_SIG_FIGS = 5
MAX_DECIMALS = 6


def _quant(px: float, sz_decimals: int, mode) -> float:
    if px <= 0:
        return 0.0
    decimals = min(MAX_DECIMALS - sz_decimals, MAX_SIG_FIGS - 1 - int(math.floor(math.log10(px))))
    decimals = max(decimals, 0)
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(px)).quantize(q, rounding=mode))


def round_price_down(px: float, sz_decimals: int) -> float:
    return _quant(px, sz_decimals, ROUND_DOWN)


def round_price_up(px: float, sz_decimals: int) -> float:
    return _quant(px, sz_decimals, ROUND_UP)


def round_price(px: float, sz_decimals: int, side: str, passive: bool = True) -> float:
    """Passive: buys round down, sells up. Aggressive (IOC): the opposite."""
    down = (side == "buy") == passive
    return round_price_down(px, sz_decimals) if down else round_price_up(px, sz_decimals)


def round_size(size: float, sz_decimals: int) -> float:
    q = Decimal(1).scaleb(-sz_decimals)
    return float(Decimal(repr(abs(size))).quantize(q, rounding=ROUND_DOWN))


def tick(px: float, sz_decimals: int) -> float:
    """Price increment at this level."""
    if px <= 0:
        return 0.0
    decimals = min(MAX_DECIMALS - sz_decimals, MAX_SIG_FIGS - 1 - int(math.floor(math.log10(px))))
    return 10.0 ** (-max(decimals, 0))
