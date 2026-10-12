"""Events flowing from an exchange adapter into the engine.

Every message names its coin and, where applicable, the client order id it belongs to;
the OMS checks each one against its own records and discards anything it did not
originate (a "foreign" message is logged, never applied).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Event:
    ts: datetime


@dataclass(frozen=True)
class OrderUpdate(Event):
    """Order lifecycle change as reported by the venue."""

    coin: str
    cloid: str
    oid: str
    status: str  # open | filled | canceled | rejected | partial
    filled_size: float
    remaining_size: float
    reason: str = ""


@dataclass(frozen=True)
class Fill(Event):
    """One execution; ``fill_id`` makes fills idempotent across redelivery."""

    coin: str
    cloid: str
    oid: str
    fill_id: str
    side: str  # buy | sell
    size: float
    price: float
    fee: float
    liquidity: str  # maker | taker


@dataclass(frozen=True)
class Disconnected(Event):
    """The streaming connection dropped; the engine must resync from REST."""

    reason: str = ""


@dataclass(frozen=True)
class Reconnected(Event):
    pass
