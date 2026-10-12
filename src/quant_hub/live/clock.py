"""Wall clock and simulated clock with the same interface (asyncio-friendly)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta


class Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SimClock(Clock):
    """Virtual time: ``sleep`` advances the clock instantly (tests, replays)."""

    def __init__(self, start: datetime):
        self._now = start
        self.on_advance = None  # optional callback(now) run after each sleep

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
        if self.on_advance:
            self.on_advance(self._now)
        await asyncio.sleep(0)

    def set(self, when: datetime) -> None:
        self._now = when
