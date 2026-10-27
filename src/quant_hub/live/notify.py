"""Alerts out, safety-only commands in.

The notifier is a separate process from the executor: the executor never listens on
the network. Inbound commands (``/status``, ``/pause``, ``/snapshot``) are accepted only
from whitelisted Telegram user ids; they read the store and at most write flags there;
resuming is a shell operation on the host, by design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx

from quant_hub.live.contracts import pause_key, state_key
from quant_hub.live.store import Store

LEVELS = ("info", "warning", "critical")


class Notifier(Protocol):
    async def send(self, level: str, title: str, text: str = "") -> None: ...


@dataclass
class LogNotifier:
    """Collects alerts in memory (tests) and prints them."""

    sent: list[tuple[str, str, str]] | None = None

    async def send(self, level: str, title: str, text: str = "") -> None:
        if self.sent is None:
            self.sent = []
        self.sent.append((level, title, text))


class TelegramNotifier:
    """Outbound alerts through the Bot API. Token and chat id come from the environment."""

    def __init__(self, token: str | None = None, chat_id: str | None = None, timeout: float = 10.0):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout
        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")

    @property
    def _base(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    async def send(self, level: str, title: str, text: str = "") -> None:
        icon = {"info": "i", "warning": "!", "critical": "!!!"}.get(level, "?")
        body = f"[{icon}] {title}\n{text}".strip()
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(
                f"{self._base}/sendMessage", json={"chat_id": self.chat_id, "text": body[:4000]}
            )
            r.raise_for_status()

    async def send_photo(self, png: bytes, caption: str = "") -> None:
        async with httpx.AsyncClient(timeout=max(self.timeout, 30.0)) as c:
            r = await c.post(
                f"{self._base}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption[:1000]},
                files={"photo": ("snapshot.png", png, "image/png")},
            )
            r.raise_for_status()


def snapshot_key(book_id: str) -> str:
    """The book's latest one-picture summary, written by analytics after the attribution."""
    return f"reports/{book_id}/snapshot.png"


class CommandPoller:
    """Long-poll the bot for ``/status``, ``/pause <book>`` and ``/snapshot`` from allowed users."""

    def __init__(
        self, store: Store, notifier: TelegramNotifier, allowed_user_ids: set[int], books: list[str]
    ):
        self.store = store
        self.notifier = notifier
        self.allowed = allowed_user_ids
        self.books = books
        self.offset = 0

    async def poll_once(self, client: httpx.AsyncClient) -> list[str]:
        r = await client.get(
            f"{self.notifier._base}/getUpdates",
            params={"offset": self.offset, "timeout": 25},
            timeout=35,
        )
        r.raise_for_status()
        handled: list[str] = []
        for upd in r.json().get("result", []):
            self.offset = max(self.offset, upd["update_id"] + 1)
            msg = upd.get("message") or {}
            uid = (msg.get("from") or {}).get("id")
            text = (msg.get("text") or "").strip()
            if uid not in self.allowed or not text.startswith("/"):
                continue
            handled.append(await self.handle(text))
        return handled

    async def handle(self, text: str) -> str:
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd == "/status":
            reply = self.status(args[0] if args else None)
        elif cmd == "/pause":
            if not args or args[0] not in self.books:
                reply = f"usage: /pause <book>; books: {', '.join(self.books)}"
            else:
                self.store.put(pause_key(args[0]), datetime.now(UTC).isoformat())
                reply = (
                    f"paused {args[0]}: no new orders; resting ones are cancelled on the next "
                    "tick. Resume from the host."
                )
        elif cmd == "/snapshot":
            books = args[:1] if args and args[0] in self.books else self.books
            missing = []
            for b in books:
                png = self.store.get(snapshot_key(b))
                if png is None:
                    missing.append(b)
                else:
                    await self.notifier.send_photo(png, f"{b} snapshot")
            if not missing:
                return f"snapshot sent: {', '.join(books)}"
            reply = f"no snapshot yet for {', '.join(missing)} (written daily after 10:10 UTC)"
        else:
            reply = "commands: /status [book], /pause <book>, /snapshot [book]"
        await self.notifier.send("info", reply)
        return reply

    def status(self, book: str | None) -> str:
        lines = []
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        for b in self.books if book is None else [book]:
            raw = self.store.get(state_key(b, today))
            if raw is None:
                lines.append(f"{b}: no session today")
                continue
            st = json.loads(raw)
            s = st.get("summary", {})
            lines.append(
                f"{b}: {st.get('status')} phase={st.get('phase')} fills={s.get('n_fills', 0)} "
                f"done={s.get('completion', 0):.0%} slip={s.get('slippage_bps', 0):.1f}bps"
                + (f" HALT: {st.get('halt_reason')}" if st.get("halt_reason") else "")
            )
            if self.store.exists(pause_key(b)):
                lines.append(f"{b}: PAUSED")
        return "\n".join(lines)
