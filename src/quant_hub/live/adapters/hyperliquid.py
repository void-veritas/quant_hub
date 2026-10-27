"""Hyperliquid adapter on the official SDK.

Credentials: ``HL_ACCOUNT_ADDRESS`` (the account whose positions are traded: a main
account or one of its sub-accounts) and ``HL_AGENT_PRIVATE_KEY`` (an API/agent wallet
approved for trading only; it cannot withdraw). A sub-account has no key of its own: its
orders are signed by its master's API wallet with ``vaultAddress`` set, otherwise they land
on the master account. ``signing_vault`` decides this from the venue's ``userRole`` at
connect and refuses any other combination. REST (``Info``) is the source of truth; the
websocket feeds ``orderUpdates`` and ``userFills`` into the event queue as a hint. Orders
carry our ``cloid`` so the venue's records can always be matched to the OMS. Nothing here
reads the main wallet.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import UTC, datetime

from quant_hub.live.adapters.base import AccountSnapshot, PlaceResult
from quant_hub.live.events import Disconnected, Event, Fill, OrderUpdate, Reconnected
from quant_hub.live.oms import Order
from quant_hub.live.policy import Quote

MAINNET = "https://api.hyperliquid.xyz"
TESTNET = "https://api.hyperliquid-testnet.xyz"


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def signing_vault(account: str, account_role: dict, agent: str, agent_role: dict) -> str | None:
    """``vaultAddress`` for orders on ``account`` signed by ``agent``, or ValueError.

    Roles are the venue's ``userRole`` answers. A main account ("user") is traded by an
    agent it approved; a sub-account by an agent its master approved, with the
    sub-account as vault. The main wallet key itself is refused (RUNBOOK: never on the VPS).
    """
    a = account.lower()
    if account_role.get("role") == "user":
        owner, vault = a, None
    elif account_role.get("role") == "subAccount":
        owner, vault = str((account_role.get("data") or {}).get("master", "")).lower(), account
    else:
        raise ValueError(f"{account} is a {account_role.get('role')!r} account: cannot trade it")
    if agent.lower() in (a, owner):
        raise ValueError("HL_AGENT_PRIVATE_KEY is the account's own key, use an API wallet")
    if agent_role.get("role") != "agent":
        raise ValueError(f"{agent} is not an approved API wallet (role {agent_role.get('role')!r})")
    approved_by = str((agent_role.get("data") or {}).get("user", "")).lower()
    if not owner or approved_by != owner:
        raise ValueError(f"API wallet {agent} is approved by {approved_by}, not by {owner or a}")
    return vault


def agent_expiry(agents: list[dict], agent: str) -> int | None:
    """``validUntil`` (ms) of our API wallet in the owner's ``extraAgents`` list, if listed."""
    for a in agents:
        if str(a.get("address", "")).lower() == agent.lower() and a.get("validUntil"):
            return int(a["validUntil"])
    return None


def parse_ledger(rows: list[dict], me: str) -> tuple[list[tuple[int, float, str]], list[str]]:
    """External cash flows of the perp account ``me`` from ``userNonFundingLedgerUpdates``.

    Returns ([(time_ms, usd, type)], unknown types). ``usd`` is the change of the perp
    account value (marginSummary.accountValue, what the session sizes on): deposits and
    transfers in are positive, withdrawals and transfers out negative. Spot-only moves are
    not flows of the perp account. A type this function does not know is reported, never
    guessed: the caller alerts so a person can look.
    """
    me = me.lower()
    out: list[tuple[int, float, str]] = []
    unknown: list[str] = []
    for r in rows:
        d, t = r.get("delta") or {}, int(r.get("time", 0))
        kind = d.get("type", "?")
        fee = float(d.get("fee", 0) or 0)
        to_me = str(d.get("destination", "")).lower() == me
        usd = None
        if kind == "deposit":
            usd = float(d["usdc"])
        elif kind == "withdraw":
            usd = -(float(d["usdc"]) + fee)
        elif kind in ("internalTransfer", "subAccountTransfer"):
            usd = float(d["usdc"]) if to_me else -(float(d["usdc"]) + fee)
        elif kind == "accountClassTransfer":  # spot <-> perp of the same account
            usd = float(d["usdc"]) if d.get("toPerp") else -float(d["usdc"])
        elif kind == "send":  # between dexes and accounts; "" is the main perp dex
            amount = float(d.get("usdcValue") or 0)
            into = to_me and d.get("destinationDex", "") == ""
            outof = str(d.get("user", "")).lower() == me and d.get("sourceDex", "") == ""
            usd = (amount if into else 0.0) - ((amount + fee) if outof else 0.0)
        elif kind == "spotTransfer":
            usd = 0.0
        else:
            unknown.append(kind)
        if usd:
            out.append((t, usd, kind))
    return out, unknown


def parse_fill(raw: dict) -> Fill | None:
    """Map an HL fill record (REST ``userFills`` or WS ``userFills``) to a :class:`Fill`."""
    cloid = raw.get("cloid")
    if not cloid:
        return None  # not ours: our orders always carry a cloid
    side = "buy" if raw.get("side") == "B" else "sell"
    return Fill(
        ts=_ms_to_dt(int(raw["time"])),
        coin=str(raw["coin"]),
        cloid=str(cloid),
        oid=str(raw.get("oid", "")),
        fill_id=str(raw.get("tid") or raw.get("hash")),
        side=side,
        size=float(raw["sz"]),
        price=float(raw["px"]),
        fee=float(raw.get("fee", 0.0)),
        liquidity="taker" if raw.get("crossed") else "maker",
    )


def parse_order_update(raw: dict) -> OrderUpdate | None:
    """Map a WS ``orderUpdates`` entry to :class:`OrderUpdate`."""
    o = raw.get("order", {})
    cloid = o.get("cloid")
    if not cloid:
        return None
    status = str(raw.get("status", "")).lower()
    mapped = {
        "open": "open",
        "filled": "filled",
        "canceled": "canceled",
        "rejected": "rejected",
    }.get(status, "canceled" if status else "open")
    size, remaining = float(o.get("origSz", 0.0)), float(o.get("sz", 0.0))
    return OrderUpdate(
        ts=_ms_to_dt(int(raw.get("statusTimestamp", o.get("timestamp", 0)))),
        coin=str(o["coin"]),
        cloid=str(cloid),
        oid=str(o.get("oid", "")),
        status=mapped,
        filled_size=max(size - remaining, 0.0),
        remaining_size=remaining,
        reason=status if mapped == "canceled" else "",
    )


def parse_place_response(resp: dict) -> PlaceResult:
    """Interpret the exchange response of a single-order ``order`` action."""
    try:
        if resp.get("status") != "ok":
            return PlaceResult(False, error=str(resp.get("response", resp))[:200])
        st = resp["response"]["data"]["statuses"][0]
    except (KeyError, IndexError, TypeError):
        return PlaceResult(False, error=f"unparseable response {str(resp)[:200]}")
    if "error" in st:
        return PlaceResult(False, error=str(st["error"])[:200])
    if "resting" in st:
        return PlaceResult(True, oid=str(st["resting"]["oid"]))
    if "filled" in st:
        return PlaceResult(
            True, oid=str(st["filled"]["oid"]), filled_size=float(st["filled"]["totalSz"])
        )
    return PlaceResult(False, error=f"unknown status {st}")


class HyperliquidAdapter:
    """Live adapter. Construct with ``paper=True`` to use the SDK read-only (no wallet)."""

    name = "hyperliquid"

    def __init__(
        self,
        address: str | None = None,
        private_key: str | None = None,
        base_url: str = MAINNET,
        paper: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.address = address or os.environ.get("HL_ACCOUNT_ADDRESS", "")
        self._key = private_key or os.environ.get("HL_AGENT_PRIVATE_KEY", "")
        self.base_url = base_url
        self.paper = paper
        self.info = None
        self.exchange = None
        self._decimals: dict[str, int] = {}
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self.connected = False
        self._loop = loop
        if not self.address:
            raise ValueError("HL_ACCOUNT_ADDRESS is required")
        if not paper and not self._key:
            raise ValueError("HL_AGENT_PRIVATE_KEY is required for live trading")

    # --- lifecycle -------------------------------------------------------------------
    async def connect(self) -> None:
        from hyperliquid.info import Info

        self.info = Info(self.base_url, skip_ws=False)
        meta = self.info.meta()
        self._decimals = {
            a["name"]: int(a["szDecimals"]) for a in meta["universe"] if not a.get("isDelisted")
        }
        if not self.paper:
            from eth_account import Account
            from hyperliquid.exchange import Exchange

            wallet = Account.from_key(self._key)
            agent_role = self.info.post("/info", {"type": "userRole", "user": wallet.address})
            vault = signing_vault(
                self.address,
                self.info.post("/info", {"type": "userRole", "user": self.address}),
                wallet.address,
                agent_role,
            )
            self._agent = wallet.address
            self._agent_owner = str((agent_role.get("data") or {}).get("user", ""))
            self.exchange = Exchange(
                wallet, self.base_url, meta=meta, vault_address=vault, account_address=self.address
            )
        self.info.subscribe({"type": "orderUpdates", "user": self.address}, self._on_order_updates)
        self.info.subscribe({"type": "userFills", "user": self.address}, self._on_user_fills)
        self.connected = True

    async def close(self) -> None:
        if self.info is not None:
            try:
                self.info.disconnect_websocket()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass
        self.connected = False

    def sz_decimals(self) -> dict[str, int]:
        return dict(self._decimals)

    # --- websocket callbacks (SDK thread) -----------------------------------------------
    def _push(self, ev: Event) -> None:
        with self._lock:
            self._events.append(ev)

    def _on_order_updates(self, msg: dict) -> None:
        for raw in msg.get("data", []) or []:
            ev = parse_order_update(raw)
            if ev is not None:
                self._push(ev)

    def _on_user_fills(self, msg: dict) -> None:
        data = msg.get("data", {}) or {}
        if data.get("isSnapshot"):
            return  # history is fetched through REST at reconcile time
        for raw in data.get("fills", []) or []:
            ev = parse_fill(raw)
            if ev is not None:
                self._push(ev)

    def drain(self) -> list[Event]:
        with self._lock:
            out, self._events = self._events, []
        ws = getattr(self.info, "ws_manager", None)
        alive = (
            getattr(getattr(ws, "ws", None), "sock", None) is not None if ws is not None else True
        )
        if self.connected and not alive:
            self.connected = False
            out.append(Disconnected(datetime.now(UTC), "websocket closed"))
        elif not self.connected and alive:
            self.connected = True
            out.append(Reconnected(datetime.now(UTC)))
        return out

    # --- REST (source of truth) ---------------------------------------------------------
    async def snapshot(self) -> AccountSnapshot:
        state = await asyncio.to_thread(self.info.user_state, self.address)
        equity = float(state["marginSummary"]["accountValue"])
        positions = {}
        for ap in state.get("assetPositions", []):
            p = ap.get("position", {})
            szi = float(p.get("szi", 0.0))
            if szi:
                positions[p["coin"]] = szi
        raw_open = await asyncio.to_thread(self.info.open_orders, self.address)
        open_orders = [
            {
                "coin": o["coin"],
                "cloid": o.get("cloid", ""),
                "oid": str(o["oid"]),
                "side": "buy" if o["side"] == "B" else "sell",
                "size": float(o.get("origSz", o["sz"])),
                "remaining": float(o["sz"]),
                "price": float(o["limitPx"]),
            }
            for o in raw_open
        ]
        return AccountSnapshot(equity, positions, open_orders, datetime.now(UTC).isoformat())

    async def quotes(self, coins: list[str]) -> dict[str, Quote]:
        """Best bid/ask from L2 snapshots (one call per coin, weight 2 each)."""
        out: dict[str, Quote] = {}
        for coin in coins:
            if coin not in self._decimals:
                continue
            book = await asyncio.to_thread(self.info.l2_snapshot, coin)
            levels = book.get("levels", [[], []])
            if levels[0] and levels[1]:
                out[coin] = Quote(float(levels[0][0]["px"]), float(levels[1][0]["px"]))
        return out

    async def capital_flows(self, start_ms: int) -> tuple[list[tuple[int, float, str]], list[str]]:
        """Deposits, withdrawals and transfers of the account since ``start_ms`` (parse_ledger)."""
        rows = await asyncio.to_thread(
            self.info.post,
            "/info",
            {"type": "userNonFundingLedgerUpdates", "user": self.address, "startTime": start_ms},
        )
        return parse_ledger(rows or [], self.address)

    async def health(self) -> dict:
        """Margin in use (share of account value) and the API wallet's expiry (live only)."""
        state = await asyncio.to_thread(self.info.user_state, self.address)
        ms = state.get("marginSummary", {})
        value = float(ms.get("accountValue", 0) or 0)
        used = float(ms.get("totalMarginUsed", 0) or 0)
        out = {"margin_used": used / value if value > 0 else 0.0, "agent_valid_until_ms": None}
        owner = getattr(self, "_agent_owner", "")
        if owner:
            agents = await asyncio.to_thread(
                self.info.post, "/info", {"type": "extraAgents", "user": owner}
            )
            out["agent_valid_until_ms"] = agent_expiry(agents or [], self._agent)
        return out

    async def fills_since(self, start_ms: int) -> list[Event]:
        raw = await asyncio.to_thread(self.info.user_fills_by_time, self.address, start_ms)
        return [f for f in (parse_fill(r) for r in raw) if f is not None]

    # --- orders ---------------------------------------------------------------------
    async def place(self, order: Order) -> PlaceResult:
        if self.paper or self.exchange is None:
            return PlaceResult(False, error="paper adapter cannot place orders")
        from hyperliquid.utils.types import Cloid

        resp = await asyncio.to_thread(
            self.exchange.order,
            order.coin,
            order.side == "buy",
            order.size,
            order.price,
            {"limit": {"tif": order.tif}},
            order.reduce_only,
            Cloid(order.cloid),
        )
        return parse_place_response(resp)

    async def cancel(self, order: Order) -> bool:
        if self.paper or self.exchange is None:
            return False
        from hyperliquid.utils.types import Cloid

        resp = await asyncio.to_thread(
            self.exchange.cancel_by_cloid, order.coin, Cloid(order.cloid)
        )
        try:
            st = resp["response"]["data"]["statuses"][0]
        except (KeyError, IndexError, TypeError):
            return False
        return st == "success"
