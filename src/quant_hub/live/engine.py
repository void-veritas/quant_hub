"""The execution session: one book, one trading date, one event loop.

Order of work each tick: apply streamed events, periodic REST reconciliation, session
guards (flags, loss limit, window), policy step, risk gate per order, persist state.
No parallel writers: a single coroutine owns the OMS. Restart at any point resumes
from ``state/<book>/<as_of>.json`` and the venue's own records.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

from quant_hub.live.adapters.base import ExchangeAdapter
from quant_hub.live.clock import Clock
from quant_hub.live.contracts import (
    FILL_COLUMNS,
    Book,
    FillRecord,
    WeightsFile,
    fills_key,
    state_key,
    validate_weights,
    weights_key,
)
from quant_hub.live.events import Disconnected, Fill, OrderUpdate, Reconnected
from quant_hub.live.log import event as log_event
from quant_hub.live.notify import LogNotifier, Notifier
from quant_hub.live.oms import OMS
from quant_hub.live.policy import Quote, RebalancePolicy
from quant_hub.live.risk import RiskController
from quant_hub.live.rounding import round_size
from quant_hub.live.store import Store


@dataclass
class SessionSummary:
    status: str  # done | no_weights | invalid_weights | halted | closed | mismatch
    n_fills: int = 0
    completion: float = 0.0  # 1 - remaining gross / planned gross
    fees: float = 0.0
    slippage_bps: float = 0.0  # notional-weighted, signed vs decision price (positive = cost)
    maker_share: float = 0.0
    planned_gross: float = 0.0
    filled_gross: float = 0.0
    halt_reason: str = ""
    rejected: int = 0
    foreign_messages: int = 0
    problems: list[str] = field(default_factory=list)


class Session:
    def __init__(
        self,
        book: Book,
        store: Store,
        adapter: ExchangeAdapter,
        as_of: str,
        clock: Clock | None = None,
        notifier: Notifier | None = None,
        log: logging.Logger | None = None,
        resync_seconds: float = 60.0,
    ):
        self.book = book
        self.store = store
        self.adapter = adapter
        self.as_of = as_of
        self.clock = clock or Clock()
        self.notifier = notifier or LogNotifier()
        self.log = log or logging.getLogger("quant_hub.live")
        self.resync_seconds = resync_seconds
        self.start, self.end = book.window(as_of)
        self.oms: OMS = OMS(book.book_id, as_of)
        self.risk: RiskController | None = None
        self.targets: dict[str, float] = {}
        self.decision_px: dict[str, float] = {}
        self.start_positions: dict[str, float] = {}
        self.planned_net = 0.0
        self.planned_gross = 0.0
        self.phase = "init"
        self.status = "init"
        self.last_quote_ts: dict[str, datetime] = {}
        self._last_resync: datetime | None = None
        self._streaming = True
        self.equity = 0.0
        self.wf: WeightsFile | None = None

    # ------------------------------------------------------------------ helpers
    def _log(self, msg: str, level: int = logging.INFO, **data) -> None:
        log_event(self.log, msg, level, book=self.book.book_id, as_of=self.as_of, **data)

    async def _alert(self, level: str, title: str, text: str = "") -> None:
        self._log(
            title, logging.WARNING if level != "info" else logging.INFO, alert=level, detail=text
        )
        try:
            await self.notifier.send(level, f"{self.book.book_id} {self.as_of}: {title}", text)
        except Exception as err:  # alerts must never take the session down
            self._log("notifier failed", logging.ERROR, error=str(err))

    def positions(self) -> dict[str, float]:
        """Ledger positions: start of session plus what we filled."""
        pos = dict(self.start_positions)
        for c, s in self.oms.filled_position().items():
            pos[c] = pos.get(c, 0.0) + s
        return pos

    # ------------------------------------------------------------------ setup
    def _previous_weights(self) -> WeightsFile | None:
        keys = [
            k
            for k in self.store.list(f"weights/{self.book.book_id}/")
            if k < weights_key(self.book.book_id, self.as_of)
        ]
        if not keys:
            return None
        raw = self.store.get(keys[-1])
        return WeightsFile.from_json(raw.decode()) if raw else None

    async def _load_weights(self, listed: set[str]) -> WeightsFile | None:
        raw = self.store.get(weights_key(self.book.book_id, self.as_of))
        if raw is None:
            await self._alert("critical", "no weights file", "holding positions, nothing traded")
            self.status = "no_weights"
            return None
        try:
            wf = WeightsFile.from_json(raw.decode())
        except (ValueError, KeyError, TypeError) as err:
            await self._alert("critical", "weights file unreadable", str(err))
            self.status = "invalid_weights"
            return None
        problems = validate_weights(
            wf, self.book, self.clock.now(), self._previous_weights(), listed
        )
        if problems:
            await self._alert("critical", "weights file rejected", "\n".join(problems[:20]))
            self.status = "invalid_weights"
            self._problems = problems
            return None
        return wf

    def _set_targets(
        self, wf: WeightsFile, quotes: dict[str, Quote], decimals: dict[str, int]
    ) -> None:
        capital = self.equity * self.book.capital_fraction
        coins = set(wf.weights) | {c for c, s in self.start_positions.items() if abs(s) > 0}
        for coin in sorted(coins):
            q = quotes.get(coin)
            if q is None:
                continue
            # plan-vs-fact is measured against the analytics decision price when given
            self.decision_px[coin] = float(wf.prices.get(coin) or q.mid)
            w = wf.weights.get(coin, 0.0)
            raw = w * capital / q.mid
            size = round_size(abs(raw), decimals.get(coin, 0)) * (1 if raw >= 0 else -1)
            pos = self.start_positions.get(coin, 0.0)
            if self.book.trade_buffer > 0 and abs(size - pos) <= self.book.trade_buffer * abs(size):
                size = pos  # inside the no-trade zone (note 016: relative buffer, trade to target)
            self.targets[coin] = size
        self.planned_net = sum(
            (t - self.start_positions.get(c, 0.0)) * self.decision_px[c]
            for c, t in self.targets.items()
        )
        self.planned_gross = sum(
            abs(t - self.start_positions.get(c, 0.0)) * self.decision_px[c]
            for c, t in self.targets.items()
        )

    # ------------------------------------------------------------------ state
    def _persist(self, summary: SessionSummary | None = None) -> None:
        body = {
            "book_id": self.book.book_id,
            "as_of": self.as_of,
            "status": self.status,
            "phase": self.phase,
            # which venue the session traded on: "hyperliquid" = live money, "paper_hl" = paper
            "adapter": getattr(self.adapter, "name", ""),
            "updated_at": self.clock.now().isoformat(),
            "equity": self.equity,
            "start_equity": self.risk.state.start_equity if self.risk else self.equity,
            "start_positions": self.start_positions,
            "targets": self.targets,
            "decision_px": self.decision_px,
            "planned_net": self.planned_net,
            "planned_gross": self.planned_gross,
            "halt_reason": self.risk.state.halt_reason if self.risk else "",
            "oms": self.oms.to_dict(),
            "summary": asdict(summary) if summary else asdict(self.summary()),
        }
        self.store.put(
            state_key(self.book.book_id, self.as_of), json.dumps(body, sort_keys=True, default=str)
        )

    def _restore(self) -> bool:
        raw = self.store.get(state_key(self.book.book_id, self.as_of))
        if raw is None:
            return False
        st = json.loads(raw)
        if st.get("status") in ("no_weights", "invalid_weights", "mismatch"):
            return False  # earlier attempt never traded; start clean
        self.oms = OMS.from_dict(st["oms"])
        self.start_positions = {k: float(v) for k, v in st["start_positions"].items()}
        self.targets = {k: float(v) for k, v in st["targets"].items()}
        self.decision_px = {k: float(v) for k, v in st["decision_px"].items()}
        self.planned_net = float(st["planned_net"])
        self.planned_gross = float(st["planned_gross"])
        self.risk = RiskController(self.book, self.store, float(st["start_equity"]))
        if st.get("halt_reason"):
            self.risk.halt(st["halt_reason"])
        self._log("resumed from state", n_orders=len(self.oms.orders), n_fills=len(self.oms.fills))
        return True

    # ------------------------------------------------------------------ reconciliation
    async def _reconcile(self, first: bool = False) -> bool:
        """Sync the OMS with the venue's records (REST wins). False = mismatch, must halt."""
        snap = await self.adapter.snapshot()
        self.equity = snap.equity
        start_ms = int(self.start.timestamp() * 1000)
        for ev in await self.adapter.fills_since(start_ms):
            if isinstance(ev, Fill):
                self.oms.on_fill(ev)
        venue_open = {o["cloid"]: o for o in snap.open_orders}
        unknown = [c for c in venue_open if c not in self.oms.orders]
        if unknown:
            await self._alert(
                "warning", "unknown open orders on the account", ", ".join(unknown[:10])
            )
        for o in self.oms.open_orders():
            v = venue_open.get(o.cloid)
            if v is None:
                # not resting on the venue: either fully filled (fills applied above) or gone
                if o.remaining > 1e-12:
                    o.status = "canceled"
                    o.reason = o.reason or "not on venue at reconcile"
            else:
                o.oid = v.get("oid", o.oid)
                o.status = "partial" if o.filled_size > 0 else "open"
        if first:
            return True
        ledger = self.positions()
        decimals = self.adapter.sz_decimals()
        for coin in set(ledger) | set(snap.positions):
            tol = 10.0 ** (-decimals.get(coin, 0)) * 1.5
            if abs(ledger.get(coin, 0.0) - snap.positions.get(coin, 0.0)) > tol:
                await self._alert(
                    "critical",
                    "position mismatch",
                    f"{coin}: ledger {ledger.get(coin, 0.0)} "
                    f"vs venue {snap.positions.get(coin, 0.0)}",
                )
                return False
        self._last_resync = self.clock.now()
        return True

    # ------------------------------------------------------------------ events
    def _apply_events(self) -> None:
        for ev in self.adapter.drain():
            if isinstance(ev, Fill):
                if self.oms.on_fill(ev) is None:
                    self._log(
                        "foreign or duplicate fill ignored",
                        logging.WARNING,
                        coin=ev.coin,
                        cloid=ev.cloid,
                        fill_id=ev.fill_id,
                    )
            elif isinstance(ev, OrderUpdate):
                if self.oms.on_update(ev) is None:
                    self._log(
                        "foreign order update ignored",
                        logging.WARNING,
                        coin=ev.coin,
                        cloid=ev.cloid,
                    )
            elif isinstance(ev, Disconnected):
                self._streaming = False
                self._log("stream disconnected", logging.WARNING, reason=ev.reason)
            elif isinstance(ev, Reconnected):
                self._streaming = True
                self._last_resync = None  # force a REST resync on the next tick
                self._log("stream reconnected")

    # ------------------------------------------------------------------ orders
    async def _cancel_all(self) -> None:
        for o in self.oms.open_orders():
            ok = await self.adapter.cancel(o)
            if ok:
                self.oms.mark(o.cloid, "canceled", reason="session cancel")
            else:
                self._log("cancel failed", logging.WARNING, cloid=o.cloid, coin=o.coin)

    async def _execute_plan(self, plan, quotes: dict[str, Quote]) -> None:
        # A cancel that fails usually means the order filled in the meantime: its
        # replacement is not sent this tick (the next tick re-plans from the fill).
        unconfirmed: set[str] = set()
        for o in plan.cancel:
            if await self.adapter.cancel(o):
                self.oms.mark(o.cloid, "canceled", reason=f"policy {plan.phase}")
            else:
                unconfirmed.add(o.coin)
            self.last_quote_ts.pop(o.cloid, None)
        # fills that arrived while quotes were refreshed or orders cancelled: the plan was
        # sized before them, so positions are re-read before any new order is checked
        planned_on = self.positions()
        self._apply_events()
        pos = self.positions()
        for it in plan.send:
            if it.coin in unconfirmed:
                self._log("send skipped: cancel not confirmed", coin=it.coin, side=it.side)
                continue
            if abs(pos.get(it.coin, 0.0) - planned_on.get(it.coin, 0.0)) > 1e-12:
                # a fill landed after the plan was sized: re-plan this coin next tick
                self._log("send skipped: position moved", coin=it.coin, side=it.side, size=it.size)
                continue
            why = self.risk.check_order(
                it.coin,
                it.side,
                it.size,
                it.price,
                self.equity,
                pos.get(it.coin, 0.0),
                self.targets.get(it.coin, 0.0),
                reduce_only=it.reduce_only,
            )
            if why:
                self.risk.state.rejected.append(f"{it.coin} {it.side} {it.size}@{it.price}: {why}")
                self._log(
                    "order rejected by risk",
                    logging.WARNING,
                    coin=it.coin,
                    side=it.side,
                    size=it.size,
                    price=it.price,
                    why=why,
                )
                continue
            o = self.oms.new_order(
                it.coin, it.side, it.size, it.price, it.tif, self.clock.now(), it.reduce_only
            )
            self._persist()  # the cloid is on disk before the venue sees it
            res = await self.adapter.place(o)
            if not res.ok:
                self.oms.mark(o.cloid, "rejected", reason=res.error)
                self._log(
                    "order rejected by venue",
                    logging.WARNING,
                    coin=it.coin,
                    cloid=o.cloid,
                    error=res.error,
                )
                continue
            self.oms.mark(o.cloid, "open", oid=res.oid)
            self.last_quote_ts[o.cloid] = self.clock.now()
            self._log(
                "order placed",
                coin=it.coin,
                side=it.side,
                size=it.size,
                price=it.price,
                tif=it.tif,
                cloid=o.cloid,
                oid=res.oid,
            )

    # ------------------------------------------------------------------ summary
    def summary(self) -> SessionSummary:
        fills = self.oms.fills
        gross = sum(f.size * f.price for f in fills)
        slip = 0.0
        if gross > 0:
            for f in fills:
                d = self.decision_px.get(f.coin)
                if d:
                    sgn = 1 if f.side == "buy" else -1
                    slip += sgn * (f.price / d - 1) * 1e4 * f.size * f.price
            slip /= gross
        remaining = sum(
            abs(t - self.positions().get(c, 0.0)) * self.decision_px.get(c, 0.0)
            for c, t in self.targets.items()
        )
        completion = 1 - remaining / self.planned_gross if self.planned_gross > 0 else 1.0
        return SessionSummary(
            status=self.status,
            n_fills=len(fills),
            completion=max(min(completion, 1.0), 0.0),
            fees=self.oms.total_fees(),
            slippage_bps=slip,
            maker_share=(sum(f.size * f.price for f in fills if f.liquidity == "maker") / gross)
            if gross
            else 0.0,
            planned_gross=self.planned_gross,
            filled_gross=gross,
            halt_reason=self.risk.state.halt_reason if self.risk else "",
            rejected=len(self.risk.state.rejected) if self.risk else 0,
            foreign_messages=self.oms.foreign,
            problems=getattr(self, "_problems", []),
        )

    def _write_fills(self) -> None:
        rows = [
            asdict(
                FillRecord(
                    self.book.book_id,
                    self.as_of,
                    f.ts.isoformat(),
                    f.coin,
                    f.cloid,
                    f.oid,
                    f.fill_id,
                    f.side,
                    f.size,
                    f.price,
                    f.fee,
                    f.liquidity,
                    self.decision_px.get(f.coin, float("nan")),
                )
            )
            for f in self.oms.fills
        ]
        df = pd.DataFrame(rows, columns=FILL_COLUMNS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.store.put(fills_key(self.book.book_id, self.as_of), buf.getvalue())

    def _done(self) -> bool:
        if self.oms.open_orders():
            return False
        pos = self.positions()
        for c, t in self.targets.items():
            gap = abs(t - pos.get(c, 0.0)) * self.decision_px.get(c, 0.0)
            if gap >= self.book.min_notional:
                return False
        return True

    # ------------------------------------------------------------------ main
    async def run(self) -> SessionSummary:
        await self.adapter.connect()
        decimals = self.adapter.sz_decimals()
        policy = RebalancePolicy(self.book, decimals)
        try:
            resumed = self._restore()
            snap = await self.adapter.snapshot()
            self.equity = snap.equity
            if not resumed:
                self.wf = await self._load_weights(set(decimals))
                if self.wf is None:
                    self._persist()
                    return self.summary()
                self.start_positions = dict(snap.positions)
                self.risk = RiskController(self.book, self.store, snap.equity)
                quotes = await self.adapter.quotes(
                    sorted(set(self.wf.weights) | set(snap.positions))
                )
                self._set_targets(self.wf, quotes, decimals)
                missing = [c for c in self.wf.weights if c not in quotes]
                if missing:
                    await self._alert(
                        "warning", "no quotes for some names, skipped", ", ".join(missing)
                    )
            if not await self._reconcile(first=not resumed):
                self.status = "mismatch"
                self.risk.halt("position mismatch at start")
                self._persist()
                return self.summary()
            self.status = "running"
            self._persist()
            await self._alert(
                "info",
                "session started",
                f"{len(self.targets)} names, planned gross {self.planned_gross:.0f}",
            )

            while True:
                now = self.clock.now()
                self._apply_events()
                connected = getattr(self.adapter, "connected", True)
                if connected and (
                    self._last_resync is None
                    or (now - self._last_resync).total_seconds() >= self.resync_seconds
                ):
                    if not await self._reconcile():
                        self.status = "mismatch"
                        self.risk.halt("position mismatch")
                        await self._cancel_all()
                        break
                why = self.risk.session_check(now, self.as_of, self.equity)
                if why is not None:
                    await self._cancel_all()
                    self.status = "halted" if self.risk.state.halted else "closed"
                    if self.risk.state.halted:
                        await self._alert("critical", "session halted", why)
                    break
                if self._done():
                    self.status = "done"
                    break
                if connected:  # while disconnected nothing is sent; resting orders stay as they are
                    quotes = await self.adapter.quotes(sorted(self.targets))
                    plan = policy.step(
                        now,
                        self.start,
                        self.end,
                        self.targets,
                        self.positions(),
                        quotes,
                        self.oms,
                        self.last_quote_ts,
                        self.planned_net,
                        self.planned_gross,
                    )
                    self.phase = plan.phase
                    await self._execute_plan(plan, quotes)
                self._persist()
                await self.clock.sleep(self.book.tick_seconds)
            # loop exit: finalise
            self._apply_events()
            await self._reconcile()
            self._write_fills()
            s = self.summary()
            self._persist(s)
            await self._alert(
                "info" if s.status == "done" else "warning",
                f"session {s.status}",
                f"fills {s.n_fills}, done {s.completion:.0%}, slippage {s.slippage_bps:.1f} bps, "
                f"maker {s.maker_share:.0%}, fees {s.fees:.2f}",
            )
            return s
        finally:
            await self.adapter.close()
