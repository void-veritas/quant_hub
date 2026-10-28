"""Executor command line.

    python -m quant_hub.live run --book books/yolo_daily.yaml --mode paper|live [--date YYYY-MM-DD]
    python -m quant_hub.live validate --book books/yolo_daily.yaml [--date ...]
    python -m quant_hub.live status --book books/yolo_daily.yaml [--date ...]
    python -m quant_hub.live cancel-all --book books/yolo_daily.yaml  (live: cancel our orders)
    python -m quant_hub.live pause|resume|kill|unkill --book ...           (flags in the store)

Store root and bucket come from ``LIVE_STORE`` and ``LIVE_BUCKET``; paper equity from
``PAPER_EQUITY``. Without ``--mode`` the mode comes from ``MODE`` (default paper), so each
executor container takes it from its own env file and two books on one server cannot share it.
Telegram alerts are enabled when ``TELEGRAM_BOT_TOKEN`` is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from quant_hub.live.contracts import (
    KILL_KEY,
    Book,
    WeightsFile,
    pause_key,
    state_key,
    validate_weights,
    weights_key,
)
from quant_hub.live.log import get_logger
from quant_hub.live.notify import LogNotifier, TelegramNotifier
from quant_hub.live.store import Store


def make_store() -> Store:
    return Store(os.environ.get("LIVE_STORE", "./store"), os.environ.get("LIVE_BUCKET") or None)


def make_notifier():
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        return TelegramNotifier()
    return LogNotifier()


def make_adapter(book: Book, mode: str, store: Store):
    if mode == "live":
        from quant_hub.live.adapters.hyperliquid import HyperliquidAdapter

        return HyperliquidAdapter()
    if mode == "paper":
        from quant_hub.live.adapters.paper_hl import PaperOnHyperliquid

        return PaperOnHyperliquid(
            book.book_id, store, float(os.environ.get("PAPER_EQUITY", "1000"))
        )
    raise ValueError(mode)


def cmd_run(args) -> int:
    from quant_hub.live.engine import Session

    book = Book.from_yaml(args.book)
    store = make_store()
    log_key = f"logs/{book.book_id}_{args.date}.jsonl"
    log = get_logger("quant_hub.live", Path(store.root) / log_key)
    adapter = make_adapter(book, args.mode, store)
    session = Session(book, store, adapter, args.date, notifier=make_notifier(), log=log)
    try:
        summary = asyncio.run(session.run())
    finally:
        # the session log lives on the server's disk only unless it is mirrored; a failed
        # upload must not change the session's exit code
        for h in log.handlers:
            h.flush()
        try:
            store.mirror(log_key)
        except Exception as err:  # noqa: BLE001
            print(f"session log not uploaded: {err}", file=sys.stderr)
    print(json.dumps(summary.__dict__, indent=1, default=str))
    # no or rejected weights: the session already sent the critical alert with the reason and
    # traded nothing, so a second "failed, exit 1" from the scheduler would only repeat it
    return 0 if summary.status in ("done", "closed", "no_weights", "invalid_weights") else 1


def cmd_validate(args) -> int:
    book = Book.from_yaml(args.book)
    store = make_store()
    raw = store.get(weights_key(book.book_id, args.date))
    if raw is None:
        print("no weights file")
        return 1
    wf = WeightsFile.from_json(raw.decode())
    problems = validate_weights(wf, book, datetime.now(UTC))
    print(f"{len(wf.weights)} names, gross {wf.gross:.3f}, net {wf.net:+.3f}")
    print(f"produced {wf.produced_at}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


def cmd_status(args) -> int:
    book = Book.from_yaml(args.book)
    store = make_store()
    raw = store.get(state_key(book.book_id, args.date))
    if raw is None:
        print("no session state for", args.date)
        return 1
    st = json.loads(raw)
    print(
        json.dumps(
            {k: st[k] for k in ("status", "phase", "updated_at", "halt_reason", "summary")},
            indent=1,
        )
    )
    print("kill:", store.exists(KILL_KEY), "pause:", store.exists(pause_key(book.book_id)))
    return 0


def cmd_flag(args) -> int:
    store = make_store()
    key = (
        KILL_KEY if args.cmd in ("kill", "unkill") else pause_key(Book.from_yaml(args.book).book_id)
    )
    if args.cmd in ("kill", "pause"):
        store.put(key, datetime.now(UTC).isoformat())
        print("set", key)
    else:
        store.delete(key)
        print("cleared", key)
    return 0


def cmd_cancel_all(args) -> int:
    from quant_hub.live.adapters.hyperliquid import HyperliquidAdapter

    async def go():
        a = HyperliquidAdapter()
        await a.connect()
        snap = await a.snapshot()
        from quant_hub.live.oms import Order

        n = 0
        for o in snap.open_orders:
            if o["cloid"]:
                ok = await a.cancel(
                    Order(o["cloid"], o["coin"], o["side"], o["size"], o["price"], "Alo", "")
                )
                n += ok
        await a.close()
        print(f"cancelled {n} of {len(snap.open_orders)} open orders")

    asyncio.run(go())
    return 0


def cmd_riskwatch(args) -> int:
    """One hourly risk check (alerts only); inside the window the session itself guards."""
    from quant_hub.live.riskwatch import (
        BookRisk,
        DayRisk,
        capital_key,
        evaluate,
        evaluate_book,
        evaluate_health,
        evaluate_listing,
        risk_key,
    )

    book = Book.from_yaml(args.book)
    store = make_store()
    ckey = capital_key(book.book_id, args.mode)
    raw = store.get(ckey)
    cap = BookRisk.from_json(raw) if raw else None
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    async def snap():
        adapter = make_adapter(book, args.mode, store)
        await adapter.connect()
        try:
            first = await adapter.snapshot()
            quotes = await adapter.quotes(sorted(first.positions)) if first.positions else {}
            # paper equity marks to the quotes just loaded, so read the account again
            account = (await adapter.snapshot()) if quotes else first
            ledger = getattr(adapter, "capital_flows", None)  # live only: paper has no deposits
            flows = await ledger(cap.since_ms) if ledger and cap else ([], [])
            check = getattr(adapter, "health", None)  # live only
            health = await check() if check else {}
            listed = set(adapter.sz_decimals()) if hasattr(adapter, "sz_decimals") else set()
            return account, {c: q.mid for c, q in quotes.items()}, flows, health, listed
        finally:
            await adapter.close()

    account, mids, (flows, unknown), health, listed = asyncio.run(snap())
    key = risk_key(book.book_id, args.date)
    raw = store.get(key)
    day = DayRisk.from_json(raw) if raw else DayRisk(account.equity)
    if cap is None:  # first check in this mode: the ledger counts from now
        cap = BookRisk(account.equity, account.equity, since_ms=now_ms)
    info = cap.apply_flows(flows, day)
    if unknown:
        info.append(f"ledger entries not understood, check the account: {sorted(set(unknown))}")
    alerts = evaluate(
        account.equity, account.positions, mids, day, book.max_abs_weight, book.max_gross
    )
    alerts += evaluate_health(health, day, now_ms)
    unlisted = evaluate_listing(account.positions, listed, day)
    alerts += unlisted
    store.put(key, day.to_json())
    book_alerts, stop = evaluate_book(
        account.equity, cap, book.review_drawdown, book.pause_drawdown
    )
    store.put(ckey, cap.to_json())
    if stop:
        store.put(pause_key(book.book_id), datetime.now(UTC).isoformat())
    alerts += book_alerts
    report = {
        "equity": account.equity,
        "baseline": day.baseline,
        "start": cap.start,
        "peak": cap.peak,
        "performance": account.equity - cap.start,  # deposits and withdrawals excluded
        "flows": cap.flows,
        "alerts": alerts,
    }
    print(json.dumps(report))
    if info:
        asyncio.run(make_notifier().send("info", f"{book.book_id}: capital", "\n".join(info)))
    if alerts:
        level = "critical" if stop or unlisted else "warning"
        asyncio.run(make_notifier().send(level, f"{book.book_id}: risk check", "\n".join(alerts)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="quant_hub.live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    for name in (
        "run",
        "validate",
        "status",
        "cancel-all",
        "pause",
        "resume",
        "kill",
        "unkill",
        "riskwatch",
    ):
        p = sub.add_parser(name)
        p.add_argument("--book", required=name not in ("kill", "unkill"))
        p.add_argument("--date", default=today)
        if name in ("run", "riskwatch"):
            # argparse does not check a default against choices: make_adapter rejects a typo
            p.add_argument(
                "--mode", choices=["paper", "live"], default=os.environ.get("MODE", "paper")
            )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "run": cmd_run,
        "validate": cmd_validate,
        "status": cmd_status,
        "cancel-all": cmd_cancel_all,
        "riskwatch": cmd_riskwatch,
    }.get(args.cmd, cmd_flag)(args)


if __name__ == "__main__":
    sys.exit(main())
