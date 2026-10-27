"""Executor scenarios on the fake exchange (STATUS M7, harness).

Every test drives a full session through `Session.run` with a simulated clock, so the
same code path that will trade paper and live is what is exercised here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from quant_hub.live.adapters.fake import FakeExchange
from quant_hub.live.clock import SimClock
from quant_hub.live.contracts import (
    KILL_KEY,
    Book,
    WeightsFile,
    fills_key,
    pause_key,
    state_key,
    validate_weights,
    weights_key,
)
from quant_hub.live.engine import Session
from quant_hub.live.events import Fill, OrderUpdate
from quant_hub.live.notify import LogNotifier
from quant_hub.live.oms import OMS, make_cloid
from quant_hub.live.policy import Quote, RebalancePolicy, passive_price
from quant_hub.live.rounding import round_price_down, round_price_up, round_size
from quant_hub.live.store import Store

AS_OF = "2026-10-12"
START = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
DECIMALS = {"BTC": 5, "ETH": 4, "SOL": 2, "XRP": 1}
QUOTES = {
    "BTC": Quote(99_990.0, 100_010.0),
    "ETH": Quote(3_999.0, 4_001.0),
    "SOL": Quote(149.95, 150.05),
    "XRP": Quote(2.4995, 2.5005),
}


def book(**kw) -> Book:
    base = dict(
        book_id="yolo_daily",
        capital_fraction=1.0,
        max_abs_weight=0.25,
        min_notional=10.0,
        cross_after_minutes=30,
        reprice_seconds=60,
        imbalance_guard=0.10,
        tick_seconds=5.0,
        max_order_notional=10_000.0,
    )
    base.update(kw)
    return Book(**base)


def weights_file(
    weights: dict, produced_at: datetime | None = None, as_of: str = AS_OF
) -> WeightsFile:
    return WeightsFile(
        book_id="yolo_daily",
        as_of=as_of,
        produced_at=(produced_at or START - timedelta(minutes=2)).isoformat(),
        weights=weights,
        prices={c: QUOTES[c].mid for c in weights if c in QUOTES},
    )


def setup(tmp_path, weights: dict | None, equity=10_000.0, clock_start=None, **xkw):
    store = Store(tmp_path / "store")
    clock = SimClock(clock_start or START + timedelta(seconds=30))
    ex = FakeExchange(equity=equity, decimals=DECIMALS, clock=clock, **xkw)
    ex.set_quotes(dict(QUOTES))
    if weights is not None:
        store.put(weights_key("yolo_daily", AS_OF), weights_file(weights).to_json())
    return store, clock, ex


def run(session: Session):
    return asyncio.run(session.run())


def session(bk, store, ex, clock, notifier=None, **kw) -> Session:
    return Session(bk, store, ex, AS_OF, clock=clock, notifier=notifier or LogNotifier(), **kw)


# ---------------------------------------------------------------- contracts


def test_weights_roundtrip_and_hash():
    wf = weights_file({"BTC": 0.1, "ETH": -0.1})
    text = wf.to_json()
    back = WeightsFile.from_json(text)
    assert back.weights == wf.weights and back.meta["sha256"]
    tampered = text.replace("0.1", "0.2", 1)
    with pytest.raises(ValueError):
        WeightsFile.from_json(tampered)


def test_validate_weights_catches_everything():
    bk = book(universe=["BTC", "ETH", "SOL"], max_weight_change=0.2)
    now = START + timedelta(minutes=1)
    assert validate_weights(weights_file({"BTC": 0.2, "ETH": -0.2}), bk, now) == []
    bad = weights_file(
        {"BTC": 0.3, "XRP": float("nan"), "ETH": 0.2}, produced_at=START - timedelta(hours=2)
    )
    problems = validate_weights(
        bad,
        bk,
        now,
        previous=weights_file({"ETH": -0.2}, as_of="2026-10-11"),
        listed={"BTC", "ETH"},
    )
    text = "\n".join(problems)
    assert "|w|=0.3000" in text and "non-finite" in text and "not in book universe" in text
    assert "stale" in text and "not listed" in text and "change 0.400" in text
    # wrong day
    assert any(
        "not today" in p
        for p in validate_weights(weights_file({"BTC": 0.1}, as_of="2026-10-11"), bk, now)
    )
    # gross cap
    big = weights_file({c: 0.25 for c in ["BTC", "ETH", "SOL"]})
    assert any("gross" in p for p in validate_weights(big, book(max_gross=0.5), now))


def test_rounding_rules():
    assert round_price_down(100_010.123, 5) == 100_010.0  # 5 sig figs -> integer at 1e5
    assert round_price_up(2.50051, 1) == 2.5006  # 6-1 = 5 decimals; sig figs allow 4 -> 2.5006
    assert round_price_down(0.0123456, 0) == 0.012345
    assert round_size(1.23456789, 4) == 1.2345
    q = Quote(2.4995, 2.5005)
    assert passive_price(q, "buy", 1) < q.ask and passive_price(q, "sell", 1) > q.bid


def test_cloid_deterministic_and_oms_rejects_foreign():
    assert make_cloid("b", AS_OF, "BTC", 1) == make_cloid("b", AS_OF, "BTC", 1)
    assert make_cloid("b", AS_OF, "BTC", 1) != make_cloid("b", AS_OF, "ETH", 1)
    oms = OMS("b", AS_OF)
    o = oms.new_order("BTC", "buy", 1.0, 100.0, "Alo", START)
    now = START
    # fill that names another coin with our cloid: ignored
    assert (
        oms.on_fill(Fill(now, "ETH", o.cloid, "1", "x1", "buy", 1.0, 100.0, 0.0, "maker")) is None
    )
    assert (
        oms.on_fill(Fill(now, "BTC", "0xdead", "1", "x2", "buy", 1.0, 100.0, 0.0, "maker")) is None
    )
    assert (
        oms.on_fill(Fill(now, "BTC", o.cloid, "1", "x3", "sell", 1.0, 100.0, 0.0, "maker")) is None
    )
    assert oms.foreign == 3
    f = Fill(now, "BTC", o.cloid, "1", "x4", "buy", 0.4, 100.0, 0.01, "maker")
    assert oms.on_fill(f) is o and oms.on_fill(f) is None  # duplicate ignored
    assert o.filled_size == 0.4 and o.status == "partial"
    # out-of-order: an "open" update after a terminal state is ignored
    oms.mark(o.cloid, "canceled")
    oms.on_update(OrderUpdate(now, "BTC", o.cloid, "1", "open", 0.4, 0.6))
    assert o.status == "canceled"
    back = OMS.from_dict(json.loads(json.dumps(oms.to_dict())))
    assert back.orders[o.cloid].filled_size == 0.4 and back.seen_fill_ids == {"x4"}


# ---------------------------------------------------------------- sessions


def test_happy_path_passive_fills(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2, "SOL": 0.1})
    bk = book()

    # the market drifts through the resting prices after a minute
    def drift(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes(
                {
                    "BTC": Quote(99_900.0, 99_920.0),
                    "ETH": Quote(4_010.0, 4_012.0),
                    "SOL": Quote(149.80, 149.90),
                }
            )

    clock.on_advance = drift
    s = run(session(bk, store, ex, clock))
    assert s.status == "done" and s.completion > 0.99 and s.maker_share == 1.0
    pos = ex.positions
    assert pos["BTC"] == pytest.approx(0.2 * 10_000 / 100_000, abs=1e-5)
    assert pos["ETH"] == pytest.approx(-0.2 * 10_000 / 4_000, abs=1e-4)
    assert all(o.tif == "Alo" for o in ex.placed)
    assert store.exists(fills_key("yolo_daily", AS_OF)) and store.exists(
        state_key("yolo_daily", AS_OF)
    )
    # passive fills at the decision mid: no slippage (negative would be price improvement)
    assert s.slippage_bps <= 0 and abs(s.slippage_bps) < 1


def test_no_weights_holds_and_alerts(tmp_path):
    store, clock, ex = setup(tmp_path, None)
    n = LogNotifier()
    s = run(session(book(), store, ex, clock, notifier=n))
    assert s.status == "no_weights" and not ex.placed
    assert any("no weights" in t for _, t, _ in n.sent)


def test_stale_weights_rejected(tmp_path):
    store, clock, ex = setup(tmp_path, None)
    store.put(
        weights_key("yolo_daily", AS_OF),
        weights_file({"BTC": 0.1}, produced_at=START - timedelta(hours=3)).to_json(),
    )
    s = run(session(book(), store, ex, clock))
    assert s.status == "invalid_weights" and not ex.placed and any("stale" in p for p in s.problems)


def test_cross_after_t_and_window_end(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2})
    bk = book(cross_after_minutes=30, tick_seconds=60.0)
    s = run(session(bk, store, ex, clock))  # market never moves: passive orders never fill
    assert s.status == "done" and s.maker_share == 0.0 and s.n_fills == 2
    tifs = [o.tif for o in ex.placed]
    assert tifs[:2] == ["Alo", "Alo"] and "Ioc" in tifs
    first_ioc = next(o for o in ex.placed if o.tif == "Ioc")
    assert datetime.fromisoformat(first_ioc.created_at) >= START + timedelta(minutes=30)
    assert s.slippage_bps > 0  # crossing pays the half spread


def test_window_end_cancels_everything(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2})
    bk = book(
        cross_after_minutes=120, tick_seconds=300.0
    )  # never aggressive inside the 60-minute window
    s = run(session(bk, store, ex, clock))
    assert s.status == "closed" and s.n_fills == 0
    assert not (asyncio.run(ex.snapshot())).open_orders and ex.cancelled


def test_reprice_follows_the_mid(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2})
    bk = book(cross_after_minutes=120, tick_seconds=30.0, reprice_seconds=60)

    def move(now):
        if START + timedelta(minutes=2) <= now < START + timedelta(minutes=3):
            ex.set_quotes({"BTC": Quote(100_990.0, 101_010.0)})  # +1%: resting bid is now far away

    clock.on_advance = move
    sess = session(bk, store, ex, clock)

    # stop after 5 minutes by pausing
    def pause(now):
        move(now)
        if now >= START + timedelta(minutes=5):
            store.put(pause_key("yolo_daily"), "x")

    clock.on_advance = pause
    s = run(sess)
    assert s.status in ("done", "halted")  # paper fills a resting order once the mid touches it
    prices = [o.price for o in ex.placed]
    assert len(prices) >= 2 and max(prices) > 100_500  # re-quoted near the new mid
    assert ex.cancelled


def test_imbalance_guard_crosses_lagging_side(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.25, "ETH": 0.25, "SOL": -0.25, "XRP": -0.25})
    bk = book(cross_after_minutes=120, tick_seconds=10.0, imbalance_guard=0.10)

    def crash(now):
        if now >= START + timedelta(minutes=1):  # everything falls: bids fill, asks hang
            ex.set_quotes(
                {
                    "BTC": Quote(98_000.0, 98_020.0),
                    "ETH": Quote(3_900.0, 3_902.0),
                    "SOL": Quote(146.0, 146.1),
                    "XRP": Quote(2.40, 2.401),
                }
            )
        if now >= START + timedelta(minutes=5):
            store.put(pause_key("yolo_daily"), "x")

    clock.on_advance = crash
    s = run(session(bk, store, ex, clock))
    iocs = [o for o in ex.placed if o.tif == "Ioc"]
    assert iocs and all(o.side == "sell" for o in iocs)  # only the lagging (short) side was crossed
    assert s.n_fills >= 3


def test_kill_flag_halts_and_cancels(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2})
    store.put(KILL_KEY, "stop")
    n = LogNotifier()
    s = run(session(book(), store, ex, clock, notifier=n))
    assert s.status == "halted" and "kill" in s.halt_reason and not ex.placed


def test_daily_loss_limit_halts(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.25})
    bk = book(daily_loss_limit=0.01, cross_after_minutes=0, tick_seconds=30.0)  # buy immediately

    def crash(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes(
                {"BTC": Quote(90_000.0, 90_010.0)}
            )  # -10% on a 25% position = -2.5% equity

    clock.on_advance = crash
    store2 = store
    # keep a second target open so the session is still running when the loss hits
    store2.put(weights_key("yolo_daily", AS_OF), weights_file({"BTC": 0.25, "ETH": 0.2}).to_json())
    ex.reject_cloids |= {make_cloid("yolo_daily", AS_OF, "ETH", n) for n in range(1, 200)}
    s = run(session(bk, store2, ex, clock, resync_seconds=30))
    assert s.status == "halted" and "loss" in s.halt_reason


def test_risk_rejects_order_past_target():
    from quant_hub.live.risk import RiskController

    rc = RiskController(book(), Store.__new__(Store), 10_000.0)
    assert rc.check_order("BTC", "buy", 0.01, 100_000.0, 10_000.0, 0.0, 0.02) is None
    assert "overshoots" in rc.check_order("BTC", "buy", 0.03, 100_000.0, 10_000.0, 0.0, 0.02)
    assert "notional" in rc.check_order("BTC", "buy", 0.2, 100_000.0, 100_000.0, 0.0, 0.25)
    assert "away from target" in rc.check_order("BTC", "sell", 0.01, 100_000.0, 10_000.0, 0.0, 0.02)
    assert "minimum" in rc.check_order("XRP", "buy", 1.0, 2.5, 10_000.0, 0.0, 10.0)
    assert "clamp" in rc.check_order("BTC", "buy", 0.03, 100_000.0, 10_000.0, 0.0, 0.03)


def test_garbage_stream_does_not_corrupt_state(tmp_path):
    store, clock, ex = setup(
        tmp_path,
        {"BTC": 0.2, "ETH": -0.2, "SOL": 0.1},
        duplicate_events=True,
        shuffle_events=True,
    )
    ex.inject = [
        ex.foreign_fill("BTC", "0x" + "ab" * 16),
        ex.foreign_fill("DOGE", "0x" + "cd" * 16),
    ]

    def drift(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes(
                {
                    "BTC": Quote(99_900.0, 99_920.0),
                    "ETH": Quote(4_010.0, 4_012.0),
                    "SOL": Quote(149.80, 149.90),
                }
            )

    clock.on_advance = drift
    s = run(session(book(), store, ex, clock))
    assert s.status == "done" and s.foreign_messages >= 2
    # ledger equals the venue exactly despite duplicates and reordering
    snap = asyncio.run(ex.snapshot())
    st = json.loads(store.get(state_key("yolo_daily", AS_OF)))
    for c, size in snap.positions.items():
        assert size == pytest.approx(st["targets"][c], abs=1e-6)


def test_lost_stream_messages_are_recovered_by_rest(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2}, drop_stream_fraction=1.0)
    bk = book(tick_seconds=20.0)

    def drift(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes({"BTC": Quote(99_900.0, 99_920.0), "ETH": Quote(4_010.0, 4_012.0)})

    clock.on_advance = drift
    s = run(session(bk, store, ex, clock, resync_seconds=60))
    assert s.status == "done" and s.n_fills == 2  # fills came from REST, not the stream


def test_disconnect_then_reconnect(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2})
    bk = book(tick_seconds=20.0)

    def script(now):
        if START + timedelta(minutes=1) <= now < START + timedelta(minutes=3) and ex.connected:
            ex.disconnect()
        if now >= START + timedelta(minutes=3) and not ex.connected:
            ex.reconnect()
            ex.set_quotes({"BTC": Quote(99_900.0, 99_920.0), "ETH": Quote(4_010.0, 4_012.0)})

    clock.on_advance = script
    s = run(session(bk, store, ex, clock))
    assert s.status == "done" and s.n_fills == 2


def test_restart_resumes_without_duplicating_orders(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2})
    bk = book(cross_after_minutes=120, tick_seconds=20.0)

    # first process: place orders, then "crash" after 1 minute (pause flag used as the stop)
    def stop(now):
        if now >= START + timedelta(minutes=1):
            raise KeyboardInterrupt

    clock.on_advance = stop
    with pytest.raises(KeyboardInterrupt):
        run(session(bk, store, ex, clock))
    placed_before = len(ex.placed)
    assert placed_before == 2 and (asyncio.run(ex.snapshot())).open_orders
    # second process on the same store and the same venue: adopts the resting orders
    clock.on_advance = None
    clock.set(START + timedelta(minutes=2))
    ex.connected = True

    def drift(now):
        if now >= START + timedelta(minutes=3):
            ex.set_quotes({"BTC": Quote(99_900.0, 99_920.0), "ETH": Quote(4_010.0, 4_012.0)})

    clock.on_advance = drift
    s = run(session(bk, store, ex, clock))
    assert s.status == "done" and len(ex.placed) == placed_before  # nothing re-sent
    assert s.n_fills == 2


def test_position_mismatch_halts(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2})
    bk = book(tick_seconds=20.0)

    def tamper(now):
        if now >= START + timedelta(minutes=1):
            ex.positions["BTC"] = (
                ex.positions.get("BTC", 0.0) + 1.0
            )  # someone traded on the account

    clock.on_advance = tamper
    n = LogNotifier()
    s = run(session(bk, store, ex, clock, notifier=n, resync_seconds=30))
    assert s.status == "mismatch" and any("mismatch" in t for _, t, _ in n.sent)


def test_rejects_are_retried_next_tick(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2}, reject_next=1)
    bk = book(tick_seconds=20.0)

    def drift(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes({"BTC": Quote(99_900.0, 99_920.0)})

    clock.on_advance = drift
    s = run(session(bk, store, ex, clock))
    assert s.status == "done" and len(ex.placed) == 2 and ex.placed[0].cloid != ex.placed[1].cloid


def test_policy_skips_dust_and_blind_coins():
    bk = book()
    pol = RebalancePolicy(bk, DECIMALS)
    oms = OMS("yolo_daily", AS_OF)
    plan = pol.step(
        START + timedelta(seconds=10),
        START,
        START + timedelta(hours=1),
        {"BTC": 0.00005, "ETH": 1.0},
        {},
        {"BTC": QUOTES["BTC"]},
        oms,
        {},
        0.0,
        1.0,
    )
    assert plan.send == []  # BTC below $10, ETH has no quote


# ---------------------------------------------------------------- hyperliquid parsing


def test_hyperliquid_parsers():
    from quant_hub.live.adapters.hyperliquid import (
        parse_fill,
        parse_order_update,
        parse_place_response,
    )

    f = parse_fill(
        {
            "coin": "SOL",
            "px": "150.1",
            "sz": "2.5",
            "side": "B",
            "time": 1760259600000,
            "oid": 123,
            "cloid": "0x" + "1" * 32,
            "tid": 777,
            "fee": "0.05",
            "crossed": False,
        }
    )
    assert f.side == "buy" and f.liquidity == "maker" and f.fill_id == "777" and f.price == 150.1
    assert parse_fill({"coin": "SOL", "px": "1", "sz": "1", "side": "A", "time": 0}) is None
    u = parse_order_update(
        {
            "order": {
                "coin": "SOL",
                "oid": 123,
                "cloid": "0x" + "1" * 32,
                "origSz": "2.5",
                "sz": "1.0",
                "timestamp": 1760259600000,
            },
            "status": "open",
            "statusTimestamp": 1760259600500,
        }
    )
    assert u.status == "open" and u.filled_size == 1.5 and u.remaining_size == 1.0
    ok = parse_place_response(
        {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 5}}]}}}
    )
    assert ok.ok and ok.oid == "5"
    err = parse_place_response(
        {
            "status": "ok",
            "response": {"data": {"statuses": [{"error": "Post only order would have crossed"}]}},
        }
    )
    assert not err.ok and "crossed" in err.error
    filled = parse_place_response(
        {
            "status": "ok",
            "response": {
                "data": {"statuses": [{"filled": {"oid": 9, "totalSz": "2.5", "avgPx": "150"}}]}
            },
        }
    )
    assert filled.ok and filled.filled_size == 2.5


def test_cli_flags_and_validate(tmp_path, monkeypatch):
    from quant_hub.live.cli import main

    monkeypatch.setenv("LIVE_STORE", str(tmp_path / "store"))
    bk = tmp_path / "book.yaml"
    bk.write_text("book_id: yolo_daily\n")
    assert main(["pause", "--book", str(bk)]) == 0
    assert (tmp_path / "store" / "flags" / "yolo_daily" / "pause").exists()
    assert main(["resume", "--book", str(bk)]) == 0
    assert not (tmp_path / "store" / "flags" / "yolo_daily" / "pause").exists()
    assert main(["validate", "--book", str(bk), "--date", AS_OF]) == 1  # no file


def test_relative_trade_buffer_skips_small_deltas(tmp_path):
    store, clock, ex = setup(tmp_path, {"BTC": 0.2, "ETH": -0.2})
    ex.positions = {"BTC": 0.0195, "ETH": -0.6}  # BTC within 10% of target, ETH far from it
    ex._cost_basis = {"BTC": 0.0195 * 100_000, "ETH": -0.6 * 4_000}
    bk = book(trade_buffer=0.10, tick_seconds=20.0)

    def drift(now):
        if now >= START + timedelta(minutes=1):
            ex.set_quotes({"ETH": Quote(4_010.0, 4_012.0)})

    clock.on_advance = drift
    s = run(session(bk, store, ex, clock))
    assert s.status == "done" and {o.coin for o in ex.placed} == {"ETH"}
    assert ex.positions["BTC"] == 0.0195


def test_fill_during_quote_refresh_does_not_double_the_position(tmp_path):
    """A fill during the quote refresh must not be followed by an IOC for the same size.

    The replacement would be sized on the position before the fill and overshoot the
    target (2026-10-26 paper run: DOGE and XMR traded 3x their target, then reversed).
    """

    class RacyExchange(FakeExchange):
        moved = False

        async def quotes(self, coins):
            # the market moves through the resting buy exactly when the IOC phase starts
            if not self.moved and self.clock.now() >= START + timedelta(minutes=1):
                self.moved = True
                self.set_quotes({"BTC": Quote(99_900.0, 99_902.0)})
            return await super().quotes(coins)

    store = Store(tmp_path / "store")
    clock = SimClock(START + timedelta(seconds=30))
    ex = RacyExchange(equity=10_000.0, decimals=DECIMALS, clock=clock)
    ex.set_quotes(dict(QUOTES))
    store.put(weights_key("yolo_daily", AS_OF), weights_file({"BTC": 0.25}).to_json())
    bk = book(cross_after_minutes=1, tick_seconds=10.0)
    run(session(bk, store, ex, clock))
    target = 0.25 * 10_000.0 / QUOTES["BTC"].mid
    assert ex.positions["BTC"] == pytest.approx(target, rel=0.02)
    bought = sum(f.size for f in ex._all_fills if f.coin == "BTC" and f.side == "buy")
    assert bought <= target * 1.02  # never bought twice


def test_store_mirror_uploads_without_rewriting(tmp_path):
    """The session log is uploaded as it stands; the open handle keeps writing to it."""

    class FakeS3:
        def __init__(self):
            self.objects = {}

        def put_object(self, Bucket, Key, Body):  # noqa: N803 (boto3 signature)
            self.objects[(Bucket, Key)] = Body

    store = Store(tmp_path / "store", bucket="b")
    store._s3 = FakeS3()
    assert store.mirror("logs/yolo_daily_2026-10-27.jsonl") is False  # no file yet
    p = store.path("logs/yolo_daily_2026-10-27.jsonl")
    p.parent.mkdir(parents=True)
    with p.open("a") as fh:
        fh.write('{"msg": "session started"}\n')
        fh.flush()
        inode = p.stat().st_ino
        assert store.mirror("logs/yolo_daily_2026-10-27.jsonl") is True
        fh.write('{"msg": "session done"}\n')
    assert p.stat().st_ino == inode
    assert p.read_text().count("\n") == 2
    assert (
        store._s3.objects[("b", "logs/yolo_daily_2026-10-27.jsonl")]
        == b'{"msg": "session started"}\n'
    )
    assert Store(tmp_path / "local").mirror("logs/x.jsonl") is False  # no bucket: nothing to do


def test_riskwatch_alerts_once_per_level_and_per_coin():
    from quant_hub.live.riskwatch import DayRisk, evaluate

    day = DayRisk(baseline=1000.0)
    mids = {"BTC": 100_000.0, "ETH": 4_000.0}
    calm = evaluate(990.0, {"BTC": 0.002, "ETH": -0.05}, mids, day, 0.25, 1.5)
    assert calm == []  # -1%, weights 0.20 / -0.20, gross 0.40
    bad = evaluate(965.0, {"BTC": 0.002, "ETH": -0.05}, mids, day, 0.25, 1.5)
    assert len(bad) == 1 and "-3.5%" in bad[0]  # crosses -2% and -3% in one message
    assert evaluate(962.0, {"BTC": 0.002, "ETH": -0.05}, mids, day, 0.25, 1.5) == []  # no repeat
    worse = evaluate(940.0, {"BTC": 0.002, "ETH": -0.05}, mids, day, 0.25, 1.5)
    assert len(worse) == 1 and "-6.0%" in worse[0]  # the -5% level, once
    big = evaluate(940.0, {"BTC": 0.004, "ETH": -0.05}, mids, day, 0.25, 1.5)
    assert big == ["BTC weight +0.43 > 1.5 x clamp 0.25"]
    lev = evaluate(940.0, {"BTC": 0.004, "ETH": -0.35}, mids, day, 0.25, 1.5)  # gross 1.91
    assert any(m.startswith("gross") for m in lev) and any(m.startswith("ETH") for m in lev)
    again = DayRisk.from_json(day.to_json())
    assert again.baseline == 1000.0 and again.alerted == day.alerted


def test_mode_comes_from_the_container_env(monkeypatch):
    # two executors on one server: each takes paper/live from its own env file, not the shell
    from quant_hub.live.cli import build_parser

    monkeypatch.delenv("MODE", raising=False)
    assert build_parser().parse_args(["run", "--book", "b.yaml"]).mode == "paper"
    monkeypatch.setenv("MODE", "live")
    assert build_parser().parse_args(["riskwatch", "--book", "b.yaml"]).mode == "live"
    assert build_parser().parse_args(["run", "--book", "b.yaml", "--mode", "paper"]).mode == "paper"


def test_sub_account_orders_are_signed_with_its_vault_address():
    # without vaultAddress the master's API wallet would trade the master account, while the
    # session reads the sub-account's positions: refuse every combination but the two valid ones
    from quant_hub.live.adapters.hyperliquid import signing_vault

    main, sub, agent, other = "0xAa", "0xBb", "0xCc", "0xDd"
    approved_by_main = {"role": "agent", "data": {"user": main.lower()}}
    assert signing_vault(main, {"role": "user"}, agent, approved_by_main) is None
    sub_role = {"role": "subAccount", "data": {"master": main}}
    assert signing_vault(sub, sub_role, agent, approved_by_main) == sub
    bad = [
        (sub, sub_role, agent, {"role": "agent", "data": {"user": other}}),  # someone else's agent
        (main, {"role": "user"}, agent, {"role": "missing"}),  # not approved (or revoked)
        (main, {"role": "user"}, main, {"role": "user"}),  # the main wallet key itself
        (sub, sub_role, main, {"role": "user"}),
        ("0xEe", {"role": "vault"}, agent, approved_by_main),
        (sub, {"role": "subAccount"}, agent, approved_by_main),  # master unknown
    ]
    for args in bad:
        with pytest.raises(ValueError):
            signing_vault(*args)


def test_book_drawdown_review_then_pause_once_until_a_new_peak():
    from quant_hub.live.riskwatch import BookRisk, evaluate_book

    b = BookRisk(800.0, 800.0)
    assert evaluate_book(900.0, b, 0.08, 0.12) == ([], False) and b.peak == 900.0
    alerts, stop = evaluate_book(830.0, b, 0.08, 0.12)  # -70 = -8.75% of the 800 start
    assert len(alerts) == 1 and "review" in alerts[0] and not stop
    assert evaluate_book(825.0, b, 0.08, 0.12) == ([], False)  # once per level
    alerts, stop = evaluate_book(800.0, b, 0.08, 0.12)  # -100 = -12.5%
    assert stop and "pause flag set" in alerts[0]
    assert evaluate_book(790.0, b, 0.08, 0.12) == ([], False)
    assert evaluate_book(901.0, b, 0.08, 0.12) == ([], False) and b.alerted == set()  # re-armed
    assert evaluate_book(100.0, BookRisk(800.0, 800.0), None, None) == ([], False)
    assert BookRisk.from_json(b.to_json()) == b


def test_riskwatch_pauses_the_book_past_its_pause_level(tmp_path, monkeypatch):
    from quant_hub.live import cli
    from quant_hub.live.adapters.base import AccountSnapshot
    from quant_hub.live.riskwatch import BookRisk, capital_key

    book = tmp_path / "b.yaml"
    book.write_text("book_id: b\nreview_drawdown: 0.08\npause_drawdown: 0.12\n")
    monkeypatch.setenv("LIVE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("LIVE_BUCKET", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    equity = {"v": 1000.0}

    class Acc:
        async def connect(self):
            pass

        async def close(self):
            pass

        async def snapshot(self):
            return AccountSnapshot(equity["v"], {}, [], "")

    monkeypatch.setattr(cli, "make_adapter", lambda *a: Acc())
    argv = ["riskwatch", "--book", str(book), "--mode", "live", "--date", "2026-10-29"]
    assert cli.main(argv) == 0
    store = cli.make_store()
    assert BookRisk.from_json(store.get(capital_key("b", "live"))).start == 1000.0
    assert not store.exists(pause_key("b"))
    equity["v"] = 870.0
    assert cli.main(argv) == 0 and store.exists(pause_key("b"))
    assert store.get(capital_key("b", "paper")) is None  # paper and live measured apart


def test_run_without_weights_alerts_once(tmp_path, monkeypatch):
    # the session's own "no weights file" alert is enough: exit 0, so the scheduler stays quiet
    from quant_hub.live import cli

    book = tmp_path / "b.yaml"
    book.write_text("book_id: b\n")
    monkeypatch.setenv("LIVE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("LIVE_BUCKET", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(cli, "make_adapter", lambda *a: FakeExchange(equity=800.0))
    assert cli.main(["run", "--book", str(book), "--date", "2026-10-28"]) == 0
    state = json.loads(cli.make_store().get(state_key("b", "2026-10-28")))
    assert state["status"] == "no_weights"


def test_ledger_flows_of_the_perp_account():
    from quant_hub.live.adapters.hyperliquid import parse_ledger

    me, other = "0xMe", "0xother"
    rows = [
        {"time": 1, "delta": {"type": "deposit", "usdc": "800.0"}},
        {"time": 2, "delta": {"type": "withdraw", "usdc": "100.0", "fee": "1.0"}},
        {
            "time": 3,
            "delta": {
                "type": "subAccountTransfer",
                "usdc": "50",
                "user": other,
                "destination": me.lower(),
            },
        },
        {
            "time": 4,
            "delta": {
                "type": "internalTransfer",
                "usdc": "20",
                "user": me,
                "destination": other,
                "fee": "0",
            },
        },
        {"time": 5, "delta": {"type": "accountClassTransfer", "usdc": "30", "toPerp": False}},
        {
            "time": 6,
            "delta": {
                "type": "send",
                "user": other,
                "destination": me,
                "sourceDex": "spot",
                "destinationDex": "",
                "token": "USDC",
                "amount": "5",
                "usdcValue": "5",
                "fee": "0",
            },
        },
        {
            "time": 7,
            "delta": {
                "type": "send",
                "user": other,
                "destination": me,
                "sourceDex": "spot",
                "destinationDex": "spot",
                "token": "USDC",
                "amount": "5",
                "usdcValue": "5",
            },
        },
        {
            "time": 8,
            "delta": {"type": "spotTransfer", "token": "X", "amount": "1", "usdcValue": "1"},
        },
        {"time": 9, "delta": {"type": "vaultDeposit", "vault": other, "usdc": "10"}},
    ]
    flows, unknown = parse_ledger(rows, me)
    assert flows == [
        (1, 800.0, "deposit"),
        (2, -101.0, "withdraw"),
        (3, 50.0, "subAccountTransfer"),
        (4, -20.0, "internalTransfer"),
        (5, -30.0, "accountClassTransfer"),
        (6, 5.0, "send"),
    ]
    assert unknown == ["vaultDeposit"]  # reported, not guessed


def test_riskwatch_counts_deposits_and_withdrawals_as_capital(tmp_path, monkeypatch):
    from quant_hub.live import cli
    from quant_hub.live.adapters.base import AccountSnapshot
    from quant_hub.live.riskwatch import BookRisk, capital_key

    book = tmp_path / "b.yaml"
    book.write_text("book_id: b\nreview_drawdown: 0.08\npause_drawdown: 0.12\n")
    monkeypatch.setenv("LIVE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("LIVE_BUCKET", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    acc = {"equity": 1000.0, "flows": []}

    class Live:
        async def connect(self):
            pass

        async def close(self):
            pass

        async def snapshot(self):
            return AccountSnapshot(acc["equity"], {}, [], "")

        async def capital_flows(self, start_ms):
            return [f for f in acc["flows"] if f[0] >= start_ms], []

    monkeypatch.setattr(cli, "make_adapter", lambda *a: Live())
    argv = ["riskwatch", "--book", str(book), "--mode", "live", "--date", "2026-10-29"]
    assert cli.main(argv) == 0
    store = cli.make_store()
    t0 = BookRisk.from_json(store.get(capital_key("b", "live"))).since_ms
    acc.update(equity=700.0, flows=[(t0 + 1, -300.0, "withdraw")])  # -30% equity, no trading loss
    assert cli.main(argv) == 0 and not store.exists(pause_key("b"))
    cap = BookRisk.from_json(store.get(capital_key("b", "live")))
    assert (cap.start, cap.peak) == (700.0, 700.0) and cap.flows == [[t0 + 1, -300.0, "withdraw"]]
    assert cli.main(argv) == 0  # the same ledger row is not booked twice
    assert BookRisk.from_json(store.get(capital_key("b", "live"))).start == 700.0
    acc["equity"] = 610.0  # a real -90 = -12.9% of the 700 start
    assert cli.main(argv) == 0 and store.exists(pause_key("b"))


def test_margin_and_key_expiry_alert_once_a_day():
    from quant_hub.live.adapters.hyperliquid import agent_expiry
    from quant_hub.live.riskwatch import DayRisk, evaluate_health

    agents = [
        {"name": "x", "address": "0xOther", "validUntil": 5},
        {"address": "0xAb", "validUntil": 10**13},
    ]
    assert agent_expiry(agents, "0xab") == 10**13 and agent_expiry(agents, "0xcd") is None
    day = DayRisk(800.0)
    now = 10**13 - 3 * 86_400_000  # three days left
    lines = evaluate_health({"margin_used": 0.6, "agent_valid_until_ms": 10**13}, day, now)
    assert len(lines) == 2 and "margin" in lines[0] and "3.0 days" in lines[1]
    assert evaluate_health({"margin_used": 0.7, "agent_valid_until_ms": 10**13}, day, now) == []
    assert (
        evaluate_health(
            {"margin_used": 0.1, "agent_valid_until_ms": now + 30 * 86_400_000}, DayRisk(1.0), now
        )
        == []
    )
    assert evaluate_health({}, DayRisk(1.0), now) == []  # paper: nothing to check
