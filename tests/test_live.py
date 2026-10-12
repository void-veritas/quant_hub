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
