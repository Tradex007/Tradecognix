"""Round 2 (25 Nov 2026): market-order price protection, shared broker throttling, points stops."""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.brokers import throttle
from app.brokers.base import BrokerError, Instrument, OrderRequest, OrderResult, protected_limit_price
from app.brokers.paper import PaperBroker
from app.data.synthetic import generate_candles
from app.engine.feeds import SyntheticFeed
from app.engine.runner import StrategyRunner
from app.strategy.parser import RuleBasedParser

P = RuleBasedParser()
TIME_ENTRY = "Buy Nifty ATM call at 9:30 AM, exit at 3:15 pm"


# ---------------- protected limit price ----------------
def test_protected_limit_price_rounds_to_tick_against_the_band():
    assert protected_limit_price(100.0, "BUY", 1.0, 0.05) == 101.0
    assert protected_limit_price(100.0, "SELL", 1.0, 0.05) == 99.0
    assert protected_limit_price(123.47, "BUY", 1.0, 0.05) == 124.70   # 124.7047 -> down to tick, never above the cap
    assert protected_limit_price(123.47, "SELL", 1.0, 0.05) == 122.25  # 122.2353 -> up to tick, never below the floor
    assert protected_limit_price(0.10, "BUY", 1.0, 0.05) == 0.10        # band < tick still trades at LTP
    assert protected_limit_price(0.05, "SELL", 50.0, 0.05) == 0.05      # never below one tick
    assert protected_limit_price(24510.0, "BUY", 0.5, 0.0) == pytest.approx(24632.55)  # bad tick -> 0.05 default
    with pytest.raises(ValueError):
        protected_limit_price(0, "BUY", 1)
    with pytest.raises(ValueError):
        protected_limit_price(float("nan"), "SELL", 1)


# ---------------- paper broker honours LIMIT ----------------
def _inst():
    return Instrument(symbol="NIFTY24DEC24500CE", token="1", exchange="NFO", instrument_type="OPTIDX", option_type="CE", strike=24500, tick_size=0.05)


def test_paper_limit_buy_never_fills_above_limit():
    px = {"v": 100.0}
    b = PaperBroker(price_source=lambda i: px["v"], slippage_pct=5.0)  # slippage would push the fill to 105
    r = b.place_order(OrderRequest(_inst(), "BUY", 75, order_type="LIMIT", price=101.0))
    assert r.status == "filled" and r.avg_price == 101.0
    px["v"] = 102.0  # market moved beyond the limit: IOC cancel, no position
    r = b.place_order(OrderRequest(_inst(), "BUY", 75, order_type="LIMIT", price=101.0))
    assert r.status == "cancelled" and "not marketable" in r.message
    assert b.positions()[0]["net_qty"] == 75


def test_paper_limit_sell_never_fills_below_limit():
    b = PaperBroker(price_source=lambda i: 100.0, slippage_pct=5.0)
    r = b.place_order(OrderRequest(_inst(), "SELL", 75, order_type="LIMIT", price=99.0))
    assert r.status == "filled" and r.avg_price == 99.0
    r = b.place_order(OrderRequest(_inst(), "SELL", 75, order_type="LIMIT", price=100.5))
    assert r.status == "cancelled"
    assert b.place_order(OrderRequest(_inst(), "SELL", 75, order_type="LIMIT", price=0)).status == "rejected"


# ---------------- runner sends protected LIMIT orders ----------------
class RecordingBroker(PaperBroker):
    def __init__(self, *a, ltp_fails=False, **kw):
        super().__init__(*a, **kw)
        self.reqs = []
        self.ltp_fails = ltp_fails

    def ltp(self, inst):
        if self.ltp_fails:
            raise BrokerError("quote down")
        return super().ltp(inst)

    def place_order(self, req):
        self.reqs.append(req)
        return super().place_order(req)


def _setup(broker, mode="paper", text=TIME_ENTRY, pct=1.0):
    s = P.parse(text).strategy
    feed = SyntheticFeed(s.underlying, seed=3, days=5)
    if broker.price_source is None:
        broker.price_source = feed.price
    trades, orders = [], []
    r = StrategyRunner(s, broker, "TCTEST", mode, on_trade=trades.append, on_order=orders.append, market_protection_pct=pct)
    r.sleep = lambda _s: None
    day = feed._base_for(datetime(2026, 11, 20, 15, 30)).index.normalize().unique()[-1]
    return s, feed, r, trades, orders, day


def _at(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return (day + pd.Timedelta(hours=h, minutes=m)).to_pydatetime()


def _enter(r, feed, day, hhmm="09:30"):
    t = _at(day, hhmm)
    feed.now = t
    return r.on_candle_close(feed.candles("5m", t), t)


def test_runner_entry_and_exit_are_protected_limits():
    b = RecordingBroker(slippage_pct=0.5)
    s, feed, r, trades, orders, day = _setup(b)
    assert _enter(r, feed, day)["action"] == "entry"
    entry = b.reqs[0]
    assert entry.order_type == "LIMIT" and entry.side == "BUY"
    ltp = feed.price(entry.instrument)
    assert ltp < entry.price <= ltp * 1.01 + 1e-9
    assert round(entry.price / 0.05, 6) == round(entry.price / 0.05)  # on the tick grid
    assert orders[0]["order_type"] == "LIMIT" and orders[0]["limit_price"] == entry.price
    t = _at(day, "15:15")
    feed.now = t
    r.on_minute(t)
    ex = b.reqs[-1]
    assert ex.order_type == "LIMIT" and ex.side == "SELL" and ex.price >= feed.price(ex.instrument) * 0.99 - 1e-9
    assert trades and trades[0]["exit_reason"] == "hard_exit"


def test_protection_disabled_sends_market():
    b = RecordingBroker(slippage_pct=0.5)
    s, feed, r, *_rest = _setup(b, pct=0)
    day = _rest[-1]
    _enter(r, feed, day)
    assert b.reqs[0].order_type == "MARKET" and b.reqs[0].price == 0


def test_default_protection_comes_from_settings():
    from app.config import get_settings

    s = P.parse(TIME_ENTRY).strategy
    r = StrategyRunner(s, PaperBroker(price_source=lambda i: 100.0), "T")
    assert r.protection_pct == get_settings().market_protection_pct == 1.0


def test_entry_without_ltp_is_refused_but_exit_falls_back_to_market():
    b = RecordingBroker(slippage_pct=0.5, ltp_fails=True)
    s, feed, r, trades, orders, day = _setup(b)
    assert _enter(r, feed, day)["action"] == "error"
    assert b.reqs == [] and r.state["position"] is None and orders[0]["status"] == "error"
    # now with a position open, the quote dies: the exit must still go out (as MARKET)
    b.ltp_fails = False
    r.state["last_bar"] = None
    t = _at(day, "09:30")
    r.on_candle_close(feed.candles("5m", t), t)
    assert r.state["position"] is not None
    b.ltp_fails = True
    t = _at(day, "15:15")
    feed.now = t
    r.on_minute(t)
    assert b.reqs[-1].order_type == "MARKET" and r.state["position"] is None


class OpenLimitBroker(RecordingBroker):
    """Live-like broker: LIMIT orders rest 'open'; cancel works unless told otherwise."""

    def __init__(self, *a, fill_on_cancel=False, cancel_unconfirmed=False, **kw):
        super().__init__(*a, **kw)
        self.state = {}
        self.fill_on_cancel = fill_on_cancel
        self.cancel_unconfirmed = cancel_unconfirmed
        self.cancels = []

    def place_order(self, req):
        self.reqs.append(req)
        if req.order_type == "LIMIT":
            oid = f"L{len(self.reqs)}"
            self.state[oid] = {"order_id": oid, "status": "open", "avg_price": 0.0, "raw": {}}
            return OrderResult(order_id=oid, status="open")
        return PaperBroker.place_order(self, req)

    def cancel_order(self, order_id):
        self.cancels.append(order_id)
        if self.cancel_unconfirmed:
            return OrderResult(order_id=order_id, status="pending")
        o = self.state[order_id]
        o.update(status="complete", avg_price=101.0) if self.fill_on_cancel else o.update(status="cancelled")
        return OrderResult(order_id=order_id, status="cancelled")

    def order_book(self):
        return list(self.state.values())


def test_live_unfilled_entry_limit_is_cancelled_and_not_booked():
    b = OpenLimitBroker(slippage_pct=0.5)
    s, feed, r, trades, orders, day = _setup(b, mode="live")
    assert _enter(r, feed, day)["action"] == "error"
    assert b.cancels == ["L1"] and r.state["position"] is None
    assert len(b.reqs) == 1  # entries are never chased with a market order
    assert not r.state["needs_reconcile"]


def test_live_limit_that_fills_during_cancel_is_booked():
    b = OpenLimitBroker(slippage_pct=0.5, fill_on_cancel=True)
    s, feed, r, trades, orders, day = _setup(b, mode="live")
    assert _enter(r, feed, day)["action"] == "entry"
    assert r.state["position"]["legs"][0]["entry"] == 101.0


def test_live_unconfirmed_cancel_halts_for_reconcile():
    b = OpenLimitBroker(slippage_pct=0.5, cancel_unconfirmed=True)
    s, feed, r, trades, orders, day = _setup(b, mode="live")
    assert _enter(r, feed, day)["action"] == "error"
    assert r.state["needs_reconcile"] and r.state["halted"]


def test_live_unfilled_exit_limit_retries_as_market():
    b = OpenLimitBroker(slippage_pct=0.5, fill_on_cancel=True)  # entry fills on cancel
    s, feed, r, trades, orders, day = _setup(b, mode="live")
    _enter(r, feed, day)
    b.fill_on_cancel = False  # exit LIMIT will not fill
    t = _at(day, "15:15")
    feed.now = t
    r.on_minute(t)
    assert [q.order_type for q in b.reqs] == ["LIMIT", "LIMIT", "MARKET"]
    assert r.state["position"] is None and trades[0]["exit_reason"] == "hard_exit"


# ---------------- throttle ----------------
@pytest.fixture()
def fake_clock(monkeypatch):
    clock = {"t": 1000.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(throttle, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(throttle, "sleep", sleep)
    monkeypatch.setattr(throttle.random, "uniform", lambda a, b: 1.0)
    throttle.reset()
    yield clock, slept
    throttle.reset()


def test_token_bucket_paces_requests(fake_clock):
    clock, slept = fake_clock
    b = throttle.configure("x", rate=2.0, capacity=2.0)
    for _ in range(6):
        b.acquire()
    # 2 burst tokens, then one every 0.5 s
    assert sum(slept) == pytest.approx(2.0)
    assert all(s == pytest.approx(0.5) for s in slept)


def test_bucket_is_shared_per_broker(fake_clock):
    assert throttle.bucket_for("angelone") is throttle.bucket_for("angelone")
    assert throttle.bucket_for("angelone").rate == 3.0
    assert throttle.bucket_for("zerodha") is not throttle.bucket_for("angelone")
    assert throttle.bucket_for("somebroker").rate == throttle.FALLBACK_RATE


def test_rate_limit_errors_back_off_exponentially(fake_clock):
    clock, slept = fake_clock
    throttle.configure("b", rate=1000.0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise BrokerError("Access denied because of exceeding access rate")
        return "ok"

    assert throttle.call("b", flaky) == "ok"
    assert [round(s, 3) for s in slept if s >= 0.9] == [1.0, 2.0, 4.0]


def test_non_rate_limit_errors_are_not_retried_and_retries_are_bounded(fake_clock):
    throttle.configure("c", rate=1000.0)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise BrokerError("invalid token")

    with pytest.raises(BrokerError):
        throttle.call("c", bad)
    assert calls["n"] == 1

    def always429():
        calls["n"] += 1
        raise BrokerError("HTTP 429 too many requests")

    calls["n"] = 0
    with pytest.raises(BrokerError):
        throttle.call("c", always429)
    assert calls["n"] == throttle.MAX_RETRIES + 1


def test_is_rate_limit_error_detection():
    import httpx

    req = httpx.Request("GET", "https://x")
    assert throttle.is_rate_limit_error(httpx.HTTPStatusError("x", request=req, response=httpx.Response(429, request=req)))
    assert throttle.is_rate_limit_error(BrokerError("Too Many Requests"))
    assert not throttle.is_rate_limit_error(httpx.HTTPStatusError("x", request=req, response=httpx.Response(500, request=req)))
    assert not throttle.is_rate_limit_error(ValueError("bad"))


def test_angel_candles_use_shared_throttle_not_fixed_sleep(fake_clock, monkeypatch):
    from app.brokers.angelone import AngelOneAdapter

    clock, slept = fake_clock
    n = {"calls": 0}

    class FakeApi:
        def getCandleData(self, p):
            n["calls"] += 1
            if n["calls"] == 2:
                return {"status": False, "message": "Access denied because of exceeding access rate", "errorcode": "AB1019"}
            return {"status": True, "errorcode": "", "data": [["2026-11-20T09:15:00+05:30", 1, 2, 0.5, 1.5, 10]]}

    a = AngelOneAdapter({})
    a._api = FakeApi()
    inst = Instrument(symbol="Nifty 50", token="99926000", exchange="NSE")
    df = a.historical_candles(inst, "1m", datetime(2026, 9, 1), datetime(2026, 11, 20))  # 3 chunks of 30 days
    assert n["calls"] == 4 and len(df) == 1  # one chunk retried after backoff
    assert any(s >= 0.9 for s in slept)  # the backoff, not a fixed 0.35 s per chunk
    import inspect

    import app.brokers.angelone as mod
    assert "time.sleep" not in inspect.getsource(mod)


def test_upstox_429_is_retried(fake_clock):
    import httpx

    from app.brokers.upstox import UpstoxAdapter

    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, json={"status": "error", "errors": [{"message": "Too Many Request Sent"}]})
        return httpx.Response(200, json={"status": "success", "data": {"NSE_INDEX:Nifty 50": {"last_price": 24500.5}}})

    a = UpstoxAdapter({"access_token": "t"}, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert a.ltp(Instrument(symbol="Nifty 50", token="NSE_INDEX|Nifty 50", exchange="NSE_INDEX")) == 24500.5
    assert hits["n"] == 2


# ---------------- points stops / targets ----------------
def test_parser_and_backtest_points_stop():
    s = P.parse("Buy nifty futures when 9 EMA crosses above 21 EMA on 5 min, stop loss 20 points, target 40 points").strategy
    assert s.exit.stop_loss_points == 20 and s.exit.target_points == 40 and s.exit.stop_loss_pct is None
    res = run_backtest(s, generate_candles("NIFTY", days=30, seed=7, end=date(2026, 11, 20)))
    for t in res.trades:
        move = t["legs"][0]["exit_ref"] - t["legs"][0]["entry_ref"]
        if t["exit_reason"] == "stop_loss":
            assert move <= -20 + 1e-6
        if t["exit_reason"] == "target":
            assert move >= 40 - 1e-6
    assert {"stop_loss", "target"} & {t["exit_reason"] for t in res.trades}


def test_runner_points_stop():
    s = P.parse("Buy Nifty ATM call at 9:30 AM, stop loss 10 points, exit at 3:15 pm").strategy
    px = {"v": 100.0}
    b = PaperBroker(price_source=lambda i: px["v"], slippage_pct=0.0)
    trades = []
    r = StrategyRunner(s, b, "T", on_trade=trades.append, market_protection_pct=1.0)
    feed = SyntheticFeed(s.underlying, seed=3, days=5)
    day = feed._base_for(datetime(2026, 11, 20, 15, 30)).index.normalize().unique()[-1]
    t = _at(day, "09:30")
    feed.now = t
    r.on_candle_close(feed.candles("5m", t), t)
    entry = r.state["position"]["legs"][0]["entry"]
    px["v"] = entry - 9
    r.on_minute(t + timedelta(minutes=1))
    assert r.state["position"] is not None
    px["v"] = entry - 10.5
    r.on_minute(t + timedelta(minutes=2))
    assert r.state["position"] is None and trades[0]["exit_reason"] == "stop_loss"
