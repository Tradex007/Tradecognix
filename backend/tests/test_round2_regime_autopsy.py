"""Round 2 (25 Nov 2026): ADX indicator, regime breakdown and whipsaw autopsy."""
import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import indicators as ind
from app.backtest.autopsy import analyse, build_autopsy, hour_buckets
from app.backtest.engine import Backtester, run_backtest
from app.backtest.regime import classify_days, regime_breakdown
from app.data.candles import resample
from app.data.synthetic import generate_candles
from app.strategy.dsl import Strategy
from app.strategy.parser import RuleBasedParser

P = RuleBasedParser()
EMA = "Buy nifty ATM call when 9 EMA crosses above 21 EMA on 5 min, exit on opposite signal"


def wilder_textbook_adx(high, low, close, n):
    """Wilder's original formulation (running *sums*, 'New Concepts', 1978), written independently."""
    L = len(close)
    tr, pdm, mdm = [None] * L, [None] * L, [None] * L
    for i in range(1, L):
        up, dn = high[i] - high[i - 1], low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    adx = [math.nan] * L
    s_tr = sum(tr[1:n + 1]); s_p = sum(pdm[1:n + 1]); s_m = sum(mdm[1:n + 1])
    dxs = []
    for i in range(n, L):
        if i > n:
            s_tr = s_tr - s_tr / n + tr[i]
            s_p = s_p - s_p / n + pdm[i]
            s_m = s_m - s_m / n + mdm[i]
        pdi, mdi = 100 * s_p / s_tr, 100 * s_m / s_tr
        dx = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
        dxs.append(dx)
        if len(dxs) == n:
            a = sum(dxs) / n
            adx[i] = a
        elif len(dxs) > n:
            a = (a * (n - 1) + dx) / n
            adx[i] = a
    return adx


def test_adx_matches_wilder_textbook_reference(candles_5m):
    df = candles_5m.iloc[:400]
    got = ind.adx(df, 14)["adx"].to_numpy()
    ref = np.array(wilder_textbook_adx(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), 14))
    assert np.isnan(got[:27]).all() and not np.isnan(got[27])  # first ADX at bar 2n-1
    np.testing.assert_allclose(got[27:], ref[27:], rtol=1e-9, atol=1e-9)


def test_adx_hand_computed_period_2():
    # period 2, 5 bars: hand-worked values
    df = pd.DataFrame({"high": [10, 12, 13, 12, 15], "low": [9, 10, 11, 10, 12], "close": [9.5, 11.5, 12.5, 10.5, 14.5]},
                      index=pd.date_range("2026-11-20 09:15", periods=5, freq="5min"))
    # bar1: +DM 2, -DM 0, TR 2.5 | bar2: +DM 1, -DM 0, TR 2 | bar3: +DM 0, -DM 1, TR 2.5 | bar4: +DM 3, -DM 0, TR 4.5
    # smoothed (avg) at bar2: TR 2.25, +DM 1.5, -DM 0 -> +DI 66.67, -DI 0 -> DX 100
    # bar3: TR 2.375, +DM .75, -DM .5 -> +DI 31.58, -DI 21.05 -> DX 20 ; ADX(bar3) = (100+20)/2 = 60
    # bar4: TR 3.4375, +DM 1.875, -DM .25 -> +DI 54.545, -DI 7.273 -> DX 76.47 ; ADX = (60+76.47)/2 = 68.235
    a = ind.adx(df, 2)
    assert math.isnan(a["adx"].iloc[2])
    assert a["adx"].iloc[3] == pytest.approx(60.0, abs=1e-9)
    assert a["adx"].iloc[4] == pytest.approx((60 + 100 * (54.545454545 - 7.272727273) / (54.545454545 + 7.272727273)) / 2, abs=1e-6)


def test_adx_trend_vs_chop():
    idx = pd.date_range("2026-11-20 09:15", periods=120, freq="5min")
    up = pd.DataFrame({"close": np.arange(120.0) + 100}, index=idx)
    up["open"], up["high"], up["low"] = up["close"] - 0.5, up["close"] + 0.5, up["close"] - 1
    chop_close = 100 + np.where(np.arange(120) % 2 == 0, 1.0, -1.0)
    chop = pd.DataFrame({"close": chop_close, "open": 100.0, "high": chop_close + 1, "low": chop_close - 1}, index=idx)
    a_up, a_chop = ind.adx(up, 14), ind.adx(chop, 14)
    assert a_up["adx"].iloc[-1] > 90 and a_up["plus_di"].iloc[-1] > a_up["minus_di"].iloc[-1]
    assert a_chop["adx"].iloc[-1] < 20


def test_adx_flat_market_is_zero_not_nan():
    idx = pd.date_range("2026-11-20 09:15", periods=60, freq="5min")
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    assert ind.adx(flat, 14)["adx"].iloc[-1] == 0.0


def test_adx_usable_in_dsl():
    s = Strategy.model_validate({**P.parse(EMA).strategy.model_dump(), "indicators": P.parse(EMA).strategy.model_dump()["indicators"] + [{"id": "adx14", "kind": "adx"}],
                                 "entry": {"long": [{"left": "ema9", "op": "crosses_above", "right": "ema21"}, {"left": "adx14", "op": "gt", "right": 25}]}})
    assert s.indicators[-1].period == 14
    res = run_backtest(s, generate_candles("NIFTY", days=20, seed=3, end=date(2026, 11, 20)))
    assert "ADX 14 is above 25" in __import__("app.strategy.explain", fromlist=["explain"]).explain(s)
    assert res.metrics["trades"] >= 0


def test_adx_filter_does_not_gate_opposite_signal_exit_or_get_mirrored():
    base = P.parse("Trade banknifty both ways when 9 EMA crosses 21 EMA on 5 min").strategy.model_dump()
    base["indicators"].append({"id": "adx14", "kind": "adx"})
    base["entry"]["long"].append({"left": "adx14", "op": "gt", "right": 25})
    base["entry"]["short"] = []  # direction=both with only the long side given: short is mirrored
    s = Strategy.model_validate(base)
    assert s.direction == "both" and not s.entry.short
    short = s.short_conditions()
    assert any(c.left == "adx14" and c.op == "gt" and c.right == 25 for c in short)  # not mirrored to "< 75"/"lt"
    assert all(c.left != "adx14" for c in s.opposite_of_long())  # exits don't wait for ADX


# ---------------- regimes ----------------
def _days_frame(day_specs):
    """day_specs: list of (pattern) -> 75 5m bars per day."""
    rows = []
    price = 100.0
    rng = np.random.default_rng(1)
    anchor = price
    for d, kind in enumerate(day_specs):
        day = pd.Timestamp("2026-11-02") + pd.Timedelta(days=d)
        for i in range(75):
            ts = day + pd.Timedelta(hours=9, minutes=15 + 5 * i)
            if kind == "trend":
                step = 0.3
            elif kind == "wild":
                step = 6.0 if (i // 5) % 2 == 0 else -6.0
            else:  # mean-reverting noise around the day's open
                if i == 0:
                    anchor = price
                step = (anchor - price) * 0.6 + float(rng.normal(0, 0.3))
            o, c = price, price + step
            rows.append((ts, o, max(o, c) + 0.05, min(o, c) - 0.05, c, 0))
            price = c
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


def test_classify_days_trending_range_volatile():
    kinds = ["trend", "trend", "chop", "chop", "trend", "chop", "chop", "trend", "chop", "wild"]
    df = _days_frame(kinds)
    reg = classify_days(df)
    assert len(reg) == 10
    assert reg["2026-11-11"] == "volatile"  # the wild day has by far the biggest range / ATR
    adx_close = ind.adx(df, 14)["adx"].groupby(df.index.normalize()).last()
    for (d, label), kind in zip(sorted(reg.items()), kinds):
        if label == "volatile":
            continue
        a = adx_close[pd.Timestamp(d)]
        assert label == ("trending" if a >= 25 else "range"), (d, a)
    assert reg["2026-11-03"] == "trending" and reg["2026-11-06"] == "trending"
    assert reg["2026-11-05"] == "range" and reg["2026-11-08"] == "range"


def test_regime_breakdown_shape_and_attribution():
    df = _days_frame(["trend", "trend", "chop", "wild"])
    regimes = classify_days(df)
    trades = [{"date": "2026-11-03", "net_pnl": 100.0}, {"date": "2026-11-03", "net_pnl": -50.0}, {"date": "2026-11-04", "net_pnl": -10.0},
              {"date": "1999-01-01", "net_pnl": 5.0}]
    rows = regime_breakdown(df, trades, regimes)
    assert [r["regime"] for r in rows] == ["trending", "range", "volatile"]
    by = {r["regime"]: r for r in rows}
    assert by["trending"]["trades"] == 2 and by["trending"]["net_pnl"] == 50.0 and by["trending"]["win_rate"] == 50.0
    assert sum(r["days"] for r in rows) == 4
    assert sum(r["share_of_days"] for r in rows) == pytest.approx(100.0, abs=0.05)
    json.dumps(rows, allow_nan=False)


def test_regimes_empty_inputs_are_json_safe():
    rows = regime_breakdown(pd.DataFrame(columns=["open", "high", "low", "close"]), [])
    assert all(r["days"] == 0 and r["share_of_days"] == 0.0 for r in rows)
    json.dumps(rows, allow_nan=False)


# ---------------- autopsy / what-ifs ----------------
def test_cooldown_blocks_reentry_after_loss():
    s = P.parse(EMA).strategy
    c = resample(generate_candles("NIFTY", days=30, seed=7, end=date(2026, 11, 20)), s.timeframe)
    base = Backtester(s).run(c)
    cool = Backtester(s, cooldown_bars_after_loss=3).run(c)
    # no trade in the cooldown run enters within 3 bars of a losing exit (bars are 5 min)
    losers = [pd.Timestamp(t["exit_time"]) for t in cool.trades if t["net_pnl"] < 0]
    for t in cool.trades:
        et = pd.Timestamp(t["entry_time"])
        assert not any(pd.Timedelta(0) <= et - x <= pd.Timedelta(minutes=15) for x in losers)
    assert losers and len(cool.trades) <= len(base.trades)


def test_entry_mask_blocks_entries():
    s = P.parse(EMA).strategy
    c = resample(generate_candles("NIFTY", days=10, seed=7, end=date(2026, 11, 20)), s.timeframe)
    none = Backtester(s, entry_mask=pd.Series(False, index=c.index)).run(c)
    assert none.trades == []
    allowed = Backtester(s, entry_mask=pd.Series(True, index=c.index)).run(c)
    assert len(allowed.trades) == len(Backtester(s).run(c).trades)


def test_autopsy_numbers_are_consistent():
    s = P.parse(EMA).strategy
    c = resample(generate_candles("NIFTY", days=60, seed=7, end=date(2026, 11, 20)), s.timeframe)
    res = run_backtest(s, c)
    a = build_autopsy(s, c, res.trades)
    assert a["whipsaw_trades"] == res.metrics["whipsaw"]["count"]
    assert a["whipsaw_loss"] == pytest.approx(res.metrics["whipsaw"]["net_loss"], abs=0.01) and a["whipsaw_loss"] <= 0
    cool = Backtester(s, cooldown_bars_after_loss=3).run(c)
    assert a["cooldown_saving"] == pytest.approx(sum(t["net_pnl"] for t in cool.trades) - res.metrics["net_pnl"], abs=0.05)
    hours = hour_buckets(res.trades)
    assert a["worst_hour_pnl"] == pytest.approx(min(hours.values()), abs=0.01)
    assert a["best_hour_pnl"] == pytest.approx(max(hours.values()), abs=0.01)
    assert len(a["worst_hour"]) == 5 and a["worst_hour"].endswith(":00")
    assert 0 <= len(a["suggestions"]) <= 3
    for sgg in a["suggestions"]:
        assert "+₹" in sgg and "on this data" in sgg
    json.dumps(a, allow_nan=False)


def test_suggestions_sorted_biggest_first():
    s = P.parse(EMA).strategy
    c = resample(generate_candles("NIFTY", days=60, seed=7, end=date(2026, 11, 20)), s.timeframe)
    a = build_autopsy(s, c, run_backtest(s, c).trades)
    amounts = [float(x.split("+₹")[1].split(" ")[0].replace(",", "")) for x in a["suggestions"]]
    assert amounts == sorted(amounts, reverse=True)


def test_autopsy_no_trades():
    s = P.parse(EMA).strategy
    c = resample(generate_candles("NIFTY", days=5, seed=7, end=date(2026, 11, 20)), s.timeframe)
    out = analyse(s, c, [])
    assert out["autopsy"]["whipsaw_trades"] == 0 and out["autopsy"]["worst_hour"] is None and out["autopsy"]["suggestions"] == []
    json.dumps(out, allow_nan=False)


def test_backtest_api_returns_regimes_and_autopsy(client):
    from conftest import register

    body = {"text": EMA, "days": 20}
    anon = client.post("/api/backtests/run", json=body).json()
    for key in ("regimes", "autopsy"):
        assert key in anon
    assert [r["regime"] for r in anon["regimes"]] == ["trending", "range", "volatile"]
    assert sum(r["days"] for r in anon["regimes"]) == 20
    assert set(anon["autopsy"]) == {"whipsaw_trades", "whipsaw_loss", "cooldown_saving", "adx_filter_saving", "worst_hour", "worst_hour_pnl",
                                    "best_hour", "best_hour_pnl", "suggestions"}
    h = register(client)
    saved = client.post("/api/backtests/run", json=body, headers=h).json()
    got = client.get(f"/api/backtests/{saved['id']}", headers=h).json()
    assert got["regimes"] == saved["regimes"] and got["autopsy"] == saved["autopsy"]
    assert "regimes" not in got["metrics"]
    listing = client.get("/api/backtests", headers=h).json()["runs"]
    assert listing and "regimes" not in listing[0]["metrics"]
    json.dumps(saved, allow_nan=False)


@pytest.mark.parametrize("tf", ["1h", "1d"])
def test_regimes_on_slow_timeframes(client, tf):
    text = "Buy nifty when 9 EMA crosses above 21 EMA on " + ("1 hour candles" if tf == "1h" else "daily candles")
    r = client.post("/api/backtests/run", json={"text": text, "days": 60})
    assert r.status_code == 200, r.text
    assert sum(x["days"] for x in r.json()["regimes"]) == 60
    json.dumps(r.json(), allow_nan=False)
