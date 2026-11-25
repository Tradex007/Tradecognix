"""Round 2 (25 Nov 2026): Strategy Copilot (POST /api/strategies/copilot), rule-based edits."""
import json

import pytest

from app.strategy.copilot import EXAMPLES, apply_rules, copilot_edit
from app.strategy.dsl import Strategy
from app.strategy.parser import RuleBasedParser

P = RuleBasedParser()
EMA = "Buy nifty ATM call when 9 EMA crosses above 21 EMA on 5 min, exit on opposite signal"
RSI = "Buy Reliance when RSI 14 goes below 30 on 15 min, exit when RSI above 70"
BOTH = "Trade banknifty both ways when 9 EMA crosses 21 EMA on 5 min"


def base(text=EMA) -> Strategy:
    return P.parse(text).strategy


def edit(instruction, text=EMA):
    r = apply_rules(base(text), instruction)
    Strategy.model_validate(r.spec)  # always a valid spec
    json.dumps(r.to_dict(), allow_nan=False)
    return r


# ---------------- exits & times ----------------
def test_exit_minutes_earlier():
    r = edit("exit 15 min earlier")
    assert r.understood and r.spec["hard_exit"] == "15:00"
    assert r.changes[0] == "Hard exit moved from 15:15 to 15:00"
    assert r.spec["entry_window"]["end"] == "14:45"  # last entry kept before the exit
    assert "always by 15:00 IST" in r.explanation


def test_exit_minutes_later_and_at_time():
    assert edit("exit 10 minutes later").spec["hard_exit"] == "15:25"
    assert edit("exit at 2:45 pm").spec["hard_exit"] == "14:45"
    assert edit("square off at 3").spec["hard_exit"] == "15:00"  # market-hours "3" = 15:00
    assert edit("square off everything by 14:30").spec["hard_exit"] == "14:30"


@pytest.mark.parametrize("bad", ["exit 600 min earlier", "exit 30 min later", "exit at 25:00", "exit at 9:15"])
def test_exit_out_of_hours_is_not_applied(bad):
    r = edit(bad)
    assert not r.understood and r.changes == [] and r.spec == base().model_dump()


def test_exit_already_there_is_understood_without_changes():
    r = edit("exit at 15:15")
    assert r.understood and r.changes == [] and "No change needed" in r.explanation


def test_entry_window_start_and_end():
    r = edit("start after 9:45")
    assert r.spec["entry_window"]["start"] == "09:45" and r.changes == ["First entry moved from 09:15 to 09:45"]
    r = edit("no entries after 13:30")
    assert r.spec["entry_window"]["end"] == "13:30" and r.changes == ["Last entry moved from 15:00 to 13:30"]
    r = edit("no new trades after 2 pm")
    assert r.spec["entry_window"]["end"] == "14:00"
    assert not edit("no entries after 15:20").understood  # after the 15:15 hard exit


# ---------------- stops, targets, risk ----------------
def test_stop_loss_pct_and_points():
    r = edit("add stop loss 1%")
    assert r.spec["exit"]["stop_loss_pct"] == 1 and r.changes == ["Stop loss set to 1%"]
    r = edit("stop loss 30 points")
    assert r.spec["exit"]["stop_loss_points"] == 30 and r.spec["exit"]["stop_loss_pct"] is None
    assert "30-point stop loss" in r.explanation
    s2 = Strategy.model_validate(edit("add stop loss 1%").spec)
    r = apply_rules(s2, "change stop loss to 25 points")
    assert r.spec["exit"]["stop_loss_pct"] is None and r.spec["exit"]["stop_loss_points"] == 25
    assert r.changes == ["Stop loss changed from 1% to 25 points"]


def test_target_and_combined_edits():
    r = edit("add a 1% stop loss and a 2% target")
    assert r.spec["exit"]["stop_loss_pct"] == 1 and r.spec["exit"]["target_pct"] == 2
    assert r.changes == ["Stop loss set to 1%", "Target set to 2%"]
    assert edit("target 50 points").spec["exit"]["target_points"] == 50


def test_trailing_stop_not_confused_with_stop_loss():
    r = edit("trailing stop 0.5%")
    assert r.spec["exit"]["trailing_sl_pct"] == 0.5 and r.spec["exit"]["stop_loss_pct"] is None


def test_max_trades_and_daily_loss_limit():
    r = edit("max 2 trades a day")
    assert r.spec["risk"]["max_trades_per_day"] == 2 and r.changes == ["Max trades a day changed from 3 to 2"]
    assert edit("only one trade per day").spec["risk"]["max_trades_per_day"] == 1
    r = edit("daily loss limit 5k")
    assert r.spec["risk"]["daily_loss_cap"] == 5000 and r.changes == ["Daily loss limit set to ₹5,000"]
    assert edit("daily loss limit ₹7,500").spec["risk"]["daily_loss_cap"] == 7500


@pytest.mark.parametrize("bad", ["stop loss 0%", "stop loss 1000%", "max 0 trades a day", "max 999 trades a day", "trailing stop 0%"])
def test_invalid_numbers_are_refused(bad):
    r = edit(bad)
    assert not r.understood and r.changes == []


# ---------------- timeframe, indicators ----------------
def test_timeframe_switches():
    r = edit("switch timeframe to 15 min")
    assert r.spec["timeframe"] == "15m" and r.changes == ["Timeframe changed from 5-minute to 15-minute candles"]
    assert edit("switch to 1 hour candles").spec["timeframe"] == "1h"
    r = edit("use daily candles")
    assert r.spec["timeframe"] == "1d" and r.spec["intraday"] is False
    r = edit("switch timeframe to 7 min")
    assert not r.understood and "not supported" in r.explanation


def test_ma_periods_rename_ids_and_conditions():
    r = edit("use 20 and 50 EMA")
    ids = [i["id"] for i in r.spec["indicators"]]
    assert ids == ["ema20", "ema50"]
    assert r.spec["entry"]["long"] == [{"left": "ema20", "op": "crosses_above", "right": "ema50"}]
    assert r.changes == ["Moving averages changed from EMA 9 / EMA 21 to EMA 20 / EMA 50"]
    assert "EMA 20 crosses above EMA 50" in r.explanation
    r = edit("use 50 and 20 sma")  # order and kind
    assert [(i["kind"], i["period"]) for i in r.spec["indicators"]] == [("sma", 20), ("sma", 50)]
    r = edit("use 21 and 9 ema")  # swap-safe renaming (no id collision mid-way)
    assert [i["id"] for i in r.spec["indicators"]] == ["ema9", "ema21"]
    assert not edit("use 9 and 9 ema").understood


def test_rsi_levels():
    r = edit("RSI 25/75", RSI)
    assert r.understood and r.changes == ["RSI levels changed: 30 → 25, 70 → 75"]
    levels = sorted({c["right"] for g in ("long", "short") for c in r.spec["entry"][g] + r.spec["exit"][g] if isinstance(c["right"], float)})
    assert levels == [25.0, 75.0]
    assert not edit("RSI 25/75").understood  # EMA strategy has no RSI
    assert not edit("rsi 50/50", RSI).understood


def test_adx_filter_added_to_both_sides_and_not_mirrored():
    r = edit("add ADX filter > 25", BOTH)
    assert {"id": "adx14", "kind": "adx", "period": 14} .items() <= r.spec["indicators"][-1].items()
    for side in ("long", "short"):
        assert {"left": "adx14", "op": "gt", "right": 25.0} in r.spec["entry"][side]
    s = Strategy.model_validate(r.spec)
    assert all(c.left != "adx14" for c in s.opposite_of_long() + s.opposite_of_short())
    assert r.changes == ["Added ADX filter: only enter when ADX(14) is above 25"]
    r2 = apply_rules(s, "adx filter > 30")
    assert r2.changes == ["ADX filter changed: only enter when ADX(14) is above 30"]
    assert sum(1 for c in r2.spec["entry"]["long"] if c["left"] == "adx14") == 1
    r3 = apply_rules(Strategy.model_validate(r2.spec), "remove adx filter")
    assert all(i["kind"] != "adx" for i in r3.spec["indicators"]) and r3.changes == ["ADX filter removed"]


# ---------------- instrument / contract ----------------
def test_switch_instrument_index_and_stock():
    r = edit("switch to banknifty")
    assert r.spec["underlying"] == "BANKNIFTY" and r.spec["option"]["expiry"] == "monthly"
    assert r.changes[0] == "Instrument changed from NIFTY to BANKNIFTY"
    r = edit("switch to sensex")
    assert r.spec["underlying"] == "SENSEX" and r.spec["exchange"] == "BSE"
    r = edit("switch to reliance")
    assert r.spec["underlying"] == "RELIANCE" and r.spec["instrument_type"] == "equity" and r.spec["option"] is None
    r = edit("switch to stock TATAELXSI")
    assert r.spec["underlying"] == "TATAELXSI"
    r = edit("switch to finnifty", RSI)  # from a stock back to an index -> options
    assert r.spec["underlying"] == "FINNIFTY" and r.spec["instrument_type"] == "option"


def test_switch_contract_and_moneyness():
    assert edit("switch to futures").spec["instrument_type"] == "future"
    r = edit("switch to ITM")
    assert r.spec["option"]["mode"] == "itm_buy" and r.changes == ["Option mode changed from ATM BUY to ITM BUY"]
    r = edit("2 strikes OTM")
    assert r.spec["option"] == {"mode": "otm_buy", "strike_steps": 2, "expiry": "weekly"}
    r = edit("switch to options", RSI)
    assert r.spec["instrument_type"] == "option" and r.spec["option"]["mode"] == "atm_buy"


def test_lots():
    assert edit("trade 2 lots").spec["risk"]["lots"] == 2


# ---------------- not understood ----------------
@pytest.mark.parametrize("junk", ["make me a sandwich", "😀😀😀", "\x00", "what is the weather"])
def test_not_understood_returns_unchanged_spec_and_examples(junk):
    r = edit(junk)
    assert r.understood is False and r.changes == [] and r.spec == base().model_dump()
    assert sum(ex in r.explanation for ex in EXAMPLES) >= 2


def test_llm_fallback_only_when_rules_fail(monkeypatch):
    import app.strategy.copilot as cp

    calls = []

    def fake(s, instruction):
        calls.append(instruction)
        d = s.model_dump()
        d["risk"]["lots"] = 4
        return cp.CopilotResult(spec=Strategy.model_validate(d).model_dump(), changes=["risk.lots changed from 1 to 4"],
                                explanation="x", understood=True, method="llm")

    monkeypatch.setattr(cp, "llm_edit", fake)
    assert copilot_edit(base(), "exit 15 min earlier").spec["hard_exit"] == "15:00" and calls == []
    r = copilot_edit(base(), "quadruple my size please")
    assert r.understood and r.spec["risk"]["lots"] == 4 and calls == ["quadruple my size please"]

    def boom(s, i):
        raise RuntimeError("quota")

    monkeypatch.setattr(cp, "llm_edit", boom)
    assert copilot_edit(base(), "quadruple my size please").understood is False


def test_llm_edit_validates_and_diffs(monkeypatch):
    import app.strategy.copilot as cp
    import app.strategy.llm as llm

    d = base().model_dump()
    d["hard_exit"] = "14:00"
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "call_openai_compatible", lambda text, system_prompt=None, user_prompt=None: json.dumps(d))
    r = cp.llm_edit(base(), "leave at two")
    assert r.method == "llm" and r.spec["hard_exit"] == "14:00" and r.changes == ['hard_exit changed from "15:15" to "14:00"']
    monkeypatch.setattr(llm, "call_openai_compatible", lambda *a, **k: '{"underlying": 5}')
    with pytest.raises(Exception):
        cp.llm_edit(base(), "nonsense")


# ---------------- API ----------------
def test_copilot_endpoint_contract(client):
    spec = base().model_dump()
    r = client.post("/api/strategies/copilot", json={"spec": spec, "instruction": "exit 15 min earlier"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"spec", "changes", "explanation", "understood"}
    assert body["understood"] is True and body["spec"]["hard_exit"] == "15:00" and body["changes"][0] == "Hard exit moved from 15:15 to 15:00"
    r = client.post("/api/strategies/copilot", json={"spec": spec, "instruction": "blah blah"})
    assert r.status_code == 200 and r.json()["understood"] is False and r.json()["spec"] == spec and r.json()["changes"] == []


@pytest.mark.parametrize("instruction", ["", "   ", "x" * 501])
def test_copilot_endpoint_400_on_empty_or_long_instruction(client, instruction):
    r = client.post("/api/strategies/copilot", json={"spec": base().model_dump(), "instruction": instruction})
    assert r.status_code == 400


@pytest.mark.parametrize("spec", [{"x": 1}, {"entry": {"long": []}}, {**base().model_dump(), "hard_exit": "99:99"}])
def test_copilot_endpoint_422_on_invalid_spec(client, spec):
    r = client.post("/api/strategies/copilot", json={"spec": spec, "instruction": "exit at 3"})
    assert r.status_code == 422
    assert r.json()["detail"]["message"] == "Invalid strategy"


def test_copilot_endpoint_is_rate_limited_for_anonymous(client):
    from app.config import get_settings

    spec = base().model_dump()
    limit = int(get_settings().rate_limit_public.split("/")[0])
    codes = [client.post("/api/strategies/copilot", json={"spec": spec, "instruction": "exit at 3"}).status_code for _ in range(limit + 1)]
    assert codes[-1] == 429 and codes[0] == 200
