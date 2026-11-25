"""Strategy Copilot: apply a plain-English edit ("exit 15 min earlier") to an existing Strategy.

Rule-based and deterministic: each handler below recognises one kind of instruction, applies it to
a working copy of the spec and *consumes* the matched words, so one message can carry several
edits ("exit at 3pm and add a 1% stop loss"). The edited spec is re-validated with the DSL; if
nothing was recognised (or the edit would make the strategy invalid) the original spec is returned
unchanged with ``understood=False``. When an LLM is configured (see llm.py) it is tried only for
instructions the rules did not understand, and its output must validate against the DSL too.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .dsl import Strategy
from .explain import TF_WORDS, explain

MAX_INSTRUCTION_CHARS = 500
EXAMPLES = [
    "exit 15 min earlier",
    "add a 1% stop loss and a 2% target",
    "max 2 trades a day",
    "switch timeframe to 15 min",
    "use 20 and 50 EMA",
    "add ADX filter > 25",
]

INDEX_WORDS = [
    (r"\bbank\s?nifty\b|\bnifty\s?bank\b", "BANKNIFTY"),
    (r"\bfin\s?nifty\b|\bnifty\s?fin(?:ancial)?(?:\s?services)?\b", "FINNIFTY"),
    (r"\bmidcp\s?nifty\b|\bmidcap\s?nifty\b|\bnifty\s?midcap(?:\s?select)?\b", "MIDCPNIFTY"),
    (r"\bsensex\b", "SENSEX"),
    (r"\bnifty(?:\s?50)?\b", "NIFTY"),
]


class CopilotError(ValueError):
    """An instruction was recognised but cannot be applied to this strategy."""


@dataclass
class CopilotResult:
    spec: dict
    changes: list[str] = field(default_factory=list)
    explanation: str = ""
    understood: bool = True
    method: str = "rules"

    def to_dict(self) -> dict:
        return {"spec": self.spec, "changes": self.changes, "explanation": self.explanation, "understood": self.understood}


# ----------------------------------------------------------------------------------------- helpers
def _mins(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _hhmm(total: int) -> str:
    return f"{total // 60:02d}:{total % 60:02d}"


def _norm_time(h: int, m: int, ap: Optional[str]) -> Optional[str]:
    ap = (ap or "").replace(".", "").lower()
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if not ap and 1 <= h <= 3:  # market hours: "3:15" means 15:15
        h += 12
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return f"{h:02d}:{m:02d}"


TIME = r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"


def _time_from(m: re.Match, g: int = 1) -> str:
    t = _norm_time(int(m.group(g)), int(m.group(g + 1) or 0), m.group(g + 2))
    if t is None or not ("09:15" <= t <= "15:30"):
        raise CopilotError(f"{m.group(0).strip()!r} is outside market hours (09:15-15:30 IST)")
    return t


def _num(s: str, unit: str | None = None) -> float:
    v = float(s.replace(",", ""))
    unit = (unit or "").lower()
    return v * (1000 if unit in ("k", "thousand") else 100000 if unit in ("l", "lakh", "lac") else 1)


def _fmt(v: Optional[float]) -> str:
    return "none" if v is None else f"{v:g}"


class _Ctx:
    """Working copy of the spec plus the change log and the not-yet-consumed instruction text."""

    def __init__(self, spec: dict, text: str):
        self.spec = spec
        self.text = text
        self.changes: list[str] = []

    def find(self, pattern: str) -> Optional[re.Match]:
        return re.search(pattern, self.text)

    def consume(self, m: re.Match) -> None:
        self.text = self.text[: m.start()] + " ; " + self.text[m.end():]


def _rename_operand(x: Any, old: str, new: str) -> Any:
    if isinstance(x, str):
        base, dot, attr = x.partition(".")
        if base == old:
            return new + dot + attr
    return x


def _rename_indicator(spec: dict, old: str, new: str) -> None:
    for ind in spec["indicators"]:
        if ind["id"] == old:
            ind["id"] = new
    for group in (spec["entry"]["long"], spec["entry"]["short"], spec["exit"]["long"], spec["exit"]["short"]):
        for c in group:
            c["left"] = _rename_operand(c["left"], old, new)
            c["right"] = _rename_operand(c["right"], old, new)


def _materialise_mirrored_side(spec: dict) -> None:
    """For direction=both with one side empty the DSL mirrors the other side at evaluation time.
    Before adding a filter or editing levels, write the mirrored side out explicitly so a filter
    like "ADX > 25" is not mirrored into "ADX < 25" for the other side."""
    if spec["direction"] != "both":
        return
    s = Strategy.model_validate(spec)
    if not spec["entry"]["short"]:
        spec["entry"]["short"] = [c.model_dump() for c in s.short_conditions()]
    elif not spec["entry"]["long"]:
        spec["entry"]["long"] = [c.model_dump() for c in s.long_conditions()]


def _active_entry_sides(spec: dict) -> list[str]:
    d = spec["direction"]
    if d in ("long", "neutral"):
        return ["long"]
    if d == "short":
        return ["short"]
    return ["long", "short"]


# ---------------------------------------------------------------------------------------- handlers
def h_hard_exit(c: _Ctx) -> None:
    m = c.find(r"\b(?:exit|square\s?-?\s?off|close(?:\s+(?:all|everything|positions?))?)\s+(\d{1,3})\s*(?:min(?:ute)?s?|m)\b\s*(earlier|later|early|late|before|after|sooner)")
    if m:
        n = int(m.group(1))
        sign = -1 if m.group(2) in ("earlier", "early", "before", "sooner") else 1
        old = c.spec["hard_exit"]
        new_m = _mins(old) + sign * n
        if not (_mins("09:16") <= new_m <= _mins("15:30")):
            raise CopilotError(f"moving the exit {n} min {'earlier' if sign < 0 else 'later'} would put it outside market hours")
        _set_hard_exit(c, _hhmm(new_m))
        c.consume(m)
        return
    m = c.find(r"\b(?:hard\s+)?(?:exit|square\s?-?\s?off|close(?:\s+(?:all|everything|positions?))?)(?:\s+(?:all|everything|positions?))?\s+(?:at|by|before)\s+" + TIME)
    if m:
        _set_hard_exit(c, _time_from(m))
        c.consume(m)


def _set_hard_exit(c: _Ctx, new: str) -> None:
    old = c.spec["hard_exit"]
    if not c.spec.get("intraday", True):
        c.changes.append("Strategy switched to intraday so the square-off time applies")
        c.spec["intraday"] = True
        if c.spec["timeframe"] == "1d":
            raise CopilotError("a daily-candle strategy is positional and has no intraday exit time; switch the timeframe first")
    if new <= c.spec["entry_window"]["start"]:
        raise CopilotError(f"the exit time {new} must be after the first entry time {c.spec['entry_window']['start']}")
    c.spec["hard_exit"] = new
    if new != old:
        c.changes.append(f"Hard exit moved from {old} to {new}")
    end = c.spec["entry_window"]["end"]
    if end >= new:
        new_end = max(_hhmm(_mins(new) - 15), c.spec["entry_window"]["start"])
        c.spec["entry_window"]["end"] = new_end
        c.changes.append(f"Last entry moved from {end} to {new_end} (15 min before the exit)")


def h_entry_window(c: _Ctx) -> None:
    m = c.find(r"\bno\s+(?:new\s+)?(?:entries|entry|trades?|positions?)\s+(?:after|post|beyond)\s+" + TIME)
    m = m or c.find(r"\b(?:stop|last)\s+(?:new\s+)?(?:entries|entry|trades?|taking trades)\s+(?:at|by|after)\s+" + TIME)
    if m:
        new = _time_from(m)
        old = c.spec["entry_window"]["end"]
        if new >= c.spec["hard_exit"]:
            raise CopilotError(f"last entry {new} must be before the exit time {c.spec['hard_exit']}")
        if new < c.spec["entry_window"]["start"]:
            raise CopilotError(f"last entry {new} is before the first entry time {c.spec['entry_window']['start']}")
        c.spec["entry_window"]["end"] = new
        if new != old:
            c.changes.append(f"Last entry moved from {old} to {new}")
        c.consume(m)
    m = c.find(r"\b(?:start|begin|enter|entries|trade|trading|only trade)\w*\s+(?:only\s+)?(?:after|from|at)\s+" + TIME)
    m = m or c.find(r"\bno\s+(?:entries|entry|trades?)\s+before\s+" + TIME)
    if m:
        new = _time_from(m)
        old = c.spec["entry_window"]["start"]
        if new >= c.spec["hard_exit"]:
            raise CopilotError(f"first entry {new} must be before the exit time {c.spec['hard_exit']}")
        c.spec["entry_window"]["start"] = new
        if c.spec["entry_window"]["end"] < new:
            c.changes.append(f"Last entry moved from {c.spec['entry_window']['end']} to {new}")
            c.spec["entry_window"]["end"] = new
        if new != old:
            c.changes.append(f"First entry moved from {old} to {new}")
        c.consume(m)


def h_trailing(c: _Ctx) -> None:
    m = c.find(r"\btrail(?:ing)?\s*(?:stop(?:\s*-?\s*loss)?|sl)?\s*(?:of|to|at|=)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent|pc\b)")
    m = m or c.find(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s*trail(?:ing)?\s*(?:stop(?:\s*-?\s*loss)?|sl)?")
    if m:
        v = float(m.group(1))
        _check_pct(v, 100, "trailing stop")
        old = c.spec["exit"].get("trailing_sl_pct")
        c.spec["exit"]["trailing_sl_pct"] = v
        c.changes.append(f"Trailing stop {'set to' if old is None else 'changed from ' + _fmt(old) + '% to'} {v:g}%")
        c.consume(m)
        return
    m = c.find(r"\b(?:remove|drop|no|disable)\s+(?:the\s+)?trail(?:ing)?\s*(?:stop(?:\s*-?\s*loss)?|sl)?")
    if m:
        if c.spec["exit"].get("trailing_sl_pct") is not None:
            c.changes.append("Trailing stop removed")
        c.spec["exit"]["trailing_sl_pct"] = None
        c.consume(m)


def _check_pct(v: float, hi: float, what: str) -> None:
    if not (0 < v <= hi):
        raise CopilotError(f"a {what} must be between 0 and {hi:g}%")


def _stop_or_target(c: _Ctx, words: str, key: str, label: str, hi_pct: float) -> None:
    unit = r"(%|percent|pc\b|points?\b|pts?\b)"
    m = c.find(r"\b(?:add|set|change|use|put|move|make|with)?\s*(?:a|the|an)?\s*(?:" + words + r")\s*(?:of|to|at|=|:)?\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*" + unit)
    m2 = None if m else c.find(r"\b(\d+(?:\.\d+)?)\s*" + unit + r"\s*(?:" + words + r")\b")
    mm = m or m2
    if mm:
        v = float(mm.group(1))
        is_pct = mm.group(2).startswith(("%", "percent", "pc"))
        ex = c.spec["exit"]
        old_pct, old_pts = ex.get(f"{key}_pct"), ex.get(f"{key}_points")
        old = f"{old_pct:g}%" if old_pct else (f"{old_pts:g} points" if old_pts else None)
        if is_pct:
            _check_pct(v, hi_pct, label)
            ex[f"{key}_pct"], ex[f"{key}_points"] = v, None
            new = f"{v:g}%"
        else:
            if not (0 < v <= 100000):
                raise CopilotError(f"a {label} in points must be positive")
            ex[f"{key}_pct"], ex[f"{key}_points"] = None, v
            new = f"{v:g} points"
        c.changes.append(f"{label.capitalize()} {'set to' if old is None else 'changed from ' + old + ' to'} {new}")
        c.consume(mm)
        return
    m = c.find(r"\b(?:remove|drop|no|disable)\s+(?:the\s+)?(?:" + words + r")\b")
    if m:
        ex = c.spec["exit"]
        if ex.get(f"{key}_pct") or ex.get(f"{key}_points"):
            c.changes.append(f"{label.capitalize()} removed")
        ex[f"{key}_pct"], ex[f"{key}_points"] = None, None
        c.consume(m)


def h_stop_loss(c: _Ctx) -> None:
    _stop_or_target(c, r"stop\s*-?\s*loss|stoploss|\bsl\b|stop", "stop_loss", "stop loss", 100)


def h_target(c: _Ctx) -> None:
    _stop_or_target(c, r"target|take\s*-?\s*profit|profit\s+target|\btp\b", "target", "target", 1000)


def h_daily_loss(c: _Ctx) -> None:
    m = c.find(r"\b(?:daily|day|per\s+day)\s*(?:max(?:imum)?\s*)?loss\s*(?:limit|cap)?\s*(?:of|to|at|=|:)?\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakh|lac|l)?\b")
    m = m or c.find(r"\bmax(?:imum)?\s+loss\s*(?:per|a)\s+day\s*(?:of|to|at|=|:)?\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakh|lac|l)?\b")
    m = m or c.find(r"\b(?:stop|quit)\s+(?:trading\s+)?(?:for\s+the\s+day\s+)?after\s+(?:a\s+)?(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakh|lac|l)?\s*loss")
    if m:
        v = _num(m.group(1), m.group(2))
        if v < 1:
            raise CopilotError("the daily loss limit must be at least ₹1")
        old = c.spec["risk"].get("daily_loss_cap")
        c.spec["risk"]["daily_loss_cap"] = v
        c.changes.append(f"Daily loss limit {'set to' if old is None else f'changed from ₹{old:,.0f} to'} ₹{v:,.0f}")
        c.consume(m)
        return
    m = c.find(r"\b(?:remove|drop|no|disable)\s+(?:the\s+)?daily\s+loss\s*(?:limit|cap)?")
    if m:
        if c.spec["risk"].get("daily_loss_cap"):
            c.changes.append("Daily loss limit removed")
        c.spec["risk"]["daily_loss_cap"] = None
        c.consume(m)


WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def h_max_trades(c: _Ctx) -> None:
    n = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)"
    m = c.find(r"\b(?:max(?:imum)?|at\s+most|only|upto|up\s+to)\s*(?:of\s*)?" + n + r"\s*(?:trades?|entries|positions?)(?:\s*(?:a|per|each)\s*day)?")
    m = m or c.find(r"\b" + n + r"\s*(?:trades?|entries)\s*(?:a|per|each)\s*day")
    m = m or c.find(r"\b(?:max(?:imum)?\s+)?trades?\s*(?:per|a)\s*day\s*(?:to|=|:)?\s*" + n)
    if m:
        raw = m.group(1)
        v = WORD_NUMS.get(raw) or int(raw)
        if not (1 <= v <= 100):
            raise CopilotError("max trades a day must be between 1 and 100")
        old = c.spec["risk"]["max_trades_per_day"]
        c.spec["risk"]["max_trades_per_day"] = v
        c.changes.append(f"Max trades a day changed from {old} to {v}")
        c.consume(m)


def h_lots(c: _Ctx) -> None:
    m = c.find(r"\b(?:trade|use|with|change\s+to|set\s+to|make\s+it)?\s*(\d{1,3})\s*(lots?|shares?|qty|quantity)\b")
    if m:
        v = int(m.group(1))
        if not (1 <= v <= 500):
            raise CopilotError("size must be between 1 and 500")
        old = c.spec["risk"]["lots"]
        c.spec["risk"]["lots"] = v
        unit = "shares" if c.spec["instrument_type"] == "equity" else "lots"
        c.changes.append(f"Position size changed from {old} to {v} {unit}")
        c.consume(m)


TF_MAP = {"1": "1m", "3": "3m", "5": "5m", "10": "10m", "15": "15m", "30": "30m", "60": "1h"}


def h_timeframe(c: _Ctx) -> None:
    tf = None
    m = c.find(r"\b(?:switch|change|move|use|go|set)?\s*(?:the\s+)?(?:timeframe|time\s*frame|tf|candles?|charts?)?\s*(?:to|on)?\s*(\d{1,3})\s*-?\s*(?:min(?:ute)?s?|m)\b\s*(?:candles?|charts?|bars?|timeframe|tf)?")
    if m and (re.search(r"timeframe|time\s*frame|\btf\b|candle|chart|bar|switch|change|\bon\b|\buse\b", m.group(0)) ):
        n = m.group(1)
        if n not in TF_MAP:
            raise CopilotError(f"{n}-minute candles are not supported (use 1, 3, 5, 10, 15, 30 min, 1h or daily)")
        tf = TF_MAP[n]
    else:
        m = c.find(r"\b(?:switch|change|move|use|go|set)?\s*(?:the\s+)?(?:timeframe|time\s*frame|tf|candles?|charts?)?\s*(?:to|on)?\s*(?:(?:1|one)\s*-?\s*(?:hour|hr|h)\b|hourly)\s*(?:candles?|charts?|bars?|timeframe|tf)?")
        if m:
            tf = "1h"
        else:
            m = c.find(r"\b(?:switch|change|move|use|go|set)?\s*(?:the\s+)?(?:timeframe|time\s*frame|tf|candles?|charts?)?\s*(?:to|on)?\s*(?:daily|1\s*d(?:ay)?|day)\s*(?:candles?|charts?|bars?|timeframe|tf)\b")
            m = m or c.find(r"\b(?:switch|change|move)\s+(?:the\s+)?(?:timeframe|time\s*frame|tf|candles?|charts?)\s+to\s+(?:daily|1\s*d(?:ay)?)\b")
            if m:
                tf = "1d"
    if tf:
        old = c.spec["timeframe"]
        c.spec["timeframe"] = tf
        if tf != old:
            c.changes.append(f"Timeframe changed from {TF_WORDS[old]} to {TF_WORDS[tf]} candles")
        if tf == "1d" and c.spec.get("intraday", True):
            c.changes.append("Daily candles make the strategy positional (no intraday square-off)")
        c.consume(m)


def h_ma_periods(c: _Ctx) -> None:
    m = c.find(r"\b(?:use|change\s+to|switch\s+to|set|make\s+it|try)?\s*(\d{1,3})\s*(?:and|&|/|,|-)\s*(\d{1,3})\s*-?\s*(?:period\s*)?(ema|sma|dma|moving averages?|mas?)\b")
    m = m or c.find(r"\b(?:use|change\s+to|switch\s+to|set|try)?\s*(ema|sma|dma)\s*(\d{1,3})\s*(?:and|&|/|,|-)\s*(?:(?:ema|sma|dma)\s*)?(\d{1,3})\b")
    if m:
        g = m.groups()
        if g[0].isdigit():
            p1, p2, kind = int(g[0]), int(g[1]), g[2]
        else:
            kind, p1, p2 = g[0], int(g[1]), int(g[2])
        kind = {"dma": "sma"}.get(kind, kind if kind in ("ema", "sma") else None)
        _set_ma_pair(c, sorted([p1, p2]), kind)
        c.consume(m)
        return
    m = c.find(r"\b(?:use|change\s+to|switch\s+to|set|try)?\s*(?:an?\s+)?(\d{1,3})\s*-?\s*(?:period\s*)?(ema|sma|dma)\b")
    m = m or c.find(r"\b(?:use|change\s+to|switch\s+to|set|try)?\s*(?:an?\s+)?(ema|sma|dma)\s*(?:period\s*)?(?:to\s*)?(\d{1,3})\b")
    if m:
        g = m.groups()
        period, kind = (int(g[0]), g[1]) if g[0].isdigit() else (int(g[1]), g[0])
        kind = "sma" if kind == "dma" else kind
        mas = [i for i in c.spec["indicators"] if i["kind"] in ("ema", "sma")]
        if len(mas) != 1:
            raise CopilotError("this strategy has " + ("no moving average" if not mas else "two moving averages; give both periods, e.g. 'use 20 and 50 EMA'"))
        _set_ma(c, mas[0], period, kind)
        c.consume(m)


def _set_ma(c: _Ctx, ind: dict, period: int, kind: Optional[str]) -> None:
    if not (1 <= period <= 500):
        raise CopilotError("moving-average periods must be between 1 and 500")
    old_desc = f"{ind['kind'].upper()} {ind['period']}"
    kind = kind or ind["kind"]
    ind["kind"], ind["period"] = kind, period
    taken = {i["id"] for i in c.spec["indicators"] if i is not ind}
    new_id = f"{kind}{period}"
    if new_id in taken:
        raise CopilotError(f"{kind.upper()} {period} is already used in this strategy")
    if new_id != ind["id"]:
        _rename_indicator(c.spec, ind["id"], new_id)
    new_desc = f"{kind.upper()} {period}"
    if new_desc != old_desc:
        c.changes.append(f"{old_desc} changed to {new_desc}")


def _set_ma_pair(c: _Ctx, periods: list[int], kind: Optional[str]) -> None:
    if periods[0] == periods[1]:
        raise CopilotError("the two moving averages need different periods")
    mas = sorted([i for i in c.spec["indicators"] if i["kind"] in ("ema", "sma")], key=lambda i: i["period"])
    if len(mas) != 2:
        raise CopilotError("this strategy does not use a two-moving-average crossover" if len(mas) < 2 else "this strategy has more than two moving averages")
    # rename via temporary ids so "9/21 -> 21/50" never collides mid-way
    for k, ind in enumerate(mas):
        _rename_indicator(c.spec, ind["id"], f"tmpma{k}")
    olds = [f"{i['kind'].upper()} {i['period']}" for i in mas]
    for k, ind in enumerate(mas):
        ind["kind"] = kind or ind["kind"]
        ind["period"] = periods[k]
        if not (1 <= periods[k] <= 500):
            raise CopilotError("moving-average periods must be between 1 and 500")
    for k, ind in enumerate(mas):
        new_id = f"{ind['kind']}{ind['period']}"
        if any(i["id"] == new_id for i in c.spec["indicators"] if i is not ind):
            raise CopilotError(f"{new_id.upper()} is already used in this strategy")
        _rename_indicator(c.spec, f"tmpma{k}", new_id)
    news = [f"{i['kind'].upper()} {i['period']}" for i in mas]
    if olds != news:
        c.changes.append(f"Moving averages changed from {olds[0]} / {olds[1]} to {news[0]} / {news[1]}")
    c.spec["name"] = re.sub(r"(EMA|SMA) \d+ / (EMA|SMA) \d+", f"{news[0]} / {news[1]}", c.spec.get("name", ""))


def h_rsi_levels(c: _Ctx) -> None:
    m = c.find(r"\brsi\s*(?:levels?)?\s*(?:to|at|of|=|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:/|-|and|&|,)\s*(\d{1,2}(?:\.\d+)?)\b")
    m = m or c.find(r"\b(?:oversold|os)\s*(?:at|to|=)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:and|,|&)?\s*(?:overbought|ob)\s*(?:at|to|=)?\s*(\d{1,2}(?:\.\d+)?)")
    if not m:
        return
    lo, hi = sorted([float(m.group(1)), float(m.group(2))])
    if not (0 < lo < 50 < hi < 100):
        raise CopilotError("RSI levels must be a low level below 50 and a high level above 50, e.g. 'RSI 30/70'")
    rsi_ids = {i["id"] for i in c.spec["indicators"] if i["kind"] == "rsi"}
    if not rsi_ids:
        raise CopilotError("this strategy does not use RSI")
    _materialise_mirrored_side(c.spec)
    changed = set()
    for group in (c.spec["entry"]["long"], c.spec["entry"]["short"], c.spec["exit"]["long"], c.spec["exit"]["short"]):
        for cond in group:
            if str(cond["left"]).split(".")[0] in rsi_ids and isinstance(cond["right"], (int, float)):
                old = float(cond["right"])
                new = lo if old < 50 else hi if old > 50 else old
                if new != old:
                    changed.add((old, new))
                cond["right"] = new
    if not changed:
        raise CopilotError("this strategy has no RSI level to change")
    c.changes.append("RSI levels changed: " + ", ".join(f"{a:g} → {b:g}" for a, b in sorted(changed)))
    c.consume(m)


def h_adx_filter(c: _Ctx) -> None:
    m = c.find(r"\b(?:add\s+(?:an?\s+)?)?adx\s*(?:\(\s*(\d{1,3})\s*\))?\s*(?:filter)?\s*(?:>|above|over|greater\s+than|>=|of|at)\s*(\d{1,2}(?:\.\d+)?)")
    m = m or c.find(r"\bonly\s+(?:trade|enter)\s+when\s+adx\s*(?:is\s+)?(?:>|above|over|greater\s+than)\s*(\d{1,3})?\s*(\d{1,2}(?:\.\d+)?)")
    if m:
        period = int(m.group(1)) if m.group(1) else 14
        level = float(m.group(2))
        if not (0 < level < 100):
            raise CopilotError("ADX level must be between 0 and 100")
        if not (2 <= period <= 100):
            raise CopilotError("ADX period must be between 2 and 100")
        ind_id = f"adx{period}"
        if not any(i["id"] == ind_id for i in c.spec["indicators"]):
            c.spec["indicators"].append({"id": ind_id, "kind": "adx", "period": period, "multiplier": None, "source": "close", "start": None, "end": None})
        if c.spec["entry"]["logic"] == "any" and sum(len(c.spec["entry"][s]) for s in ("long", "short")) > 1:
            raise CopilotError("entries use 'any' logic, so a filter cannot be added; make all entry conditions required first")
        _materialise_mirrored_side(c.spec)
        replaced = False
        for side in _active_entry_sides(c.spec):
            conds = c.spec["entry"][side]
            existing = [x for x in conds if x["left"] == ind_id and x["op"] in ("gt", "gte")]
            if existing:
                existing[0]["right"], replaced = level, True
            else:
                conds.append({"left": ind_id, "op": "gt", "right": level})
        c.changes.append(f"{'ADX filter changed' if replaced else 'Added ADX filter'}: only enter when ADX({period}) is above {level:g}")
        c.consume(m)
        return
    m = c.find(r"\b(?:remove|drop|no|disable)\s+(?:the\s+)?adx(?:\s+filter)?")
    if m:
        ids = {i["id"] for i in c.spec["indicators"] if i["kind"] == "adx"}
        if ids:
            for side in ("long", "short"):
                c.spec["entry"][side] = [x for x in c.spec["entry"][side] if str(x["left"]).split(".")[0] not in ids]
            for grp in ("long", "short"):
                if any(str(x["left"]).split(".")[0] in ids or str(x["right"]).split(".")[0] in ids for x in c.spec["exit"][grp]):
                    raise CopilotError("ADX is also used in an exit rule; edit that rule first")
            c.spec["indicators"] = [i for i in c.spec["indicators"] if i["id"] not in ids]
            c.changes.append("ADX filter removed")
        c.consume(m)


def h_instrument(c: _Ctx) -> None:
    from .parser import EQUITY_ALIASES

    target = None
    m = None
    for pat, sym in INDEX_WORDS:
        m = c.find(r"\b(?:switch|change|move|trade|use|go)?\s*(?:the\s+)?(?:instrument|underlying|index|symbol)?\s*(?:to|on)?\s*(?:" + pat + r")")
        if m:
            target, is_index = sym, True
            break
    if not target:
        for alias, sym in sorted(EQUITY_ALIASES.items(), key=lambda kv: -len(kv[0])):
            m = c.find(r"\b(?:switch|change|move|trade|use|go)?\s*(?:the\s+)?(?:instrument|underlying|stock|symbol)?\s*(?:to|on)?\s*" + re.escape(alias) + r"\b")
            if m:
                target, is_index = sym, False
                break
    if not target:
        m = c.find(r"\b(?:switch|change|move)\s+(?:the\s+)?(?:instrument|underlying|stock|symbol)?\s*to\s+(?:the\s+)?stock\s+([a-z][a-z0-9&-]{1,19})\b")
        m = m or c.find(r"\b(?:switch|change|move)\s+(?:the\s+)?(?:stock|symbol)\s+to\s+([a-z][a-z0-9&-]{1,19})\b")
        if m:
            target, is_index = re.sub(r"[^A-Z0-9&]", "", m.group(1).upper()), False
    if not target:
        return
    old = c.spec["underlying"]
    c.spec["underlying"] = target
    if target != old:
        c.changes.append(f"Instrument changed from {old} to {target}")
        c.spec["name"] = c.spec.get("name", "").replace(old, target, 1)
    c.spec["exchange"] = "BSE" if target == "SENSEX" else "NSE"
    was = c.spec["instrument_type"]
    if not is_index and was in ("option", "index") and not re.search(r"option|future|\bfut\b", c.text):
        c.spec["instrument_type"], c.spec["option"] = "equity", None
        c.changes.append(f"Contract type changed from {_TYPE_WORDS[was]} to shares (cash equity)")
    elif is_index and was == "equity" and not re.search(r"option|future|\bfut\b", c.text):
        c.spec["instrument_type"] = "option"
        c.spec["option"] = c.spec.get("option") or {"mode": "atm_buy", "strike_steps": 1, "expiry": "weekly"}
        c.changes.append("Contract type changed from shares to ATM options (buy)")
    if c.spec.get("option") and target in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY") and c.spec["option"].get("expiry") == "weekly":
        c.spec["option"]["expiry"] = "monthly"
        c.changes.append(f"Option expiry set to monthly ({target} has no weekly expiry)")
    c.consume(m)


_TYPE_WORDS = {"option": "options", "future": "futures", "equity": "shares", "index": "futures"}


def h_contract(c: _Ctx) -> None:
    m = c.find(r"\b(?:switch|change|move|trade|use|go)?\s*(?:to|with|in)?\s*(?:the\s+)?(futures?|fut|options?|shares|cash|equity)\b")
    if m and not re.search(r"\b(?:atm|itm|otm)\b", m.group(0)):
        word = m.group(1)
        new = "future" if word.startswith("fut") else "option" if word.startswith("option") else "equity"
        was = c.spec["instrument_type"]
        if new == "option" and was != "option":
            c.spec["option"] = c.spec.get("option") or {"mode": "atm_buy", "strike_steps": 1, "expiry": "monthly" if c.spec["underlying"] in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY") else "weekly"}
        if new != "option":
            c.spec["option"] = None
        c.spec["instrument_type"] = new
        if new != (was if was != "index" else "future"):
            c.changes.append(f"Contract type changed from {_TYPE_WORDS[was]} to {_TYPE_WORDS[new]}" + (" (buy ATM)" if new == "option" else ""))
        c.consume(m)


def h_moneyness(c: _Ctx) -> None:
    m = c.find(r"\b(?:(\d{1,2})\s*(?:strikes?|steps?)\s*)?(atm|itm|otm|at[- ]the[- ]money|in[- ]the[- ]money|out[- ]of[- ]the[- ]money)\b(?:\s*(?:by\s*)?(\d{1,2})\s*(?:strikes?|steps?))?")
    if not m:
        return
    word = m.group(2).replace(" ", "-")
    mny = {"at-the-money": "atm", "in-the-money": "itm", "out-of-the-money": "otm"}.get(word, word)
    steps = m.group(1) or m.group(3)
    if c.spec["instrument_type"] != "option":
        c.spec["instrument_type"] = "option"
        c.spec["option"] = {"mode": "atm_buy", "strike_steps": 1, "expiry": "monthly" if c.spec["underlying"] in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY") else "weekly"}
        c.changes.append("Contract type changed to options")
    o = c.spec["option"]
    mode = o["mode"]
    side = "sell" if mode.endswith("_sell") else "buy"
    if mode.startswith(("straddle", "strangle")):
        if mny == "atm":
            new_mode = f"straddle_{side}"
        else:
            new_mode = f"strangle_{side}"
    else:
        new_mode = f"{mny}_{side}"
    if steps:
        n = int(steps)
        if not (0 <= n <= 20):
            raise CopilotError("strike steps must be between 0 and 20")
        if o["strike_steps"] != n:
            c.changes.append(f"Strike distance changed from {o['strike_steps']} to {n} step(s)")
        o["strike_steps"] = n
    if new_mode != mode:
        c.changes.append(f"Option mode changed from {mode.replace('_', ' ').upper()} to {new_mode.replace('_', ' ').upper()}")
    o["mode"] = new_mode
    c.consume(m)


def h_positional(c: _Ctx) -> None:
    m = c.find(r"\b(?:make\s+it|switch\s+to|go)?\s*(positional|overnight|carry\s+forward|swing)\b")
    if m and c.spec.get("intraday", True):
        c.spec["intraday"] = False
        c.changes.append("Strategy made positional (no intraday square-off)")
        c.consume(m)
        return
    m = c.find(r"\b(?:make\s+it|switch\s+to|go)\s*intraday\b")
    if m:
        if c.spec["timeframe"] == "1d":
            raise CopilotError("daily-candle strategies are positional; switch the timeframe first")
        if not c.spec.get("intraday", True):
            c.spec["intraday"] = True
            c.changes.append(f"Strategy made intraday (square off by {c.spec['hard_exit']})")
        c.consume(m)


# Order matters: more specific phrases first so they consume their words before looser ones.
HANDLERS: list[Callable[[_Ctx], None]] = [
    h_trailing, h_daily_loss, h_stop_loss, h_target, h_hard_exit, h_entry_window, h_max_trades,
    h_adx_filter, h_rsi_levels, h_ma_periods, h_timeframe, h_moneyness, h_instrument, h_contract, h_lots, h_positional,
]


def _normalise(text: str) -> str:
    t = " " + text.lower().replace("₹", " rs ").replace("–", "-").replace("—", "-") + " "
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    return re.sub(r"\s+", " ", t)


def _not_understood(s: Strategy, reason: str) -> CopilotResult:
    ex = "; ".join(f'"{e}"' for e in EXAMPLES[:3])
    return CopilotResult(spec=s.model_dump(), changes=[], understood=False,
                         explanation=f"{reason} Try an instruction like {ex}. Current strategy: {explain(s)}")


def apply_rules(s: Strategy, instruction: str) -> CopilotResult:
    """Apply a rule-based edit. Never raises for user input; returns understood=False instead."""
    ctx = _Ctx(copy.deepcopy(s.model_dump()), _normalise(instruction))
    try:
        for h in HANDLERS:
            h(ctx)
    except CopilotError as e:
        return _not_understood(s, f"Could not apply that change: {e}.")
    if not ctx.changes:
        unchanged = ctx.text != _normalise(instruction)  # recognised, but already set that way
        if unchanged:
            return CopilotResult(spec=s.model_dump(), changes=[], understood=True, explanation="No change needed: the strategy already does that. " + explain(s))
        return _not_understood(s, f"I could not understand {instruction.strip()[:80]!r}.")
    try:
        new = Strategy.model_validate(ctx.spec)
    except ValidationError as e:
        return _not_understood(s, f"Could not apply that change: {e.errors(include_url=False)[0]['msg']}.")
    return CopilotResult(spec=new.model_dump(), changes=_dedupe(ctx.changes), explanation=explain(new), understood=True)


def _dedupe(xs: list[str]) -> list[str]:
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


# --------------------------------------------------------------------------------- LLM fallback
COPILOT_PROMPT = """You edit an Indian trader's algorithmic trading strategy. You get the current strategy as JSON
(matching the given JSON Schema) and one plain-English edit instruction. Return ONLY the full updated
strategy JSON object (no prose, no markdown). Change only what the instruction asks; keep everything else
identical. Times are 24h "HH:MM" IST. If the instruction is not a strategy edit, return the strategy unchanged."""


def _diff(old: dict, new: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a == b or k == "description":
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f"{prefix}{k}.")
        else:
            show = lambda v: json.dumps(v, ensure_ascii=False)[:80]  # noqa: E731
            out.append(f"{prefix}{k} changed from {show(a)} to {show(b)}")
    return out


def llm_edit(s: Strategy, instruction: str) -> Optional[CopilotResult]:
    from ..config import get_settings
    from .dsl import strategy_json_schema
    from .llm import _extract_json, call_gemini, call_openai_compatible, llm_available

    if not llm_available():
        return None
    user = (f"JSON Schema:\n{json.dumps(strategy_json_schema())}\n\nCurrent strategy JSON:\n{json.dumps(s.model_dump())}"
            f"\n\nInstruction:\n{instruction}")
    call = call_gemini if get_settings().gemini_api_key else call_openai_compatible
    raw = call(instruction, system_prompt=COPILOT_PROMPT, user_prompt=user)
    data = _extract_json(raw)
    new = Strategy.model_validate(data)
    changes = _diff(s.model_dump(), new.model_dump())
    if not changes:
        return None
    return CopilotResult(spec=new.model_dump(), changes=changes[:10], explanation=explain(new), understood=True, method="llm")


def copilot_edit(s: Strategy, instruction: str, use_llm: bool = True) -> CopilotResult:
    res = apply_rules(s, instruction)
    if res.understood or not use_llm:
        return res
    try:
        llm = llm_edit(s, instruction)
    except Exception:  # network, quota, bad JSON, invalid spec: keep the rule-based answer
        llm = None
    return llm or res
