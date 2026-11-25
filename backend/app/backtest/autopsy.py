"""Whipsaw autopsy: where a backtest lost money to chop, and which simple fix would have helped.

- Whipsaw trades: losing trades closed by a signal within `whipsaw_bars` bars (same definition as
  metrics.detect_whipsaws).
- cooldown_saving: net P&L of the same backtest with re-entry blocked for 3 bars after a losing
  exit, minus the original net P&L (positive = the cooldown would have helped).
- adx_filter_saving: same, with entries skipped on bars where ADX(14) on the strategy timeframe
  (at the signal bar, so no look-ahead) is below 20. Bars where ADX has not warmed up are allowed.
- worst/best hour: entry-hour buckets (IST) of net P&L.
- suggestions: up to 3 plain-English fixes with a positive impact on this data, biggest first.

Every number is finite and rounded, so the result is always JSON-serialisable.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Optional

import pandas as pd

from .. import indicators as ind
from ..strategy.dsl import Strategy
from .metrics import detect_whipsaws

log = logging.getLogger("tradecognix.autopsy")
COOLDOWN_BARS = 3
ADX_FILTER_PERIOD = 14
ADX_FILTER_LEVEL = 20.0


def _f(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return round(x, 2) if math.isfinite(x) else 0.0


def rupees(x: float) -> str:
    sign = "-" if x < 0 else "+"
    return f"{sign}₹{abs(x):,.0f}"


def _net(trades: list[dict]) -> float:
    return float(sum(t["net_pnl"] for t in trades))


def hour_buckets(trades: list[dict]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for t in trades:
        hh = str(t.get("entry_time", ""))[11:13]
        if hh.isdigit():
            out[f"{hh}:00"] += float(t["net_pnl"])
    return dict(out)


def _rerun(strategy: Strategy, candles: pd.DataFrame, data_source: str, bt_kwargs: dict, **what_if) -> Optional[float]:
    from .engine import Backtester

    try:
        res = Backtester(strategy, **bt_kwargs, **what_if).run(candles, data_source)
    except Exception:  # a what-if must never break the main result
        log.exception("autopsy re-run failed")
        return None
    return _net(res.trades)


def build_autopsy(strategy: Strategy, candles: pd.DataFrame, trades: list[dict], data_source: str = "synthetic",
                  bt_kwargs: dict | None = None, whipsaw_bars: int = 2) -> dict:
    bt_kwargs = dict(bt_kwargs or {})
    base = _net(trades)
    whips = detect_whipsaws(trades, whipsaw_bars)
    whip_loss = sum(trades[k]["net_pnl"] for k in whips)

    cooldown_saving = adx_saving = 0.0
    if trades:
        alt = _rerun(strategy, candles, data_source, bt_kwargs, cooldown_bars_after_loss=COOLDOWN_BARS)
        cooldown_saving = (alt - base) if alt is not None else 0.0
        adx = ind.adx(candles, ADX_FILTER_PERIOD)["adx"]
        mask = ~(adx < ADX_FILTER_LEVEL)  # NaN (warm-up) -> allowed
        alt = _rerun(strategy, candles, data_source, bt_kwargs, entry_mask=mask)
        adx_saving = (alt - base) if alt is not None else 0.0

    hours = hour_buckets(trades)
    worst = min(hours.items(), key=lambda kv: (kv[1], kv[0])) if hours else None
    best = max(hours.items(), key=lambda kv: (kv[1], kv[0])) if hours else None

    ideas: list[tuple[float, str]] = []
    if cooldown_saving > 0.5:
        ideas.append((cooldown_saving, f"Add a {COOLDOWN_BARS}-bar cooldown after a loss: {rupees(cooldown_saving)} on this data"))
    if adx_saving > 0.5:
        ideas.append((adx_saving, f"Skip entries when ADX({ADX_FILTER_PERIOD}) is below {ADX_FILTER_LEVEL:g}: {rupees(adx_saving)} on this data"))
    if worst and worst[1] < -0.5 and len(hours) > 1:
        ideas.append((-worst[1], f"Avoid entries in the {worst[0]} hour: {rupees(-worst[1])} on this data"))
    ideas.sort(key=lambda x: -x[0])

    return {
        "whipsaw_trades": len(whips),
        "whipsaw_loss": _f(whip_loss),
        "cooldown_saving": _f(cooldown_saving),
        "adx_filter_saving": _f(adx_saving),
        "worst_hour": worst[0] if worst else None,
        "worst_hour_pnl": _f(worst[1]) if worst else 0.0,
        "best_hour": best[0] if best else None,
        "best_hour_pnl": _f(best[1]) if best else 0.0,
        "suggestions": [s for _, s in ideas[:3]],
    }


def analyse(strategy: Strategy, candles: pd.DataFrame, trades: list[dict], data_source: str = "synthetic",
            bt_kwargs: dict | None = None, whipsaw_bars: int = 2) -> dict:
    """{"regimes": [...], "autopsy": {...}} for a finished backtest. Never raises."""
    from .regime import regime_breakdown

    out: dict[str, Any] = {}
    try:
        out["regimes"] = regime_breakdown(candles, trades)
    except Exception:
        log.exception("regime breakdown failed")
        out["regimes"] = []
    try:
        out["autopsy"] = build_autopsy(strategy, candles, trades, data_source, bt_kwargs, whipsaw_bars)
    except Exception:
        log.exception("autopsy failed")
        out["autopsy"] = None
    return out
