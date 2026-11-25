"""Market-regime breakdown of a backtest: how the strategy did on trending, range-bound and
volatile session days.

Classification per session day, from the underlying's candles on the strategy timeframe:
1. ``volatile`` when the day's range divided by the daily ATR known *before* that day (Wilder,
   14 sessions; the first day uses its own true range) is at or above the 80th percentile of that
   ratio over the sample;
2. otherwise ``trending`` when ADX(14) on the strategy timeframe at the session's last bar is >= 25;
3. otherwise ``range`` (including days where ADX has not warmed up yet).
Trades are attributed to the day they were entered.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import indicators as ind

REGIMES = ("trending", "range", "volatile")
VOLATILE_PERCENTILE = 80.0
TREND_ADX = 25.0
ADX_PERIOD = 14


def _finite(x: float, nd: int = 2) -> float:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else 0.0


def classify_days(candles: pd.DataFrame, adx_period: int = ADX_PERIOD, trend_adx: float = TREND_ADX,
                  volatile_pct: float = VOLATILE_PERCENTILE) -> dict[str, str]:
    """{"YYYY-MM-DD": "trending" | "range" | "volatile"} for every session day in `candles`."""
    if candles is None or candles.empty:
        return {}
    df = candles[["open", "high", "low", "close"]].astype(float)
    day = df.index.normalize()
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    rng = daily["high"] - daily["low"]
    prev_close = daily["close"].shift(1)
    tr = pd.concat([rng, (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_known = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)  # information available before the day
    atr_known = atr_known.fillna(tr)
    ratio = (rng / atr_known.replace(0, np.nan)).fillna(0.0)
    threshold = float(np.percentile(ratio.to_numpy(), volatile_pct)) if len(ratio) else math.inf
    adx_val = ind.adx(df, adx_period)["adx"]
    adx_close = adx_val.groupby(day).last()
    out: dict[str, str] = {}
    for d in daily.index:
        r = float(ratio.loc[d])
        a = adx_close.get(d, np.nan)
        if len(daily) > 1 and r >= threshold and r > 0:  # a one-day sample has nothing to rank against
            label = "volatile"
        elif a is not None and not pd.isna(a) and float(a) >= trend_adx:
            label = "trending"
        else:
            label = "range"
        out[pd.Timestamp(d).date().isoformat()] = label
    return out


def regime_breakdown(candles: pd.DataFrame, trades: list[dict], day_regimes: dict[str, str] | None = None) -> list[dict]:
    """One row per regime (always all three, in a fixed order) with days, trades, net P&L, win rate."""
    day_regimes = day_regimes if day_regimes is not None else classify_days(candles)
    total_days = len(day_regimes)
    days = defaultdict(int)
    for r in day_regimes.values():
        days[r] += 1
    pnl, n, wins = defaultdict(float), defaultdict(int), defaultdict(int)
    for t in trades:
        d = t.get("date") or str(t.get("entry_time", ""))[:10]
        r = day_regimes.get(d)
        if r is None:
            continue
        n[r] += 1
        pnl[r] += float(t["net_pnl"])
        wins[r] += 1 if t["net_pnl"] > 0 else 0
    return [
        {
            "regime": r,
            "days": int(days[r]),
            "trades": int(n[r]),
            "net_pnl": _finite(pnl[r]),
            "win_rate": _finite(wins[r] / n[r] * 100) if n[r] else 0.0,
            "share_of_days": _finite(days[r] / total_days * 100) if total_days else 0.0,
        }
        for r in REGIMES
    ]
