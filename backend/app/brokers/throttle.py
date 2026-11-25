"""Shared per-broker request throttling: a token bucket per broker plus exponential backoff on
rate-limit errors (HTTP 429, "too many requests", Angel's "exceeding access rate", ...).

Every adapter routes its market-data calls (historical candles, LTP) through
``BrokerAdapter.throttled``, so all sessions and backtests in this process share one budget per
broker instead of each hammering the API (previously only Angel One slept a fixed 0.35 s between
candle chunks). Per process, like the API rate limits; a multi-process deployment needs a shared
store.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Conservative requests/second per broker for data endpoints (published limits are higher for
# some, but candle history endpoints are the strictest: Angel ~3/s, Kite 3/s).
DEFAULT_RATES: dict[str, float] = {
    "angelone": 3.0, "zerodha": 3.0, "upstox": 10.0, "fyers": 8.0, "dhan": 5.0,
}
FALLBACK_RATE = 3.0
MAX_RETRIES = 4
BACKOFF_BASE = 1.0  # seconds; doubles per retry: 1, 2, 4, 8 (+/- 20% jitter)
BACKOFF_CAP = 20.0

RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "rate-limit", "ratelimit", "access rate", "throttl",
                      "request limit", "too many request")

# Indirection so tests can replace the clock and the sleep.
monotonic: Callable[[], float] = time.monotonic
sleep: Callable[[float], None] = time.sleep


class TokenBucket:
    def __init__(self, rate: float, capacity: float | None = None):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate))
        self.tokens = self.capacity
        self.updated = monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_acquire(self) -> float:
        """Take a token if available (returns 0.0), else return the seconds to wait."""
        with self._lock:
            self._refill(monotonic())
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            return (1.0 - self.tokens) / self.rate

    def acquire(self) -> float:
        """Block until a token is available. Returns the total time waited."""
        waited = 0.0
        while True:
            wait = self.try_acquire()
            if wait <= 0:
                return waited
            sleep(wait)
            waited += wait


_buckets: dict[str, TokenBucket] = {}
_lock = threading.Lock()


def bucket_for(broker_key: str) -> TokenBucket:
    with _lock:
        b = _buckets.get(broker_key)
        if b is None:
            b = _buckets[broker_key] = TokenBucket(DEFAULT_RATES.get(broker_key, FALLBACK_RATE))
        return b


def configure(broker_key: str, rate: float, capacity: float | None = None) -> TokenBucket:
    with _lock:
        b = _buckets[broker_key] = TokenBucket(rate, capacity)
        return b


def reset() -> None:
    with _lock:
        _buckets.clear()


def is_rate_limit_error(e: BaseException) -> bool:
    for attr in ("status_code", "status", "code"):
        v = getattr(e, attr, None)
        if v == 429 or v == "429":
            return True
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = str(e).lower()
    return any(m in msg for m in RATE_LIMIT_MARKERS)


def backoff_delay(attempt: int) -> float:
    return min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt)) * random.uniform(0.8, 1.2)


def call(broker_key: str, fn: Callable[..., T], *args: Any, retries: int = MAX_RETRIES, **kwargs: Any) -> T:
    """Run `fn` under the broker's token bucket, retrying rate-limit errors with backoff."""
    bucket = bucket_for(broker_key)
    attempt = 0
    while True:
        bucket.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt >= retries or not is_rate_limit_error(e):
                raise
            sleep(backoff_delay(attempt))
            attempt += 1
