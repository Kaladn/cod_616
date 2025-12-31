"""HotSpare: in-memory time-windowed hot-spare buffer for events.

Implements a fixed-window ring buffer (time-based) with deterministic drop
policy and a 30-second default window. Thread-safe and resilient.
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# Module-level configurable constant
MAX_EVENTS = 50000

class HotSpare:
    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        # store tuples (ts, event)
        self._buf: Deque[Tuple[float, Dict]] = deque()
        self._is_takeover = False
        self.dropped_newest = 0

    def _now(self) -> float:
        return time.time()

    def _prune(self, now: float) -> None:
        # remove older than window_seconds
        cutoff = now - self.window_seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def mirror(self, event: Dict, ts: float | None = None) -> None:
        try:
            now = float(ts) if ts is not None else self._now()
            with self._lock:
                # prune first using provided/now
                self._prune(now)
                # enforce max
                if len(self._buf) >= MAX_EVENTS:
                    # drop newest (the incoming event) deterministically
                    self.dropped_newest += 1
                    return
                # append
                self._buf.append((now, dict(event)))
                # prune again in case window_seconds < 0 or ts far in the future
                self._prune(now)
        except Exception:
            # never crash; ignore mirroring errors
            return

    def takeover(self) -> None:
        try:
            with self._lock:
                self._is_takeover = True
        except Exception:
            return

    def release(self) -> None:
        try:
            with self._lock:
                self._is_takeover = False
        except Exception:
            return

    def drain(self) -> List[Dict]:
        try:
            with self._lock:
                # return chronological order (oldest -> newest) by sorting
                sorted_buf = sorted(self._buf, key=lambda x: x[0])
                out = [event for (_ts, event) in sorted_buf]
                if self._is_takeover:
                    # clear only in takeover
                    self._buf.clear()
                return out
        except Exception:
            return []

    def stats(self) -> Dict:
        try:
            with self._lock:
                if self._buf:
                    # compute chronological oldest/newest
                    ts_list = [t for (t, _e) in self._buf]
                    oldest = min(ts_list)
                    newest = max(ts_list)
                else:
                    oldest = None
                    newest = None
                return {
                    'buffer_count': len(self._buf),
                    'oldest_ts': oldest,
                    'newest_ts': newest,
                    'is_takeover': bool(self._is_takeover),
                    'window_seconds': float(self.window_seconds),
                    'dropped_newest': int(self.dropped_newest),
                }
        except Exception:
            # on error return safe defaults
            return {
                'buffer_count': 0,
                'oldest_ts': None,
                'newest_ts': None,
                'is_takeover': False,
                'window_seconds': float(self.window_seconds),
                'dropped_newest': int(self.dropped_newest),
            }