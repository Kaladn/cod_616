from __future__ import annotations

import time
from typing import Callable, Iterable, List, Tuple, Optional

from forge_memory.core.record import ForgeRecord
from forge_memory.core.binary_log import BinaryLog


Predicate = Callable[[ForgeRecord], bool]
Extractor = Callable[[ForgeRecord], any]


class QueryLayer:
    """
    High-level memory query engine for ForgeRecords.

    Streaming, predicate-based, zero-copy where possible.
    """

    def __init__(self, binary_log: BinaryLog):
        self.binary_log = binary_log

    # -------------------------------------------------------------
    # Core iterators
    # -------------------------------------------------------------

    def scan(self) -> Iterable[ForgeRecord]:
        """Stream all ForgeRecords in record order."""
        return self.binary_log.read_all()

    def where(self, predicate: Predicate) -> Iterable[ForgeRecord]:
        """Filter with arbitrary predicate."""
        for r in self.scan():
            if predicate(r):
                yield r

    # -------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------

    def where_flag(self, flag: str) -> Iterable[ForgeRecord]:
        """Filter windows where operator_flags contains flag."""
        return self.where(
            lambda r: flag in r.context.get("flags", {}).get("operator_flags", [])
        )

    def time_range(self, start_ts: float, end_ts: float) -> Iterable[ForgeRecord]:
        """Filter windows by timestamp inclusive."""
        return self.where(lambda r: start_ts <= r.timestamp <= end_ts)

    def eomm_above(self, threshold: float) -> Iterable[ForgeRecord]:
        return self.where(
            lambda r: r.error_metrics.get("eomm_score", 0) > threshold
        )

    def eomm_below(self, threshold: float) -> Iterable[ForgeRecord]:
        return self.where(
            lambda r: r.error_metrics.get("eomm_score", 0) <= threshold
        )

    def success_only(self) -> Iterable[ForgeRecord]:
        """Filter to successful windows (EOMM <= threshold)."""
        return self.where(lambda r: r.success)

    def failures_only(self) -> Iterable[ForgeRecord]:
        """Filter to failed windows (EOMM > threshold)."""
        return self.where(lambda r: not r.success)

    def by_session(self, session_id: str) -> Iterable[ForgeRecord]:
        """Filter by session ID."""
        return self.where(
            lambda r: r.context.get("session", {}).get("session_id") == session_id
        )

    def by_pulse(self, pulse_id: int) -> Iterable[ForgeRecord]:
        """Filter by pulse ID."""
        return self.where(lambda r: r.pulse_id == pulse_id)

    # -------------------------------------------------------------
    # Extractors
    # -------------------------------------------------------------

    def extract(self, extractor: Extractor, source: Optional[Iterable[ForgeRecord]] = None) -> Iterable:
        if source is None:
            source = self.scan()
        for r in source:
            yield extractor(r)

    # -------------------------------------------------------------
    # SQL-ish combinators
    # -------------------------------------------------------------

    def take(self, n: int, iterable: Optional[Iterable[ForgeRecord]] = None):
        if iterable is None:
            iterable = self.scan()
        count = 0
        for r in iterable:
            if count >= n:
                break
            yield r
            count += 1

    def count(self, iterable: Optional[Iterable[ForgeRecord]] = None) -> int:
        """Count records in iterable."""
        if iterable is None:
            return len(self.binary_log)
        return sum(1 for _ in iterable)

    # -------------------------------------------------------------
    # Convenience queries
    # -------------------------------------------------------------

    def windows_with_flag(self, flag: str, limit: Optional[int] = None):
        it = self.where_flag(flag)
        if limit:
            return list(self.take(limit, it))
        return list(it)

    def high_confidence_anomalies(self, threshold: float = 0.75, limit: Optional[int] = None):
        it = self.eomm_above(threshold)
        if limit:
            return list(self.take(limit, it))
        return list(it)

    def eomm_time_series(self) -> List[Tuple[float, float]]:
        """Return (timestamp, eomm_score) pairs."""
        return list(
            self.extract(
                lambda r: (r.timestamp, r.error_metrics.get("eomm_score", 0))
            )
        )

    def last_n_minutes(self, minutes: int) -> Iterable[ForgeRecord]:
        now = time.time()
        start = now - (minutes * 60)
        return self.time_range(start, now)

    def manipulation_events_last(self, minutes: int) -> List[ForgeRecord]:
        return list(
            self.where(
                lambda r: r.error_metrics.get("eomm_score", 0) > 0.5
                and r.timestamp >= time.time() - minutes * 60
            )
        )

    def flag_frequency(self) -> dict[str, int]:
        """Count occurrences of each operator flag."""
        flags_count = {}
        for r in self.scan():
            flags = r.context.get("flags", {}).get("operator_flags", [])
            for flag in flags:
                flags_count[flag] = flags_count.get(flag, 0) + 1
        return flags_count

    def eomm_stats(self) -> dict[str, float]:
        """Calculate EOMM score statistics."""
        scores = [r.error_metrics.get("eomm_score", 0) for r in self.scan()]
        if not scores:
            return {"min": 0, "max": 0, "mean": 0, "count": 0}
        
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }

    def session_summary(self) -> dict[str, int]:
        """Count records per session."""
        sessions = {}
        for r in self.scan():
            session_id = r.context.get("session", {}).get("session_id", "unknown")
            sessions[session_id] = sessions.get(session_id, 0) + 1
        return sessions
