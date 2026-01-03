"""
Recognition Field v2 — Temporal cognition engine for Forge memory.

Provides:
  - Anomaly streams (threshold-based)
  - Flag-based queries (find INSTA_MELT, etc.)
  - Time-based queries (last N minutes)
  - Burst detection (contiguous high-anomaly windows)
  - Session summaries (per-session stats)
  - Feature extraction (time series, operator frequency)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Set

import sys
from pathlib import Path

# Allow imports from forge_memory
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.record import ForgeRecord
from query.query_layer import QueryLayer


@dataclass
class Burst:
    """
    A contiguous sequence of high-anomaly windows.
    
    Represents temporal patterns where manipulation events cluster together.
    """
    start_ts: float
    end_ts: float
    records: List[ForgeRecord]
    peak_eomm: float
    operators: Set[str]

    @property
    def length(self) -> int:
        """Number of windows in this burst."""
        return len(self.records)

    @property
    def duration(self) -> float:
        """Time span of this burst in seconds."""
        return self.end_ts - self.start_ts

    def __repr__(self) -> str:
        return (
            f"Burst(length={self.length}, duration={self.duration:.1f}s, "
            f"peak_eomm={self.peak_eomm:.2f}, operators={self.operators})"
        )


@dataclass
class SessionSummary:
    """
    Per-session statistics and fingerprint.
    
    Tracks manipulation patterns, operator usage, and EOMM distribution
    for a single gaming session.
    """
    session_id: str
    window_count: int
    anomaly_count: int
    high_conf_count: int
    avg_eomm: float
    max_eomm: float
    operator_histogram: Dict[str, int]
    map_name: Optional[str]
    mode_name: Optional[str]

    @property
    def manipulation_rate(self) -> float:
        """Percentage of windows with EOMM > 0.5."""
        if self.window_count == 0:
            return 0.0
        return (self.anomaly_count / self.window_count) * 100

    @property
    def high_conf_rate(self) -> float:
        """Percentage of windows with EOMM > 0.75."""
        if self.window_count == 0:
            return 0.0
        return (self.high_conf_count / self.window_count) * 100

    def __repr__(self) -> str:
        return (
            f"SessionSummary(session={self.session_id}, windows={self.window_count}, "
            f"manip_rate={self.manipulation_rate:.1f}%, avg_eomm={self.avg_eomm:.2f})"
        )


class RecognitionField:
    """
    Recognition Field v2 — Temporal pattern recognition on top of QueryLayer.

    This layer turns raw memory cells (ForgeRecords) into:
      - anomaly streams
      - temporal bursts
      - session summaries
      - operator fingerprints

    Usage:
        rf = RecognitionField(query_layer)
        
        # Find all INSTA_MELT windows
        insta_melt = rf.windows_with_flag("INSTA_MELT")
        
        # Find manipulation in last 5 minutes
        recent_manip = rf.manipulation_last_minutes(5)
        
        # Detect temporal bursts
        bursts = rf.find_bursts(threshold=0.75, min_len=3)
        
        # Session summary
        summary = rf.summarize_session("live_session_20251203_060841")
    """

    def __init__(self, query_layer: QueryLayer) -> None:
        self.q = query_layer

    # ---------------------------------------------------------
    # Core anomaly streams
    # ---------------------------------------------------------

    def anomaly_stream(self, threshold: float = 0.5) -> Iterable[ForgeRecord]:
        """Stream all windows where EOMM > threshold."""
        for r in self.q.scan():
            if r.error_metrics["eomm_score"] > threshold:
                yield r

    def high_confidence_stream(self, threshold: float = 0.75) -> Iterable[ForgeRecord]:
        """Stream all windows where EOMM > threshold (high confidence anomalies)."""
        return self.anomaly_stream(threshold)

    # ---------------------------------------------------------
    # Flag-based & time-based views
    # ---------------------------------------------------------

    def windows_with_flag(
        self,
        flag: str,
        threshold: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[ForgeRecord]:
        """
        All windows where operator_flags contains `flag`.
        If threshold is provided, additionally require EOMM > threshold.
        """
        out: List[ForgeRecord] = []
        for r in self.q.scan():
            flags = r.context["flags"].get("operator_flags", [])
            if flag not in flags:
                continue
            if threshold is not None and r.error_metrics["eomm_score"] <= threshold:
                continue
            out.append(r)
            if limit is not None and len(out) >= limit:
                break
        return out

    def manipulation_last_minutes(
        self,
        minutes: int,
        threshold: float = 0.5,
    ) -> List[ForgeRecord]:
        """
        All windows in the last N minutes where EOMM > threshold.
        Uses QueryLayer.last_n_minutes as a coarse pre-filter when available.
        """
        # Use time-filtered scan first to cut search space
        records = self.q.last_n_minutes(minutes)
        return [
            r for r in records
            if r.error_metrics["eomm_score"] > threshold
        ]

    # ---------------------------------------------------------
    # Temporal bursts (core temporal pattern)
    # ---------------------------------------------------------

    def find_bursts(
        self,
        threshold: float = 0.75,
        min_len: int = 3,
        max_gap_sec: float = 2.0,
    ) -> List[Burst]:
        """
        Find contiguous bursts of high-confidence anomalies.

        A burst is:
          - consecutive high-anomaly windows (EOMM > threshold)
          - gaps between windows <= max_gap_sec
          - length >= min_len

        This detects temporal clustering of manipulation events.
        """
        anomalies = list(self.high_confidence_stream(threshold))
        if not anomalies:
            return []

        bursts: List[Burst] = []
        current: List[ForgeRecord] = [anomalies[0]]

        for prev, cur in zip(anomalies, anomalies[1:]):
            dt = cur.timestamp - prev.timestamp
            if dt <= max_gap_sec:
                current.append(cur)
            else:
                # Close current burst
                if len(current) >= min_len:
                    bursts.append(self._build_burst(current))
                current = [cur]

        # Handle tail
        if len(current) >= min_len:
            bursts.append(self._build_burst(current))

        return bursts

    def _build_burst(self, records: List[ForgeRecord]) -> Burst:
        """Construct a Burst from a list of ForgeRecords."""
        start_ts = records[0].timestamp
        end_ts = records[-1].timestamp
        peak_eomm = max(r.error_metrics["eomm_score"] for r in records)
        ops: Set[str] = set()
        for r in records:
            ops.update(r.error_metrics.get("operators", {}).keys())
        return Burst(
            start_ts=start_ts,
            end_ts=end_ts,
            records=records,
            peak_eomm=peak_eomm,
            operators=ops,
        )

    # ---------------------------------------------------------
    # Session-level summaries
    # ---------------------------------------------------------

    def summarize_session(self, session_id: str) -> SessionSummary:
        """
        Summarize a single session by session_id.
        Session id lives in: record.context["session"]["session_id"]
        """
        windows: List[ForgeRecord] = []
        for r in self.q.scan():
            sess = r.context.get("session", {})
            if sess.get("session_id") == session_id:
                windows.append(r)

        if not windows:
            return SessionSummary(
                session_id=session_id,
                window_count=0,
                anomaly_count=0,
                high_conf_count=0,
                avg_eomm=0.0,
                max_eomm=0.0,
                operator_histogram={},
                map_name=None,
                mode_name=None,
            )

        window_count = len(windows)
        eomms = [r.error_metrics["eomm_score"] for r in windows]
        max_eomm = max(eomms)
        avg_eomm = sum(eomms) / window_count

        anomaly_count = sum(1 for v in eomms if v > 0.5)
        high_conf_count = sum(1 for v in eomms if v > 0.75)

        # Operator histogram
        hist: Dict[str, int] = {}
        for r in windows:
            ops = r.error_metrics.get("operators", {})
            for op_name in ops.keys():
                hist[op_name] = hist.get(op_name, 0) + 1

        sess0 = windows[0].context.get("session", {})
        map_name = sess0.get("map_name")
        mode_name = sess0.get("mode_name")

        return SessionSummary(
            session_id=session_id,
            window_count=window_count,
            anomaly_count=anomaly_count,
            high_conf_count=high_conf_count,
            avg_eomm=avg_eomm,
            max_eomm=max_eomm,
            operator_histogram=hist,
            map_name=map_name,
            mode_name=mode_name,
        )

    def summarize_all_sessions(self, limit: Optional[int] = None) -> List[SessionSummary]:
        """
        Summarize all sessions found in memory.

        Warning: does a full scan; fine for current scale.
        """
        by_session: Dict[str, List[ForgeRecord]] = {}

        for r in self.q.scan():
            sess = r.context.get("session", {})
            sid = sess.get("session_id")
            if not sid:
                continue
            by_session.setdefault(sid, []).append(r)

        summaries: List[SessionSummary] = []
        for sid, windows in by_session.items():
            if not windows:
                continue

            window_count = len(windows)
            eomms = [r.error_metrics["eomm_score"] for r in windows]
            max_eomm = max(eomms)
            avg_eomm = sum(eomms) / window_count

            anomaly_count = sum(1 for v in eomms if v > 0.5)
            high_conf_count = sum(1 for v in eomms if v > 0.75)

            hist: Dict[str, int] = {}
            for r in windows:
                ops = r.error_metrics.get("operators", {})
                for op_name in ops.keys():
                    hist[op_name] = hist.get(op_name, 0) + 1

            sess0 = windows[0].context.get("session", {})
            map_name = sess0.get("map_name")
            mode_name = sess0.get("mode_name")

            summaries.append(
                SessionSummary(
                    session_id=sid,
                    window_count=window_count,
                    anomaly_count=anomaly_count,
                    high_conf_count=high_conf_count,
                    avg_eomm=avg_eomm,
                    max_eomm=max_eomm,
                    operator_histogram=hist,
                    map_name=map_name,
                    mode_name=mode_name,
                )
            )

            if limit is not None and len(summaries) >= limit:
                break

        return summaries

    # ---------------------------------------------------------
    # Feature extraction for plotting
    # ---------------------------------------------------------

    def eomm_time_series(self) -> List[Tuple[float, float]]:
        """(timestamp, eomm_score) pairs for all windows."""
        return [
            (r.timestamp, r.error_metrics["eomm_score"])
            for r in self.q.scan()
        ]

    def operator_frequency(self) -> Dict[str, int]:
        """Global histogram of operator usage across all windows."""
        hist: Dict[str, int] = {}
        for r in self.q.scan():
            ops = r.error_metrics.get("operators", {})
            for name in ops.keys():
                hist[name] = hist.get(name, 0) + 1
        return hist
