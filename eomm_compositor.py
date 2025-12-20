"""Robust EOMM compositor stub used in test contexts.
A conservative implementation that does not require external config files.
"""
from typing import List

class EommCompositor:
    def __init__(self, config_path: str = None):
        self.config_path = config_path

    def compose_window(self, operator_results: List[object], window_start_epoch: float,
                       window_end_epoch: float, session_id: str, frame_count: int, session_tracker=None):
        # Conservative composite: max operator confidence
        max_conf = 0.0
        flags = []
        for op in operator_results:
            max_conf = max(max_conf, getattr(op, 'confidence', 0.0))
            flags.extend(getattr(op, 'flags', []))

        # Minimal telemetry window structure sufficient for tests
        return {
            "window_start_epoch": window_start_epoch,
            "window_end_epoch": window_end_epoch,
            "window_duration_ms": int((window_end_epoch - window_start_epoch) * 1000),
            "operator_results": operator_results,
            "eomm_composite_score": max_conf,
            "eomm_flags": list(set(flags)),
            "session_id": session_id,
            "frame_count": frame_count,
            "metadata": {}
        }

