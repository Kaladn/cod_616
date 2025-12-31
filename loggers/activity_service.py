"""Activity logger — disk-observer implementation.

This service reads the activity log files on disk and emits events based
on real filesystem changes. It intentionally does not start subprocesses
or provide alternate runtime modes.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

IDLE_THRESHOLD_PULSES = 50
STALL_THRESHOLD_PULSES = 100


def get_activity_log_path() -> Path:
    script_dir = Path(__file__).parent
    logs_dir = (script_dir / ".." / "logs" / "activity").resolve()
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"user_activity_{today}.jsonl"


class ActivityService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        # config_path and use_subprocess are accepted for compatibility but ignored
        self.config_path = config_path
        self.use_subprocess = use_subprocess
        self._running = False

        self._log_path = get_activity_log_path()
        self._pulses_unchanged = 0
        self._last_size: Optional[int] = None
        self._last_mtime: Optional[float] = None
        self._current_day = datetime.now().strftime("%Y%m%d")

    def start(self) -> None:
        logging.info("[ActivityService] start (disk-observer)")
        self._running = True

    def stop(self) -> None:
        logging.info("[ActivityService] stop")
        self._running = False

    def is_running(self) -> bool:
        return bool(self._running)

    def _resolve_path(self) -> Path:
        # Recompute path on day rollover
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_day:
            self._current_day = today
            self._last_size = None
            self._last_mtime = None
        self._log_path = get_activity_log_path()
        return self._log_path

    def poll(self) -> Optional[dict]:
        try:
            path = self._resolve_path()
            if not path.exists():
                self._pulses_unchanged += 1
                # Emit activity_failed immediately if file missing
                return {
                    "event_type": "activity_failed",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "detail": "file_missing",
                }

            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime

            # first-time observation
            if self._last_size is None:
                self._last_size = size
                self._last_mtime = mtime
                self._pulses_unchanged = 0
                return None

            if size > self._last_size or mtime > (self._last_mtime or 0):
                # Growth detected
                self._last_size = size
                self._last_mtime = mtime
                self._pulses_unchanged = 0
                return {
                    "event_type": "activity_detected",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "file_size": size,
                    "mtime": mtime,
                }

            # unchanged
            self._pulses_unchanged += 1
            if self._pulses_unchanged >= STALL_THRESHOLD_PULSES:
                return {
                    "event_type": "activity_stalled",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "pulses_unchanged": self._pulses_unchanged,
                }
            if self._pulses_unchanged >= IDLE_THRESHOLD_PULSES:
                return {
                    "event_type": "activity_idle",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "pulses_unchanged": self._pulses_unchanged,
                }

            return None
        except Exception as exc:
            logging.warning(f"[ActivityService] poll error: {exc}")
            return {
                "event_type": "activity_failed",
                "timestamp": time.time(),
                "file_path": str(self._log_path),
                "detail": "exception",
                "error": str(exc),
            }
