"""Network logger — disk-observer implementation.

This service reads network capture/log files on disk and emits events based
on real filesystem changes. It intentionally does not launch or wrap
external capture processes.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

IDLE_THRESHOLD_PULSES = 50
STALL_THRESHOLD_PULSES = 100


def get_network_log_path(config_path: Optional[str]) -> Path:
    if config_path:
        return Path(config_path)
    script_dir = Path(__file__).parent
    logs_dir = (script_dir / ".." / "logs" / "network").resolve()
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"network_capture_{today}.jsonl"


class NetworkService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        self.config_path = config_path
        self.use_subprocess = use_subprocess
        self._running = False

        self._log_path = get_network_log_path(config_path)
        self._pulses_unchanged = 0
        self._last_size: Optional[int] = None
        self._last_mtime: Optional[float] = None
        self._current_day = datetime.now().strftime("%Y%m%d")

    def start(self) -> None:
        logging.info("[NetworkService] start (disk-observer)")
        self._running = True

    def stop(self) -> None:
        logging.info("[NetworkService] stop")
        self._running = False

    def is_running(self) -> bool:
        return bool(self._running)

    def _resolve_path(self) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_day:
            self._current_day = today
            self._last_size = None
            self._last_mtime = None
        self._log_path = get_network_log_path(self.config_path)
        return self._log_path

    def poll(self) -> Optional[dict]:
        try:
            path = self._resolve_path()
            if not path.exists():
                self._pulses_unchanged += 1
                return {
                    "event_type": "network_failed",
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
                    "event_type": "network_active",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "file_size": size,
                    "mtime": mtime,
                }

            # unchanged
            self._pulses_unchanged += 1
            if self._pulses_unchanged >= STALL_THRESHOLD_PULSES:
                return {
                    "event_type": "network_stalled",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "pulses_unchanged": self._pulses_unchanged,
                }
            if self._pulses_unchanged >= IDLE_THRESHOLD_PULSES:
                return {
                    "event_type": "network_idle",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "pulses_unchanged": self._pulses_unchanged,
                }

            return None
        except Exception as exc:
            logging.warning(f"[NetworkService] poll error: {exc}")
            return {
                "event_type": "network_failed",
                "timestamp": time.time(),
                "file_path": str(self._log_path),
                "detail": "exception",
                "error": str(exc),
            }
