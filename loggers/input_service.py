"""Input logger — disk-observer implementation.

This service reads input log files on disk and emits events for newly
appended records. It intentionally does not start subprocesses or provide
alternate runtime modes.
"""
from __future__ import annotations

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

IDLE_THRESHOLD_PULSES = 50
STALL_THRESHOLD_PULSES = 100


def get_input_log_path(config_path: Optional[str]) -> Path:
    if config_path:
        return Path(config_path)
    script_dir = Path(__file__).parent
    logs_dir = (script_dir / ".." / "logs" / "input").resolve()
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"input_events_{today}.jsonl"


class InputService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        # keep signature for compatibility; ignore use_subprocess
        self.config_path = config_path
        self.use_subprocess = use_subprocess
        self._running = False

        self._log_path = get_input_log_path(config_path)
        self._read_offset: int = 0
        self._pulses_unchanged = 0
        self._current_day = datetime.now().strftime("%Y%m%d")

    def start(self) -> None:
        logging.info("[InputService] start (disk-observer)")
        self._running = True

    def stop(self) -> None:
        logging.info("[InputService] stop")
        self._running = False

    def is_running(self) -> bool:
        return bool(self._running)

    def _resolve_path(self) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_day:
            self._current_day = today
            # rollover: reset offset
            self._read_offset = 0
        self._log_path = get_input_log_path(self.config_path)
        return self._log_path

    def poll(self) -> Optional[List[dict]]:
        try:
            path = self._resolve_path()
            if not path.exists():
                self._pulses_unchanged += 1
                return [{
                    "event_type": "input_failed",
                    "timestamp": time.time(),
                    "file_path": str(path),
                    "detail": "file_missing",
                }]

            current_size = path.stat().st_size
            if current_size < self._read_offset:
                # file rotated/truncated
                self._read_offset = 0

            # Treat existing content as an initial snapshot: do not emit existing records
            if self._read_offset == 0:
                if current_size == 0:
                    # nothing to read yet
                    return None
                # consume existing content as initial snapshot
                self._read_offset = current_size
                self._pulses_unchanged = 0
                return None

            if current_size == self._read_offset:
                self._pulses_unchanged += 1
                if self._pulses_unchanged >= STALL_THRESHOLD_PULSES:
                    return [{
                        "event_type": "input_stalled",
                        "timestamp": time.time(),
                        "file_path": str(path),
                        "pulses_unchanged": self._pulses_unchanged,
                    }]
                if self._pulses_unchanged >= IDLE_THRESHOLD_PULSES:
                    return [{
                        "event_type": "input_idle",
                        "timestamp": time.time(),
                        "file_path": str(path),
                        "pulses_unchanged": self._pulses_unchanged,
                    }]
                return None

            # read new data
            events: List[dict] = []
            with open(path, 'r', encoding='utf-8') as fh:
                fh.seek(self._read_offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        events.append({
                            "event_type": "input_event",
                            "timestamp": time.time(),
                            "file_path": str(path),
                            "record": obj,
                        })
                    except Exception as e:
                        return [{
                            "event_type": "input_failed",
                            "timestamp": time.time(),
                            "file_path": str(path),
                            "detail": "json_parse_error",
                            "error": str(e),
                            "line": line,
                        }]
                self._read_offset = fh.tell()

            # reset unchanged counter
            self._pulses_unchanged = 0
            return events if events else None
        except Exception as exc:
            logging.warning(f"[InputService] poll error: {exc}")
            return [{
                "event_type": "input_failed",
                "timestamp": time.time(),
                "file_path": str(self._log_path),
                "detail": "exception",
                "error": str(exc),
            }]
