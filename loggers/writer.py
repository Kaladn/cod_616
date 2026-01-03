"""Simple pulse-based, append-only JSONL writer for loggers.

- One writer per (category, prefix) produces files under `logs_data/<category>/`.
- Filename: `{prefix}_{MM-DD-YY}.jsonl` using Eastern Time day boundaries (00:01 starts new day).
- Writes are canonical JSON (UTF-8, no whitespace, sorted keys).
- Maintains a rolling SHA-256 chain stored as `_sha` in each JSON line.
- Single writer thread owns the file handle; producers call `enqueue(event: dict)`.

Constraints: stdlib only, boring, auditable, no retries, no background threads beyond the single writer thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from queue import Queue, Empty
from typing import Any, Dict, Optional
import hashlib

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

EST_ZONE = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone(timedelta(hours=-5))

_logger = logging.getLogger(__name__)

_INITIAL_SHA = "0" * 64


def _ensure_finite_numbers(obj: Any):
    # Recursively ensure no NaN/Inf
    if isinstance(obj, float):
        if not (obj == obj and abs(obj) != float("inf")):
            raise ValueError("Non-finite float not allowed in payload")
    elif isinstance(obj, dict):
        for v in obj.values():
            _ensure_finite_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _ensure_finite_numbers(v)


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    _ensure_finite_numbers(obj)
    s = json.dumps(obj, separators=(",",":"), sort_keys=True, ensure_ascii=False)
    return s.encode("utf-8")


def _day_key_for(dt_est: datetime) -> str:
    """Return the string key for the calendar day, applying 00:01 start rule.

    If local EST time is before 00:01, treat as previous day.
    """
    # normalize to date in EST
    if isinstance(dt_est, datetime):
        local = dt_est
    else:
        raise TypeError("dt_est must be datetime")
    t = local.timetz()
    # If time < 00:01, shift back one day
    if local.time() < dt_time(0, 1):
        local = local - timedelta(days=1)
    return local.strftime("%m-%d-%y")


class DailyJSONWriter:
    """Pulse-based append-only writer.

    Usage:
        w = DailyJSONWriter(category='activity', prefix='act_log')
        w.start()
        w.enqueue({'timestamp': ..., 'event': ...})
        w.stop()
    """

    def __init__(self, category: str, prefix: str, data_root: str = "logs_data", pulse_interval: float = 0.1):
        self.category = category
        self.prefix = prefix
        self.data_root = data_root
        self.pulse_interval = float(pulse_interval)

        self._queue: Queue = Queue(maxsize=10000)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # writer state
        self._file = None
        self._current_day = None
        self._prev_sha = _INITIAL_SHA

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name=f"DailyJSONWriter-{self.category}-{self.prefix}", daemon=True)
            self._thread.start()
            _logger.info("DailyJSONWriter started: %s/%s", self.category, self.prefix)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        # wait for thread
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # flush remaining
        try:
            self._drain_and_write_once()
        except Exception:
            _logger.exception("Error while flushing on stop")
        finally:
            self._close_file()
            _logger.info("DailyJSONWriter stopped: %s/%s", self.category, self.prefix)

    def enqueue(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        # Quick validation
        _ensure_finite_numbers(event)
        self._queue.put(event)

    # Internal utilities
    def _ensure_dir(self) -> None:
        d = os.path.join(self.data_root, self.category)
        os.makedirs(d, exist_ok=True)

    def _open_file_for_day(self, day_key: str):
        self._ensure_dir()
        filename = f"{self.prefix}_{day_key}.jsonl"
        path = os.path.join(self.data_root, self.category, filename)
        # open in text append mode
        f = open(path, "a", encoding="utf-8")
        return f

    def _current_est_datetime(self) -> datetime:
        now_utc = datetime.now(timezone.utc)
        try:
            now_est = now_utc.astimezone(EST_ZONE)
        except Exception:
            # fallback to naive local time if zone conversion fails
            now_est = datetime.now()
        return now_est

    def _rotate_if_needed(self):
        now_est = self._current_est_datetime()
        day_key = _day_key_for(now_est)
        if day_key != self._current_day:
            # rotate
            self._close_file()
            try:
                self._file = self._open_file_for_day(day_key)
                self._current_day = day_key
            except Exception:
                _logger.exception("Failed to open log file for day %s", day_key)
                self._file = None

    def _close_file(self):
        try:
            if self._file:
                try:
                    self._file.flush()
                    try:
                        os.fsync(self._file.fileno())
                    except Exception:
                        pass
                finally:
                    self._file.close()
        finally:
            self._file = None

    def _drain_queue(self) -> list:
        items = []
        while True:
            try:
                item = self._queue.get_nowait()
                items.append(item)
            except Empty:
                break
        return items

    def _drain_and_write_once(self) -> None:
        items = self._drain_queue()
        if not items:
            return
        # ensure file open and rotation
        self._rotate_if_needed()
        if not self._file:
            _logger.error("No file available for writing; dropping %d events", len(items))
            return
        try:
            for event in items:
                try:
                    payload_bytes = _canonical_json_bytes(event)
                    # compute next sha = sha256(prev_sha_hex + payload_bytes)
                    m = hashlib.sha256()
                    m.update(self._prev_sha.encode("ascii"))
                    m.update(payload_bytes)
                    new_sha = m.hexdigest()
                    # write line with _sha
                    line_obj = dict(event)
                    line_obj["_sha"] = new_sha
                    line_line = json.dumps(line_obj, separators=(",",":"), sort_keys=True, ensure_ascii=False)
                    self._file.write(line_line + "\n")
                    # update chain
                    self._prev_sha = new_sha
                except Exception:
                    _logger.exception("Failed to serialize or write event: %s", event)
                    # continue with next event
                    continue
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass
        except Exception:
            _logger.exception("Unexpected error during write pulse")

    def _run(self) -> None:
        # initialize
        while self._running:
            try:
                # rotation and writing
                self._rotate_if_needed()
                self._drain_and_write_once()
            except Exception:
                _logger.exception("Writer encountered an unexpected error")
            time.sleep(self.pulse_interval)
        # final flush on exit
        try:
            self._drain_and_write_once()
        except Exception:
            _logger.exception("Error while flushing at thread exit")


# light CLI for manual quick tests (not used by tests)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    w = DailyJSONWriter("activity", "act_log")
    w.start()
    try:
        for i in range(5):
            w.enqueue({"timestamp": int(time.time() * 1000), "i": i})
            time.sleep(0.02)
        time.sleep(0.2)
    finally:
        w.stop()
