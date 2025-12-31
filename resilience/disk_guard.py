"""DiskGuard: monitor free disk space for logs_data and emit warnings via DailyJSONWriter.

- Monitors `logs_data` path by default.
- Computes avg daily usage from last N days (default 7); falls back to 1GB/day if insufficient history.
- Emits WARNING when free_bytes < 3 * avg_daily_usage_bytes; CRITICAL when free_bytes < 1 * avg_daily_usage_bytes.
- Writes warnings to `logs_data/system/syswarn_MM-DD-YY.jsonl` through DailyJSONWriter(category='system', prefix='syswarn').
- Never crashes; catches all exceptions in loop.

This module is standalone and uses only stdlib (and the existing DailyJSONWriter).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional
import shutil
import json

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

EST = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone(timedelta(hours=-5))

_logger = logging.getLogger(__name__)

# fallback avg bytes per day when insufficient history
FALLBACK_AVG_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


class DiskGuard:
    def __init__(self, logs_path: str = 'logs_data', check_interval: float = 60.0, days: int = 7,
                 disk_usage_fn: Optional[Callable[[str], shutil._ntuple_diskusage]] = None):
        self.logs_path = logs_path
        self.check_interval = float(check_interval)
        self.days = int(days)
        self.disk_usage_fn = disk_usage_fn or shutil.disk_usage

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_free = None
        self._last_avg = None
        self._last_severity = None
        self._last_emit_time = {"warning": 0.0, "critical": 0.0}

        # writer lazily created
        self._writer = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name='DiskGuard', daemon=True)
            self._thread.start()
            _logger.info('DiskGuard started')

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # stop writer if exists
        if self._writer:
            try:
                stop_fn = getattr(self._writer, 'stop', None)
                if callable(stop_fn):
                    stop_fn()
            except Exception:
                _logger.exception('Failed to stop system writer')
        _logger.info('DiskGuard stopped')

    def stats(self) -> dict:
        return {
            'last_free_bytes': self._last_free,
            'last_avg_daily_usage_bytes': self._last_avg,
            'last_severity': self._last_severity,
        }

    # helper to lazily get writer
    def _get_writer(self):
        if self._writer:
            return self._writer
        try:
            from loggers.writer import DailyJSONWriter
            # choose a writer pulse interval related to our check interval to ensure timely flushes in tests
            writer_pulse = max(0.05, min(1.0, self.check_interval))
            w = DailyJSONWriter(category='system', prefix='syswarn', data_root=self.logs_path, pulse_interval=writer_pulse)
            w.start()
            self._writer = w
            return w
        except Exception:
            _logger.exception('Failed to create system DailyJSONWriter')
            return None

    def _now_iso_est(self) -> str:
        try:
            now = datetime.now(timezone.utc).astimezone(EST)
        except Exception:
            now = datetime.now()
        return now.isoformat()

    def _compute_avg_daily_usage(self) -> float:
        # look at files under logs_path/* and aggregate by date token in filename ending with _MM-DD-YY.jsonl
        sizes_by_date = defaultdict(int)
        if not os.path.isdir(self.logs_path):
            return FALLBACK_AVG_BYTES
        for root, dirs, files in os.walk(self.logs_path):
            for fn in files:
                # we expect pattern *_MM-DD-YY.jsonl
                if not fn.endswith('.jsonl'):
                    continue
                parts = fn.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                date_part = parts[1].rsplit('.', 1)[0]
                # validate date format MM-DD-YY
                try:
                    dt = datetime.strptime(date_part, '%m-%d-%y')
                except Exception:
                    # fallback: use file modified date
                    try:
                        p = os.path.join(root, fn)
                        mtime = os.path.getmtime(p)
                        dt = datetime.fromtimestamp(mtime)
                    except Exception:
                        continue
                # use date token as key
                key = dt.strftime('%Y-%m-%d')
                try:
                    p = os.path.join(root, fn)
                    sizes_by_date[key] += os.path.getsize(p)
                except Exception:
                    _logger.exception('Failed to stat file %s', fn)
                    continue
        sorted_dates = sorted(sizes_by_date.items(), reverse=True)
        if not sorted_dates:
            return FALLBACK_AVG_BYTES
        # take up to self.days recent dates
        total = 0
        count = 0
        for k, v in sorted_dates[:self.days]:
            total += v
            count += 1
        if count == 0:
            return FALLBACK_AVG_BYTES
        return total // count

    def _emit_warning(self, severity: str, free_bytes: int, avg_bytes: int, threshold: int):
        # only emit at most once per interval unless severity escalates; use check_interval as base
        now = time.time()
        prev_time = self._last_emit_time.get(severity, 0.0)
        if now - prev_time < self.check_interval and self._last_severity == severity:
            return
        w = self._get_writer()
        if not w:
            _logger.warning('No system writer available; cannot emit %s', severity)
            return
        event = {
            'timestamp': self._now_iso_est(),
            'free_bytes': int(free_bytes),
            'avg_daily_usage_bytes': int(avg_bytes),
            'threshold_bytes': int(threshold),
            'severity': severity,
        }
        try:
            w.enqueue(event)
            # best-effort immediate flush so warnings are visible quickly
            try:
                flush_fn = getattr(w, '_drain_and_write_once', None)
                if callable(flush_fn):
                    flush_fn()
            except Exception:
                _logger.exception('Failed to flush writer after enqueue')
            self._last_emit_time[severity] = now
            self._last_severity = severity
        except Exception:
            _logger.exception('Failed to enqueue warning event')

    def _run_once(self):
        # compute avg daily usage
        try:
            avg = int(self._compute_avg_daily_usage())
            self._last_avg = avg
        except Exception:
            _logger.exception('Failed to compute avg daily usage; using fallback')
            avg = FALLBACK_AVG_BYTES
            self._last_avg = avg
            # one-time informational warning about baseline
            self._emit_warning('warning', 0, avg, 3 * avg)

        # check disk usage
        try:
            usage = self.disk_usage_fn(self.logs_path)
            free = usage.free
            self._last_free = int(free)
        except Exception:
            _logger.exception('disk_usage check failed')
            return

        warn_thr = 3 * avg
        crit_thr = 1 * avg
        if free < crit_thr:
            self._emit_warning('critical', free, avg, crit_thr)
        elif free < warn_thr:
            self._emit_warning('warning', free, avg, warn_thr)

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                self._run_once()
            except Exception:
                _logger.exception('Unexpected error in DiskGuard loop')
            time.sleep(self.check_interval)
