"""PulseBus: in-memory router for events with bounded buffers and pulse-driven forwarding.

Design decisions (explicit):
- Per-topic bounded deque buffer (thread-safe via locks) to avoid blocking producers.
- Deterministic drop-oldest policy when buffer is full.
- Pulse thread wakes at `pulse_interval` and forwards buffered events to registered writers.
- Writers are any objects exposing `enqueue(event: dict)`; PulseBus does not assume file I/O.
- PulseBus never crashes: all exceptions are caught, logged, and processing continues.
- Provides lightweight metrics for published, dropped, and forwarded counts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Dict, Tuple, Any, Optional

_logger = logging.getLogger(__name__)


class PulseBus:
    def __init__(self, data_root: str = 'logs_data', pulse_interval: float = 0.1, default_buffer_size: int = 1024):
        self.data_root = data_root
        self.pulse_interval = float(pulse_interval)
        self.default_buffer_size = int(default_buffer_size)

        # buffers keyed by (category, prefix) -> {'lock': Lock, 'deque': deque, 'maxlen': int}
        self._buffers: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # writers keyed by (category, prefix) -> writer
        self._writers: Dict[Tuple[str, str], Any] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._global_lock = threading.Lock()

        # metrics
        self.metrics = {
            'published': 0,
            'dropped': 0,
            'forwarded': 0,
        }

    def start(self) -> None:
        with self._global_lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name='PulseBus', daemon=True)
            self._thread.start()
            _logger.info('PulseBus started (pulse_interval=%s)', self.pulse_interval)

    def stop(self) -> None:
        with self._global_lock:
            if not self._running:
                return
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # stop and close any writers we own
        with self._global_lock:
            for key, writer in list(self._writers.items()):
                try:
                    stop_fn = getattr(writer, 'stop', None)
                    if callable(stop_fn):
                        stop_fn()
                except Exception:
                    _logger.exception('Failed to stop writer for %s/%s', key[0], key[1])
        _logger.info('PulseBus stopped')

    def register_writer(self, category: str, prefix: str, writer: Any, buffer_size: Optional[int] = None) -> None:
        key = (category, prefix)
        with self._global_lock:
            self._writers[key] = writer
            if key not in self._buffers:
                size = buffer_size if buffer_size is not None else self.default_buffer_size
                self._buffers[key] = {
                    'lock': threading.Lock(),
                    'deque': deque(),
                    'maxlen': int(size)
                }
        _logger.info('Writer registered for %s/%s (buffer=%s)', category, prefix, buffer_size or self.default_buffer_size)

    def _ensure_writer(self, category: str, prefix: str):
        key = (category, prefix)
        with self._global_lock:
            if key in self._writers:
                return self._writers[key]
            # lazily create a DailyJSONWriter
            try:
                from loggers.writer import DailyJSONWriter
                writer = DailyJSONWriter(category=category, prefix=prefix, data_root=self.data_root, pulse_interval=self.pulse_interval)
                writer.start()
                self._writers[key] = writer
                # ensure buffer exists
                if key not in self._buffers:
                    self._buffers[key] = {
                        'lock': threading.Lock(),
                        'deque': deque(),
                        'maxlen': self.default_buffer_size
                    }
                _logger.info('Lazily created writer for %s/%s', category, prefix)
                return writer
            except Exception:
                _logger.exception('Failed to lazily create writer for %s/%s', category, prefix)
                return None

    def publish(self, event: dict, category: str, prefix: str) -> None:
        """Enqueue event non-blocking; drop newest if buffer full; never raise."""
        try:
            if not isinstance(event, dict):
                _logger.error('PulseBus.publish called with non-dict event; ignoring')
                return
            key = (category, prefix)
            with self._global_lock:
                if key not in self._buffers:
                    # lazily create buffer
                    self._buffers[key] = {
                        'lock': threading.Lock(),
                        'deque': deque(),
                        'maxlen': self.default_buffer_size
                    }
                buf = self._buffers[key]
            # drop oldest policy: if buffer full, remove oldest to make room
            with buf['lock']:
                if len(buf['deque']) >= buf['maxlen']:
                    buf['deque'].popleft()  # Drop oldest event
                    self.metrics['dropped'] += 1
                    _logger.warning('Buffer full for %s/%s: dropped oldest event', category, prefix)
                buf['deque'].append(event)
                self.metrics['published'] += 1
        except Exception:
            _logger.exception('Unexpected error in publish; event dropped')
            # never re-raise
            return

    def _drain_buffer(self, key: Tuple[str, str]) -> list:
        buf = self._buffers.get(key)
        if buf is None:
            return []
        items = []
        with buf['lock']:
            while buf['deque']:
                items.append(buf['deque'].popleft())
        return items

    def _run(self) -> None:
        while self._running:
            try:
                keys = list(self._buffers.keys())
                for key in keys:
                    try:
                        items = self._drain_buffer(key)
                        if not items:
                            continue
                        # ensure writer exists (lazy creation)
                        writer = self._writers.get(key)
                        if not writer:
                            writer = self._ensure_writer(key[0], key[1])
                        if not writer:
                            # still no writer available; drop events
                            _logger.warning('No writer for %s/%s; dropping %d events', key[0], key[1], len(items))
                            self.metrics['dropped'] += len(items)
                            continue
                        for ev in items:
                            try:
                                writer.enqueue(ev)
                                self.metrics['forwarded'] += 1
                            except Exception:
                                _logger.exception('Writer raised while enqueuing event for %s/%s', key[0], key[1])
                                # do not crash; events are lost if writer fails
                        # After forwarding batch to writer, trigger a writer flush (best-effort)
                        try:
                            # call internal flush method if available
                            flush_fn = getattr(writer, '_drain_and_write_once', None)
                            if callable(flush_fn):
                                flush_fn()
                        except Exception:
                            _logger.exception('Writer flush failed for %s/%s', key[0], key[1])
                    except Exception:
                        _logger.exception('Error while draining buffer for %s/%s', key[0], key[1])
                time.sleep(self.pulse_interval)
            except Exception:
                _logger.exception('PulseBus main loop encountered an unexpected error')
        # final drain on shutdown (best-effort)
        try:
            for key in list(self._buffers.keys()):
                items = self._drain_buffer(key)
                writer = self._writers.get(key)
                if writer:
                    for ev in items:
                        try:
                            writer.enqueue(ev)
                            self.metrics['forwarded'] += 1
                        except Exception:
                            _logger.exception('Writer raised during final flush for %s/%s', key[0], key[1])
        except Exception:
            _logger.exception('Error during final flush')

    def stats(self) -> dict:
        with self._global_lock:
            copy = dict(self.metrics)
            # per-buffer sizes
            copy['buffers'] = {k: len(v['deque']) for k, v in self._buffers.items()}
        return copy
