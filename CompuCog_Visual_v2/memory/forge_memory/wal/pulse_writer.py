from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Iterable, Optional

from forge_memory.core.record import ForgeRecord
from forge_memory.core.string_dict import StringDictionary
from forge_memory.core.binary_log import BinaryLog
from forge_memory.bridge.schema_map import TrueVisionSchemaMap
from forge_memory.wal.wal_writer import WALWriter


class PulseConfig:
    """
    Thresholds for forming and flushing pulses.

    All units:
      - max_records_per_pulse: count
      - max_bytes_per_pulse: bytes
      - max_age_ms_per_pulse: milliseconds
    """

    def __init__(
        self,
        max_records_per_pulse: int = 128,
        max_bytes_per_pulse: int = 512 * 1024,
        max_age_ms_per_pulse: int = 250,
    ) -> None:
        self.max_records_per_pulse = max_records_per_pulse
        self.max_bytes_per_pulse = max_bytes_per_pulse
        self.max_age_ms_per_pulse = max_age_ms_per_pulse


class PulseWriter:
    """
    Orchestrator:

        TrueVision window → schema_map → ForgeRecord
        → buffer → WAL entry (pulse) → BinaryLog append_batch

    This class owns:
      - pulse buffering
      - pulse_id assignment
      - per-record seq assignment
      - flush decision (count / bytes / age)
      - ordering guarantees to WAL + BinaryLog

    It does NOT:
      - invent any schema (delegates to TrueVisionSchemaMap)
      - do recovery (WALReader/WALReplayer handle that)
      - manage checkpoints (just notifies via hook)
    """

    def __init__(
        self,
        *,
        wal_writer: WALWriter,
        binary_log: BinaryLog,
        string_dict: StringDictionary,
        schema_map: TrueVisionSchemaMap,
        worker_id: int = 0,
        config: Optional[PulseConfig] = None,
        checkpoint_callback: Optional[
            callable
        ] = None,  # optional hook: fn(pulse_id, record_count, total_bytes)
    ) -> None:
        self.wal_writer = wal_writer
        self.binary_log = binary_log
        self.string_dict = string_dict
        self.schema_map = schema_map

        self.worker_id = int(worker_id) & 0xFF
        self.config = config or PulseConfig()
        self.checkpoint_callback = checkpoint_callback

        # Pulse + sequence state
        # Start pulse_id at WALWriter's last_pulse_id so we don't collide after restart.
        self._pulse_id_counter = getattr(wal_writer, "last_pulse_id", 0)
        self._seq_counter: int = 0

        # Buffering
        self._buffer: List[ForgeRecord] = []
        self._buffer_bytes: int = 0
        self._buffer_first_wallclock: Optional[float] = None

        # Synchronization
        self._lock = threading.Lock()
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_window(self, window: Dict[str, Any]) -> None:
        """
        Ingest a single TrueVision window.

        This:
          1) Validates and translates via schema_map.
          2) Constructs a ForgeRecord (without pulse_id yet).
          3) Assigns worker_id + seq.
          4) Estimates serialized size (once) for byte-threshold.
          5) Buffers and flushes if needed.
        """
        if self._closed:
            raise RuntimeError("PulseWriter is closed; cannot submit new windows")

        # 1–3. Translate + record construction
        rec = self._window_to_record(window)

        # 4–5. Buffer and maybe flush
        with self._lock:
            self._buffer_append(rec)
            self._maybe_flush_locked()

    def flush(self, reason: str = "manual") -> None:
        """
        Force flush current buffer into a pulse (if non-empty).

        'reason' is purely for logging/observability upstream.
        """
        del reason  # not used here, but kept for callers' semantic clarity

        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """
        Flush any remaining records and mark writer as closed.

        No further windows may be submitted after this.
        """
        with self._lock:
            if not self._closed:
                self._flush_locked()
                self._closed = True
                # Save string dictionary
                self.string_dict.save()

    # ------------------------------------------------------------------
    # Internal helpers — window → record
    # ------------------------------------------------------------------

    def _window_to_record(self, window: Dict[str, Any]) -> ForgeRecord:
        """
        Translate a TrueVision window to a fully-formed ForgeRecord,
        except for pulse_id which is assigned at flush time.
        """
        rec_dict = self.schema_map.window_to_record_dict(window)

        # ForgeRecord.from_dict handles all core fields.
        record = ForgeRecord.from_dict(rec_dict)

        # Fill per-writer fields here (pulse_id is filled later).
        record.worker_id = self.worker_id
        record.seq = self._next_seq()

        return record

    def _next_seq(self) -> int:
        self._seq_counter = (self._seq_counter + 1) & 0xFFFFFFFF
        return self._seq_counter

    # ------------------------------------------------------------------
    # Internal helpers — buffering + thresholds
    # ------------------------------------------------------------------

    def _buffer_append(self, record: ForgeRecord) -> None:
        """Append record to buffer and update thresholds."""
        self._buffer.append(record)

        # Estimate bytes by performing a one-time serialization.
        # This also warms StringDictionary entries, which is fine.
        serialized = record.serialize(self.string_dict)
        self._buffer_bytes += len(serialized)

        if self._buffer_first_wallclock is None:
            self._buffer_first_wallclock = time.time()

    def _buffer_age_ms(self) -> float:
        if self._buffer_first_wallclock is None:
            return 0.0
        return (time.time() - self._buffer_first_wallclock) * 1000.0

    def _should_flush_locked(self) -> bool:
        """
        Decide whether to flush based on buffer size/bytes/age thresholds.
        Assumes caller holds _lock.
        """
        if not self._buffer:
            return False

        cfg = self.config

        if len(self._buffer) >= cfg.max_records_per_pulse:
            return True

        if self._buffer_bytes >= cfg.max_bytes_per_pulse:
            return True

        if self._buffer_age_ms() >= cfg.max_age_ms_per_pulse:
            return True

        return False

    def _maybe_flush_locked(self) -> None:
        if self._should_flush_locked():
            self._flush_locked()

    # ------------------------------------------------------------------
    # Internal helpers — pulse commit path
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """
        Flush current buffer into a single pulse.

        Steps:
          1) Snapshot buffered records.
          2) Clear buffer state.
          3) Assign pulse_id and attach to records.
          4) Write WAL entry.
          5) Append batch to BinaryLog.
          6) Notify checkpoint callback (if any).
        Caller must hold _lock.
        """
        if not self._buffer:
            return

        # 1) Snapshot
        records: List[ForgeRecord] = list(self._buffer)
        total_bytes = self._buffer_bytes

        # 2) Clear buffer
        self._buffer = []
        self._buffer_bytes = 0
        self._buffer_first_wallclock = None

        # 3) Assign pulse_id
        self._pulse_id_counter += 1
        pulse_id = self._pulse_id_counter

        # Attach pulse_id to each record (used mainly on BinaryLog side / queries)
        for rec in records:
            rec.pulse_id = pulse_id

        # 4) WAL entry (durable)
        #    If this throws, we let the exception propagate:
        #    it's a hard failure condition, and BinaryLog MUST NOT see those records.
        self.wal_writer.write_entry(pulse_id, records)

        # 5) BinaryLog append (permanent store)
        #    If this throws after WAL commit, recovery will reapply from WAL.
        self.binary_log.append_batch(records)

        # 6) Checkpoint hook
        if self.checkpoint_callback is not None:
            try:
                self.checkpoint_callback(
                    pulse_id, len(records), total_bytes
                )
            except Exception:
                # Never let checkpoint callback break core durability path
                pass
