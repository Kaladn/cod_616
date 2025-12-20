# 🫀 PulseWriter — Durability Orchestrator

**Last Updated**: December 3, 2025  
**Status**: LOCKED

---

## 1. Role in the Stack

Data path:

```text
TrueVision window
  → schema_map (TrueVisionWindow → ForgeRecord)
  → PulseWriter (batching, pulse_id, seq, worker_id)
  → WAL (wal.bin entry)
  → BinaryLog (records.bin append_batch)
```

**PulseWriter is the ONLY component allowed to:**

* Construct ForgeRecords from windows (using schema_map)
* Decide pulse boundaries (when to flush)
* Call WAL append
* Call BinaryLog append_batch

Everything else is either upstream (TrueVision) or downstream (WAL, BinaryLog, Query).

---

## 2. Core Concepts

### 2.1 Pulse

A **pulse** is a batch of ForgeRecords that move atomically through durability:

```text
Pulse = {
  pulse_id: uint64,
  records: [ForgeRecord, ...],
}
```

In WAL:

* One pulse → one WAL entry
  * `PULSE_ID` (uint64)
  * `RECORD_COUNT` (uint32)
  * concatenated ForgeRecords

In BinaryLog:

* The same records are appended in the same order as a batch.

**Invariant:**

For any `pulse_id`, its records are either:

* fully present in WAL and BinaryLog, or
* fully present only in WAL (and replayable), or
* absent from both (never committed).

Never BinaryLog-only.

---

### 2.2 IDs and Counters

* `pulse_id` → `uint64`, strictly monotonic per WAL file / data_dir
* `worker_id` → `uint8`, fixed per PulseWriter instance (0–255)
* `seq` → `uint32`, monotonically incrementing per ForgeRecord created by this PulseWriter

**seq rule:**

* Starts at 0 or 1 when PulseWriter is initialized.
* Increments by 1 for every ForgeRecord (across pulses).
* Wrap-around (2³²) is allowed; no correctness assumptions depend on strict global uniqueness.

---

## 3. PulseWriter Interface

### 3.1 Construction

```python
PulseWriter(
    wal_writer: WALWriter,
    binary_log: BinaryLog,
    string_dict: StringDictionary,
    worker_id: int,
    config: PulseConfig,
    schema_map: TrueVisionSchemaMap
)
```

* `wal_writer` → implementation of WAL spec (write_entry, truncate, etc.)
* `binary_log` → BinaryLog instance
* `string_dict` → shared StringDictionary for ForgeRecord
* `worker_id` → 0–255
* `config` → thresholds (see below)
* `schema_map` → pure translator: TrueVision window → ForgeRecord dict

### 3.2 Public Methods

* `submit_window(window: dict) -> None`
  * Called by TrueVision pipeline for each window.
  * Translates via schema_map; appends ForgeRecord to in-memory buffer; may trigger auto-flush.

* `flush(reason: str = "manual") -> None`
  * Forces current buffer into a pulse (if non-empty).
  * `reason` is informational (for logs/metrics only).

* `close() -> None`
  * Flushes any remaining buffered windows.
  * Blocks until last pulse has been written to WAL and BinaryLog.
  * After close, `submit_window` must raise or be a no-op.

---

## 4. Pulse Boundaries & Thresholds

Defined in `PulseConfig`:

```python
PulseConfig = {
    "max_records_per_pulse": int,   # e.g. 128
    "max_bytes_per_pulse": int,     # e.g. 512 * 1024
    "max_age_ms_per_pulse": int,    # e.g. 250 (buffer time)
}
```

**Buffering rules:**

PulseWriter maintains:

* `buffer_records: list[ForgeRecord]`
* `buffer_bytes: int` (approx total serialized size; tracked incrementally)
* `buffer_first_ts: float | None` (ts_start of first buffered window)

On each `submit_window`:

1. Window → `ForgeRecord` via schema_map.
2. Append to `buffer_records`.
3. Update `buffer_bytes` and `buffer_first_ts` if needed.
4. Check thresholds:
   * If `len(buffer_records) >= max_records_per_pulse` → flush.
   * If `buffer_bytes >= max_bytes_per_pulse` → flush.
   * If `now - buffer_first_ts >= max_age_ms_per_pulse` → flush.

On `flush()` (manual or threshold):

* If buffer is empty → no-op.
* Else → form a pulse and commit (see next section).

---

## 5. Pulse Commit Protocol

**Goal:**

Make every pulse durably recoverable before it ever hits BinaryLog.

### 5.1 Steps (normal execution)

When `flush()` decides to commit a pulse:

1. Acquire `pulse_lock`.

2. Snapshot:
   * `records = buffer_records[:]`
   * `record_count = len(records)`
   * If `record_count == 0`:
     * Release lock, return no-op.

3. Compute:
   * `pulse_id = last_pulse_id + 1`

4. Clear live buffer:
   * `buffer_records = []`
   * `buffer_bytes = 0`
   * `buffer_first_ts = None`
     *(important: after snapshot, so new incoming windows start a fresh pulse)*

5. **WAL append:**
   * Call `wal_writer.write_entry(pulse_id, records)`
     * This:
       * builds `entry_body`
       * computes checksum
       * writes full entry
       * updates WAL header (ENTRY_COUNT, optionally LAST_CHECKPOINT_PULSE_ID)
       * fsyncs
   * If this step fails:
     * No attempt is made to write to BinaryLog.
     * Pulse is considered **not committed**.
     * Caller may:
       * log error
       * stop ingestion
       * or retry (implementation choice)

6. **BinaryLog append:**
   * Call `binary_log.append_batch(records)`
   * This should not be fsync'd on every pulse (WAL is your durability), but BinaryLog should be flushed periodically or upon checkpoint.

7. Update in-memory state:
   * `last_pulse_id = pulse_id`
   * notify `CheckpointManager` of:
     * `pulses += 1`
     * `records += record_count`
     * `bytes += sum(len(r.serialize(...)))` (or approximated from buffer_bytes snapshot)

8. Release `pulse_lock`.

**Invariant:**

If BinaryLog append fails after WAL write succeeds, WAL still holds a full, valid pulse entry. Recovery will reapply it.

---

## 6. Crash Invariants

### 6.1 Crash before flush

* Buffered windows (not yet flushed to WAL) are lost.
* Acceptable, as they were never durable.
* Upstream can tolerate this as "at most X ms of data loss" determined by `max_age_ms_per_pulse`.

### 6.2 Crash during WAL write

* If crash happens before `write_entry` completes (and fsync returns):
  * WAL entry will be either:
    * non-existent, or
    * truncated/corrupt
  * Recovery scan will ignore it per WAL spec (length/CRC failure).

### 6.3 Crash after WAL write, before BinaryLog append

* WAL entry is fully valid.
* BinaryLog may or may not contain these records.
* On recovery:
  * WAL replay reads entry.
  * Replays records into BinaryLog.
  * Pulse becomes fully committed.

### 6.4 Crash after BinaryLog append, before metadata update/checkpoint

* WAL and BinaryLog both contain the pulse.
* Recovery may reapply the same WAL entry:
  * Implementation must either:
    * detect duplication (idempotent append), or
    * accept benign duplication for now (simpler initial system).
  * **Recommended:** store `last_applied_pulse_id` in metadata; only apply WAL entries with `pulse_id > last_applied_pulse_id`.

---

## 7. Interaction with Checkpointing

PulseWriter should not implement checkpoint logic itself but must provide metrics:

* On each committed pulse:
  * `CheckpointManager.on_pulse_committed(pulse_id, record_count, total_bytes)`

CheckpointManager:

* Tracks:
  * pulses since checkpoint
  * bytes since checkpoint
* When thresholds met:
  * invokes `checkpoint(pulse_id)` flow (as defined in WAL spec), which:
    * updates WAL header's LAST_CHECKPOINT_PULSE_ID
    * truncates WAL
    * persists metadata state

PulseWriter does not truncate WAL; it only reports pulses.

---

## 8. Integration with schema_map

PulseWriter MUST NEVER invent field mappings.

It must:

1. Accept raw window `dict` from TrueVision.

2. Call:
   ```python
   rec_dict = schema_map.truevision_window_to_record_dict(window)
   record = ForgeRecord.from_dict(rec_dict)
   ```

3. Fill:
   * `pulse_id` later (not in rec_dict)
   * `worker_id` from its own config
   * `seq` from internal counter

**schema_map is the only source of truth** for:

* success logic (EOMM threshold)
* params/context structure
* error_metrics content

PulseWriter just pipes those into ForgeRecord.

---

## 9. Threading & Concurrency

* Exactly **one** PulseWriter instance should own a particular WAL + BinaryLog pair.

* Multiple producer threads (e.g., TrueVision worker threads) may call `submit_window`, but:
  * Either:
    * they send windows via a thread-safe queue to a single PulseWriter thread, or
    * `submit_window` itself is synchronized (acquires `buffer_lock`).

* `flush()` and `close()` must also grab the same lock used by `submit_window` to avoid races.

Lock separation:

* `buffer_lock` → protects in-memory pulse buffer operations (submit & flush).
* `pulse_lock` → may be the same as `buffer_lock` or separate; but WAL/BinaryLog commits must be serialized.

---

## 10. Audit Checklist

When implementing `pulse_writer.py`, verify:

- [ ] Only one PulseWriter per WAL + BinaryLog pair
- [ ] `pulse_id` strictly monotonic (no skips, no duplicates in normal operation)
- [ ] `seq` increments per ForgeRecord (across pulse boundaries)
- [ ] `worker_id` is constant for this PulseWriter instance
- [ ] All three thresholds respected (max_records, max_bytes, max_age_ms)
- [ ] `buffer_first_ts` set on first window, cleared on flush
- [ ] Snapshot buffer before clearing (prevent race)
- [ ] WAL write happens before BinaryLog append (never reverse order)
- [ ] If WAL write fails, BinaryLog is not touched
- [ ] If BinaryLog append fails, WAL entry is already durable (recoverable)
- [ ] `pulse_lock` prevents concurrent flush/commit
- [ ] `close()` flushes remaining buffered records
- [ ] After `close()`, `submit_window` is disabled
- [ ] CheckpointManager notified after each successful commit
- [ ] No field mapping logic inside PulseWriter (schema_map only)
- [ ] All ForgeRecord fields filled correctly (pulse_id, worker_id, seq)

---

## 11. Next Steps

This PulseWriter specification is now **LOCKED**.

The complete durability stack is now spec'd:

```text
TrueVision → schema_map → PulseWriter → WAL → BinaryLog → Query
```

Next options:

* **Option A**: Implement WAL (wal_writer.py, wal_reader.py, wal_replayer.py)
* **Option B**: Implement PulseWriter (pulse_writer.py)
* **Option C**: Implement schema_map converter (schema_map.py: truevision_window_to_record_dict)

Recommended order:

1. WAL implementation (foundation for durability)
2. schema_map implementation (translation layer)
3. PulseWriter implementation (orchestrator)

---

**END OF PULSEWRITER SPECIFICATION**
