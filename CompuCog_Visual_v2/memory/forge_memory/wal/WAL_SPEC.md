# FORGE MEMORY SYSTEM - WAL SPECIFICATION

## Overview

**File:** `wal.bin`  
**Role:** Append-only journal of *pulses* (batches of ForgeRecords)  
**Guarantees:**

- Every record that reaches BinaryLog **must first** be fully written + fsync'd to WAL
- After a crash, any committed pulses in WAL are either:
  - fully replayable into BinaryLog, or
  - ignored cleanly (if partial/corrupt)
- WAL is periodically **checkpointed** and **truncated** to prevent unbounded growth

---

## WAL File Format

### Header (20 bytes, fixed at offset 0)

| Offset | Size | Field                    | Type   | Notes                              |
|--------|------|--------------------------|--------|------------------------------------|
| 0      | 4    | MAGIC                    | uint32 | `0x57414C00` (`"WAL\0"`)          |
| 4      | 2    | VERSION                  | uint16 | WAL schema version (`0x0001`)     |
| 6      | 2    | RESERVED                 | uint16 | Must be 0                         |
| 8      | 8    | LAST_CHECKPOINT_PULSE_ID | uint64 | Highest pulse fully checkpointed  |
| 16     | 4    | ENTRY_COUNT              | uint32 | Number of valid entries written   |

**Header rewrite triggers:**
- New entry appended (ENTRY_COUNT increments)
- Checkpoint written (LAST_CHECKPOINT_PULSE_ID changes)

---

### Entry Format (variable, starts at offset 20)

Each entry is self-contained with embedded checksum:

| Offset | Size | Field        | Type   | Notes                                                |
|--------|------|--------------|--------|------------------------------------------------------|
| 0      | 4    | ENTRY_LENGTH | uint32 | Total bytes (header+records+checksum)               |
| 4      | 8    | PULSE_ID     | uint64 | Pulse sequence ID (monotonic)                       |
| 12     | 4    | RECORD_COUNT | uint32 | Number of ForgeRecords in this entry                |
| 16     | N    | RECORD_DATA  | bytes  | Concatenated ForgeRecord binaries                   |
| 16+N   | 4    | CHECKSUM     | uint32 | CRC32 of entry_body                                 |

**Checksum calculation:**
```
entry_body = PULSE_ID (8) + RECORD_COUNT (4) + RECORD_DATA (N)
ENTRY_LENGTH = 4 + len(entry_body) + 4
CHECKSUM = CRC32(entry_body)
```

**RECORD_DATA encoding:**
- Concatenated ForgeRecords: `record_1 || record_2 || ... || record_k`
- Each ForgeRecord is self-framing (includes RECORD_LENGTH in header)
- To parse: read ForgeRecord header → get length → deserialize → advance → repeat

---

## WAL Writer Protocol

### Initialization

**If file doesn't exist:**
1. Create with modest initial size (few MB)
2. Write header:
   - MAGIC = `0x57414C00`
   - VERSION = `1`
   - RESERVED = `0`
   - LAST_CHECKPOINT_PULSE_ID = `0`
   - ENTRY_COUNT = `0`
3. fsync header

**If file exists:**
1. Open with MMapFile (mode='r+')
2. Read & validate header (MAGIC, VERSION)
3. Prepare for append at current_offset

---

### write_entry(pulse_id, records) → entry_offset

**Atomic append protocol:**

1. **Acquire lock** (single writer)
2. Serialize all records: `record_data = concat(r.serialize(string_dict) for r in records)`
3. Build entry_body: `struct.pack("<Q I", pulse_id, record_count) + record_data`
4. Compute checksum: `checksum = CRC32(entry_body)`
5. Build entry_data: `struct.pack("<I", entry_length) + entry_body + checksum_bytes`
6. Write entry_data at current_offset
7. Update:
   - `current_offset += entry_length`
   - `ENTRY_COUNT += 1` in header
8. Rewrite WAL header with new ENTRY_COUNT
9. **fsync** to guarantee durability
10. Release lock
11. Return entry_offset

**Invariants:**
- Entry never partially visible as "valid"
- If crash during write: incomplete entry detected and ignored
- If crash after write but before header update: scan discovers entry anyway

---

## WAL Reader & Recovery

### Header Load

1. Read first 20 bytes
2. Parse MAGIC, VERSION, RESERVED, LAST_CHECKPOINT_PULSE_ID, ENTRY_COUNT
3. If MAGIC invalid or file too short → treat WAL as empty

---

### Entry Scan Algorithm

```python
offset = 20
entries = []

while offset + 12 <= file_size:
    # Read ENTRY_LENGTH
    entry_length = read_uint32(offset)
    
    if entry_length == 0:
        break  # Invalid
    
    if offset + entry_length > file_size:
        # Partial trailing entry from crash
        break  # Ignore this and anything beyond
    
    # Read entire entry
    entry_bytes = read_bytes(offset, entry_length)
    entry_body = entry_bytes[4:-4]
    checksum_stored = unpack_uint32(entry_bytes[-4:])
    
    # Verify checksum
    checksum_calc = CRC32(entry_body)
    if checksum_calc != checksum_stored:
        # Corrupted trailing entry
        break
    
    # Parse
    pulse_id, record_count = unpack("<Q I", entry_body[:12])
    record_data = entry_body[12:]
    
    entries.append((pulse_id, record_count, record_data, offset, entry_length))
    offset += entry_length

return entries, last_good_offset
```

**Key behavior:**
- Any corruption/truncation stops at that entry
- Previous entries remain valid
- ENTRY_COUNT is advisory; scan determines real valid entries

---

## Recovery Protocol (Startup after crash)

### Metadata Sources

- **Metadata file**: last known committed `checkpoint_pulse_id`
- **BinaryLog** (`records.bin`): durable record store
- **wal.bin**: journal of pulses written before crash

### Recovery Steps

1. Open WAL in read mode (WALReader)
2. Load WAL header: `wal_last_checkpoint = LAST_CHECKPOINT_PULSE_ID`
3. Load system Metadata: `meta_checkpoint = metadata.checkpoint_pulse_id`
4. Determine checkpoint: `checkpoint = max(wal_last_checkpoint, meta_checkpoint)`
5. Scan WAL entries
6. For each valid entry `(pulse_id, record_count, record_data, ...)`:
   - If `pulse_id <= checkpoint`: skip (already durable)
   - Else:
     - Parse record_data into ForgeRecords
     - Append batch to BinaryLog: `binary_log.append_batch(records)`
     - Update indexes if present
7. After all entries applied:
   - Update Metadata: `checkpoint_pulse_id = last_applied_pulse_id`
   - Optionally checkpoint + truncate WAL

**Result:**
- BinaryLog caught up with all fully persisted pulses
- Partial entries ignored
- No duplicate replay

---

## Checkpoint & Truncation

### Checkpoint Policy

Trigger checkpoint when:
```
pulse_count_since_checkpoint >= 1000
OR
bytes_since_checkpoint >= 100 MB
```

### Checkpoint Procedure

1. Ensure all pulses ≤ `pulse_id` written to BinaryLog + indexes
2. Call `wal_writer.write_checkpoint(pulse_id)`:
   - Update `LAST_CHECKPOINT_PULSE_ID` in WAL header
   - fsync
3. Update Metadata: `metadata.update_checkpoint(pulse_id)`
4. Truncate WAL: `wal_writer.truncate(keep_after_pulse_id=pulse_id)`:
   - Scan entries
   - Find first entry with `entry.pulse_id >= keep_after_pulse_id`
   - Copy those entries to `wal_new.bin`
   - Write new header with updated checkpoint
   - Atomically replace `wal.bin` with `wal_new.bin`
   - fsync directory
5. Reset counters:
   - `pulse_count_since_checkpoint = 0`
   - `bytes_since_checkpoint = 0`

---

## Crash Scenarios

### A: Crash before writing any entry
- WAL header exists, ENTRY_COUNT=0
- Recovery scan sees 0 entries → nothing to replay

### B: Crash during entry write
- ENTRY_LENGTH partially written or body truncated
- Recovery: `offset + ENTRY_LENGTH > file_size` OR checksum mismatch
- Stops scanning, ignores partial entry
- All previous entries safe

### C: Crash after entry write, before header update
- Entry bytes fully valid, but ENTRY_COUNT not incremented
- Recovery scans by length (ignores ENTRY_COUNT)
- Entry discovered and replayed once

### D: Crash after checkpoint, before truncation
- LAST_CHECKPOINT_PULSE_ID updated
- Old entries still present in WAL
- Recovery uses checkpoint to skip old entries
- Next truncate() removes them

---

## Invariants

1. **Durability**: Every entry in BinaryLog first exists in WAL + fsync'd
2. **Atomicity**: Entries are all-or-nothing (checksum validates completeness)
3. **Ordering**: Pulse IDs monotonic, replay maintains order
4. **Idempotency**: Recovery never replays same pulse twice
5. **Crash-safety**: Partial writes detected and ignored, valid entries preserved
6. **Bounded growth**: Checkpoint + truncation keeps WAL size reasonable

---

## Constants

```python
MAGIC_WAL = 0x57414C00  # "WAL\0"
WAL_VERSION = 0x0001
WAL_HEADER_SIZE = 20  # bytes
WAL_CHECKPOINT_THRESHOLD_PULSES = 1000
WAL_CHECKPOINT_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB
```
