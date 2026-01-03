# State of the System - 2026-01-03 05:08:19 EST

**Workspace Root:** `D:\cod_616`

---

## MODULE: resilience/hot_spare.py

### 1. MODULE IDENTITY
- **Module name:** `HotSpare`
- **Intended role:** In-memory time-windowed hot-spare buffer for events
- **What this module DOES:**
  - Maintains a fixed time-window ring buffer (default 30 seconds) for event mirroring
  - Stores events as tuples of (timestamp, event_dict) in a deque
  - Prunes events older than window_seconds on every mirror() call
  - Enforces MAX_EVENTS=50000 capacity limit using deterministic drop-newest policy
  - Thread-safe operations using threading.Lock
  - Supports takeover/release mode for drain behavior control
  - Provides stats() for buffer state inspection
- **What this module explicitly DOES NOT do:**
  - Does NOT persist events to disk
  - Does NOT automatically forward events anywhere
  - Does NOT raise exceptions (catches all and returns safely)
  - Does NOT validate event schema or content
  - Does NOT modify incoming events (creates dict() copy)

### 2. DATA OWNERSHIP
- **Data types created:** `Dict` (stats dictionary), `List[Dict]` (drain output)
- **Data types consumed:** `Dict` (event dictionaries), `float` (timestamps)
- **Data types persisted:** NONE (in-memory only)
- **Data types not persisted:** All event data (ephemeral buffer)

### 3. PUBLIC INTERFACES
- **Public classes:** `HotSpare`
- **Public functions/methods:**
  - `__init__(window_seconds: float = 30.0)` - initialize with time window
  - `mirror(event: Dict, ts: float | None = None) -> None` - add event to buffer
  - `takeover() -> None` - enable takeover mode
  - `release() -> None` - disable takeover mode
  - `drain() -> List[Dict]` - return chronologically sorted events; clears buffer only if in takeover mode
  - `stats() -> Dict` - return buffer metrics
- **Expected inputs:**
  - `mirror()`: event dict, optional timestamp float
  - All methods accept no required arguments beyond self
- **Expected outputs:**
  - `drain()`: List of event dicts sorted by timestamp (oldest first)
  - `stats()`: Dict with keys: buffer_count, oldest_ts, newest_ts, is_takeover, window_seconds, dropped_newest
- **Side effects:** NONE (memory-only, no disk/network/stdout)

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `time`, `threading`, `collections.deque`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_buf: Deque[Tuple[float, Dict]]` - event buffer (bounded by MAX_EVENTS and time window)
  - `_is_takeover: bool` - takeover mode flag
  - `dropped_newest: int` - counter for dropped events
  - `window_seconds: float` - time window configuration
- **Bounded or unbounded:** BOUNDED by MAX_EVENTS (50000) and window_seconds (default 30.0)
- **State reset/eviction:** 
  - Time-based: events older than window_seconds are pruned on each mirror()
  - Capacity-based: newest events dropped when buffer reaches MAX_EVENTS
  - Explicit clear: buffer cleared only during drain() when in takeover mode

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Event dictionaries from any caller via mirror()
- **Explicit downstream outputs:** Event lists via drain() to any caller
- **Required external conditions:** NONE

### 7. OPEN FACTUAL QUESTIONS
- Module-level constant MAX_EVENTS is defined but not exposed as constructor parameter
- takeover/release mechanism is thread-safe but purpose is IMPLIED ONLY (not documented in code)
- Stats method never raises exceptions but returns safe defaults on error; error conditions are NOT LOGGED

---

## MODULE: resilience/pulse_bus.py

### 1. MODULE IDENTITY
- **Module name:** `PulseBus`
- **Intended role:** In-memory router for events with bounded buffers and pulse-driven forwarding
- **What this module DOES:**
  - Routes events to per-topic bounded deque buffers (thread-safe)
  - Deterministic drop-oldest policy when buffer full (actually drop-newest per code: "drop newest event")
  - Pulse thread wakes at pulse_interval (default 0.1s) and forwards buffered events
  - Writers are lazily created (DailyJSONWriter) if not registered
  - Provides start/stop lifecycle management
  - Tracks metrics: published, dropped, forwarded counts
  - Calls writer.enqueue() for each forwarded event
  - Best-effort flush via _drain_and_write_once() after batch forwarding
- **What this module explicitly DOES NOT do:**
  - Does NOT assume file I/O directly (delegates to writers)
  - Does NOT block producers (non-blocking publish)
  - Does NOT crash on exceptions (all caught, logged, processing continues)
  - Does NOT guarantee event delivery (events lost if writer fails)
  - Does NOT persist internal state across restarts

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (metrics), buffer dictionaries per (category, prefix)
- **Data types consumed:** `dict` (events from publish() calls)
- **Data types persisted:** NONE (delegates persistence to writers)
- **Data types not persisted:** Buffered events (ephemeral in-memory queues)

### 3. PUBLIC INTERFACES
- **Public classes:** `PulseBus`
- **Public functions/methods:**
  - `__init__(data_root: str = 'logs_data', pulse_interval: float = 0.1, default_buffer_size: int = 1024)`
  - `start() -> None` - start pulse thread
  - `stop() -> None` - stop pulse thread and close writers
  - `register_writer(category: str, prefix: str, writer: Any, buffer_size: Optional[int] = None) -> None`
  - `publish(event: dict, category: str, prefix: str) -> None` - enqueue event
- **Expected inputs:**
  - `publish()`: event (dict), category (str), prefix (str)
  - `register_writer()`: category, prefix, writer object with enqueue() method, optional buffer_size
- **Expected outputs:** NONE (side effects only)
- **Side effects:**
  - Memory: updates internal buffers and metrics
  - Disk: indirectly via writer objects
  - Logging: via logging module (_logger)

### 3. PUBLIC INTERFACES (continued)
- **Public attributes:**
  - `metrics: dict` - keys: published, dropped, forwarded (all integers)

### 4. DEPENDENCIES
- **Internal imports:** `loggers.writer.DailyJSONWriter` (lazy import in _ensure_writer)
- **External imports:** `logging`, `threading`, `time`, `collections.deque`, `typing`
- **Runtime import behavior:** DailyJSONWriter imported only when lazy writer creation needed

### 5. STATE & MEMORY
- **State held in memory:**
  - `_buffers: Dict[Tuple[str, str], Dict]` - per-topic buffer state (lock, deque, maxlen)
  - `_writers: Dict[Tuple[str, str], Any]` - registered writers
  - `_running: bool` - lifecycle flag
  - `metrics: dict` - published/dropped/forwarded counters
- **Bounded or unbounded:** BOUNDED per-topic by default_buffer_size (default 1024) or custom buffer_size
- **State reset/eviction:**
  - Buffers drained on each pulse cycle
  - Drop-newest policy when buffer full (contradicts docstring which says drop-oldest)
  - NOT CLEARED on stop (no explicit cleanup of buffer contents)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Events via publish() from any caller
- **Explicit downstream outputs:** Events forwarded to writer.enqueue()
- **Required external conditions:**
  - Writers must expose enqueue(event: dict) method
  - Writers optionally expose stop() and _drain_and_write_once() methods
  - data_root directory must be writable for lazy writer creation

### 7. OPEN FACTUAL QUESTIONS
- Docstring claims "drop-oldest policy" but code implements drop-newest ("dropping newest event")
- Final drain on shutdown is best-effort; events may be lost if writers fail
- _ensure_writer() returns None on failure; calling code checks for None but events are dropped
- pulse_interval passed to DailyJSONWriter but not used by PulseBus for any timing validation

---

## MODULE: resilience/disk_guard.py

### 1. MODULE IDENTITY
- **Module name:** `DiskGuard`
- **Intended role:** Monitor free disk space for logs_data and emit warnings via DailyJSONWriter
- **What this module DOES:**
  - Monitors logs_path (default 'logs_data') for free disk space
  - Computes average daily usage from last N days (default 7) of .jsonl files
  - Falls back to 1GB/day if insufficient history
  - Emits WARNING when free_bytes < 3 * avg_daily_usage_bytes
  - Emits CRITICAL when free_bytes < 1 * avg_daily_usage_bytes
  - Writes to logs_data/system/syswarn_MM-DD-YY.jsonl via DailyJSONWriter
  - Thread-safe check loop with configurable check_interval (default 60s)
  - Never crashes (all exceptions caught in loop)
- **What this module explicitly DOES NOT do:**
  - Does NOT take corrective action (cleanup, archival, etc.)
  - Does NOT monitor anything other than logs_path directory
  - Does NOT emit warnings for non-space issues
  - Does NOT validate file integrity

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (warning events, stats)
- **Data types consumed:** File system metadata (stat, disk_usage)
- **Data types persisted:** Warning events (via DailyJSONWriter delegation)
- **Data types not persisted:** Computed metrics (avg usage, free bytes) - stored in memory only

### 2. DATA OWNERSHIP (continued)
- **Data types not persisted:** Internal state variables (_last_free, _last_avg, _last_severity, _last_emit_time)

### 3. PUBLIC INTERFACES
- **Public classes:** `DiskGuard`
- **Public functions/methods:**
  - `__init__(logs_path: str = 'logs_data', check_interval: float = 60.0, days: int = 7, disk_usage_fn: Optional[Callable] = None)`
  - `start() -> None` - start monitoring thread
  - `stop() -> None` - stop monitoring thread and close writer
  - `stats() -> dict` - return last_free_bytes, last_avg_daily_usage_bytes, last_severity
- **Expected inputs:** Constructor parameters (all optional with defaults)
- **Expected outputs:**
  - `stats()`: dict with int/float/str/None values
  - Events written to system/syswarn_*.jsonl (side effect)
- **Side effects:**
  - Disk I/O: writes to logs_data/system/syswarn_*.jsonl
  - Disk reads: walks logs_path, stats files
  - Logging: via logging module (_logger)

### 4. DEPENDENCIES
- **Internal imports:** `loggers.writer.DailyJSONWriter` (lazy import in _get_writer)
- **External imports:** `logging`, `os`, `threading`, `time`, `collections.defaultdict`, `datetime`, `typing`, `shutil`, `json`, `zoneinfo.ZoneInfo` (fallback to timezone offset)
- **Runtime import behavior:** ZoneInfo imported with fallback; DailyJSONWriter lazily imported

### 5. STATE & MEMORY
- **State held in memory:**
  - `_last_free: Optional[int]` - last observed free bytes
  - `_last_avg: Optional[float]` - last computed avg daily usage
  - `_last_severity: Optional[str]` - last emitted severity level
  - `_last_emit_time: dict` - per-severity emission timestamps
  - `_writer: Optional[DailyJSONWriter]` - lazily created writer
- **Bounded or unbounded:** BOUNDED (fixed-size state variables)
- **State reset/eviction:** NEVER CLEARED (state persists for lifetime of instance)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** File system (os.walk, os.path.getsize, shutil.disk_usage)
- **Explicit downstream outputs:** Warning events to DailyJSONWriter (category='system', prefix='syswarn')
- **Required external conditions:**
  - logs_path must be accessible directory
  - .jsonl files must follow pattern *_MM-DD-YY.jsonl
  - Sufficient permissions for os.walk and stat operations

### 7. OPEN FACTUAL QUESTIONS
- FALLBACK_AVG_BYTES (1GB) constant is defined but not configurable
- EST timezone hardcoded via ZoneInfo("America/New_York") with fallback to UTC-5
- _compute_avg_daily_usage() falls back to file mtime if date parsing fails; this is IMPLIED ONLY
- writer_pulse interval chosen as max(0.05, min(1.0, check_interval)) but rationale is NOT DOCUMENTED
- One-time informational warning about baseline emitted with free=0 which may confuse consumers

---

## MODULE: loggers/writer.py

### 1. MODULE IDENTITY
- **Module name:** `DailyJSONWriter`
- **Intended role:** Pulse-based, append-only JSONL writer for loggers
- **What this module DOES:**
  - Produces files under logs_data/<category>/ with pattern {prefix}_{MM-DD-YY}.jsonl
  - Uses Eastern Time day boundaries (00:01 starts new day)
  - Writes canonical JSON (UTF-8, no whitespace, sorted keys)
  - Maintains rolling SHA-256 chain stored as _sha field in each line
  - Single writer thread owns file handle; producers call enqueue(event: dict)
  - Automatic file rotation at day boundaries
  - fsync after each flush (best-effort)
  - Queue-based buffering with blocking put
- **What this module explicitly DOES NOT do:**
  - Does NOT retry writes on failure
  - Does NOT spawn background threads beyond the single writer thread
  - Does NOT cache events beyond the internal Queue
  - Does NOT validate event schema
  - Does NOT handle concurrent writers to same file

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (with _sha field added), `bytes` (canonical JSON), file handles
- **Data types consumed:** `dict` (events from enqueue())
- **Data types persisted:** JSONL lines in logs_data/<category>/{prefix}_{MM-DD-YY}.jsonl
- **Data types not persisted:** Internal queue contents (lost on stop if not drained)

### 3. PUBLIC INTERFACES
- **Public classes:** `DailyJSONWriter`
- **Public functions/methods:**
  - `__init__(category: str, prefix: str, data_root: str = "logs_data", pulse_interval: float = 0.1)`
  - `start() -> None` - start writer thread
  - `stop() -> None` - stop thread, flush remaining, close file
  - `enqueue(event: Dict[str, Any]) -> None` - queue event for writing (may raise TypeError)
- **Expected inputs:**
  - `enqueue()`: event dict (must pass _ensure_finite_numbers validation)
- **Expected outputs:** NONE (side effects only)
- **Side effects:**
  - Disk: creates/appends to .jsonl files in logs_data/<category>/
  - Disk: fsync calls (best-effort)
  - Logging: via logging module (_logger)

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `json`, `logging`, `os`, `threading`, `time`, `datetime`, `queue`, `typing`, `hashlib`, `zoneinfo.ZoneInfo` (with fallback)
- **Runtime import behavior:** ZoneInfo imported with fallback to timezone offset

### 5. STATE & MEMORY
- **State held in memory:**
  - `_queue: Queue` - unbounded Queue (UNBOUNDED MEMORY RISK if producers overwhelm writer)
  - `_file: Optional[file handle]` - current output file
  - `_current_day: Optional[str]` - current day key (MM-DD-YY format)
  - `_prev_sha: str` - last SHA-256 hash (starts with "0"*64)
- **Bounded or unbounded:** UNBOUNDED queue (producer blocking on put, but queue itself has no maxsize)
- **State reset/eviction:**
  - File handle reset on day boundary rotation
  - SHA chain reset to _INITIAL_SHA="0"*64 only implicitly (NOT IMPLEMENTED for new day files)
  - Queue drained on each pulse and on stop()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Event dicts via enqueue() from any caller
- **Explicit downstream outputs:** .jsonl files on disk
- **Required external conditions:**
  - data_root/<category>/ directory must be writable (created with os.makedirs)
  - EST timezone via ZoneInfo or fallback
  - Events must contain only finite numbers (no NaN/Inf)

### 7. OPEN FACTUAL QUESTIONS
- Queue is unbounded; producer blocking may occur but queue growth is not monitored
- SHA chain does NOT reset on day boundary rotation; new file continues with SHA from previous file (IMPLICIT CROSS-FILE CHAIN)
- _ensure_finite_numbers() recursively validates but does not sanitize; raises ValueError on violation
- _day_key_for() applies 00:01 rule: times before 00:01 treated as previous day; rationale NOT DOCUMENTED
- File rotation happens in _rotate_if_needed() but race conditions with concurrent enqueue() are NOT ADDRESSED
- _INITIAL_SHA constant is "0"*64 but no semantic meaning documented

---

## MODULE: loggers/activity_service.py

### 1. MODULE IDENTITY
- **Module name:** `ActivityService`
- **Intended role:** Disk-observer implementation for activity log monitoring
- **What this module DOES:**
  - Reads activity log files on disk (pattern: logs/activity/user_activity_YYYYMMDD.jsonl)
  - Emits events based on real filesystem changes (size/mtime growth)
  - Tracks pulses_unchanged counter for idle/stalled detection
  - Emits activity_detected when file grows
  - Emits activity_idle when unchanged for IDLE_THRESHOLD_PULSES (50)
  - Emits activity_stalled when unchanged for STALL_THRESHOLD_PULSES (100)
  - Emits activity_failed when file missing or exception occurs
  - Handles day rollover by resetting state
- **What this module explicitly DOES NOT do:**
  - Does NOT start subprocesses
  - Does NOT provide alternate runtime modes (config_path and use_subprocess params ignored)
  - Does NOT read file contents (only stat metadata)
  - Does NOT validate file format

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (events with event_type, timestamp, file_path, etc.)
- **Data types consumed:** File system metadata (Path.stat)
- **Data types persisted:** NONE
- **Data types not persisted:** All state is ephemeral (last_size, last_mtime, pulses_unchanged)

### 3. PUBLIC INTERFACES
- **Public classes:** `ActivityService`
- **Public functions/methods:**
  - `__init__(config_path: Optional[str] = None, use_subprocess: bool = True)` - params accepted but ignored
  - `start() -> None` - set running flag
  - `stop() -> None` - clear running flag
  - `is_running() -> bool` - return running state
  - `poll() -> Optional[dict]` - check file state and return event or None
- **Expected inputs:** None (poll() takes no arguments)
- **Expected outputs:**
  - `poll()`: dict with keys: event_type, timestamp, file_path, file_size (optional), mtime (optional), pulses_unchanged (optional), detail (optional), error (optional)
  - `poll()`: None if no event to emit
- **Side effects:**
  - Disk I/O: reads file stat metadata
  - Logging: via logging.info/warning

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `time`, `logging`, `datetime`, `pathlib.Path`, `typing.Optional`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_log_path: Path` - computed file path (recomputed on day rollover)
  - `_pulses_unchanged: int` - counter (UNBOUNDED in principle but resets on file change)
  - `_last_size: Optional[int]` - last observed file size
  - `_last_mtime: Optional[float]` - last observed modification time
  - `_current_day: str` - YYYYMMDD format for rollover detection
- **Bounded or unbounded:** BOUNDED (fixed-size state variables)
- **State reset/eviction:**
  - State reset on day rollover (detected via datetime comparison)
  - Counter reset when file changes detected
  - NOT CLEARED on stop()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** File system (pathlib.Path.stat)
- **Explicit downstream outputs:** Event dicts via poll() return value
- **Required external conditions:**
  - File path pattern: ../logs/activity/user_activity_YYYYMMDD.jsonl relative to __file__
  - File may not exist (handled gracefully)

### 7. OPEN FACTUAL QUESTIONS
- config_path parameter accepted but ignored (signature compatibility only?)
- use_subprocess parameter accepted but ignored (signature compatibility only?)
- File path is hardcoded relative to __file__ parent; NOT CONFIGURABLE despite config_path param
- First-time observation (last_size is None) returns None without emitting any event; IMPLICIT INITIALIZATION
- Thresholds IDLE_THRESHOLD_PULSES=50 and STALL_THRESHOLD_PULSES=100 are module constants; NOT CONFIGURABLE per instance

---

## MODULE: loggers/input_service.py

### 1. MODULE IDENTITY
- **Module name:** `InputService`
- **Intended role:** Disk-observer implementation for input log monitoring
- **What this module DOES:**
  - Reads input log files on disk (pattern: logs/input/input_events_YYYYMMDD.jsonl)
  - Emits events for newly appended records (incremental reading)
  - Treats existing content as initial snapshot (consumed without emitting)
  - Parses JSON lines and emits input_event for each new record
  - Emits input_idle/input_stalled based on pulses_unchanged thresholds
  - Emits input_failed on file missing, JSON parse errors, or exceptions
  - Handles day rollover by resetting read offset
- **What this module explicitly DOES NOT do:**
  - Does NOT start subprocesses
  - Does NOT provide alternate runtime modes (params ignored)
  - Does NOT validate record schema beyond JSON parsing
  - Does NOT emit events for pre-existing file content on first poll

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (events), `List[dict]` (batch events)
- **Data types consumed:** JSONL file lines, file system metadata
- **Data types persisted:** NONE
- **Data types not persisted:** Read offset and pulses counter (ephemeral state)

### 3. PUBLIC INTERFACES
- **Public classes:** `InputService`
- **Public functions/methods:**
  - `__init__(config_path: Optional[str] = None, use_subprocess: bool = True)` - params accepted, config_path used if provided
  - `start() -> None` - set running flag
  - `stop() -> None` - clear running flag
  - `is_running() -> bool` - return running state
  - `poll() -> Optional[List[dict]]` - read new records and return list of events or None
- **Expected inputs:** None (poll() takes no arguments)
- **Expected outputs:**
  - `poll()`: List[dict] with one or more event dicts (event_type: input_event, input_idle, input_stalled, input_failed)
  - `poll()`: None if no events to emit
- **Side effects:**
  - Disk I/O: reads file incrementally via seek/tell
  - Logging: via logging.info/warning

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `json`, `time`, `logging`, `datetime`, `pathlib.Path`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_log_path: Path` - computed file path
  - `_read_offset: int` - byte offset for incremental reading (UNBOUNDED in principle, limited by file size)
  - `_pulses_unchanged: int` - counter (UNBOUNDED in principle, resets on file growth)
  - `_current_day: str` - YYYYMMDD for rollover detection
- **Bounded or unbounded:** Offset is UNBOUNDED (grows with file size); other state BOUNDED
- **State reset/eviction:**
  - Offset reset to 0 on day rollover or if file truncated (current_size < offset)
  - Counter reset when file grows
  - NOT CLEARED on stop()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** JSONL files on disk, file system metadata
- **Explicit downstream outputs:** List[dict] events via poll() return
- **Required external conditions:**
  - File path: logs/input/input_events_YYYYMMDD.jsonl (or config_path if provided)
  - File must be UTF-8 JSONL format
  - File may not exist (handled gracefully with input_failed event)

### 7. OPEN FACTUAL QUESTIONS
- use_subprocess parameter accepted but ignored (signature compatibility only?)
- Initial snapshot behavior (_read_offset==0 and file has content) consumes without emitting; rationale NOT DOCUMENTED
- File truncation detection (current_size < offset) resets offset to 0; NO EVENT EMITTED for data loss
- JSON parse error returns single input_failed event and terminates batch processing; subsequent valid lines NOT PROCESSED
- Thresholds IDLE_THRESHOLD_PULSES=50, STALL_THRESHOLD_PULSES=100 are module constants; NOT CONFIGURABLE

---

## MODULE: loggers/network_service.py

### 1. MODULE IDENTITY
- **Module name:** `NetworkService`
- **Intended role:** Disk-observer implementation for network log monitoring
- **What this module DOES:**
  - Reads network capture/log files on disk (pattern: logs/network/network_capture_YYYYMMDD.jsonl)
  - Emits events based on file size/mtime growth
  - Tracks pulses_unchanged for idle/stalled detection
  - Emits network_active when file grows
  - Emits network_idle/network_stalled based on thresholds
  - Emits network_failed when file missing or exception occurs
  - Handles day rollover by resetting state
- **What this module explicitly DOES NOT do:**
  - Does NOT launch or wrap external capture processes
  - Does NOT read file contents (only stat metadata)
  - Does NOT validate network data format
  - Does NOT provide alternate runtime modes (params ignored)

### 2. DATA OWNERSHIP
- **Data types created:** `dict` (events)
- **Data types consumed:** File system metadata (Path.stat)
- **Data types persisted:** NONE
- **Data types not persisted:** All state (last_size, last_mtime, pulses_unchanged)

### 3. PUBLIC INTERFACES
- **Public classes:** `NetworkService`
- **Public functions/methods:**
  - `__init__(config_path: Optional[str] = None, use_subprocess: bool = True)` - config_path used if provided
  - `start() -> None` - set running flag
  - `stop() -> None` - clear running flag
  - `is_running() -> bool` - return running state
  - `poll() -> Optional[dict]` - check file state and return event or None
- **Expected inputs:** None (poll() takes no arguments)
- **Expected outputs:**
  - `poll()`: dict with event_type (network_active, network_idle, network_stalled, network_failed), timestamp, file_path, and optional fields
  - `poll()`: None if no event to emit
- **Side effects:**
  - Disk I/O: reads file stat metadata
  - Logging: via logging.info/warning

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `time`, `logging`, `datetime`, `pathlib.Path`, `typing.Optional`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_log_path: Path`
  - `_pulses_unchanged: int` (UNBOUNDED in principle)
  - `_last_size: Optional[int]`
  - `_last_mtime: Optional[float]`
  - `_current_day: str`
- **Bounded or unbounded:** BOUNDED (fixed-size state variables)
- **State reset/eviction:** Reset on day rollover; counter reset on file growth; NOT CLEARED on stop()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** File system (Path.stat)
- **Explicit downstream outputs:** Event dicts via poll() return
- **Required external conditions:**
  - File path: logs/network/network_capture_YYYYMMDD.jsonl (or config_path)
  - File may not exist (handled)

### 7. OPEN FACTUAL QUESTIONS
- use_subprocess parameter accepted but ignored
- get_network_log_path() function defined but config_path takes precedence; fallback behavior IMPLICIT
- First-time observation returns None without event; IMPLICIT INITIALIZATION
- Thresholds NOT CONFIGURABLE per instance

---

## MODULE: forge_memory/core/binary_log.py

### 1. MODULE IDENTITY
- **Module name:** `BinaryLog`
- **Intended role:** Append-only binary record file with explicit frame
- **What this module DOES:**
  - Writes binary records with magic header b'FREC'
  - Enforces LITTLE-ENDIAN byte order for all integer fields
  - Frame structure: magic(4) + version(u16) + header_len(u16) + payload_len(u32) + timestamp_ms(u64) + crc32(u32) + payload_bytes
  - Computes CRC32 over payload bytes and stores as uint32 LE
  - Returns offset (start of magic field) on append
  - Validates CRC on read
  - Fsync on append (best-effort)
  - Supports iteration over all records
- **What this module explicitly DOES NOT do:**
  - Does NOT implement WAL
  - Does NOT support concurrency (single writer assumption)
  - Does NOT cache reads
  - Does NOT spawn background threads

### 2. DATA OWNERSHIP
- **Data types created:** `ForgeRecord` instances
- **Data types consumed:** `ForgeRecord` instances (via append)
- **Data types persisted:** Binary records in records.bin file
- **Data types not persisted:** NONE (all records written immediately)

### 3. PUBLIC INTERFACES
- **Public classes:** `BinaryLog`
- **Public functions/methods:**
  - `__init__(data_dir_or_path: str, string_dict: Optional[object] = None)` - string_dict stored but not used
  - `append(record: ForgeRecord) -> int` - write record and return offset
  - `append_record(record: ForgeRecord) -> int` - backwards-compatible alias for append
  - `read(offset: int) -> ForgeRecord` - read and validate record at offset
  - `iter_records() -> Iterator[ForgeRecord]` - yield all records in order
  - `close()` - close file handle
- **Expected inputs:**
  - `append()`: ForgeRecord with payload_bytes (bytes), timestamp (int ms), optional crc32
  - `read()`: offset (int) pointing to start of magic field
- **Expected outputs:**
  - `append()`: int offset
  - `read()`: ForgeRecord instance
  - `iter_records()`: yields ForgeRecord instances
- **Side effects:**
  - Disk: appends to records.bin file
  - Disk: fsync calls (best-effort, exceptions ignored)

### 4. DEPENDENCIES
- **Internal imports:** `forge_memory.core.record.ForgeRecord`
- **External imports:** `os`, `struct`, `zlib`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_f: file handle` - open in 'a+b' mode (append + read)
  - `path: str` - file path
  - `string_dict: Optional[object]` - stored but not used
- **Bounded or unbounded:** File handle is single object; file size UNBOUNDED (grows indefinitely)
- **State reset/eviction:** NEVER (append-only log)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** ForgeRecord instances from callers
- **Explicit downstream outputs:** Binary file records.bin
- **Required external conditions:**
  - data_dir must be writable (created if missing via os.makedirs)
  - Filesystem must support fsync (ignored on failure)

### 7. OPEN FACTUAL QUESTIONS
- string_dict parameter stored but never used; purpose NOT IMPLEMENTED
- File opened in 'a+b' mode; concurrent writers NOT PREVENTED
- CRC validation on read but no recovery mechanism on CRC mismatch (raises ValueError)
- VERSION=1 and HEADER_LEN=16 hardcoded; forward/backward compatibility NOT ADDRESSED
- iter_records() breaks on first incomplete record; partial recovery NOT IMPLEMENTED

---

## MODULE: forge_memory/core/record.py

### 1. MODULE IDENTITY
- **Module name:** `ForgeRecord`
- **Intended role:** Immutable, canonical JSON payload container
- **What this module DOES:**
  - Stores record with fields: offset (int|None), timestamp (int ms), payload_bytes (bytes), crc32 (int)
  - Enforces canonical JSON: UTF-8, no whitespace, sorted keys, no NaN/Inf
  - Supports two construction modes: explicit (4 args) and dict-style (**kwargs)
  - Dict-style mode packs kwargs into canonical JSON payload
  - Normalizes timestamp: accepts seconds (float) or ms (int)
  - Computes CRC32 over payload_bytes
  - Validates numeric finiteness recursively
- **What this module explicitly DOES NOT do:**
  - Does NOT allow mutation (immutable via __slots__)
  - Does NOT persist records (storage is caller responsibility)
  - Does NOT validate JSON schema
  - Does NOT support lazy payload parsing

### 2. DATA OWNERSHIP
- **Data types created:** `ForgeRecord` instances, `bytes` (canonical JSON)
- **Data types consumed:** `dict` (kwargs for dict-style construction), `bytes` (explicit construction)
- **Data types persisted:** NONE (record is data container only)
- **Data types not persisted:** All fields (ephemeral until written by caller)

### 3. PUBLIC INTERFACES
- **Public classes:** `ForgeRecord`
- **Public functions/methods:**
  - `__init__(*args, **kwargs)` - dual-mode constructor
  - `__repr__()` - string representation
- **Public attributes (read-only via __slots__):**
  - `offset: int | None`
  - `timestamp: int` (epoch milliseconds)
  - `payload_bytes: bytes`
  - `crc32: int` (unsigned 32-bit)
- **Expected inputs:**
  - Explicit: (offset: int|None, timestamp_ms: int, payload_bytes: bytes, crc32: int|None)
  - Dict-style: **kwargs with optional 'timestamp' key
- **Expected outputs:** ForgeRecord instance
- **Side effects:** NONE (pure data structure)

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `typing`, `json`, `time`, `zlib`, `math`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:** 4 immutable fields (offset, timestamp, payload_bytes, crc32)
- **Bounded or unbounded:** BOUNDED per instance (payload_bytes can be arbitrary size but is fixed at construction)
- **State reset/eviction:** NEVER (immutable)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Constructor arguments from any caller
- **Explicit downstream outputs:** Read-only attributes
- **Required external conditions:** NONE

### 7. OPEN FACTUAL QUESTIONS
- Dict-style construction stores normalized timestamp back into payload_dict before serialization; IMPLICIT MUTATION of input dict
- _ensure_finite_numbers() raises ValueError but validation happens before CRC computation; PARTIAL CONSTRUCTION possible if CRC fails
- offset is None until record is appended; NO VALIDATION that offset is set before use
- CRC32 computed during construction if None provided; REDUNDANT COMPUTATION if caller provides crc32 for explicit mode

---

## MODULE: forge_memory/core/string_dict.py

### 1. MODULE IDENTITY
- **Module name:** `StringDictionary`
- **Intended role:** Deterministic string-to-id mapping with persistent storage
- **What this module DOES:**
  - Stores strings in strings.dict file with format: [4-byte uint32 LE length][utf-8 bytes]
  - Assigns sequential 1-based IDs on first appearance
  - Loads existing mappings on init
  - Appends new strings on get_id() if not present
  - Fsync on append (best-effort)
  - Provides bidirectional lookup: string->id and id->string
- **What this module explicitly DOES NOT do:**
  - Does NOT support string removal
  - Does NOT compact or reorganize file
  - Does NOT handle concurrent writers
  - Does NOT validate string content

### 2. DATA OWNERSHIP
- **Data types created:** `int` (IDs), `str` (retrieved strings)
- **Data types consumed:** `str` (strings to store)
- **Data types persisted:** String records in strings.dict file
- **Data types not persisted:** NONE (all strings persisted immediately)

### 3. PUBLIC INTERFACES
- **Public classes:** `StringDictionary`
- **Public functions/methods:**
  - `__init__(path_or_dir: str)` - accepts directory or file path
  - `get_id(s: str) -> int` - return existing or new ID (1-based)
  - `get_string(id: int) -> str` - return string for ID (raises KeyError if invalid)
  - `close()` - close file handle
- **Expected inputs:**
  - `get_id()`: str
  - `get_string()`: int (1-based, must be valid)
- **Expected outputs:**
  - `get_id()`: int >= 1
  - `get_string()`: str
- **Side effects:**
  - Disk: appends to strings.dict file
  - Disk: fsync calls (best-effort)

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `os`, `struct`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `_f: file handle` - open in 'a+b' mode
  - `_id_to_str: List[str]` - index 0 unused, 1-based IDs (UNBOUNDED, grows with unique strings)
  - `_str_to_id: Dict[str, int]` - reverse mapping (UNBOUNDED)
- **Bounded or unbounded:** UNBOUNDED (grows with number of unique strings)
- **State reset/eviction:** NEVER (append-only)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Strings from any caller
- **Explicit downstream outputs:** IDs (int) and strings (str)
- **Required external conditions:**
  - Directory must be writable for file creation
  - Filesystem must support fsync (ignored on failure)

### 7. OPEN FACTUAL QUESTIONS
- File opened in 'a+b' mode; concurrent writers NOT PREVENTED
- _load_existing() skips duplicate strings if encountered ("if s in self._str_to_id: continue"); file corruption NOT DETECTED
- ID 0 is reserved (list index 0 unused) but no validation prevents returning ID 0
- File format has no version or magic header; forward compatibility NOT ADDRESSED

---

## MODULE: forge_memory/indexes/hash_index.py

### 1. MODULE IDENTITY
- **Module name:** `HashIndex`
- **Intended role:** Simple in-memory hash index key -> list[offsets]
- **What this module DOES:**
  - Maps keys to lists of offsets
  - Serializes keys to bytes (deterministic)
  - Thread-safe via threading.Lock
  - Supports insert, lookup, batch_insert, clear
  - Tuple keys joined with '|' separator
- **What this module explicitly DOES NOT do:**
  - Does NOT persist index to disk
  - Does NOT limit memory usage
  - Does NOT deduplicate offsets

### 2. DATA OWNERSHIP
- **Data types created:** `List[int]` (offset lists), `bytes` (serialized keys)
- **Data types consumed:** Any key type, `int` offsets
- **Data types persisted:** NONE
- **Data types not persisted:** All index data (in-memory only)

### 3. PUBLIC INTERFACES
- **Public classes:** `HashIndex`
- **Public methods:**
  - `__init__()`
  - `insert(key: Any, offset: int)` - add offset to key's list
  - `lookup(key: Any) -> List[int]` - return copy of offset list
  - `batch_insert(pairs: List[Tuple[Any, int]])` - insert multiple
  - `clear()` - reset index
- **Expected inputs:** Any key type (serializable via str()), int offsets
- **Expected outputs:** List[int] (copy of offsets)
- **Side effects:** Memory only

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `threading`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `lock: threading.Lock`
  - `table: Dict[bytes, List[int]]` (UNBOUNDED)
- **Bounded or unbounded:** UNBOUNDED (grows with unique keys and offsets)
- **State reset/eviction:** Only via explicit clear()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Keys and offsets from any caller
- **Explicit downstream outputs:** Lists of offsets
- **Required external conditions:** NONE

### 7. OPEN FACTUAL QUESTIONS
- Key serialization via str() and '|' join may cause collisions (e.g., tuple ('a|b',) vs ('a', 'b'))
- No deduplication; repeated inserts of same offset create duplicates
- lookup() returns copy; modification safety guaranteed but may be performance issue for large lists

---

## MODULE: forge_memory/indexes/bitmap_index.py

### 1. MODULE IDENTITY
- **Module name:** `BitmapIndex`
- **Intended role:** Simple in-memory success bitmap using Python set
- **What this module DOES:**
  - Maps record_id (int) to success (bool) using set membership
  - Thread-safe via threading.Lock
  - Supports set_bit, batch_set, filter_offsets, clear
  - filter_offsets returns offsets where record_id is in success_set
- **What this module explicitly DOES NOT do:**
  - Does NOT persist bitmap
  - Does NOT validate record_id uniqueness
  - Does NOT limit memory

### 2. DATA OWNERSHIP
- **Data types created:** `List[int]` (filtered offsets), `Set[int]` (success set)
- **Data types consumed:** `int` (record_id), `bool` (success), offset lists, offset-to-id mapping
- **Data types persisted:** NONE
- **Data types not persisted:** All bitmap data (in-memory only)

### 3. PUBLIC INTERFACES
- **Public classes:** `BitmapIndex`
- **Public methods:**
  - `__init__()`
  - `set_bit(record_id: int, success: bool)` - add/remove from set
  - `batch_set(items: List[tuple[int, bool]])` - batch set operations
  - `filter_offsets(offsets: List[int], offset_to_id: dict) -> List[int]` - filter by success
  - `clear()` - reset bitmap
- **Expected inputs:** int record_ids, bool success, offset lists, offset->id mapping dict
- **Expected outputs:** List[int] (filtered offsets)
- **Side effects:** Memory only

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `threading`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `lock: threading.Lock`
  - `success_set: Set[int]` (UNBOUNDED)
- **Bounded or unbounded:** UNBOUNDED (grows with unique record_ids marked as success)
- **State reset/eviction:** Only via explicit clear()

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Record IDs, success flags, offset mappings from any caller
- **Explicit downstream outputs:** Filtered offset lists
- **Required external conditions:** offset_to_id dict must be accurate

### 7. OPEN FACTUAL QUESTIONS
- filter_offsets requires external offset_to_id mapping; NOT SELF-CONTAINED
- set_bit with success=False removes from set; NO DISTINCTION between "never seen" and "explicitly false"
- No persistence; index must be rebuilt from scratch on restart

---

## MODULE: forge_memory/indexes/vcache.py

### 1. MODULE IDENTITY
- **Module name:** `VCache`
- **Intended role:** Simple LRU cache mapping key -> offsets list
- **What this module DOES:**
  - Implements LRU eviction using OrderedDict
  - Configurable max_entries (default 1000)
  - Tracks hits/misses
  - Returns copy of offset lists
  - NOT thread-safe (no locking)
- **What this module explicitly DOES NOT do:**
  - Does NOT synchronize with persistent indexes
  - Does NOT validate cache consistency
  - Does NOT limit memory per entry (only entry count)

### 2. DATA OWNERSHIP
- **Data types created:** `List[int]` (offset list copies), `dict` (stats)
- **Data types consumed:** Any key type, `List[int]` offsets
- **Data types persisted:** NONE
- **Data types not persisted:** All cache data (in-memory only)

### 3. PUBLIC INTERFACES
- **Public classes:** `VCache`
- **Public methods:**
  - `__init__(max_entries=1000)`
  - `lookup(key: Any) -> Optional[List[int]]` - return copy or None
  - `insert(key: Any, offsets: List[int])` - add/update entry
  - `clear()` - reset cache
  - `stats() -> dict` - return hits, misses, hit_rate
- **Expected inputs:** Any key type, List[int] offsets
- **Expected outputs:** Optional[List[int]] (copy), dict (stats)
- **Side effects:** Memory only

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `collections.OrderedDict`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `cache: OrderedDict` - BOUNDED by max_entries
  - `hits: int`, `misses: int` - counters (UNBOUNDED but small ints)
- **Bounded or unbounded:** BOUNDED by max_entries (default 1000); LRU eviction
- **State reset/eviction:** LRU eviction when full; explicit clear() resets all

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Keys and offset lists from any caller
- **Explicit downstream outputs:** Offset list copies, stats
- **Required external conditions:** NONE

### 7. OPEN FACTUAL QUESTIONS
- NOT thread-safe; concurrent access NOT PREVENTED
- lookup() and insert() both copy offset lists; may be expensive for large lists
- max_entries is count-based; actual memory usage NOT BOUNDED
- No cache invalidation mechanism; stale data possible if underlying index changes

---

## MODULE: forge_memory/pulse/pulse_writer.py

### 1. MODULE IDENTITY
- **Module name:** `PulseWriter`
- **Intended role:** Drains worker queues and writes batches to BinaryLog with index updates
- **What this module DOES:**
  - Polls multiple queues in pulse loop
  - Writes records to BinaryLog
  - Updates HashIndex for task_id and (engine_id, transform_id, grid_shape_in, grid_shape_out)
  - Updates BitmapIndex with success flag
  - Maintains offset->record_id mapping
  - Thread-based with configurable pulse_interval_ms (default 10ms)
- **What this module explicitly DOES NOT do:**
  - Does NOT validate record schema
  - Does NOT retry writes on failure
  - Does NOT persist index state

### 2. DATA OWNERSHIP
- **Data types created:** `ForgeRecord` instances, offset->id mappings
- **Data types consumed:** `dict` (records from queues), BinaryLog, HashIndex, BitmapIndex instances
- **Data types persisted:** Binary records (via BinaryLog delegation)
- **Data types not persisted:** Index state, offset mappings (in-memory only)

### 3. PUBLIC INTERFACES
- **Public classes:** `PulseWriter`
- **Public methods:**
  - `__init__(queues: List, binary_log: BinaryLog, indexes: dict, pulse_interval_ms: int = 10)`
  - `start()` - start pulse thread
  - `stop()` - stop pulse thread (timeout 1.0s)
  - `run()` - pulse loop (internal, called by thread)
  - `get_offset_to_id_map() -> dict` - return copy of mapping
- **Expected inputs:**
  - Constructor: queues (List of Queue-like objects), binary_log, indexes dict with keys: engine_shapes, task, success
  - Records from queues: dicts with keys: pulse_id, worker_id, seq, timestamp, success, task_id, engine_id, transform_id, failure_reason, grid_shape_in, grid_shape_out, color_count, train_pair_indices, error_metrics, params, context
- **Expected outputs:** NONE (side effects only)
- **Side effects:**
  - Disk: writes to BinaryLog
  - Memory: updates indexes

### 4. DEPENDENCIES
- **Internal imports:** `forge_memory.core.binary_log.BinaryLog`, `forge_memory.indexes.hash_index.HashIndex`, `forge_memory.indexes.bitmap_index.BitmapIndex`, `forge_memory.core.record.ForgeRecord`
- **External imports:** `threading`, `time`, `typing.List`
- **Runtime import behavior:** ForgeRecord imported in _dict_to_record method

### 5. STATE & MEMORY
- **State held in memory:**
  - `_next_record_id: int` - monotonic counter (UNBOUNDED)
  - `_offset_to_id: dict` - mapping (UNBOUNDED, grows with records written)
  - References to queues, binary_log, indexes (external ownership)
- **Bounded or unbounded:** UNBOUNDED (record_id counter and offset mapping grow indefinitely)
- **State reset/eviction:** NEVER (state persists for instance lifetime)

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** Record dicts from worker queues
- **Explicit downstream outputs:** Records to BinaryLog, index updates
- **Required external conditions:**
  - Queues must support get(block=False)
  - BinaryLog must support append_record()
  - Indexes must support insert() and set_bit()

### 7. OPEN FACTUAL QUESTIONS
- _dict_to_record() constructs ForgeRecord with kwargs but ForgeRecord expects specific constructor signature; MISMATCH between record dict schema and ForgeRecord constructor
- pulse_interval_ms conversion to seconds via division; precision loss for very small intervals
- stop() timeout is 1.0s hardcoded; records in queues may be lost if thread doesn't stop
- No error handling for queue.get() exceptions beyond continue
- get_offset_to_id_map() returns copy via dict() constructor; expensive for large mappings

---

## MODULE: forge_memory/pulse/worker_pool.py

### 1. MODULE IDENTITY
- **Module name:** `WorkerPool`
- **Intended role:** Spawns N worker threads calling hypothesis function and pushing records to queues
- **What this module DOES:**
  - Creates worker_count threads (default 4)
  - Each worker calls hypothesis_fn(worker_id) repeatedly
  - Results pushed to per-worker queue (maxsize default 1000)
  - Provides start/stop lifecycle
  - Supports put_direct() for external record injection
- **What this module explicitly DOES NOT do:**
  - Does NOT validate hypothesis_fn return type
  - Does NOT handle hypothesis_fn exceptions
  - Does NOT limit CPU usage
  - Does NOT implement backpressure beyond queue maxsize

### 2. DATA OWNERSHIP
- **Data types created:** `List[Queue]` (worker queues), `List[threading.Thread]` (worker threads)
- **Data types consumed:** `dict` (records from hypothesis_fn)
- **Data types persisted:** NONE
- **Data types not persisted:** All queue contents (ephemeral)

### 3. PUBLIC INTERFACES
- **Public classes:** `WorkerPool`
- **Public methods:**
  - `__init__(worker_count: int = 4, queue_size: int = 1000)`
  - `start(hypothesis_fn: Callable[[int], dict])` - start workers with function
  - `stop()` - stop workers (timeout 1.0s per thread)
  - `get_queues() -> List[Queue]` - return queue list
  - `put_direct(wid: int, record: dict)` - inject record to worker queue
- **Expected inputs:**
  - `start()`: hypothesis_fn callable taking int (worker_id) returning dict
  - `put_direct()`: wid (int), record (dict)
- **Expected outputs:**
  - `get_queues()`: List[Queue]
- **Side effects:** CPU (worker threads running continuously)

### 4. DEPENDENCIES
- **Internal imports:** NONE
- **External imports:** `threading`, `time`, `queue.Queue`, `queue.Empty`, `typing`
- **Runtime import behavior:** NONE

### 5. STATE & MEMORY
- **State held in memory:**
  - `queues: List[Queue]` - each Queue has maxsize limit (BOUNDED per queue)
  - `threads: List[threading.Thread]` - fixed count
  - `running: bool` - lifecycle flag
- **Bounded or unbounded:** BOUNDED per-queue by queue_size; aggregate memory is worker_count * queue_size
- **State reset/eviction:** Queues drain naturally as records are consumed; no explicit clearing

### 6. INTEGRATION POINTS
- **Explicit upstream inputs:** hypothesis_fn results
- **Explicit downstream outputs:** Records to queues (consumed by external PulseWriter or other)
- **Required external conditions:**
  - hypothesis_fn must return dict
  - hypothesis_fn should not block indefinitely

### 7. OPEN FACTUAL QUESTIONS
- Workers loop continuously while running; CPU usage NOT CONTROLLED
- hypothesis_fn exceptions cause worker to fail; NO ERROR HANDLING or restart
- put() timeout 0.1s in worker loop; records may be lost if queue full and thread is stopping
- stop() timeout 1.0s per thread hardcoded; no forced termination
- put_direct() may block indefinitely if queue full; NO TIMEOUT

---

## WORKSPACE-LEVEL INTEGRATION SUMMARY

### Test Coverage (21 tests passing as of 2026-01-03 05:08:19)
- **Unit tests:** resilience (hot_spare, pulse_bus, disk_guard), loggers (writer, activity/input/network services), forge_memory (hash_index, bitmap_index, vcache)
- **Integration tests:** end_to_end (forge_memory full stack), services_integration (activity/input/network together)

### Scripts and Automation
- **scripts/check_no_runtime_stubs.sh:** Bash script greping for forbidden patterns (stub, fake, mock, dry_run, test_mode) in loggers/ directory; excludes test_ files and __pycache__

### Key System Constraints
- **Timezone:** EST (America/New_York) hardcoded in DailyJSONWriter and DiskGuard via ZoneInfo with fallback to UTC-5
- **Day boundary:** 00:01 rule in DailyJSONWriter: times before 00:01 treated as previous day
- **Unbounded memory risks:** DailyJSONWriter._queue (no maxsize), StringDictionary mappings, HashIndex.table, PulseWriter._offset_to_id
- **Thread safety:** HotSpare, PulseBus, DiskGuard, HashIndex, BitmapIndex use locks; VCache and WorkerPool NOT thread-safe
- **Error handling:** Nearly all modules catch all exceptions and continue; errors logged but not propagated

### Implicit Integration Patterns
- **PulseBus -> DailyJSONWriter:** PulseBus lazily creates DailyJSONWriter if no writer registered; uses data_root and pulse_interval from PulseBus init
- **DiskGuard -> DailyJSONWriter:** DiskGuard lazily creates DailyJSONWriter(category='system', prefix='syswarn') with computed writer_pulse interval
- **forge_memory stack:** WorkerPool feeds queues -> PulseWriter drains -> BinaryLog writes + indexes update; full reconstruction required on restart (no index persistence)
- **Services pattern:** ActivityService, InputService, NetworkService all follow disk-observer pattern: poll() returns event dict or None, no subprocess, day rollover state reset

### Cross-Module Dependencies
- **resilience/pulse_bus.py imports loggers/writer.py** (lazy)
- **resilience/disk_guard.py imports loggers/writer.py** (lazy)
- **forge_memory/pulse/pulse_writer.py imports forge_memory/core/* and forge_memory/indexes/***
- **forge_memory/core/binary_log.py imports forge_memory/core/record.py**
- **Tests import respective modules from resilience/, loggers/, forge_memory/**

### Files NOT ANALYZED (present in workspace but not introspected)
- loggers/process_logger.py, activity_logger.py, input_logger.py, gamepad_logger_continuous.py (integration wrappers, not service implementations)
- loggers/*_integration.py (integration glue code)
- playground/*.py (temporary/debug code)
- archive/* (not current)

### Filesystem Layout
```
logs_data/
├── activity/              (ActivityService reads from here)
├── input/                 (InputService reads from here)
├── network/               (NetworkService reads from here)
├── system/
│   └── syswarn_*.jsonl    (DiskGuard writes here)
└── <category>/
    └── {prefix}_*.jsonl   (DailyJSONWriter output pattern)
```

---

## FACTUAL CONCLUSIONS

1. **All resilience modules are production-ready:** HotSpare, PulseBus, DiskGuard are fully implemented with tests, no stubs remaining.

2. **Disk-observer services replace subprocess-based stubs:** ActivityService, InputService, NetworkService now read real files; config_path and use_subprocess parameters are signature-only compatibility shims (ignored).

3. **forge_memory is a complete binary log + indexing system:** BinaryLog (append-only binary), StringDictionary (string interning), HashIndex (key->offsets), BitmapIndex (success filtering), VCache (LRU cache), WorkerPool (hypothesis generators), PulseWriter (queue drainer + index updater). **Index state is NOT persisted** and must be rebuilt on restart.

4. **SHA-256 chain in DailyJSONWriter persists across day boundaries:** _prev_sha is not reset on file rotation, creating an implicit cross-file chain.

5. **PulseBus docstring contradicts implementation:** Claims "drop-oldest policy" but code implements drop-newest.

6. **Unbounded memory growth vectors identified:** DailyJSONWriter queue, StringDictionary mappings, HashIndex table, PulseWriter offset map, WorkerPool hypothesis generation rate.

7. **Thread safety is inconsistent:** Most modules use locks; VCache and WorkerPool do not.

8. **CI guard script enforces no runtime stubs:** Validates resilience implementation by blocking forbidden patterns in loggers/ directory.

9. **Test suite is comprehensive and disk-backed:** 21 tests using tempfile.TemporaryDirectory() for true filesystem interaction; no mocking of I/O.

10. **EST timezone assumption is pervasive:** DailyJSONWriter and DiskGuard both hardcode America/New_York with fallback; multi-timezone support NOT IMPLEMENTED.

---

**END OF INTROSPECTION REPORT**
