# CompuCog Contract Atlas
**The Cognitive Organism Wiring Diagram**

Version: 1.0  
Date: 2025-12-04  
Status: Master Reference Document

---

## Purpose

This document defines **every component connection** in CompuCog and the **exact contracts** each interface must obey. It prevents type drift, adapter violations, and contract mutations that cause system collapse.

---

# Component Connection Graph

```
┌─────────────────┐
│ ChronosManager  │ (Deterministic Time)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EventManager   │ ◄──────────────────┐
└────────┬────────┘                    │
         │                             │
         │                    ┌────────┴────────┐
         │                    │ SensorRegistry  │
         │                    └────────┬────────┘
         │                             │
         │          ┌──────────────────┼──────────────────┐
         │          │                  │                  │
         │   ┌──────▼──────┐    ┌─────▼─────┐    ┌──────▼──────┐
         │   │  Activity   │    │   Input   │    │   Process   │
         │   │   Adapter   │    │  Adapter  │    │   Adapter   │
         │   └─────────────┘    └───────────┘    └─────────────┘
         │          │                  │                  │
         │   ┌──────▼──────┐    ┌─────▼─────┐           │
         │   │  Network    │    │  Gamepad  │           │
         │   │   Adapter   │    │  Adapter  │           │
         │   └─────────────┘    └───────────┘           │
         │                                               │
         ▼                                               │
┌─────────────────┐                                     │
│  FusionEngine   │ ◄───────────────────────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CapsuleBuilder  │ (6-1-6 Temporal Windows)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ForgeMemory    │ (Persistent Storage)
└─────────────────┘
```

---

# Contract Definitions

## 1. ChronosManager → EventManager

**Contract Type:** Temporal Contract

**ChronosManager OUTPUTS:**
- `timestamp: float` - Unix epoch time or deterministic simulation time
- `window_id: int` - Monotonic window counter
- `window_start: float` - Window boundary start
- `window_end: float` - Window boundary end

**EventManager EXPECTS:**
- Timestamps MUST be monotonically increasing within each stream
- Timestamps MUST be floats, never strings
- Window IDs MUST align with Chronos windows

**If Contract Violated:**
- Event timeline becomes unsorted
- 6-1-6 capsules break temporal ordering
- Fusion loses time-alignment
- Memory corruption

---

## 2. SensorRegistry → EventManager

**Contract Type:** Event Contract (CRITICAL)

**SensorRegistry OUTPUTS:**
- `Event` object ONLY
  - `event_id: str`
  - `timestamp: float`
  - `source_id: str`
  - `tags: List[str]`
  - `metadata: dict`
  - `pulse_id: Optional[str]`
  - `nvme_ref: Optional[Tuple[int, int, int]]`

**EventManager EXPECTS:**
- Event object, NEVER raw dict
- All Event fields must be present
- source_id must be registered
- timestamp must be valid float

**If Contract Violated:**
- EventManager crashes or hangs
- Downstream fusion receives malformed data
- Capsule construction fails
- Memory writes corrupt data

**ENFORCEMENT RULE:**
```python
# ✅ CORRECT
def convert_to_event(self, data: dict) -> Event:
    return Event(tags=[...], metadata={...}, ...)

# ❌ WRONG - NEVER RETURN DICT
def convert_to_event(self, data: dict) -> dict:
    return {"tags": [...], "data": {...}}
```

---

## 3. LoggerAdapters → SensorRegistry

**Contract Type:** Adapter Contract

**LoggerAdapters OUTPUT:**
- Raw sensor data as dict (internally)
- Converted to Event via `convert_to_event()`

**SensorRegistry EXPECTS:**
- Adapter inherits from `SensorAdapter` ABC
- Implements all abstract methods:
  - `initialize() -> bool`
  - `start() -> bool`
  - `stop() -> bool`
  - `get_latest_data() -> dict`
  - `convert_to_event(data: dict) -> Event`

**Adapter Lifecycle:**
```
initialize() -> start() -> [poll loop: get_latest_data() -> convert_to_event()] -> stop()
```

**If Contract Violated:**
- Sensor fails to register
- Events never generated
- Registry poll loop skips sensor

---

## 4. Activity Logger → ActivityLoggerAdapter

**Contract Type:** File Tailing Contract

**Activity Logger OUTPUTS:**
- JSONL file: `logs/activity/user_activity_YYYYMMDD.jsonl`
- Schema:
  ```json
  {
    "timestamp": "ISO-8601",
    "active_window_title": "str",
    "active_process": "str",
    "idle_seconds": int
  }
  ```

**ActivityLoggerAdapter EXPECTS:**
- Valid JSON per line
- timestamp field present
- Handles missing fields gracefully

**Adapter OUTPUTS:**
- Event with `source_id="activity_monitor"`
- tags: `["activity", "window"]`
- metadata: full activity dict

---

## 5. Input Logger → InputLoggerAdapter

**Contract Type:** File Tailing Contract

**Input Logger OUTPUTS:**
- JSONL file: `logs/input/input_activity_YYYYMMDD.jsonl`
- Schema:
  ```json
  {
    "timestamp": "ISO-8601",
    "keyboard_events": int,
    "mouse_events": int,
    "total_events": int
  }
  ```

**InputLoggerAdapter OUTPUTS:**
- Event with `source_id="input_monitor"`
- tags: `["input", "aggregate"]`

---

## 6. Process Logger → ProcessLoggerAdapter

**Contract Type:** File Tailing Contract

**Process Logger OUTPUTS:**
- JSONL file: `logs/process/process_activity_YYYYMMDD.jsonl`
- Schema:
  ```json
  {
    "timestamp": "ISO-8601",
    "process_name": "str",
    "pid": int,
    "suspicious": bool
  }
  ```

**ProcessLoggerAdapter OUTPUTS:**
- Event with `source_id="process_monitor"`
- tags: `["process", "spawn"]`

---

## 7. Network Logger → NetworkLoggerAdapter

**Contract Type:** PowerShell File Tailing Contract

**Network Logger OUTPUTS:**
- JSONL file: `logs/network/telemetry_YYYYMMDD.jsonl`
- Schema:
  ```json
  {
    "timestamp": "ISO-8601",
    "local_address": "str",
    "remote_address": "str",
    "state": "str"
  }
  ```

**NetworkLoggerAdapter OUTPUTS:**
- Event with `source_id="network_monitor"`
- tags: `["network", "connection"]`

---

## 8. Gamepad Logger → GamepadLoggerAdapter

**Contract Type:** High-Frequency File Tailing Contract

**Gamepad Logger OUTPUTS:**
- JSONL file: `logs/gamepad/gamepad_stream_YYYYMMDD_HHMMSS.jsonl`
- Schema:
  ```json
  {
    "timestamp": "ISO-8601",
    "left_stick": [float, float],
    "right_stick": [float, float],
    "buttons": dict
  }
  ```

**GamepadLoggerAdapter OUTPUTS:**
- Event with `source_id="gamepad_monitor"`
- tags: `["gamepad", "controller"]`
- Sample rate: 60Hz

---

## 9. EventManager → FusionEngine

**Contract Type:** Batch Event Contract

**EventManager OUTPUTS:**
- Events grouped by time window
- All events within window have aligned timestamps
- Each event has valid source_id, tags, metadata

**FusionEngine EXPECTS:**
- Events from multiple sensors in same time window
- Vision events (if available)
- Input events (if available)
- Network events (if available)
- Gamepad events (if available)

**If Contract Violated:**
- Cross-modal correlation fails
- Cheat detection loses context
- Causal chains break

---

## 10. FusionEngine → CapsuleBuilder

**Contract Type:** Multi-Modal Fusion Contract

**FusionEngine OUTPUTS:**
- Fused event dict containing:
  - `vision: dict`
  - `input: dict`
  - `network: dict`
  - `audio: dict` (future)
  - `process: dict`
  - `gamepad: dict`

**CapsuleBuilder EXPECTS:**
- At least ONE sensor stream present
- All present streams have valid data
- Timestamps aligned

---

## 11. CapsuleBuilder → ForgeMemory

**Contract Type:** Capsule Storage Contract

**CapsuleBuilder OUTPUTS:**
- Capsule object:
  - `anchor_event: Event`
  - `before_events: List[Event]` (up to 6)
  - `after_events: List[Event]` (up to 6)
  - `window_id: int`

**ForgeMemory EXPECTS:**
- Serializable capsule structure
- Valid anchor event
- Before/after events in temporal order

---

# 12. EventManager → ForgeMemory Contract

**Contract Type:** Forge Gateway Contract (CRITICAL)

**EventManager OUTPUTS:**
- ForgeRecord objects written to BinaryLog
- All sensory events permanently stored
- Causal ordering preserved

**ForgeMemory EXPECTS:**
- ForgeRecord with 44-byte header + variable tail
- All required fields populated
- Serializable context/params
- Monotonic timestamps within each source

**Event → ForgeRecord Mapping:**

| Event Field  | ForgeRecord Field | Required | Mapping Rule                                      |
|--------------|-------------------|----------|---------------------------------------------------|
| `event_id`   | `seq`             | YES      | EventManager global sequence counter              |
| `timestamp`  | `timestamp`       | YES      | ChronosManager timestamp (float, Unix epoch)      |
| `source_id`  | `task_id`         | YES      | Direct mapping (sensor source identifier)         |
| N/A          | `engine_id`       | YES      | Fixed: "event_pipeline_v1" (string ref)           |
| `tags`       | `params`          | YES      | Convert to dict: `{"tags": ["tag1", "tag2"]}`     |
| `metadata`   | `context`         | YES      | Direct mapping (must be JSON-serializable)        |
| N/A          | `success`         | YES      | Always True (events are not tasks)                |
| N/A          | `worker_id`       | YES      | hash(source_id) % 256 for stable worker ID        |
| N/A          | `pulse_id`        | YES      | EventManager pulse counter (incremented per write)|
| N/A          | `transform_id`    | YES      | "sensor_event" (fixed string ref)                 |
| N/A          | `failure_reason`  | NO       | Always None for events                            |
| N/A          | `grid_shape_in`   | YES      | (0, 0) for non-vision sensors                     |
| N/A          | `grid_shape_out`  | YES      | (0, 0) for non-vision sensors                     |
| N/A          | `color_count`     | YES      | 0 for non-vision sensors                          |
| N/A          | `train_pair_indices` | YES   | Empty list for events                             |
| N/A          | `error_metrics`   | YES      | Empty dict for events                             |

**Ordering Guarantees:**
- EventManager enforces **total causal ordering** via global `seq` counter
- ChronosManager ensures **monotonic timestamps** within each sensor stream
- BinaryLog appends are **atomic** (thread-safe via lock)
- 6-1-6 CapsuleBuilder consumes Forge in strict temporal sequence

**If Contract Violated:**
- Forge writes fail with serialization errors
- Capsule construction breaks (missing records)
- Fusion loses temporal alignment
- Forensic analysis becomes impossible

**ENFORCEMENT RULE:**
```python
# ✅ CORRECT - EventManager converts and writes
event = Event(...)
forge_record = event_manager._event_to_forge_record(event)
binary_log.append(forge_record)

# ❌ WRONG - Direct Forge writes bypassing EventManager
forge_record = ForgeRecord(...)
binary_log.append(forge_record)  # FORBIDDEN
```

---

# 13. Forbidden Architectural Patterns

**The following patterns are STRICTLY PROHIBITED:**

## ❌ NEVER DO THESE:

1. **Adapters writing to Forge directly**
   - Causes: Ordering violations, missing seq numbers, broken capsules
   - Violates: Single gateway principle

2. **Components creating ForgeRecord objects**
   - Causes: Schema drift, mapping inconsistency, invalid field values
   - Violates: EventManager owns conversion

3. **Bypassing EventManager for sensory data**
   - Causes: Lost events, incomplete timeline, fusion gaps
   - Violates: Single write path contract

4. **Writing JSONL files anywhere in the system**
   - Causes: Duplicate storage, desync, no causal ordering
   - Violates: Forge as single source of truth

5. **Direct BinaryLog writes outside EventManager**
   - Causes: Race conditions, corrupted sequences, broken atomicity
   - Violates: Gateway contract

6. **Subprocess loggers that write files**
   - Causes: I/O bottlenecks, desync, orphaned processes
   - Violates: Unified memory architecture

7. **Adapters returning ForgeRecord instead of Event**
   - Causes: Contract confusion, adapter coupling to Forge schema
   - Violates: Clean separation of concerns

8. **Using time.time() instead of chronos.now()**
   - Causes: Temporal misalignment, unsorted timeline, broken windows
   - Violates: Deterministic time contract

9. **Modifying ForgeRecord after creation**
   - Causes: Checksum failures, corrupted writes, invalid binary data
   - Violates: Immutable record contract

10. **Skipping EventManager registration for sources**
    - Causes: Rejected events, silent data loss, missing streams
    - Violates: Source registration contract

---

# Expansion Slots

## Reserved Sensor Types (Not Yet Implemented)

- `BIOMETRIC_1`, `BIOMETRIC_2`, `BIOMETRIC_3`: Eye tracking, heart rate, etc.
- `ENVIRONMENTAL_1`, `ENVIRONMENTAL_2`: Room temp, lighting, etc.
- `CUSTOM_1` through `CUSTOM_15`: User-defined sensors

**Contract Rule:**
All future sensors MUST follow the SensorAdapter contract.
All MUST output Event objects via `convert_to_event()`.

---

# Critical Contract Violations to Avoid

## ❌ NEVER DO THESE:

1. **Return dict instead of Event from convert_to_event()**
   - Causes: EventManager crash, fusion failure, memory corruption

2. **Pass raw dict to EventManager.record_event()**
   - Causes: Type errors, missing attributes, downstream failure

3. **Use time.time() instead of chronos.now()**
   - Causes: Temporal misalignment, broken capsules

4. **Modify Event objects after creation**
   - Causes: Cache invalidation, memory inconsistency

5. **Skip sensor registration**
   - Causes: EventManager rejects events, silent data loss

6. **Return non-serializable objects in Event.metadata**
   - Causes: ForgeMemory write failure, corrupted logs

---

# Contract Enforcement

## Runtime Validation (Implemented)

`SensorAdapter.record_event()` now enforces:
```python
if not isinstance(event, Event):
    raise TypeError(
        f"Contract violation: {self.__class__.__name__}.convert_to_event() "
        f"returned {type(event).__name__}, expected Event object."
    )
```

## Future Enforcement

- Type checking at component boundaries
- Schema validation for Event metadata
- Temporal alignment verification
- Contract violation detection in CI/CD

---

# Usage

## For Developers

When adding new sensors:
1. Check this document for the adapter contract
2. Implement SensorAdapter ABC
3. Ensure convert_to_event() returns Event object
4. Test with minimal test harness
5. Validate contract compliance

## For Debugging

When system fails:
1. Check this document for the violated contract
2. Identify which component's output doesn't match next component's input
3. Fix the contract violation
4. Re-test end-to-end

---

# Version History

- **v1.1 (2025-12-04)**: Forge Integration Contract
  - Added Event → ForgeRecord mapping (Contract #12)
  - Added Forbidden Architectural Patterns section
  - Defined EventManager as Forge Gateway
  - Specified ordering guarantees
  - Documented field-by-field conversion rules
  - Added enforcement examples

- **v1.0 (2025-12-04)**: Initial Contract Atlas
  - Mapped 11 core contracts
  - Defined 5 logger adapter contracts
  - Added expansion slot rules
  - Implemented Event contract enforcement

---

**This document is the single source of truth for CompuCog component contracts.**

**Any code that violates these contracts will cause system failure.**

**When in doubt, consult this atlas.**

**EventManager is the ONLY legal gateway to Forge Memory.**
