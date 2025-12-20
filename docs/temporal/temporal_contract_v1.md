# Temporal Contract Spec — Multi-Stream Forensic Capture

**Version:** 1.0  
**Scope:** All capture streams (activity/input/process/network/TrueVision).  
**Objective:** Deterministic post-capture alignment without cross-stream mutation during capture.

---

## 1. Event Envelope

**Format:** JSONL (1 event per line).  
**Order:** Append-only; file order is event order. No rewrites.

**Required fields:**

* `source` (string) — canonical stream id.
* `session_id` (string) — assigned at session start; stable for run.
* `seq` (int) — per-source, per-session; strictly increasing; starts at 1.
* `t_monotonic_ns` (int) — primary alignment time; monotonic nanoseconds.
* `t_utc` (string) — ISO8601 UTC with microsecond precision, trailing `Z`.
* `event_id` (string) — unique within dataset; recommended format: `{session_id}:{source}:{seq:012d}`.
* `payload` (object) — source-owned immutable content.
* `meta` (object) — `schema_version`, `logger_version`, optional diagnostic fields.

---

## 2. Clock Contract

**Primary alignment axis:** `t_monotonic_ns`.

**Clock sources:**

* `t_monotonic_ns` MUST be sourced from a monotonic clock (Python: `time.monotonic_ns()`).
* `t_utc` MUST be UTC wall-clock (system UTC) or derived via a single session mapping.

**Capture-time constraints:**

* No “correction” of timestamps at capture time.
* No cross-stream timestamp substitution.
* Capture emits observed time only.

---

## 3. Sequence Contract

* `seq` increments by 1 per emitted event for that `(session_id, source)`.
* Missing events are represented as gaps in `seq` (no backfill).
* Reordering is prohibited.
* Duplicate `seq` for a `(session_id, source)` is invalid.

---

## 4. Heartbeats and Liveness

Each source MUST emit heartbeat events at a configured interval.

**Default:** ≤ 5s.

Heartbeat payload MUST include:

* `type = "heartbeat"`
* `uptime_ns`
* `last_seq_emitted` (or equivalent)

Heartbeat uses the same envelope and sequencing.

---

## 5. Error Observability

On local failures, the source MUST emit an explicit error event.

**Error payload MUST include:**

* `type = "error"`
* `error_code`
* `message`
* `context` (optional)

Do not hide failures via retries without emitting evidence of the fault.

---

## 6. Session Lifecycle

A session has explicit boundaries.

**Required session events per source:**

* `type = "session_start"` (first event or explicit)
* `type = "session_end"` (explicit)

`session_start` payload MUST include:

* `process_id` (optional)
* `host_id` (optional)
* `source_config_hash` (recommended)

`session_end` payload MUST include:

* `final_seq`
* `reason` (normal/abort/error)

---

## 7. Storage and Rotation

* Streams may rotate files; rotation MUST produce a manifest record:

  * `{session_id, source, file_id, seq_min, seq_max, t_monotonic_min, t_monotonic_max, sha256}`

Rotation preserves per-file event order and allows deterministic reconstruction.

---

## 8. Validation (capture-side)

Each emitted event MUST pass validation before flush:

* required fields present
* `seq > 0`
* `t_monotonic_ns > 0`
* `t_utc` parseable ISO8601 UTC
* `payload` is an object
* `meta.schema_version` present

On validation failure:

* emit a `type="validation_error"` event
* include failing field list and raw snippet if available

---

## 9. Post-Capture Alignment Guidance

* Canonical alignment uses `t_monotonic_ns`.
* `t_utc` is for human correlation and cross-session coarse anchoring.
* Drift assessment is computed offline; capture does not adjust for drift.

---

## 10. Tolerances (defaults)

* Event jitter expected: ≤ 10 ms typical; higher acceptable under load.
* Cross-stream correlation tolerance: 500 ms default; investigation-configurable.

---

## 11. Security / Forensic Integrity

* Each stream SHOULD produce periodic integrity checkpoints:

  * rolling hashes or per-file sha256 manifests.
* Derived artifacts MUST reference:

  * source file ids
  * seq ranges
  * schema versions
  * mapping parameters

---

## 12. Versioning

* `meta.schema_version` REQUIRED.
* Backward-incompatible changes increment major version.
* Maintain contract changelog.

---

## Addendum — Gamepad Logging Requirements (Temporal Contract v1.0)

This addendum extends the prior Temporal Contract with **mandatory identifiers and sequencing rules specific to gamepad input capture**. It introduces no changes to existing fields; it only constrains `payload` content for the `source = "gamepad"` stream.

### A1. Gamepad Source Identification

Each gamepad event **MUST** include stable device identifiers in `payload.device`.

**Rules:**

* `device_class` MUST be `"gamepad"`.
* `device_index` MUST be stable for the session (0-based).
* `vendor_id` / `product_id` MUST be hex strings if available.
* `instance_guid` MUST be stable for the physical device across reconnects within the session.
* `driver` MUST identify the API (`xinput`, `dinput`, `hidraw`, etc.).
* `os_device_path` SHOULD be included when available (Windows/HID).

---

### A2. Gamepad Event Types

`payload.type` MUST be one of:

* `"button"` — digital button press/release
* `"axis"` — analog axis update
* `"connection"` — device connect/disconnect
* `"heartbeat"` — periodic liveness record (see §4 in main spec)

---

### A3. Button Events

**Rules:**

* `button.id` MUST be canonicalized (e.g., `A`, `B`, `X`, `Y`, `LB`, `RB`, `START`).
* `button.code` MUST be the raw driver index.
* `state` MUST be `"pressed"` or `"released"`.

---

### A4. Axis Events

**Rules:**

* `axis.id` MUST be canonical (`LX`, `LY`, `RX`, `RY`, `LT`, `RT`).
* `value` MUST be normalized to `[-1.0, 1.0]` (or `[0.0, 1.0]` for triggers).
* `raw` SHOULD be included when available.
* Deadzone handling MUST be logged, not hidden.

---

### A5. Connection Events

**Rules:**

* MUST emit on connect and disconnect.
* Disconnects MUST NOT reset `seq`; gaps are expected.

---

### A6. Sequencing and Timing Guarantees (Gamepad-Specific)

* `seq` ordering reflects **capture order**, not physical action order.
* Multiple axis updates MAY share the same `t_monotonic_ns` if sampled in one poll cycle.
* Button and axis events MUST NOT be merged into composite records at capture time.

---

### A7. Anti-Ambiguity Constraints

* No inference at capture time (no gesture detection, no combo synthesis).
* No fusion with other inputs at capture time.
* Gamepad logs remain **independent, timestamp-complete, and device-identifiable**.

---

### A8. Offline Fusion Implication (Non-NormATIVE)

During offline analysis:

* Gamepad events align via `t_monotonic_ns`.
* Device identity enables multi-controller discrimination.
* Button/axis streams can be windowed and correlated with TrueVision frames without ambiguity.

---

**End of Addendum**
