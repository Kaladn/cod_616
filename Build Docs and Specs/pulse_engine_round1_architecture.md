# COD_616 Pulse Engine — Round 1 Architecture Specification

## Executive Summary

This document specifies the architecture for a **7-layer pulse_engine** that coordinates resilience and routing for write-only JSONL logging in COD_616. The design eliminates the "single brain" supervisor pattern in favor of a **layered signal-driven architecture** where each layer processes signals from the previous layer and emits new signals, with all coordination happening via pulse ticks and a deterministic action executor.

---

## 1. Signal Schema

All signals are **JSON-serializable dictionaries** with a common base structure and layer-specific extensions.

### 1.1 Base Signal Structure

```python
{
    "signal_type": str,        # e.g., "writer_stalled", "route_to_hotspare"
    "severity": str,           # "info" | "warning" | "critical"
    "source_layer": int,       # 1-7, layer that emitted this signal
    "timestamp": float,        # time.time() when signal was created
    "pulse_id": int,           # monotonic pulse counter
    "context": dict            # layer-specific payload
}
```

### 1.2 Signal Types by Layer

#### Layer 1: Failure Detection
- `writer_stalled` — writer has not flushed in N pulses
  - context: `{"category": str, "last_flush_pulse": int, "lag_pulses": int}`
- `writer_exception` — writer raised an exception
  - context: `{"category": str, "exception_type": str, "exception_msg": str}`
- `buffer_growth` — buffer size exceeding threshold
  - context: `{"category": str, "buffer_size": int, "threshold": int}`
- `heartbeat_missed` — expected heartbeat not received
  - context: `{"component": str, "last_heartbeat_pulse": int}`

#### Layer 2: Event Routing
- `route_to_primary` — route events to primary writer
  - context: `{"category": str}`
- `route_to_hotspare` — route events to hotspare only
  - context: `{"category": str, "reason": str}`
- `route_to_quarantine` — quarantine events (drop with logging)
  - context: `{"category": str, "reason": str}`
- `route_drop` — drop events silently
  - context: `{"category": str, "reason": str}`

#### Layer 3: Writer Management
- `writer_healthy` — writer is operating normally
  - context: `{"category": str, "lag_pulses": int, "last_rotation": float}`
- `writer_degraded` — writer is slow but functional
  - context: `{"category": str, "lag_pulses": int}`
- `writer_failed` — writer is non-functional
  - context: `{"category": str, "failure_reason": str}`
- `writer_rotation_due` — writer should rotate to new day file
  - context: `{"category": str, "current_day": str, "target_day": str}`

#### Layer 4: HotSpare Management
- `hotspare_mirror_active` — hotspare is mirroring events
  - context: `{"buffer_count": int, "oldest_ts": float, "newest_ts": float}`
- `hotspare_takeover_initiated` — hotspare takeover started
  - context: `{"trigger_reason": str, "buffer_count": int}`
- `hotspare_takeover_complete` — hotspare takeover finished, events drained
  - context: `{"events_drained": int, "drain_duration_ms": float}`
- `hotspare_release` — hotspare released back to mirror mode
  - context: `{"buffer_count": int}`
- `hotspare_overflow` — hotspare dropped events due to MAX_EVENTS
  - context: `{"dropped_newest": int}`

#### Layer 5: Disk Pressure
- `disk_pressure_normal` — disk usage is normal
  - context: `{"free_bytes": int, "avg_daily_usage": int, "days_remaining": float}`
- `disk_pressure_warning` — disk usage approaching limits
  - context: `{"free_bytes": int, "avg_daily_usage": int, "days_remaining": float}`
- `disk_pressure_critical` — disk usage critical
  - context: `{"free_bytes": int, "avg_daily_usage": int, "days_remaining": float}`

#### Layer 6: Policy
- `policy_normal` — no degradation needed
  - context: `{}`
- `policy_degrade_low_priority` — drop low-priority categories
  - context: `{"categories_to_drop": list[str]}`
- `policy_degrade_all_but_critical` — only allow critical categories
  - context: `{"categories_allowed": list[str]}`
- `policy_emergency_shutdown` — prepare for graceful shutdown
  - context: `{"reason": str, "shutdown_delay_seconds": float}`

#### Layer 7: Escalation
- `escalation_syswarn` — emit system warning event
  - context: `{"severity": str, "message": str, "source_signals": list[dict]}`
- `escalation_forensic_marker` — emit forensic marker for post-mortem
  - context: `{"marker_type": str, "details": dict}`

---

## 2. Layer Contracts

Each layer is a **pure function** that takes:
- **Engine state** (persistent across pulses)
- **Input signals** (from previous layer or external sources)
- **Context snapshot** (immutable per-tick: pulse_id, timestamp, stats)

And produces:
- **Output signals** (for next layer)
- **State mutations** (applied after all layers execute)

### 2.1 Layer 1: Failure Detection

**Purpose:** Detect anomalies in writers, buffers, and components.

**Inputs:**
- Engine state: writer stats (last_flush_pulse, exception_count, buffer_sizes)
- Context snapshot: current pulse_id, timestamp
- External signals: none (reads state only)

**Outputs:**
- Failure signals: `writer_stalled`, `writer_exception`, `buffer_growth`, `heartbeat_missed`

**Allowed Actions:**
- Read writer stats
- Compare pulse counters
- Emit failure signals

**Forbidden Actions:**
- Modify writer state
- Make routing decisions
- Call writer methods

**Thresholds:**
- Writer stalled: no flush in 50 pulses (5 seconds at 100ms pulse)
- Buffer growth: buffer size > 1000 events
- Heartbeat missed: no heartbeat in 100 pulses (10 seconds)

---

### 2.2 Layer 2: Event Routing

**Purpose:** Decide where events should flow based on failure signals.

**Inputs:**
- Engine state: current routing table (category → destination)
- Input signals: failure signals from Layer 1
- Context snapshot: current pulse_id, timestamp

**Outputs:**
- Routing signals: `route_to_primary`, `route_to_hotspare`, `route_to_quarantine`, `route_drop`

**Allowed Actions:**
- Read failure signals
- Emit routing signals
- Update routing table (via state mutations)

**Forbidden Actions:**
- Directly route events (only emit signals)
- Call writer or hotspare methods
- Make policy decisions

**Routing Logic:**
- If `writer_stalled` or `writer_exception` → `route_to_hotspare`
- If `buffer_growth` critical → `route_to_quarantine`
- If policy says drop → `route_drop`
- Otherwise → `route_to_primary`

---

### 2.3 Layer 3: Writer Management

**Purpose:** Track writer health, lag, and rotation needs.

**Inputs:**
- Engine state: writer health status, rotation timestamps
- Input signals: failure signals from Layer 1, routing signals from Layer 2
- Context snapshot: current pulse_id, timestamp, current_day

**Outputs:**
- Writer status signals: `writer_healthy`, `writer_degraded`, `writer_failed`, `writer_rotation_due`

**Allowed Actions:**
- Read writer stats
- Emit writer status signals
- Schedule rotation (via state mutations)

**Forbidden Actions:**
- Directly call writer methods
- Modify routing table
- Make policy decisions

**Health Classification:**
- Healthy: lag < 10 pulses, no exceptions in last 100 pulses
- Degraded: lag 10-50 pulses, or 1-3 exceptions in last 100 pulses
- Failed: lag > 50 pulses, or 4+ exceptions in last 100 pulses

---

### 2.4 Layer 4: HotSpare Management

**Purpose:** Ensure last 30 seconds are preserved; manage takeover/release.

**Inputs:**
- Engine state: hotspare mode (mirror | takeover), takeover trigger timestamp
- Input signals: routing signals from Layer 2, writer status from Layer 3
- Context snapshot: current pulse_id, timestamp

**Outputs:**
- HotSpare signals: `hotspare_mirror_active`, `hotspare_takeover_initiated`, `hotspare_takeover_complete`, `hotspare_release`, `hotspare_overflow`

**Allowed Actions:**
- Read hotspare stats
- Emit hotspare signals
- Trigger takeover/release (via state mutations)

**Forbidden Actions:**
- Directly call hotspare methods (only emit signals)
- Make routing decisions
- Make policy decisions

**Takeover Logic:**
- Initiate takeover if `writer_failed` signal received
- Drain hotspare buffer to PulseBus after 30-second window
- Release hotspare when writer recovers (healthy for 10 consecutive pulses)

---

### 2.5 Layer 5: Disk Pressure

**Purpose:** Observe disk trends and emit warnings.

**Inputs:**
- Engine state: DiskGuard stats (free_bytes, avg_daily_usage)
- Input signals: none (reads DiskGuard state only)
- Context snapshot: current pulse_id, timestamp

**Outputs:**
- Disk pressure signals: `disk_pressure_normal`, `disk_pressure_warning`, `disk_pressure_critical`

**Allowed Actions:**
- Read DiskGuard stats
- Emit disk pressure signals

**Forbidden Actions:**
- Modify disk state
- Make routing or policy decisions
- Throttle writes

**Thresholds:**
- Normal: free_bytes > 3× avg_daily_usage
- Warning: free_bytes between 2× and 3× avg_daily_usage
- Critical: free_bytes < 2× avg_daily_usage

---

### 2.6 Layer 6: Policy

**Purpose:** Decide what is allowed under stress.

**Inputs:**
- Engine state: current policy mode (normal | degrade_low | degrade_all | emergency)
- Input signals: disk pressure from Layer 5, writer status from Layer 3, hotspare status from Layer 4
- Context snapshot: current pulse_id, timestamp

**Outputs:**
- Policy signals: `policy_normal`, `policy_degrade_low_priority`, `policy_degrade_all_but_critical`, `policy_emergency_shutdown`

**Allowed Actions:**
- Read all input signals
- Emit policy signals
- Update policy mode (via state mutations)

**Forbidden Actions:**
- Directly modify routing table (only emit signals)
- Call writer or hotspare methods

**Policy Escalation:**
- Normal: no pressure, all writers healthy
- Degrade low priority: disk warning OR 1+ writers degraded
- Degrade all but critical: disk critical OR 2+ writers failed
- Emergency shutdown: disk critical AND hotspare overflow AND 3+ writers failed

**Priority Categories:**
- Critical: `activity`, `system`
- High: `input`, `network`, `process`
- Low: `gamepad`, `truevision`

---

### 2.7 Layer 7: Escalation

**Purpose:** Surface alerts, warnings, forensic markers.

**Inputs:**
- Engine state: escalation history (to avoid spam)
- Input signals: all signals from Layers 1-6
- Context snapshot: current pulse_id, timestamp

**Outputs:**
- Escalation signals: `escalation_syswarn`, `escalation_forensic_marker`

**Allowed Actions:**
- Read all input signals
- Emit escalation signals
- Emit events to PulseBus (via action executor)

**Forbidden Actions:**
- Fix anything
- Make routing or policy decisions
- Directly call writer methods

**Escalation Rules:**
- Emit `escalation_syswarn` for any critical severity signal
- Emit `escalation_forensic_marker` for writer failures, hotspare takeovers, policy changes
- Rate-limit: max 1 syswarn per category per 100 pulses

---

## 3. Engine Tick Contract

The engine executes a **deterministic tick** on every pulse (~100ms).

### 3.1 Tick Sequence

```
1. Capture context snapshot (pulse_id, timestamp, current_day, stats)
2. Execute Layer 1 (Failure Detection) → emit signals
3. Execute Layer 2 (Event Routing) → emit signals
4. Execute Layer 3 (Writer Management) → emit signals
5. Execute Layer 4 (HotSpare Management) → emit signals
6. Execute Layer 5 (Disk Pressure) → emit signals
7. Execute Layer 6 (Policy) → emit signals
8. Execute Layer 7 (Escalation) → emit signals
9. Execute Action Executor (apply all state mutations and actions)
10. Increment pulse_id
```

### 3.2 Action Executor

The **Action Executor** is the **only component** that can:
- Modify engine state
- Call writer methods
- Call hotspare methods
- Emit events to PulseBus

It processes signals in order and applies actions deterministically.

**Action Types:**
- `update_routing_table` — change category routing
- `trigger_hotspare_takeover` — call `hotspare.takeover()`
- `trigger_hotspare_release` — call `hotspare.release()`
- `drain_hotspare_to_pulsebus` — call `hotspare.drain()` and publish to PulseBus
- `emit_syswarn_event` — publish system warning to PulseBus
- `update_policy_mode` — change policy mode
- `update_writer_stats` — record writer health changes

**Determinism:**
- Actions are applied in the order signals were emitted (Layer 1 → 7)
- If multiple signals request conflicting actions, **last signal wins**
- All state mutations are applied atomically after all layers execute

---

## 4. State Model

### 4.1 Engine State (Persistent Across Pulses)

```python
{
    "pulse_id": int,                          # monotonic pulse counter
    "routing_table": dict[str, str],          # category → destination ("primary" | "hotspare" | "quarantine" | "drop")
    "writer_stats": dict[str, dict],          # category → {last_flush_pulse, exception_count, buffer_size, health}
    "hotspare_mode": str,                     # "mirror" | "takeover"
    "hotspare_takeover_pulse": int | None,    # pulse_id when takeover started
    "policy_mode": str,                       # "normal" | "degrade_low" | "degrade_all" | "emergency"
    "escalation_history": dict[str, int],     # signal_type → last_emitted_pulse_id
    "disk_stats": dict,                       # {free_bytes, avg_daily_usage, last_check_pulse}
}
```

### 4.2 Context Snapshot (Immutable Per Tick)

```python
{
    "pulse_id": int,                          # current pulse
    "timestamp": float,                       # time.time()
    "current_day": str,                       # YYYY-MM-DD in Eastern time
    "pulsebus_stats": dict,                   # stats from PulseBus (buffer sizes, drop counts)
    "hotspare_stats": dict,                   # stats from HotSpare (buffer_count, oldest_ts, newest_ts, dropped_newest)
    "diskguard_stats": dict,                  # stats from DiskGuard (free_bytes, avg_daily_usage)
}
```

---

## 5. Coupling Risks + Mitigations

### 5.1 Risk: Layer 2 (Routing) Directly Modifies Routing Table

**Symptom:** Layer 2 calls `engine.routing_table["activity"] = "hotspare"` directly.

**Impact:** Breaks deterministic action execution; state changes happen mid-tick.

**Mitigation:** Layer 2 emits `route_to_hotspare` signal; Action Executor applies routing change after all layers execute.

---

### 5.2 Risk: Layer 4 (HotSpare Management) Calls `hotspare.takeover()` Directly

**Symptom:** Layer 4 calls `hotspare.takeover()` during tick.

**Impact:** Breaks signal-driven architecture; creates hidden dependencies.

**Mitigation:** Layer 4 emits `hotspare_takeover_initiated` signal; Action Executor calls `hotspare.takeover()` after all layers execute.

---

### 5.3 Risk: Layer 7 (Escalation) Reads Layer 3 State Directly

**Symptom:** Layer 7 accesses `engine.writer_stats` instead of reading signals.

**Impact:** Creates backward dependency; Layer 7 now depends on Layer 3 state schema.

**Mitigation:** Layer 3 emits `writer_failed` signal with all necessary context; Layer 7 reads signals only.

---

### 5.4 Risk: Circular Signal Dependencies

**Symptom:** Layer 6 emits `policy_degrade_low_priority` → Layer 2 emits `route_drop` → Layer 6 reads `route_drop` and changes policy again.

**Impact:** Non-deterministic behavior; infinite loops.

**Mitigation:** Layers only read signals from **previous layers** in the current tick. Layer 6 cannot read Layer 2 signals from the same tick.

---

### 5.5 Risk: Action Executor Becomes a "Manager of Managers"

**Symptom:** Action Executor contains complex logic for deciding which actions to apply.

**Impact:** Violates "no manager of managers" constraint; logic leaks out of layers.

**Mitigation:** Action Executor is **dumb**: it applies actions in signal order, last-signal-wins for conflicts. All logic stays in layers.

---

## 6. Observability Plan

### 6.1 Counters (Prometheus-style, in-memory)

```python
pulse_engine_ticks_total                    # total pulses executed
pulse_engine_tick_duration_seconds          # histogram of tick duration
pulse_engine_signals_emitted_total          # counter by signal_type, severity, source_layer
pulse_engine_actions_executed_total         # counter by action_type
pulse_engine_layer_duration_seconds         # histogram by layer (1-7)
pulse_engine_routing_table_changes_total    # counter by category
pulse_engine_hotspare_takeovers_total       # counter
pulse_engine_hotspare_releases_total        # counter
pulse_engine_policy_mode_changes_total      # counter by mode
pulse_engine_escalations_total              # counter by escalation_type
```

### 6.2 Metrics Emission

- Emit metrics snapshot to PulseBus every 100 pulses (10 seconds)
- Category: `system`, severity: `info`
- Format: `{"metric_name": value, "labels": {...}}`

### 6.3 Forensic Markers

- Emit forensic marker on:
  - Writer failure
  - HotSpare takeover
  - Policy mode change
  - Emergency shutdown
- Category: `system`, severity: `critical`
- Include: full signal history for last 10 pulses, engine state snapshot

### 6.4 Health Check API

```python
engine.health() -> dict:
    {
        "pulse_id": int,
        "uptime_seconds": float,
        "policy_mode": str,
        "hotspare_mode": str,
        "writers_healthy": int,
        "writers_degraded": int,
        "writers_failed": int,
        "disk_pressure": str,
        "last_tick_duration_ms": float,
    }
```

---

## 7. Failure Cascade Analysis

### 7.1 Scenario: Single Writer Fails

**Trigger:** `activity` writer raises exception.

**Expected Behavior:**
1. Layer 1 detects `writer_exception` → emits signal
2. Layer 2 receives signal → emits `route_to_hotspare` for `activity`
3. Layer 3 marks writer as `failed`
4. Layer 4 initiates hotspare takeover
5. Layer 7 emits forensic marker
6. Action Executor: routes `activity` to hotspare, calls `hotspare.takeover()`
7. Events continue flowing to hotspare; last 30s preserved

**Failure Cascade Risk:** None. Other writers unaffected.

---

### 7.2 Scenario: Disk Pressure Critical

**Trigger:** DiskGuard reports free_bytes < 2× avg_daily_usage.

**Expected Behavior:**
1. Layer 5 emits `disk_pressure_critical`
2. Layer 6 receives signal → emits `policy_degrade_all_but_critical`
3. Layer 2 receives policy signal → emits `route_drop` for low-priority categories
4. Layer 7 emits `escalation_syswarn`
5. Action Executor: updates routing table to drop `gamepad`, `truevision`
6. Critical categories (`activity`, `system`) continue writing

**Failure Cascade Risk:** Low. Non-critical categories dropped cleanly.

---

### 7.3 Scenario: HotSpare Overflow

**Trigger:** HotSpare buffer exceeds MAX_EVENTS during takeover.

**Expected Behavior:**
1. HotSpare drops newest events, increments `dropped_newest` counter
2. Layer 4 reads hotspare stats → emits `hotspare_overflow`
3. Layer 6 receives signal → escalates policy to `degrade_all_but_critical`
4. Layer 7 emits `escalation_syswarn` and forensic marker
5. Action Executor: updates policy mode, emits syswarn event

**Failure Cascade Risk:** Medium. If overflow persists, policy escalates to emergency shutdown.

---

### 7.4 Scenario: Pulse Loop Stalls

**Trigger:** Pulse loop blocks for >1 second (e.g., writer fsync hangs).

**Expected Behavior:**
1. **Prevention:** All writer operations are non-blocking; fsync happens in pulse loop but with timeout.
2. If pulse stalls, watchdog thread (external to pulse_engine) detects missed pulses.
3. Watchdog emits `heartbeat_missed` signal to Layer 1 on next pulse.
4. Layer 1 → Layer 2 → Layer 4: route all categories to hotspare.
5. Layer 7 emits forensic marker.

**Failure Cascade Risk:** High if no watchdog. **Mitigation:** Add external watchdog thread that monitors pulse_id increments.

---

### 7.5 Scenario: All Writers Fail Simultaneously

**Trigger:** Filesystem becomes read-only; all writers raise exceptions.

**Expected Behavior:**
1. Layer 1 emits `writer_exception` for all categories
2. Layer 2 routes all categories to hotspare
3. Layer 3 marks all writers as failed
4. Layer 4 initiates hotspare takeover
5. Layer 5 may emit `disk_pressure_critical` (if related)
6. Layer 6 escalates to `policy_emergency_shutdown`
7. Layer 7 emits forensic marker
8. Action Executor: routes all to hotspare, initiates graceful shutdown after 30s

**Failure Cascade Risk:** None. HotSpare preserves last 30s; system shuts down gracefully.

---

## 8. Policy Boundaries

### 8.1 Decision: When to Route to HotSpare

**Made by:** Layer 2 (Event Routing)

**Timing:** Immediately upon receiving `writer_failed` or `writer_stalled` signal

**Rationale:** Routing is a mechanical response to failure detection; no policy judgment needed.

---

### 8.2 Decision: When to Drop Low-Priority Categories

**Made by:** Layer 6 (Policy)

**Timing:** After evaluating disk pressure + writer health + hotspare status

**Rationale:** Dropping categories is a policy decision that requires holistic view of system stress.

---

### 8.3 Decision: When to Initiate HotSpare Takeover

**Made by:** Layer 4 (HotSpare Management)

**Timing:** Immediately upon receiving `route_to_hotspare` signal for any category

**Rationale:** Takeover is a mechanical response to routing change; ensures last 30s are preserved.

---

### 8.4 Decision: When to Emit System Warning

**Made by:** Layer 7 (Escalation)

**Timing:** Immediately upon receiving any critical severity signal

**Rationale:** Escalation is observability, not policy; emits warnings for all critical events.

---

### 8.5 Decision: When to Initiate Emergency Shutdown

**Made by:** Layer 6 (Policy)

**Timing:** Only after disk critical AND hotspare overflow AND 3+ writers failed

**Rationale:** Shutdown is the most severe policy decision; requires multiple failure conditions.

---

## 9. Alternative Decompositions

### 9.1 Merge Layer 3 (Writer Management) and Layer 4 (HotSpare Management)?

**Argument:** Both manage write destinations; could be unified.

**Rejection:** Writer management tracks health of primary writers; HotSpare management tracks backup buffer. Merging creates a "manager of managers" that handles both primary and backup, violating separation of concerns.

---

### 9.2 Merge Layer 5 (Disk Pressure) into Layer 1 (Failure Detection)?

**Argument:** Disk pressure is a failure signal.

**Rejection:** Disk pressure is **not a failure**; it's a trend observation. Layer 1 detects acute failures (exceptions, stalls); Layer 5 observes chronic conditions (disk usage). Merging conflates reactive and proactive monitoring.

---

### 9.3 Split Layer 6 (Policy) into "Degradation Policy" and "Shutdown Policy"?

**Argument:** Degradation and shutdown are different concerns.

**Rejection:** Both are policy decisions based on system stress. Splitting creates two layers that read the same inputs and make similar decisions. Keep unified; use policy mode enum to distinguish behavior.

---

### 9.4 Eliminate Layer 7 (Escalation) and Emit Warnings from Each Layer?

**Argument:** Each layer knows when to escalate.

**Rejection:** Escalation requires holistic view of all signals to avoid spam and ensure forensic markers include full context. Distributed escalation leads to redundant warnings and incomplete forensic data.

---

## 10. Open Questions for Round 2

1. **File structure:** Should each layer be a separate file, or group related layers?
2. **Action Executor implementation:** Single function with switch/case, or action registry?
3. **Signal passing:** List of dicts, or custom Signal class with methods?
4. **State persistence:** In-memory only, or periodic snapshots to disk?
5. **Watchdog thread:** Should watchdog be part of pulse_engine, or external?
6. **TrueVision stub:** What category name, what placeholder fields?

---

## End of Round 1 Specification

This architecture is **ready for pseudocode** if approved. Awaiting engineering QC and Round 2 prompt.
