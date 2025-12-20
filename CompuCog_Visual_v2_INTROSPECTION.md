# CompuCog_Visual_v2 — Introspection Report

> **ARCHIVED (2025-12-20):** The `CompuCog_Visual_v2` package has been archived and removed from `main`. A backup branch `backup/pre-removal-compucog_visual_v2` contains the full package if you need to restore or inspect it.
> Use the canonical TrueVision implementation under `TruVision_files/` and the `gaming/` package for ongoing development.

**Workspace Root:** D:\cod_616\CompuCog_Visual_v2
**Analysis Date:** December 19, 2025

---

## 1. MODULE IDENTITY

- **Module name:** CompuCog Visual v2 (TrueVision / EventManager / Forge Memory integration)
- **Intended role:** Live capture and analysis of game visual data (TrueVision) with event recording, operator-based detection, and persistent storage (Forge Memory). Implements 6-1-6 temporal capsules and an event pipeline for downstream forensic analysis.

**What this module DOES:**
- Captures screen frames via `FrameCapture` and converts to compact grids with `FrameToGrid`.
- Runs a set of vision operators (CrosshairLock, HitRegistration, DeathEvent, EdgeEntry) over frame sequences to detect manipulation patterns.
- Composes operator scores into an EOMM composite via `EommCompositor`.
- Records windows and telemetry to Forge Memory using `PulseWriter` / `BinaryLog` (persistent binary records).
- Records events and builds 6-1-6 capsules via `EventManager` and `ChronosManager` (deterministic timestamps).
- Provides a live harness `CognitiveHarness` that ties capture → operators → event recording → Forge writes.

**What this module explicitly DOES NOT do (based on code/comments):**
- Does NOT integrate activity/input/network/gamepad loggers by default (logger modules exist but are not initialized).
- Does NOT implement audio capture (explicitly missing).
- Does NOT implement cross-modal event fusion (no correlation logic beyond vision operators).
- Does NOT manage daemonized logger processes (`CompuCogLogger` exists as separate system, not invoked).

---

## 2. DATA OWNERSHIP

**Data types this module creates:**
- FrameGrid objects (small 32×32 numpy arrays with palette info)
- Telemetry windows (TelemetryWindow / dict with operator scores, eomm, timestamps)
- ForgeRecords (binary records persisted via PulseWriter/BinaryLog)
- Event objects (EventManager events, capsules)
- Logs and metadata saved under `forge_data/` and `logs/` (where loggers are enabled)

**Data types this module consumes:**
- YAML configuration: `gaming/config/truevision_integration.yaml`, operator configs, eomm composer config
- Raw screen frames (via FrameCapture)
- Operator result objects (from operator modules)

**Data types this module persists:**
- Binary Forge data (`forge_data/records.bin`, WAL files, strings.dict)
- Event memory (in-memory EventManager; persisted via Forge writes when windows submitted)

**Data types this module does not persist:**
- Raw video frames (frames converted to grid objects; raw images not persisted by default)
- Per-frame operator debug outputs (not written unless operators explicitly do so)

---

## 3. PUBLIC INTERFACES

**Public classes (not exhaustive):**
- `CognitiveHarness` (`gaming/truevision_event_live.py`) — main runtime harness that orchestrates capture, operators, event recording, and Forge writes.
- `ChronosManager` (`event_system/chronos_manager.py`) — deterministic time provider.
- `EventManager` (`event_system/event_manager.py`) — event recording and capsule construction.
- `FrameCapture`, `FrameToGrid` (`core/frame_to_grid.py`) — capture and grid conversion.
- Operator classes in `operators/` (e.g., `CrosshairLockOperator`, `HitRegistrationOperator`, `DeathEventOperator`, `EdgeEntryOperator`).
- `EommCompositor` (`compositor/eomm_compositor.py`) — produces composite EOMM score.
- Forge memory helpers: `build_truevision_forge_pipeline`, `PulseWriter`, `BinaryLog` (under `memory/forge_memory`).

**Public functions / methods (examples):**
- `CognitiveHarness.run(duration, verbose)` — run live capture loop.
- `CognitiveHarness.process_window(window)` — submit a window to Forge and record events.
- `EventManager.record_event()` / `attach_event_to_chain()` — event APIs.
- `FrameCapture.capture()` / `FrameToGrid.convert()` — capture and conversion.

**Expected inputs and outputs:**
- Inputs: Frames (RGB arrays), YAML configs, operator configs
- Outputs: Forge binary records, Event objects, console logs

**Side effects:**
- Disk I/O: Forge binary files and WAL; potential logs if loggers enabled
- Memory: EventManager in-memory state, frame buffer (bounded), PulseWriter buffers
- Stdout: Extensive initialization and progress prints
- Network: None by default (network logger is an optional external script)

---

## 4. DEPENDENCIES

**Internal imports (within CompuCog_Visual_v2):**
- event_system.* (ChronosManager, EventManager)
- memory.forge_memory.* (PulseWriter, BinaryLog)
- core.frame_to_grid (FrameCapture, FrameToGrid)
- operators.* (CrosshairLockOperator, etc.)
- compositor.eomm_compositor
- baselines.session_baseline

**External imports (stdlib / third-party observed):**
- Stdlib: `sys`, `time`, `argparse`, `pathlib`, `typing`, `yaml` (PyYAML)
- Third-party: `numpy`, `PIL` (Pillow), `mss` or `opencv-python` (FrameCapture), any operator-specific libraries

**Runtime import behavior:**
- Config files loaded at runtime from `gaming/config/*.yaml` (no dynamic remote loading)
- Logger modules exist but are not initialized by the harness (explicit imports present in `COMPONENT_MAP.md` recommendations)

---

## 5. STATE & MEMORY

**State held in memory (examples):**
- `CognitiveHarness.frame_buffer` — list of recent FrameGrid objects (bounded by `max_buffer_size`)
- `PulseWriter` buffers (batched records before flush)
- `BinaryLog` memory-mapped file handle
- `EventManager` in-memory event store and chains
- `SessionBaselineTracker` state (if enabled)

**Bounded vs Unbounded:**
- Frame buffer is bounded (`max_buffer_size`, default 10)
- PulseWriter batching controls memory for writes
- EventManager stores events in memory; retention policy not explicit in code

**State reset / eviction:**
- `frame_buffer` uses manual pop(0) to evict oldest frames when exceeding max size
- `PulseWriter.flush()` and `close()` invoked on shutdown for durability
- `SessionBaselineTracker` is constructed at startup and may be re-created in some code paths (documented bug)

---

## 6. INTEGRATION POINTS

**Explicit upstream inputs expected:**
- `gaming/config/truevision_integration.yaml` and operator-specific YAMLs
- Display or screen capture permission (mss/opencv works on host)
- Optional: CompuCogLogger config for loggers if they are integrated

**Explicit downstream outputs produced:**
- Forge binary records in `forge_data/`
- EventManager events and capsules
- Telemetry windows passed to PulseWriter

**Required external conditions:**
- Filesystem write permissions to `forge_data/` and `logs/`
- Monitor/display accessible for frame capture
- Python environment with `numpy`, `PIL`, `PyYAML`, and screen capture library

---

## 7. OPEN FACTUAL QUESTIONS

**Declared but unused code paths:**
- Logger modules under `loggers/` exist but are not initialized by `CognitiveHarness`.
- `CompuCogLogger` daemon tools exist but are not invoked by the harness.

**Defined but unraised exceptions:**
- Operators have no specific custom exceptions defined; operator failures are not caught (can raise runtime exceptions and bubble up).

**Referenced but missing components:**
- Audio capture component: explicitly noted as not present in `COMPONENT_MAP.md` and no `loggers/audio` exists.
- Cross-modal fusion rules: `gaming/config/fusion_rules.yaml` is missing.

**Implicit assumptions required for correct operation:**
- System assumes a working display capture pipeline (mss/opencv) with stable frame timing.
- SessionBaselineTracker state is assumed to persist across windows, but the code path may re-create it if disabled (bug reported).
- Operators assume three-frame sequences for detection (window size = 3 minimum).

---

**End of report.**

---

# Integration Planning Notes (user-requested)

Below are concise, prioritized items (actionable) to help plan and schedule integration work for CompuCog_Visual_v2 into the main workspace. These are derived from the facts above and `COMPONENT_MAP.md` and are intended to be used as an integration checklist.

1. Integrate loggers (Activity, Input, Process, Network, Gamepad) and register sources with `EventManager` (high priority).
2. Add `gaming/config/loggers.yaml` and `gaming/config/system.yaml` and move hardcoded values into config.
3. Fix `SessionBaselineTracker` re-creation bug and ensure baseline persistence across windows.
4. Implement basic cross-modal fusion rules (aim lock + no mouse input → aimbot event) and add `gaming/config/fusion_rules.yaml`.
5. Implement audio capture component and logger (microphone capture, VAD) and add to harness as a source.
6. Add operator result caching and error handling to improve robustness and performance.
7. Add tests for integrated pipelines (logger start/stop, Forge write/flush, event chain creation).
8. Create a phased timeline and estimate (COMPONENT_MAP contains estimates already; validate locally).

---

*Note:* This file contains factual extraction followed by a short planning summary as requested by the user. The factual report section respects the UNIVERSAL COPILOT INTROSPECTION PROMPT v1.1 rules (no opinions, no change suggestions). The planning notes are user-directed next steps to help schedule work.
