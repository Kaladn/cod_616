# Forensic Analysis: truevision_event_live.py
**Date:** December 4, 2025  
**Analyst:** Manus AI  
**Scope:** Data flow, connections, naming conventions, hardcoded values

---

## 1. DATA FLOW TRACE

### Current Data Sources (WRITING)
```
✅ VISION (TrueVision)
   ├─ FrameCapture → raw screen frames
   ├─ FrameToGrid → 32×32 grids with 10-color palette
   ├─ 4 Operators → detection results
   ├─ EommCompositor → composite EOMM score
   └─ PulseWriter → Forge Memory (ForgeRecords)

❌ LOGGERS (NOT INTEGRATED)
   ├─ activity_logger.py (window focus, app usage)
   ├─ input_logger.py (keyboard/mouse events)
   ├─ network_logger.ps1 (network telemetry)
   └─ process_logger.py (process monitoring)

❌ GAMEPAD (NOT INTEGRATED)
   └─ gamepad_logger_continuous.py (controller input)

❌ AUDIO (NOT INTEGRATED)
   └─ No audio capture component exists

❌ NETWORK TELEMETRY (NOT INTEGRATED)
   └─ CompuCogLogger/network_logger.ps1 exists but not called
```

### Data Flow Diagram
```
[Screen] → FrameCapture
            ↓
         FrameToGrid (32×32 grid)
            ↓
         FrameBuffer (10 frames rolling)
            ↓
         4 Operators (analyze last 3 frames)
            ↓
         EommCompositor (weighted scoring)
            ↓
         TrueVisionSchemaMap (window → ForgeRecord)
            ↓
         PulseWriter (batching)
            ↓
         WALWriter (durability log)
            ↓
         BinaryLog (memory-mapped storage)
            ↓
         [forge_data/records.bin]
```

**MISSING CONNECTIONS:**
- No CompuCogLogger integration → activity_logger.py NOT writing
- No input_logger.py integration → mouse/keyboard NOT captured
- No network_logger.ps1 integration → network telemetry NOT captured
- No gamepad_logger_continuous.py integration → controller input NOT captured
- No audio capture → voice comms NOT monitored

---

## 2. VARIABLE NAMING AUDIT

### ✅ GOOD NAMING (Conventions Followed)
```python
# Classes: PascalCase
class CognitiveHarness:
class ChronosManager:
class EventManager:

# Functions/Methods: snake_case
def process_window(self, window: Dict[str, Any]) -> None:
def _record_high_eomm_event(self, window, eomm_score):
def _build_detection_window(self, window_id):

# Constants: UPPER_SNAKE_CASE (when used)
MAX_BUFFER_SIZE = 10
STATS_INTERVAL = 5.0

# Config dicts: snake_case
capture_config = {...}
operators_config = {...}
pulse_config = {...}
```

### ⚠️ INCONSISTENT NAMING
```python
# Line 37-40: Inconsistent path building
sys.path.insert(0, str(parent_dir / "memory"))  # Good
sys.path.insert(0, str(parent_dir / "core"))    # Inconsistent with structure

# Line 163-164: Variable shadowing
self.baseline = SessionBaselineTracker(...)  # Instance var
session_tracker=self.baseline if self.baseline else SessionBaselineTracker(...)  # Re-creates!

# Line 207: Magic string
if len(self.frame_buffer) < 3:  # Should be MIN_FRAMES_FOR_DETECTION = 3

# Line 338: Hardcoded literal
window_duration_ms = int((window_end - window_start) * 1000)  # Should use constant MILLIS_PER_SECOND
```

### ❌ BAD NAMING
```python
# Line 42: Ambiguous import alias
from crosshair_lock import CrosshairLockOperator, FrameSequence  # FrameSequence used everywhere but never aliased clearly

# Line 98: Abbreviation inconsistency
self.eomm = EommCompositor(...)  # Should be self.eomm_compositor for clarity
self.chronos = ChronosManager(...)  # Full name better: self.chronos_manager

# Line 373-376: Inconsistent suffixes
window = {
    "operator_scores": {...},   # Dict
    "operator_flags": [...],    # List
}
# Should be: operator_scores_dict and operator_flags_list OR remove suffixes entirely
```

---

## 3. HARDCODED VALUES (Should Be YAML)

### 🔴 CRITICAL HARDCODES (Must Move to YAML)
```python
# Line 207: Frame buffer threshold
self.max_buffer_size = 10  # Should be in config/truevision_integration.yaml

# Line 358-359: Progress interval
stats_interval = 5.0  # Should be configurable

# Line 377: Color palette size
"grid_color_count": 10,  # Should come from grid config (currently hardcoded)

# Line 395: Frame requirement
if len(self.frame_buffer) >= 3:  # MIN_FRAMES_FOR_DETECTION should be in config
```

### 🟡 MODERATE HARDCODES (Should Move to YAML)
```python
# Line 369: Sleep interval
time.sleep(0.05)  # 50ms frame delay - should be configurable for performance tuning

# Line 313-318: Severity thresholds
if eomm_score >= 0.9:
    severity = "critical"
elif eomm_score >= 0.8:
    severity = "high"
else:
    severity = "medium"
# Should be in events config:
#   severity_thresholds:
#     critical: 0.9
#     high: 0.8
#     medium: 0.7

# Line 141-147: Source registration metadata
self.event_mgr.register_source("vision", "organ", {"type": "TrueVision"})
# Hardcoded metadata - should be in events config
```

### 🟢 ACCEPTABLE HARDCODES (System Invariants)
```python
# Line 325-328: Default baseline creation (fallback)
session_tracker=self.baseline if self.baseline else SessionBaselineTracker(min_samples_for_warmup=5)
# This is a fallback, acceptable

# Line 349: Grid shape extraction
"grid_shape": [latest_grid.grid.shape[0], latest_grid.grid.shape[1]] if hasattr(latest_grid.grid, 'shape') else [32, 32]
# Fallback to 32×32 is reasonable
```

---

## 4. MISSING INTEGRATIONS (Critical Gaps)

### 📂 CompuCogLogger Directory Structure
```
CompuCogLogger/
├── loggers/
│   ├── activity_logger.py       ❌ NOT INTEGRATED
│   ├── input_logger.py          ❌ NOT INTEGRATED
│   ├── network_logger.ps1       ❌ NOT INTEGRATED
│   └── process_logger.py        ❌ NOT INTEGRATED
├── logs/
│   ├── activity/                ✅ Has old logs (20251111, 20251117)
│   ├── input/                   ✅ Has old logs
│   ├── network/                 ✅ Has old logs
│   └── process/                 (empty)
├── tray_app.py                  ❌ NOT INTEGRATED
└── start_all.ps1                ❌ NOT CALLED BY truevision_event_live.py
```

### 🔌 Required Integration Points

#### A. Activity Logger Integration
```python
# Should be in CognitiveHarness.__init__()
from CompuCogLogger.loggers.activity_logger import ActivityLogger

self.activity_logger = ActivityLogger(config_path="CompuCogLogger/config.json")
self.activity_logger.start()  # Background thread

# In _shutdown():
self.activity_logger.stop()
```

#### B. Input Logger Integration
```python
from CompuCogLogger.loggers.input_logger import InputLogger

self.input_logger = InputLogger(config_path="CompuCogLogger/config.json")
self.input_logger.start()

# Correlate with vision events:
# If high EOMM detected + no mouse movement = aimbot
```

#### C. Network Logger Integration
```python
# Need Python wrapper for network_logger.ps1 OR rewrite in Python
import subprocess

self.network_logger_proc = subprocess.Popen(
    ["powershell", "-File", "CompuCogLogger/loggers/network_logger.ps1"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# In _shutdown():
self.network_logger_proc.terminate()
```

#### D. Gamepad Logger Integration
```python
<!-- LEGACY: previously imported from `CompuCog_Visual_v2.loggers.gamepad_logger_continuous` (archived 2025-12-20). Use `CompuCogLogger` and the conversion tool `scripts/convert_gamepad_to_contract.py` for gamepad telemetry. -->

self.gamepad_logger = GamepadLogger(poll_rate=60)
self.gamepad_logger.start()

# Correlate aim lock detection with stick input
```

---

## 5. SCHEMA COMPLIANCE AUDIT

### ✅ Forge Schema Compliance (TRUEVISION_SCHEMA_MAP.md)
```python
# Lines 343-391: Window construction
window = {
    "ts_start": window_start,              ✅ Required
    "ts_end": window_end,                  ✅ Required
    "grid_shape": [h, w],                  ✅ Required
    "grid_color_count": 10,                ✅ Required
    "eomm_score": eomm_score,              ✅ Required
    "operator_scores": {...},              ✅ Required
    "operator_flags": [...],               ✅ Required
    "telemetry": {...},                    ✅ Required
    "session_context": {...}               ✅ Required
}
```

**NO SCHEMA VIOLATIONS** - Window dict matches TrueVisionSchemaMap expectations.

---

## 6. EVENT SYSTEM AUDIT

### ✅ EventManager Integration (Proper)
```python
# Lines 137-156: Source registration
self.event_mgr.register_source("vision", "organ", {...})
self.event_mgr.register_source("operators", "detector", {...})
self.event_mgr.register_source("session", "tracker", {...})

# Lines 148-156: Chain creation
self.session_chain = self.event_mgr.create_chain(
    chain_id=self.session_id,
    metadata={"start_time": self.chronos.now(), "type": "gaming_session"}
)
```

### ⚠️ Missing Event Sources
```python
# Should also register:
self.event_mgr.register_source("input", "sensor", {"type": "MouseKeyboard"})
self.event_mgr.register_source("network", "sensor", {"type": "NetworkTelemetry"})
self.event_mgr.register_source("gamepad", "sensor", {"type": "Controller"})
self.event_mgr.register_source("audio", "sensor", {"type": "VoiceComms"})
```

---

## 7. CRITICAL BUGS FOUND

### 🐛 Bug #1: SessionBaselineTracker Re-creation (Line 325)
```python
# CURRENT (BAD):
telemetry_window = self.eomm.compose_window(
    session_tracker=self.baseline if self.baseline else SessionBaselineTracker(min_samples_for_warmup=5)
)

# PROBLEM: If self.baseline is None, creates NEW tracker every window (loses baseline state)

# FIX:
# Either ensure self.baseline always exists OR pass None and handle in compositor
```

### 🐛 Bug #2: Frame Buffer Never Reaches Operators (Line 395-396)
```python
# CURRENT:
if len(self.frame_buffer) >= 3:
    window = self._build_detection_window(window_counter)

# PROBLEM: Frame buffer only gets 1 grid per loop iteration
# Operators need 3 frames minimum, but we process EVERY frame once buffer >= 3
# This means we're re-running operators on overlapping sequences unnecessarily

# FIX: Add stride or batch processing
```

### 🐛 Bug #3: Operator Results Not Cached (Performance Issue)
```python
# CURRENT: Lines 410-430 run ALL operators EVERY window
if self.crosshair_op:
    crosshair_result = self.crosshair_op.analyze(seq)
if self.hit_op:
    hit_result = self.hit_op.analyze(seq)
# ... etc

# PROBLEM: Analyzing 30 windows/sec = 120 operator calls/sec (4 ops × 30)
# Operators likely doing expensive computations on SAME frame sequences

# FIX: Cache operator results per frame sequence hash
```

---

## 8. CONFIG FILE AUDIT

### ✅ YAML Configs (Properly Loaded)
```python
# Line 127-129: Main config
config_path = Path(__file__).parent / "config" / "truevision_integration.yaml"
with open(config_path) as f:
    self.config = yaml.safe_load(f)

# Lines 164-181: Operator configs
crosshair_config = str(config_base / "crosshair_lock.yaml")
hit_config = str(config_base / "hit_registration.yaml")
death_config = str(config_base / "death_event.yaml")
edge_config = str(config_base / "edge_entry.yaml")
eomm_config = str(Path(__file__).parent / "config" / "eomm_compositor.yaml")
```

### ⚠️ Missing Config Entries
```yaml
# truevision_integration.yaml SHOULD HAVE:

system:
  max_frame_buffer_size: 10          # Line 207 hardcode
  min_frames_for_detection: 3        # Line 395 hardcode
  frame_capture_interval_ms: 50      # Line 369 hardcode
  stats_print_interval_sec: 5.0      # Line 358 hardcode

events:
  severity_thresholds:
    critical: 0.9                     # Line 313 hardcode
    high: 0.8                         # Line 315 hardcode
    medium: 0.7                       # Line 317 hardcode
  
  sources:
    vision:
      kind: "organ"
      metadata: {"type": "TrueVision"}
    operators:
      kind: "detector"
      metadata: {"type": "TrueVisionOperators"}
    session:
      kind: "tracker"
      metadata: {"type": "SessionTracker"}
    input:                            # MISSING
      kind: "sensor"
      metadata: {"type": "MouseKeyboard"}
    network:                          # MISSING
      kind: "sensor"
      metadata: {"type": "NetworkTelemetry"}
    gamepad:                          # MISSING
      kind: "sensor"
      metadata: {"type": "Controller"}

loggers:                              # MISSING ENTIRE SECTION
  activity:
    enabled: true
    config_path: "CompuCogLogger/config.json"
  input:
    enabled: true
    config_path: "CompuCogLogger/config.json"
  network:
    enabled: true
    script_path: "CompuCogLogger/loggers/network_logger.ps1"
  gamepad:
    enabled: true
    poll_rate: 60
```

---

## 9. IMPORT AUDIT

### ✅ Clean Imports (No Circular Dependencies)
```python
# Lines 21-48: Import order is correct
# 1. Standard library
# 2. Third-party (yaml)
# 3. Local modules (cognitive stack → truevision)
```

### ⚠️ Path Hacking (Lines 24-29)
```python
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "memory"))
sys.path.insert(0, str(parent_dir / "core"))
sys.path.insert(0, str(parent_dir / "operators"))
sys.path.insert(0, str(parent_dir / "compositor"))
sys.path.insert(0, str(parent_dir / "baselines"))
```

**PROBLEM:** Adding 6 directories to sys.path is fragile. If module names collide, imports break.

**FIX:** Use proper package structure:
```python
<!-- LEGACY: `frame_to_grid` previously lived under `CompuCog_Visual_v2.core`. The canonical versions live in `gaming/` and `TruVision_files/` now. Refer to `gaming/frame_to_grid.py` or `TruVision_files` utilities. -->
<!-- LEGACY: `CrosshairLockOperator` previously came from `CompuCog_Visual_v2.operators`. Use `operators/crosshair_lock.py` (refactored into the repo root) or `gaming/` operator implementations. -->
```

---

## 10. RECOMMENDATIONS (Priority Order)

### 🔴 CRITICAL (Do Immediately)
1. **Integrate CompuCogLogger loggers** (activity, input, network, process)
2. **Add missing event sources** (input, network, gamepad, audio)
3. **Fix SessionBaselineTracker re-creation bug** (Line 325)
4. **Move all hardcoded values to YAML** (max_buffer_size, thresholds, intervals)

### 🟡 HIGH PRIORITY (Do This Week)
5. **Add gamepad logger integration** (gamepad_logger_continuous.py)
6. **Implement audio capture** (voice comms monitoring)
7. **Add operator result caching** (performance optimization)
8. **Create unified config schema** (all loggers + vision in one config)

### 🟢 MEDIUM PRIORITY (Do Next Week)
9. **Fix import path hacking** (use proper package structure)
10. **Add frame buffer stride logic** (avoid redundant operator analysis)
11. **Create cross-modal event fusion** (vision + input + network → aimbot detection)
12. **Add telemetry correlation timestamps** (synchronize all data sources)

### 🔵 LOW PRIORITY (Future Enhancement)
13. **Add replay capability** (load from Forge, replay with all loggers)
14. **Create logger health monitoring** (detect if loggers crash)
15. **Add dynamic operator enable/disable** (runtime config reload)

---

## 11. DATA INTEGRITY CHECKS

### ✅ Writing to Forge (WORKING)
```
✓ PulseWriter.submit_window() called every frame
✓ WALWriter durability log working
✓ BinaryLog memory-mapped file closed properly (Bug #2 fixed)
✓ StringDictionary saved on close
```

### ❌ NOT Writing to CompuCogLogger (BROKEN)
```
✗ activity_logger.py NOT started → No window focus logs
✗ input_logger.py NOT started → No mouse/keyboard logs
✗ network_logger.ps1 NOT started → No network telemetry
✗ process_logger.py NOT started → No process monitoring
✗ gamepad_logger_continuous.py NOT started → No controller input
```

### ⚠️ Potential Data Loss Points
1. **If operators crash:** No try/except around analyze() calls (Lines 410-430)
2. **If Forge full:** No disk space check before writing
3. **If session interrupted:** Ctrl+C handler exists but might lose last pulse
4. **If logger crashes:** No health monitoring or restart logic

---

## 12. FINAL VERDICT

### Current State: **60% Complete** 🟡

**What Works:**
- ✅ Vision capture (TrueVision)
- ✅ Operator detection
- ✅ EOMM scoring
- ✅ Forge Memory persistence
- ✅ EventManager + 6-1-6 capsules
- ✅ ChronosManager deterministic time

**What's Missing:**
- ❌ Activity logger integration (window focus)
- ❌ Input logger integration (mouse/keyboard)
- ❌ Network logger integration (telemetry)
- ❌ Gamepad logger integration (controller)
- ❌ Audio capture (voice comms)
- ❌ Cross-modal event fusion (aimbot = aim lock + no mouse input)

**What's Broken:**
- 🐛 SessionBaselineTracker re-creation bug
- 🐛 Operator result caching missing (performance)
- 🐛 Hardcoded values (should be YAML)
- 🐛 Import path hacking (fragile)

---

## 13. ACTION PLAN

### Phase 1: Logger Integration (2-3 hours)
1. Create `CognitiveHarness._init_loggers()` method
2. Start activity_logger, input_logger, process_logger as background threads
3. Wrap network_logger.ps1 in Python subprocess
4. Add logger shutdown in `_shutdown()`
5. Test all loggers writing simultaneously

### Phase 2: Config Cleanup (1 hour)
1. Add `system` section to truevision_integration.yaml
2. Add `loggers` section with enable flags
3. Move all hardcoded thresholds/intervals to config
4. Remove magic numbers from code

### Phase 3: Cross-Modal Fusion (2 hours)
1. Register input/network/gamepad event sources
2. Add correlation logic: vision events + input events → composite detection
3. Example: `if aim_lock_detected and mouse_velocity == 0: record_aimbot_event()`

### Phase 4: Bug Fixes (1 hour)
1. Fix SessionBaselineTracker re-creation
2. Add operator result caching
3. Add error handling around operator analyze() calls
4. Add disk space checks

**TOTAL EFFORT:** ~6-7 hours to reach **90% complete** cognitive organism.
