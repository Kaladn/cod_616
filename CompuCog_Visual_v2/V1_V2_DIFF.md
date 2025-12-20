# CompuCog_Visual v1 → v2 File Comparison

## Files in v1 NOT in v2

### Root Level (CompuCog_Visual/)
- ❌ `delta_extractor.py` - **MISSING** (delta frame extraction for video analysis)
- ❌ `requirements.txt` - **MISSING** (Python dependencies for v1)
- ❌ `test_visual.py` - **MISSING** (v1 test harness)
- ❌ `video_operators.py` - **MISSING** (video processing operators)
- ❌ `visual_fingerprint.py` - **MISSING** (fingerprinting utilities)
- ❌ `visual_sensor.py` - **MISSING** (sensor integration layer)

### Root Directories (CompuCog_Visual/)
- ❌ `config/` - **MISSING** (contains truevision_config.yaml, visual_config.yaml)
- ❌ `security/` - **MISSING** (security operators and whitelist.yaml)
- ❌ `utils/` - **MISSING** (empty directory, can ignore)

### gaming/ Subdirectory
- ✅ All v1 gaming/ Python files exist in v2
- ❌ `gaming/analysis/` - **MISSING** (deep_analysis_20251129_120513.json output)
- ❌ `gaming/operators/` - **MISSING** (10 operator files duplicated in v2/operators/)

### gaming/operators/ (v1 structure)
These files exist in v1 `gaming/operators/` but v2 moved them to root `operators/`:
- `base_operator.py`
- `color_shift.py`
- `crosshair_lock.py`
- `crosshair_motion.py`
- `death_event.py`
- `edge_entry.py`
- `flicker_detector.py`
- `hit_registration.py`
- `hud_stability.py`
- `peripheral_flash.py`

**Note**: v2 correctly moved operators to root-level `operators/` directory (better structure)

---

## Files in v2 NOT in v1

### Root Level (CompuCog_Visual_v2/)
- ✅ `launch_truevision.ps1` - **NEW** (unified launcher for full pipeline)
- ✅ `USAGE.md` - **NEW** (comprehensive system documentation)
- ✅ `compucog_schema.py` - **NEW** (copied from root for logger compatibility)

### Root Directories (CompuCog_Visual_v2/)
- ✅ `loggers/` - **NEW** (5 telemetry loggers: input, activity, process, network, gamepad)
- ✅ `logs/` - **NEW** (logger output directories: input/, activity/, process/, network/, gamepad/)
- ✅ `memory/` - **NEW** (scaffolding for Memory layer - v2 cognitive enhancement)
- ✅ `reasoning/` - **NEW** (scaffolding for Reasoning layer - v2 cognitive enhancement)

### gaming/ Subdirectory
- ✅ `analyze_correlations.py` - **NEW** (correlate visual detection with logger telemetry)

---

## Critical Missing Components (Need Migration)

### 1. **config/ directory**
**Status**: ❌ MISSING  
**Impact**: HIGH  
**Files**:
- `truevision_config.yaml` (exists in v2 gaming/ - OK)
- `visual_config.yaml` (NOT in v2 - may be legacy)

**Action**: Check if `visual_config.yaml` is needed, or if `truevision_config.yaml` replaced it

---

### 2. **security/ directory**
**Status**: ❌ MISSING  
**Impact**: UNKNOWN  
**Files**:
- `config.yaml`
- `whitelist.yaml`
- `operators/` (security-specific operators)
- `logs/`

**Action**: Investigate if security operators are separate system or part of TrueVision

---

### 3. **Root-level utilities**
**Status**: ❌ MISSING  
**Impact**: MEDIUM-HIGH  

| File | Purpose | Action |
|------|---------|--------|
| `delta_extractor.py` | Extract frame deltas for video analysis | Determine if needed for v2 |
| `visual_fingerprint.py` | Fingerprinting utilities | May be replaced by `analyze_fingerprints.py` |
| `visual_sensor.py` | Sensor integration layer | Check if replaced by `gaming_sensor.py` |
| `video_operators.py` | Video processing operators | Determine if still used |
| `test_visual.py` | v1 test harness | Check if replaced by `truevision_smoke_test.py` |

---

## Recommendations

### Immediate Actions:
1. ✅ **DONE**: Operators moved to root-level `operators/` (better structure)
2. ✅ **DONE**: All telemetry loggers integrated (input, activity, process, network, gamepad)
3. ✅ **DONE**: Unified launcher created (`launch_truevision.ps1`)
4. ✅ **DONE**: Comprehensive documentation (`USAGE.md`)

### Pending Review:
1. ⏳ **Investigate `security/` directory** - Determine if needed for v2
2. ⏳ **Check `visual_config.yaml`** - May be legacy, replaced by `truevision_config.yaml`
3. ⏳ **Review root-level utilities** - Determine which are still needed vs replaced
4. ⏳ **Migrate missing utilities if needed** - Copy delta_extractor, visual_fingerprint, etc.

### v2 Enhancements (Not in v1):
- ✅ Memory layer scaffolding (pattern retention across sessions)
- ✅ Reasoning layer scaffolding (causal inference)
- ✅ Complete telemetry integration (5 loggers + visual detection)
- ✅ Correlation analysis tool (analyze_correlations.py)
- ✅ One-command launch system

---

## Summary

**v2 Status**: 🟢 **OPERATIONAL** (core TrueVision + full telemetry working)

**Missing from v1**:
- `security/` directory (unknown impact - needs investigation)
- Some root-level utilities (may be legacy or replaced)
- `config/` directory (but truevision_config.yaml exists in gaming/)

**v2 is functionally complete** for TrueVision detection + telemetry fusion. The missing v1 files need review to determine if they're:
1. Legacy components (pre-TrueVision)
2. Replaced by v2 equivalents
3. Still needed (require migration)

**Next Step**: User to clarify purpose of `security/` directory and confirm which v1 utilities are still required.
