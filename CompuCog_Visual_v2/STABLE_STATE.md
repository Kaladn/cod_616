# CompuCog Visual v2.0 — Stable State Registry

**Current Stable Version**: v2.0.1  
**Last Updated**: December 2, 2025

---

## Version History

### v2.0.1 — Continuous Gamepad Telemetry (Current Stable)
**Date**: December 2, 2025  
**Status**: ✅ Stable  

**Components**:
- TrueVision v2.0 detection pipeline (10 operators, EOMM compositor)
- 5 telemetry loggers: input, activity, process, network, gamepad_continuous
- Unified launcher: `launch_truevision.ps1`
- Analysis tools: `analyze_correlations.py`, `show_math.py`
- Memory & reasoning scaffolding (directories prepared)

**Key Changes**:
- ✅ Continuous gamepad event stream (microsecond timestamps)
- ✅ Event counter CLI (no spam)
- ✅ Removed old aggregated gamepad logger
- ✅ Gamepad logs to `logs/gamepad/gamepad_stream_YYYYMMDD.jsonl`
- ✅ 1,478 events logged per test session

**Validated**:
- ✅ Xbox One controller detection working
- ✅ Button, axis (stick/trigger), hat (D-pad) events captured
- ✅ 60Hz polling with <20ms event timestamps
- ✅ Correlation-ready format (ISO timestamps, JSONL)

**Files**:
```
CompuCog_Visual_v2/
├── loggers/
│   └── gamepad_logger_continuous.py  [NEW - continuous event stream]
├── gaming/
│   ├── show_math.py                  [NEW - mathematical analysis]
│   ├── analyze_correlations.py       [EXISTING - telemetry correlation]
│   ├── truevision_live_run.py
│   └── truevision_smoke_test.py
├── launch_truevision.ps1
└── USAGE.md
```

**Rollback Command**:
```powershell
# To revert to v2.0.0 (aggregated gamepad):
git checkout v2.0.0-tag
# OR restore from backup:
Copy-Item C:\Backups\CompuCog_Visual_v2_v2.0.0\* -Destination CompuCog_Visual_v2\ -Recurse -Force
```

---

### v2.0.0 — Initial v2 Migration (Previous Stable)
**Date**: November 2025  
**Status**: ⚠️ Superseded by v2.0.1  

**Components**:
- TrueVision v2.0 detection pipeline
- 5 telemetry loggers: input, activity, process, network, gamepad_aggregated
- Memory & reasoning scaffolding

**Key Changes**:
- ✅ Migrated all v1 files to v2 structure
- ✅ Added memory/ and reasoning/ directories
- ✅ Integrated 5 telemetry loggers
- ✅ Created unified launcher
- ✅ 7-minute live capture validated (165 windows, 68.5% EOMM)
- ⚠️ Gamepad used 3-second aggregated sampling (replaced in v2.0.1)

**Rollback Command**:
```powershell
# Restore v2.0.0 from git tag:
git checkout v2.0.0-tag
```

---

### v1.0.0 — TrueVision v1 Baseline
**Date**: October 2025  
**Status**: ⚠️ Legacy (use v2.x)  

**Components**:
- TrueVision v1.0 detection pipeline
- Bot match baselines (match1_casual_bots)
- Initial EOMM compositor

**Key Changes**:
- ✅ 10 detection operators validated
- ✅ EOMM weighted scoring implemented
- ✅ Session baseline tracking (Welford's algorithm)
- ✅ Smoke test: 8 windows (50.9% EOMM on easy bots)

**Rollback Command**:
```powershell
# Use v1 directory:
cd CompuCog_Visual
python gaming/truevision_live_run.py --duration 7
```

---

## Rollback Procedures

### Quick Rollback (File-Level)
If a single component fails, restore just that file:

```powershell
# Example: Restore old aggregated gamepad logger
Copy-Item C:\Backups\CompuCog_Visual_v2_v2.0.0\loggers\gamepad_logger.py `
  -Destination CompuCog_Visual_v2\loggers\
```

### Full Version Rollback (Git Tags)
If using Git version control:

```powershell
# View available versions
git tag

# Rollback to specific version
git checkout v2.0.0

# Return to latest
git checkout main
```

### Manual Rollback (Backup Restore)
If no Git, use file backups:

```powershell
# Create backup before changes
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item CompuCog_Visual_v2 -Destination "C:\Backups\CompuCog_Visual_v2_$timestamp" -Recurse

# Restore from backup
$backupPath = "C:\Backups\CompuCog_Visual_v2_20251201_143022"
Remove-Item CompuCog_Visual_v2 -Recurse -Force
Copy-Item $backupPath -Destination CompuCog_Visual_v2 -Recurse
```

---

## Component Stability Status

| Component | v2.0.1 Status | Last Tested | Notes |
|-----------|---------------|-------------|-------|
| TrueVision Detection | ✅ Stable | Dec 2, 2025 | 165 windows, 68.5% EOMM |
| Input Logger | ✅ Stable | Nov 2025 | 3-second samples |
| Activity Logger | ✅ Stable | Nov 2025 | 3-second samples |
| Process Logger | ✅ Stable | Nov 2025 | Continuous |
| Network Logger | ✅ Stable | Nov 2025 | Continuous |
| Gamepad Logger (Continuous) | ✅ Stable | Dec 2, 2025 | 1,478 events/session |
| Correlation Analysis | ✅ Stable | Nov 2025 | UTF-8 BOM handling |
| Math Analysis | ✅ Stable | Dec 2, 2025 | Grid/operator breakdown |
| Launch Script | ✅ Stable | Nov 2025 | 5 loggers + TrueVision |
| Memory Layer | 🚧 Planned | TBD | Scaffolding only |
| Reasoning Layer | 🚧 Planned | TBD | Scaffolding only |

---

## Testing Checklist (Pre-Release)

Before tagging a new stable version:

- [ ] **Smoke Test**: Run `truevision_smoke_test.py` (8 windows expected)
- [ ] **Live Capture**: Run `launch_truevision.ps1 -Duration 7` (verify all loggers start)
- [ ] **Gamepad Test**: Run `gamepad_logger_continuous.py --poll-rate 60` for 30 seconds (press buttons/sticks)
- [ ] **Log Validation**: Check all log files generated in `logs/` subdirectories
- [ ] **Correlation Test**: Run `analyze_correlations.py` on live capture output
- [ ] **Math Analysis**: Run `show_math.py` on telemetry (verify grid/operator display)
- [ ] **Error Check**: No unhandled exceptions in any logger output
- [ ] **Performance**: TrueVision maintains ~30 FPS capture rate

---

## Known Issues

### v2.0.1
- None currently

### v2.0.0
- ⚠️ Gamepad logger used 3-second aggregation (fixed in v2.0.1)
- ⚠️ Network logs had UTF-8 BOM causing parse errors (fixed in correlation tool)

---

## Emergency Contacts

**System Owner**: L.A. Mercey / Cortex Evolved  
**Documentation**: `USAGE.md`, `V1_V2_DIFF.md`  
**Support**: Check `README.md` in each component directory

---

## Version Tagging Convention

```
vMAJOR.MINOR.PATCH
├── MAJOR: Breaking changes (v1 → v2 migration)
├── MINOR: New features (continuous gamepad logging)
└── PATCH: Bug fixes (error handling improvements)
```

**Example**:
- `v2.0.0`: Initial v2 structure
- `v2.0.1`: Added continuous gamepad logging
- `v2.1.0`: Would add memory layer implementation
- `v3.0.0`: Would introduce breaking API changes
