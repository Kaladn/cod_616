# TrueVision v2.0 - Complete System Usage Guide

## 🎯 Quick Start (Single Command)

```powershell
cd C:\Users\mydyi\Desktop\CompuCog_Deployment\CompuCog_Visual_v2
.\launch_truevision.ps1 -Duration 15
```

This launches the **full telemetry pipeline** (input + activity + process + network + visual detection) for a 15-minute match capture.

---

## 📦 What's Included

**TrueVision v2.0** is a **standalone, self-contained EOMM detection system** with:

### **Visual Detection Layer**
- 10 detection operators (4 primary + 6 auxiliary)
- EOMM scoring via weighted compositor
- Session baseline tracking (Welford's algorithm)
- Frame capture at ~30 FPS (2560×1440 → 32×32 ARC grid)

### **Telemetry Logger Layer**
- **Input Logger**: Keyboard/mouse events, idle time, audio/camera status
- **Activity Logger**: Active window, process name, CPU/memory usage
- **Process Logger**: Running processes, system resources
- **Network Logger**: Bytes/packets sent/received, active connections

### **Analysis Tools**
- **extract_baselines.py**: Separate easy/hard bot fingerprints
- **fuse_telemetry.py**: Merge visual detection with logger telemetry
- **truevision_smoke_test.py**: Validate pipeline with synthetic data

---

## 🚀 Usage Scenarios

### **Scenario 1: Capture Live Match (Full Telemetry)**

```powershell
# Start everything for 15-minute match
.\launch_truevision.ps1 -Duration 15

# Output files created:
#   gaming/telemetry/truevision_live_20251202_HHMMSS.jsonl  (visual detection)
#   logs/input/input_activity_20251202.jsonl                (keyboard/mouse)
#   logs/activity/user_activity_20251202.jsonl              (windows/processes)
#   logs/process/process_activity_20251202.jsonl            (system resources)
#   logs/network/telemetry_20251202.jsonl                   (network packets)
```

### **Scenario 2: Quick Validation Run (5 minutes)**

```powershell
.\launch_truevision.ps1 -Duration 5
```

### **Scenario 3: Visual Detection Only (Loggers Already Running)**

```powershell
# If loggers already running from previous session
.\launch_truevision.ps1 -Duration 10 -SkipLoggers

# Or run directly
cd gaming
python truevision_live_run.py --duration 10
```

### **Scenario 4: Smoke Test (Validate Pipeline)**

```powershell
cd gaming
python truevision_smoke_test.py

# Output: 8 synthetic windows with known manipulation patterns
# Validates: operators + compositor + baseline tracking + JSONL export
```

---

## 🧪 Analysis Workflow

### **Step 1: Capture Match Data**

```powershell
.\launch_truevision.ps1 -Duration 15
```

**Output**: `gaming/telemetry/truevision_live_20251202_150022.jsonl`

---

### **Step 2: Extract Bot Baselines (If Calibrating)**

```powershell
cd gaming
python extract_baselines.py telemetry\match1_casual_bots.jsonl

# Creates 3 files:
#   match1_casual_bots_easy_bots.json      (easy bot fingerprints)
#   match1_casual_bots_hard_bots.json      (hard bot fingerprints)
#   match1_casual_bots_thresholds.json     (recommended detection thresholds)
```

**Use case**: Bot matches with known easy→hard difficulty transitions provide ground truth for "fair" vs "elevated difficulty" baselines.

---

### **Step 3: Fuse Telemetry (Complete Forensic Context)**

```powershell
python fuse_telemetry.py telemetry\truevision_live_20251202_150022.jsonl

# Creates: truevision_live_20251202_150022_FUSED.jsonl
# Each window now includes:
#   - Visual detection (EOMM score + flags)
#   - Input telemetry (mouse/keyboard events)
#   - Network telemetry (bytes/packets)
#   - Activity telemetry (active window + CPU/memory)
```

**Why fuse?** To correlate visual manipulation with system behavior:
- Visual spike + network jitter = server-side manipulation
- Visual spike + input resistance = client-side aim suppression
- Visual spike + CPU spike = engine processing non-standard logic

---

## 📊 Output Format

### **Visual Telemetry (TelemetryWindow)**

```json
{
  "window_start_epoch": 1733152210.5,
  "window_end_epoch": 1733152211.5,
  "window_duration_ms": 1000,
  "operator_results": [
    {
      "operator_name": "crosshair_lock",
      "confidence": 0.65,
      "flags": ["AIM_RESISTANCE"],
      "metrics": {"on_target_pct": 0.80, "motion_avg": 0.05},
      "metadata": {}
    }
  ],
  "eomm_composite_score": 0.75,
  "eomm_flags": ["AIM_RESISTANCE", "HITBOX_DRIFT"],
  "session_id": "live_session_20251202_150022",
  "frame_count": 30,
  "metadata": {
    "ttk_mean": 2.5,
    "ttd_mean": 1.8,
    "ttk_variance": 0.3
  }
}
```

### **Fused Telemetry (TelemetryWindow + Loggers)**

```json
{
  ... (all fields above) ...
  "input_telemetry": {
    "idle_seconds": 0,
    "is_active": true,
    "audio_active": true,
    "camera_active": false
  },
  "network_telemetry": {
    "bytes_sent": 1203,
    "bytes_recv": 4891,
    "packets_sent": 8,
    "packets_recv": 24,
    "connections_active": 1
  },
  "activity_telemetry": {
    "active_window_title": "Call of Duty",
    "active_process_name": "cod.exe",
    "cpu_percent": 32.0,
    "memory_percent": 18.5
  }
}
```

---

## 🔍 Detection Operators

### **Primary Operators (EOMM Core)**

| Operator | Weight | Detects | Key Flags |
|----------|--------|---------|-----------|
| **crosshair_lock** | 30% | Aim assist/resistance | AIM_RESISTANCE |
| **hit_registration** | 30% | Hitbox manipulation | HITBOX_DRIFT, GHOST_HITS |
| **death_event** | 25% | TTK/TTD anomalies | INSTA_MELT, DAMAGE_SUPPRESSION |
| **edge_entry** | 15% | Spawn bias | SPAWN_BIAS, SPAWN_PRESSURE |

### **Auxiliary Operators (Additional Context)**

| Operator | Detects | Key Flags |
|----------|---------|-----------|
| **crosshair_motion** | Micro-corrections | MICRO_ADJUSTMENT |
| **color_shift** | Visibility manipulation | VISIBILITY_BOOST |
| **flicker_detector** | Frame stutters | FRAME_STUTTER |
| **hud_stability** | UI jitter | HUD_JITTER |
| **peripheral_flash** | Distraction events | DISTRACTION_EVENT |

---

## 🧬 EOMM Scoring

**Composite Score Calculation:**
```
EOMM = (crosshair × 0.30) + (hit_reg × 0.30) + (death × 0.25) + (edge × 0.15)
```

**Thresholds:**
- `EOMM < 0.3`: Normal gameplay
- `0.3 ≤ EOMM < 0.5`: Suspicious patterns
- `0.5 ≤ EOMM < 0.8`: Likely manipulation
- `EOMM ≥ 0.8`: Strong manipulation evidence

**Flags are raised when:**
- Any operator confidence exceeds 0.5
- Multiple operators detect anomalies simultaneously
- Session baseline variance exceeds 2.5 sigma

---

## 📁 Directory Structure

```
CompuCog_Visual_v2/
├── core/                           # Frame capture + grid conversion
│   ├── frame_capture.py
│   └── frame_to_grid.py
├── operators/                      # Detection operators (10 total)
│   ├── crosshair_lock.py
│   ├── hit_registration.py
│   ├── death_event.py
│   ├── edge_entry.py
│   └── ... (6 auxiliary)
├── compositor/                     # EOMM scoring
│   └── eomm_compositor.py
├── baselines/                      # Session tracking
│   └── session_baseline.py
├── memory/                         # [v2] Pattern retention (pending)
├── reasoning/                      # [v2] Causal inference (pending)
├── gaming/                         # Gaming pipelines
│   ├── truevision_live_run.py     # Main capture entry point
│   ├── truevision_smoke_test.py   # Validation harness
│   ├── extract_baselines.py       # Bot fingerprint extraction
│   ├── fuse_telemetry.py          # Visual + logger fusion
│   ├── truevision_config.yaml     # Detection thresholds
│   └── telemetry/                 # Output JSONL files
├── loggers/                        # Telemetry loggers
│   ├── input_logger.py            # Keyboard/mouse
│   ├── activity_logger.py         # Windows/processes
│   ├── process_logger.py          # System resources
│   └── network_logger.ps1         # Network packets
├── logs/                           # Logger output
│   ├── input/
│   ├── activity/
│   ├── process/
│   └── network/
├── launch_truevision.ps1           # One-command launcher
├── compucog_schema.py              # Data structures
└── README.md
```

---

## 🛠️ Configuration

**Edit detection thresholds:**
```powershell
notepad gaming\truevision_config.yaml
```

**Key settings:**
- `capture.target_fps`: Frame capture rate (default: 30)
- `operators.crosshair_lock.confidence_threshold`: Aim assist threshold (default: 0.5)
- `operators.hit_registration.ghost_hit_threshold`: Hitbox drift threshold (default: 0.3)
- `eomm_weights`: Operator weights for composite score

---

## 🎮 Real-World Usage Examples

### **Example 1: Bot Match Calibration**

```powershell
# Capture 15 minutes of bot matches (easy → hard difficulty)
.\launch_truevision.ps1 -Duration 15

# Rename for clarity
cd gaming\telemetry
Move-Item truevision_live_20251202_150022.jsonl match1_casual_bots.jsonl

# Extract baselines
cd ..
python extract_baselines.py telemetry\match1_casual_bots.jsonl

# Result: Thresholds for "fair" vs "elevated difficulty"
```

### **Example 2: Human Match Evidence Collection**

```powershell
# Capture suspicious human match
.\launch_truevision.ps1 -Duration 20

# Fuse with telemetry for complete forensic context
cd gaming
python fuse_telemetry.py telemetry\truevision_live_20251202_183045.jsonl

# Result: Complete evidence package with visual + input + network + activity data
```

### **Example 3: Continuous Monitoring**

```powershell
# Run loggers once at system startup
cd loggers
Start-Process python -ArgumentList "input_logger.py" -WindowStyle Hidden
Start-Process python -ArgumentList "activity_logger.py" -WindowStyle Hidden
Start-Process python -ArgumentList "process_logger.py" -WindowStyle Hidden
Start-Process powershell -ArgumentList "-File network_logger.ps1" -WindowStyle Hidden

# Then run visual detection for each match
cd ..\gaming
python truevision_live_run.py --duration 15  # Match 1
python truevision_live_run.py --duration 12  # Match 2
python truevision_live_run.py --duration 18  # Match 3

# All matches share same logger telemetry stream
```

---

## 🔧 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'mss'"**

**Fix:**
```powershell
pip install mss pywin32 psutil pyyaml
```

### **Issue: Loggers not capturing data**

**Check if running:**
```powershell
Get-Process python | Where-Object {$_.CommandLine -like "*logger*"}
```

**Restart if needed:**
```powershell
.\launch_truevision.ps1 -Duration 5  # (will start loggers automatically)
```

### **Issue: "Permission denied" for network logger**

**Run PowerShell as Administrator** (network packet capture requires elevation)

---

## 📈 Performance Impact

**Resource Usage (All Loggers + Visual Detection):**
- CPU: ~5-8% (background loggers + frame capture)
- RAM: ~200-300 MB
- Disk: ~50 KB/minute (JSONL output)
- Network: No impact (passive monitoring only)

**Frame Rate Impact:**
- TrueVision captures screen at 30 FPS
- Zero impact on game FPS (runs in separate process)
- Screen capture uses `mss` (fast, hardware-accelerated)

---

## 🚦 System Requirements

- **OS**: Windows 10/11 (PowerShell 5.1+)
- **Python**: 3.9+ 
- **Dependencies**: mss, pywin32, psutil, pyyaml
- **Privileges**: Administrator (for network logger only)
- **Disk Space**: ~50 MB per hour of capture

---

## 🔒 Privacy & Security

**What's logged:**
- Screen content (converted to 32×32 grid, no raw pixels stored)
- Keyboard/mouse events (timestamps only, no keystrokes recorded)
- Active window titles
- Network packet counts (no payload inspection)

**What's NOT logged:**
- Raw screenshots
- Keystrokes or passwords
- Packet contents
- Personal identifying information

**All data stays local** - nothing is transmitted off your machine.

---

## 📝 Version History

**v2.0.0** (Current)
- Full telemetry integration (input + activity + process + network)
- One-command launcher (`launch_truevision.ps1`)
- Standalone self-contained system
- Memory and Reasoning layers scaffolded (pending implementation)

**v1.0.0** (2025-12-02)
- Initial TrueVision release
- 4 primary + 6 auxiliary operators
- EOMM compositor + session baselines
- Live capture + smoke test validated

---

## 🆘 Support

**Quick reference:**
- Smoke test: `cd gaming; python truevision_smoke_test.py`
- Live capture: `.\launch_truevision.ps1 -Duration 15`
- Baseline extraction: `python extract_baselines.py <file>.jsonl`
- Telemetry fusion: `python fuse_telemetry.py <file>.jsonl`

**All output is JSONL** - one detection window per line, easily parseable.
