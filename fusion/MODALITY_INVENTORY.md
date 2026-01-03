# Fusion Modality Inventory

**Status**: Complete for v1.0 (January 3, 2026)

## Current Modalities (6)

| # | Modality | Logger | Output Pattern | Key Fields |
|---|----------|--------|----------------|------------|
| 1 | **Vision** | TrueVision | `truevision_*.jsonl` | window_start_epoch, operator_results, eomm_score, flags |
| 2 | **Activity** | activity_logger.py | `user_activity_*.jsonl` | timestamp, windowTitle, processName, idleSeconds |
| 3 | **Gamepad** | gamepad_logger.py | `gamepad_stream_*.jsonl` | timestamp, event, button/axis, value |
| 4 | **Input** | input_logger.py | `input_activity_*.jsonl` | timestamp, keystroke_count, mouse_movement, audio_active |
| 5 | **Network** | network_logger.ps1 | `telemetry_*.jsonl` | Timestamp, RemoteAddress, State, Protocol, ProcessName |
| 6 | **Process** | process_logger.py | `process_activity_*.jsonl` | timestamp, pid, process_name, origin, flagged |

## Proposed Future Modalities

### High Priority

| Modality | Purpose | Schema (proposed) |
|----------|---------|-------------------|
| **Audio Game** | Gunfire, explosions, footsteps | timestamp, audio_event, frequency_band, amplitude, duration_ms |
| **Audio Comms** | Voice chat activity (not content) | timestamp, voice_active, speaker_id, volume_level |
| **Video Index** | Frame hashes for replay sync | timestamp, frame_hash, resolution, brightness_mean |
| **GPU Metrics** | Frame time, VRAM, utilization | timestamp, frame_time_ms, gpu_util_pct, vram_used_mb |
| **Memory Pressure** | RAM usage, page faults | timestamp, ram_used_mb, page_faults, working_set_mb |

### Medium Priority

| Modality | Purpose | Schema (proposed) |
|----------|---------|-------------------|
| **Mouse Raw** | Raw mouse deltas (not coords) | timestamp, dx, dy, buttons, dpi_setting |
| **Keyboard Timing** | Key timing (not content) | timestamp, key_down_duration_ms, inter_key_delay_ms |
| **Display** | Resolution, HDR, refresh rate | timestamp, resolution, refresh_hz, hdr_enabled |
| **Overlay** | OBS, Discord, GeForce overlay | timestamp, overlay_app, visible, position |
| **Anti-Cheat** | Kernel AC events | timestamp, ac_event, module_name, status |

### Low Priority (Forensic Only)

| Modality | Purpose | Schema (proposed) |
|----------|---------|-------------------|
| **Clipboard** | Copy/paste events (not content) | timestamp, clipboard_event, content_type, size_bytes |
| **USB Events** | Device connect/disconnect | timestamp, usb_event, device_class, vid_pid |
| **Power** | Battery, power state | timestamp, power_state, battery_pct, charging |
| **System Time** | NTP sync, clock drift | timestamp, ntp_offset_ms, drift_rate |

## Fusion Pipeline

```
Loggers (6+)          timeline_merger.py         fusion.py
──────────────        ──────────────────        ────────────────
truevision_*.jsonl  ─┐
user_activity_*.jsonl ─┤
gamepad_stream_*.jsonl ─┼──> fused_events.jsonl ──> fusion_blocks.jsonl
input_activity_*.jsonl ─┤        (flat)               (6-1-6 windows)
telemetry_*.jsonl ─┤
process_activity_*.jsonl ─┘
```

## Directory Structures Supported

### Flat Structure
```
session_dir/
├── truevision_events.jsonl
├── gamepad_events.jsonl
├── network_events.jsonl
├── input_events.jsonl
├── process_events.jsonl
└── activity_events.jsonl
```

### CompuCog_Visual_v2 Structure
```
gaming/telemetry/           # TrueVision
├── truevision_live_*.jsonl
└── truevision_smoke_*.jsonl

logs/                       # Other modalities
├── activity/
│   └── user_activity_*.jsonl
├── gamepad/
│   └── gamepad_stream_*.jsonl
├── input/
│   └── input_activity_*.jsonl
├── network/
│   └── telemetry_*.jsonl
└── process/
    └── process_activity_*.jsonl
```

## Usage

```powershell
# Basic (flat structure)
python fusion.py --session D:\sessions\game1

# CompuCog_Visual_v2 structure
python fusion.py --session D:\CompuCog_Visual_v2\gaming\telemetry --logs-dir D:\CompuCog_Visual_v2\logs

# Custom FPS
python fusion.py --session D:\sessions\game1 --fps 120
```

## Alignment Metrics

- **Alignment Score**: % of modalities present across 13-frame window
- **Resonance Coherence**: Cross-modal temporal correlation (6-1-6 pattern)
- **Modality Coverage**: Per-modality fill rate
