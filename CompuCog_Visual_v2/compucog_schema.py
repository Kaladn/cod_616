"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog — Sovereign Cognitive Defense System                           ║
║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
║                                                                              ║
║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
║                                                                              ║
║     "We use unconventional digital wisdom —                                  ║
║        because conventional digital wisdom doesn't protect anyone."         ║
║                                                                              ║
║     This software is proprietary and confidential.                           ║
║     Unauthorized access, copying, modification, or distribution             ║
║     is strictly prohibited and may violate applicable laws.                  ║
║                                                                              ║
║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

"""
CompuCog Unified Telemetry Schema - I/O Contract
Version: 1.0.0
Date: November 11, 2025

THIS IS THE SINGLE SOURCE OF TRUTH FOR ALL TELEMETRY FORMATS.
All loggers MUST produce data matching these schemas.
All consumers MUST parse data according to these schemas.

NO FILE-SPECIFIC FORMATS. NO LOCAL CONVENTIONS. GLOBAL CONTRACT ONLY.
"""

from typing import TypedDict, Optional, List
from datetime import datetime

# ============================================================================
# RAW TELEMETRY SCHEMAS (Logger Output)
# ============================================================================

class NetworkEvent(TypedDict):
    """Single network connection event - emitted by NetworkTelemetry logger"""
    Timestamp: str          # ISO 8601: "2025-11-10T04:19:00.123456"
    LocalAddress: str       # "192.168.1.100" or "::"
    LocalPort: int          # 56263
    RemoteAddress: str      # "142.250.80.46" or "::"
    RemotePort: int         # 443
    State: str              # "Established", "Bound", "Listen", etc.
    Protocol: str           # "TCP" or "UDP"
    PID: int                # 8184
    ProcessName: str        # "chrome.exe"


class InputMetrics(TypedDict):
    """Input activity metrics - emitted by InputMetrics logger"""
    timestamp: str                  # ISO 8601
    keystroke_count: int            # Keypresses in 3-sec window
    mouse_click_count: int          # Mouse clicks
    mouse_movement_distance: float  # Pixels traveled (privacy: no coords)
    idle_seconds: float             # Seconds since last input
    audio_active: bool              # Speaker/headphone output active
    camera_active: bool             # Webcam active
    audio_device_name: Optional[str]    # Device name or None
    camera_device_name: Optional[str]   # Device name or None


class ActiveWindow(TypedDict):
    """Active window context - emitted by UserActivity logger"""
    timestamp: str          # ISO 8601
    windowTitle: str        # "Visual Studio Code"
    processName: str        # "Code.exe"
    executablePath: str     # "C:\\Program Files\\..."
    idleSeconds: float      # Idle time


class ProcessEvent(TypedDict):
    """Process spawn event - emitted by ProcessMonitor logger"""
    timestamp: str          # ISO 8601
    pid: int                # Process ID
    process_name: str       # "notepad.exe"
    command_line: str       # Full command line
    parent_pid: int         # Parent process ID
    origin: str             # "user_initiated" | "system" | "service"
    flagged: bool           # True if suspicious


class GamingFingerprint(TypedDict):
    """Gaming visual fingerprint - emitted by TrueVision gaming_sensor"""
    timestamp: str                      # ISO 8601 (window start time)
    t_start: float                      # Unix timestamp start
    t_end: float                        # Unix timestamp end
    frame_count: int                    # Frames analyzed in window
    features: dict                      # Feature scores (flicker, HUD, crosshair, etc)
    detections: List[dict]              # Operator detections with confidence/features


# ============================================================================
# UNIFIED TELEMETRY BLOCK (Merger Output)
# ============================================================================

class UnifiedTelemetryBlock(TypedDict):
    """
    3-second time-aligned block combining all telemetry streams.
    
    This is the OUTPUT of merge_telemetry.py and INPUT to feature_extraction.py.
    This is the ONLY format the ML pipeline accepts.
    """
    timeblock_id: str                       # "2025-11-10T17:56:57"
    timestamp: str                          # ISO 8601 block start time
    
    # Input stream
    input_metrics: InputMetrics
    
    # Activity stream
    active_window: Optional[ActiveWindow]   # None if no window change in block
    
    # Network stream
    network_events: List[NetworkEvent]      # All connections in 3-sec window
    
    # Process stream
    process_events: List[ProcessEvent]      # All spawns in 3-sec window
    
    # Gaming stream (TrueVision visual reasoning)
    gaming_fingerprints: List[GamingFingerprint]  # Visual fingerprints in 3-sec window
    
    # Correlation metadata (optional)
    session_id: Optional[str]               # User session identifier
    user_identity: Optional[str]            # Username (if available)


# ============================================================================
# LOGGER OUTPUT RULES
# ============================================================================

"""
NETWORK TELEMETRY (NetworkTelemetry/telemetry_logger.ps1)
---------------------------------------------------------
Output: NetworkTelemetry/logs/telemetry_YYYYMMDD.jsonl
Format: ONE NetworkEvent PER LINE (JSONL)

CORRECT:
{"Timestamp":"2025-11-10T04:19:00.123456","LocalAddress":"::","LocalPort":56263,...}
{"Timestamp":"2025-11-10T04:19:00.223456","LocalAddress":"192.168.1.100",...}

INCORRECT (current bug):
{"value":[{...},{...}],"Count":86}  # ARRAY WRAPPER - WRONG!

Fix: PowerShell logger must emit individual events, not array snapshots.


INPUT METRICS (InputMetrics/input_logger.py)
----------------------------------------------
Output: InputMetrics/logs/input_activity_YYYYMMDD.jsonl
Format: ONE InputMetrics PER LINE (JSONL)

CORRECT:
{"timestamp":"2025-11-10T17:56:57","keystroke_count":5,"mouse_click_count":2,...}
{"timestamp":"2025-11-10T17:57:00","keystroke_count":0,"mouse_click_count":0,...}


USER ACTIVITY (UserActivity/activity_logger.py)
------------------------------------------------
Output: UserActivity/user_activity_YYYYMMDD.jsonl
Format: ONE ActiveWindow PER LINE (JSONL)

CORRECT:
{"timestamp":"2025-11-10T17:56:57","windowTitle":"Chrome","processName":"chrome.exe",...}
{"timestamp":"2025-11-10T17:57:05","windowTitle":"Code","processName":"Code.exe",...}


PROCESS MONITOR (ProcessMonitor/process_logger.py)
---------------------------------------------------
Output: ProcessMonitor/logs/process_activity_YYYYMMDD.jsonl
Format: ONE ProcessEvent PER LINE (JSONL)

CORRECT:
{"timestamp":"2025-11-10T18:23:15","pid":12345,"process_name":"notepad.exe",...}
{"timestamp":"2025-11-10T18:23:42","pid":12346,"process_name":"calc.exe",...}
"""


# ============================================================================
# MERGER CONTRACT (ML/merge_telemetry.py)
# ============================================================================

"""
INPUT: 4 JSONL files (one per logger)
OUTPUT: 1 JSONL file with UnifiedTelemetryBlock per line

Process:
1. Read all 4 streams
2. Align events into 3-second time blocks
3. Combine events that fall within same block
4. Emit ONE UnifiedTelemetryBlock per line

Output: ML/datasets/unified_raw/telemetry_unified_YYYYMMDD.jsonl

CORRECT OUTPUT FORMAT:
{"timeblock_id":"2025-11-10T17:56:57","timestamp":"2025-11-10T17:56:57","input_metrics":{...},"active_window":{...},"network_events":[...],"process_events":[...]}
{"timeblock_id":"2025-11-10T17:57:00","timestamp":"2025-11-10T17:57:00","input_metrics":{...},"active_window":null,"network_events":[...],"process_events":[...]}
"""


# ============================================================================
# FEATURE EXTRACTION CONTRACT (ML/feature_extraction.py)
# ============================================================================

"""
INPUT: UnifiedTelemetryBlock (from unified JSONL)
OUTPUT: numpy array of 48 features (schema v1.0)

Features are extracted according to FROZEN schema v1.0:
- Input features: 10 dimensions
- Network features: 15 dimensions
- Activity features: 10 dimensions
- Process features: 8 dimensions
- Cross-correlation features: 5 dimensions

TOTAL: 48 features per 3-second block

Feature extraction MUST handle:
- Missing data (null active_window, empty network_events list)
- Partial data (only 2/4 streams available)
- Out-of-range values (normalize/clip appropriately)
"""


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_network_event(event: dict) -> bool:
    """Validate NetworkEvent schema compliance"""
    required = ['Timestamp', 'LocalAddress', 'LocalPort', 'RemoteAddress', 
                'RemotePort', 'State', 'Protocol', 'PID', 'ProcessName']
    return all(k in event for k in required)


def validate_input_metrics(metrics: dict) -> bool:
    """Validate InputMetrics schema compliance"""
    required = ['timestamp', 'keystroke_count', 'mouse_click_count', 
                'mouse_movement_distance', 'idle_seconds', 'audio_active', 
                'camera_active']
    return all(k in metrics for k in required)


def validate_active_window(window: dict) -> bool:
    """Validate ActiveWindow schema compliance"""
    required = ['timestamp', 'windowTitle', 'processName']
    return all(k in window for k in required)


def validate_process_event(event: dict) -> bool:
    """Validate ProcessEvent schema compliance"""
    required = ['timestamp', 'pid', 'process_name']
    return all(k in event for k in required)


def validate_gaming_fingerprint(fingerprint: dict) -> bool:
    """Validate GamingFingerprint schema compliance"""
    required = ['timestamp', 't_start', 't_end', 'frame_count', 'features', 'detections']
    return all(k in fingerprint for k in required)


def validate_unified_block(block: dict) -> bool:
    """Validate UnifiedTelemetryBlock schema compliance"""
    required = ['timeblock_id', 'timestamp', 'input_metrics', 
                'active_window', 'network_events', 'process_events', 'gaming_fingerprints']
    if not all(k in block for k in required):
        return False
    
    # Validate nested schemas
    if not validate_input_metrics(block['input_metrics']):
        return False
    
    if block['active_window'] is not None:
        if not validate_active_window(block['active_window']):
            return False
    
    for net_event in block['network_events']:
        if not validate_network_event(net_event):
            return False
    
    for proc_event in block['process_events']:
        if not validate_process_event(proc_event):
            return False
    
    return True


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
LOGGER IMPLEMENTATION:
----------------------
from compucog_schema import NetworkEvent
import json

# Create event
event: NetworkEvent = {
    'Timestamp': datetime.now().isoformat(),
    'LocalAddress': '192.168.1.100',
    'LocalPort': 52301,
    # ... other fields
}

# Write to JSONL
with open(f'telemetry_{date}.jsonl', 'a') as f:
    f.write(json.dumps(event) + '\n')


MERGER IMPLEMENTATION:
----------------------
from compucog_schema import UnifiedTelemetryBlock

# Create unified block
block: UnifiedTelemetryBlock = {
    'timeblock_id': '2025-11-10T17:56:57',
    'timestamp': '2025-11-10T17:56:57',
    'input_metrics': {...},
    'active_window': {...},
    'network_events': [...],
    'process_events': [...]
}

# Validate before writing
if validate_unified_block(block):
    with open(f'telemetry_unified_{date}.jsonl', 'a') as f:
        f.write(json.dumps(block) + '\n')


FEATURE EXTRACTION IMPLEMENTATION:
-----------------------------------
from compucog_schema import UnifiedTelemetryBlock
import json

# Read unified blocks
with open('telemetry_unified_20251110.jsonl') as f:
    for line in f:
        block: UnifiedTelemetryBlock = json.loads(line)
        features = extract_features_v1_0(block)
        # features is numpy array[48]
"""


# ============================================================================
# SCHEMA VERSION HISTORY
# ============================================================================

SCHEMA_VERSION = "1.0.0"
SCHEMA_DATE = "2025-11-11"

"""
Version 1.0.0 (2025-11-11):
- Initial unified schema definition
- 48-feature extraction (v1.0)
- 3-second time block alignment
- JSONL format standardization

Future versions will maintain backward compatibility or create new feature
extraction functions (extract_features_v2_0) with parallel model trees.
"""
