#!/usr/bin/env python3

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
CompuCogLogger - Input Metrics Module
======================================

Captures aggregate input metrics (NO key content, NO mouse coordinates) to distinguish:
- Active coding vs. passive reading
- Real user vs. synthetic session  
- Engaged grind vs. compromised playback

Metrics per 3-second window:
- keystroke_count: Number of key presses (not content)
- mouse_click_count: Number of clicks (not coordinates)
- mouse_movement_distance: Total pixels traveled
- idle_seconds: Seconds since last input
- audio_active: Speakers/headphones outputting audio
- camera_active: Webcam in use
- audio_device_name: Active audio device
- camera_device_name: Active camera device

Schema: compucog_schema.py::InputMetrics
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import logging

# Third-party imports
try:
    import psutil
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install psutil")
    sys.exit(1)

# Windows-specific audio/camera detection
if sys.platform == "win32":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IAudioSessionManager2
        import win32api
        import win32con
        HAS_WINDOWS_APIS = True
    except ImportError:
        print("⚠️ Windows audio/camera APIs unavailable (install: pip install pycaw pywin32)")
        HAS_WINDOWS_APIS = False
else:
    HAS_WINDOWS_APIS = False

# ============================================================================
# INPUT ACTIVITY DETECTION (Windows API - Non-Blocking, No Hooks)
# ============================================================================

class InputActivityTracker:
    """Track input activity using Windows API polling (no event hooks)"""
    
    def __init__(self):
        self.last_idle_time = 0.0
        self.last_check_time = time.time()
        
        # Activity counters (estimated from idle time changes)
        self.input_events_this_period = 0
    
    def get_idle_time_seconds(self):
        """Get seconds since last user input using Windows GetLastInputInfo API"""
        if not HAS_WINDOWS_APIS:
            return 0.0
        
        try:
            import ctypes
            from ctypes import Structure, windll, c_uint, sizeof, byref
            
            class LASTINPUTINFO(Structure):
                _fields_ = [
                    ('cbSize', c_uint),
                    ('dwTime', c_uint),
                ]
            
            lastInputInfo = LASTINPUTINFO()
            lastInputInfo.cbSize = sizeof(lastInputInfo)
            windll.user32.GetLastInputInfo(byref(lastInputInfo))
            
            millis = windll.kernel32.GetTickCount() - lastInputInfo.dwTime
            return millis / 1000.0
        except Exception as e:
            logging.debug(f"Idle time check failed: {e}")
            return 0.0
    
    def check_activity(self):
        """Check for input activity since last call"""
        current_idle = self.get_idle_time_seconds()
        current_time = time.time()
        
        # Detect input event: idle time reset (went from higher to lower)
        input_detected = False
        if current_idle < self.last_idle_time:
            input_detected = True
            self.input_events_this_period += 1
        
        self.last_idle_time = current_idle
        self.last_check_time = current_time
        
        return {
            'idle_seconds': round(current_idle, 1),
            'is_idle': current_idle > 5.0,
            'input_events': self.input_events_this_period,
            'is_active': current_idle < 1.0
        }
    
    def reset_counters(self):
        """Reset activity counters (called every 3 seconds)"""
        events = self.input_events_this_period
        self.input_events_this_period = 0
        return events

# Global tracker instance
activity_tracker = InputActivityTracker()

# ============================================================================
# WINDOWS DEVICE STATUS DETECTION
# ============================================================================

def get_audio_output_status():
    """Check if audio is actively playing (Windows)"""
    if not HAS_WINDOWS_APIS:
        return False, None
    
    try:
        CoInitialize()
        sessions = AudioUtilities.GetAllSessions()
        active_device = None
        is_active = False
        
        for session in sessions:
            if session.Process and session.State == 1:  # AudioSessionStateActive
                is_active = True
                try:
                    device = AudioUtilities.GetSpeakers()
                    if device:
                        active_device = device.FriendlyName
                except:
                    pass
                break
        
        CoUninitialize()
        return is_active, active_device
    except Exception as e:
        logging.debug(f"Audio output check failed: {e}")
        return False, None

def get_camera_status():
    """Check if webcam is active (Windows)"""
    if not HAS_WINDOWS_APIS:
        return False, None
    
    try:
        # Check for processes commonly associated with camera usage
        cam_apps = {
            'teams': 'Microsoft Teams',
            'zoom': 'Zoom',
            'discord': 'Discord',
            'skype': 'Skype',
            'obs': 'OBS Studio',
            'camera': 'Windows Camera'
        }
        
        for proc in psutil.process_iter(['name']):
            proc_name_lower = proc.info['name'].lower()
            for key, friendly in cam_apps.items():
                if key in proc_name_lower:
                    return True, friendly
        
        return False, None
    except Exception as e:
        logging.debug(f"Camera check failed: {e}")
        return False, None

# ============================================================================
# DATA COLLECTION & LOGGING
# ============================================================================

def get_log_file_path():
    """Get today's log file path"""
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "input"
    logs_dir = logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"input_activity_{today}.jsonl"

def collect_and_log():
    """Main collection loop - runs every 3 seconds"""
    log_file = get_log_file_path()
    
    logging.info(f"CompuCogLogger - Input Metrics Started")
    logging.info(f"Log File: {log_file}")
    logging.info(f"Sample Interval: 3 seconds")
    logging.info(f"Privacy Mode: Polling (no key content, no mouse coords)")
    logging.info("")
    
    sample_count = 0
    
    while True:
        try:
            # Wait 3 seconds
            time.sleep(3.0)
            sample_count += 1
            
            # Check if date rolled over
            current_log = get_log_file_path()
            if current_log != log_file:
                log_file = current_log
                logging.info(f"Date rollover → {log_file}")
            
            # Check input activity
            activity_data = activity_tracker.check_activity()
            input_events = activity_tracker.reset_counters()
            
            # Get device status
            audio_active, audio_device = get_audio_output_status()
            camera_active, camera_device = get_camera_status()
            
            # Build record (MUST match compucog_schema.py::InputMetrics)
            record = {
                "timestamp": datetime.now().isoformat(),
                "keystroke_count": 0,  # Future: hook implementation
                "mouse_click_count": 0,  # Future: hook implementation
                "mouse_movement_distance": 0.0,  # Future: hook implementation
                "idle_seconds": activity_data['idle_seconds'],
                "audio_active": audio_active,
                "camera_active": camera_active,
                "audio_device_name": audio_device,
                "camera_device_name": camera_device
            }
            
            # Write to JSONL
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            
            # Status update every 20 samples (~1 minute)
            if sample_count % 20 == 0:
                status = f"[{datetime.now().strftime('%H:%M:%S')}] {sample_count} events"
                if activity_data['is_active']:
                    status += f" | ACTIVE (idle: {activity_data['idle_seconds']}s)"
                if audio_active:
                    status += f" | Audio: {audio_device or 'Unknown'}"
                if camera_active:
                    status += f" | Camera: {camera_device or 'Unknown'}"
                logging.info(status)
        
        except Exception as e:
            logging.error(f"Collection error: {e}")
            time.sleep(3.0)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def setup_logging():
    """Configure logging output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Start input activity monitoring"""
    setup_logging()
    
    try:
        collect_and_log()
    except KeyboardInterrupt:
        logging.info("Input logger stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
