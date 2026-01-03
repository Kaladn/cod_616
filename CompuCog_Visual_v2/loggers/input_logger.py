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

# Common utilities for all loggers
try:
    from logger_common import write_safe, startup_cleanup, get_timestamp
except ImportError:
    # Fallback if running standalone
    pass

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
    
    com_initialized = False
    try:
        CoInitialize()
        com_initialized = True
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
        
        return is_active, active_device
    except Exception as e:
        logging.debug(f"Audio output check failed: {e}")
        return False, None
    finally:
        if com_initialized:
            try:
                CoUninitialize()
            except:
                pass

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


def collect_and_log(output_path: str = None):
    """
    IMMORTAL collection loop - runs FOREVER.
    
    Philosophy: NEVER exit, NEVER crash. Log or continue trying.
    Auto-deletes data older than 1 day on startup.
    """
    log_file = Path(output_path) if output_path else get_log_file_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-cleanup old data (older than 1 day)
    try:
        logs_base = log_file.parent.parent
        startup_cleanup(logs_base)
    except:
        pass
    
    logging.info(f"CompuCogLogger - Input Metrics [IMMORTAL]")
    logging.info(f"Log File: {log_file}")
    logging.info(f"Sample Interval: 3 seconds")
    logging.info(f"Privacy Mode: Polling (no key content, no mouse coords)")
    logging.info("")
    
    sample_count = 0
    
    # IMMORTAL LOOP - while True FOREVER
    while True:
        try:
            time.sleep(3.0)
            sample_count += 1
            
            # Check if date rolled over (safe)
            if not output_path:
                try:
                    current_log = get_log_file_path()
                    if current_log != log_file:
                        log_file = current_log
                        logging.info(f"Date rollover → {log_file}")
                except Exception:
                    pass
            
            # Check input activity (safe)
            try:
                activity_data = activity_tracker.check_activity()
                input_events = activity_tracker.reset_counters()
            except Exception:
                activity_data = {'idle_seconds': 0, 'is_active': False}
                input_events = 0
            
            # Get device status (safe - COM wrapped)
            try:
                audio_active, audio_device = get_audio_output_status()
            except Exception:
                audio_active, audio_device = False, None
            
            try:
                camera_active, camera_device = get_camera_status()
            except Exception:
                camera_active, camera_device = False, None
            
            # Build record
            record = {
                "timestamp": datetime.now().isoformat(),
                "keystroke_count": 0,
                "mouse_click_count": 0,
                "mouse_movement_distance": 0.0,
                "idle_seconds": activity_data.get('idle_seconds', 0),
                "audio_active": audio_active,
                "camera_active": camera_active,
                "audio_device_name": audio_device,
                "camera_device_name": camera_device
            }
            
            # Write (immortal)
            write_safe(log_file, record)
            
            # Status update every 20 samples
            if sample_count % 20 == 0:
                status = f"[{datetime.now().strftime('%H:%M:%S')}] {sample_count} events"
                if activity_data.get('is_active'):
                    status += f" | ACTIVE"
                logging.info(status)
        
        except KeyboardInterrupt:
            # IMMORTAL - ignore Ctrl+C
            logging.warning("Ctrl+C ignored - IMMORTAL mode")
            time.sleep(0.1)
            continue
        
        except Exception as e:
            # ANY error - log and continue
            logging.error(f"Loop error (continuing): {e}")
            time.sleep(3.0)
            continue

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
    import argparse
    parser = argparse.ArgumentParser(description="Input Logger")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        collect_and_log(output_path=args.output)
    except KeyboardInterrupt:
        logging.info("Input logger stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
