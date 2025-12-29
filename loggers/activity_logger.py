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
CompuCogLogger - User Activity Module
======================================

Tracks active window focus and process context for behavioral fingerprinting.
Captures human-readable process names and window titles (no obscure IDs).

Schema: compucog_schema.py::ActiveWindow

Fields:
- timestamp: ISO 8601 timestamp
- windowTitle: Active window title
- processName: Process name (e.g., "chrome.exe")
- executablePath: Full path to executable
- idleSeconds: Seconds since last input
"""

import os
import json
import psutil
import time
import logging
from datetime import datetime
from pathlib import Path
import ctypes
from ctypes import wintypes

# Windows API structures for idle detection
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.UINT),
        ('dwTime', wintypes.DWORD),
    ]

def get_idle_duration():
    """Get seconds since last user input (keyboard/mouse)"""
    lastInputInfo = LASTINPUTINFO()
    lastInputInfo.cbSize = ctypes.sizeof(lastInputInfo)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lastInputInfo))
    millis = ctypes.windll.kernel32.GetTickCount() - lastInputInfo.dwTime
    return millis / 1000.0

def get_active_window():
    """Get currently active window title and process PID"""
    try:
        import win32gui
        import win32process
        
        hwnd = win32gui.GetForegroundWindow()
        if hwnd == 0:
            return None, None, None
            
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        window_title = win32gui.GetWindowText(hwnd)
        
        return hwnd, pid, window_title
    except Exception as e:
        logging.debug(f"Error getting active window: {e}")
        return None, None, None

def get_process_info(pid):
    """Get process name and path from PID"""
    try:
        p = psutil.Process(pid)
        name = p.name()
        path = p.exe() if p.exe() else ""
        return name, path
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None, None

def get_log_file_path():
    """Get today's log file path"""
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "activity"
    logs_dir = logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"user_activity_{today}.jsonl"

def collect_and_log():
    """Main collection loop - runs every 3 seconds"""
    log_file = get_log_file_path()
    
    logging.info(f"CompuCogLogger - User Activity Started")
    logging.info(f"Log File: {log_file}")
    logging.info(f"Sample Interval: 3 seconds")
    logging.info("")
    
    sample_count = 0
    
    try:
        while True:
            time.sleep(3.0)
            sample_count += 1
            
            # Check if date rolled over
            current_log = get_log_file_path()
            if current_log != log_file:
                log_file = current_log
                logging.info(f"Date rollover → {log_file}")
            
            # Get idle duration
            idle_seconds = get_idle_duration()
            
            # Get active window info
            hwnd, pid, window_title = get_active_window()
            
            if pid:
                process_name, process_path = get_process_info(pid)
            else:
                process_name, process_path = (None, None)
            
            # Build log entry (MUST match compucog_schema.py::ActiveWindow)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "windowTitle": window_title or "",
                "processName": process_name or "Unknown",
                "executablePath": process_path or "",
                "idleSeconds": round(idle_seconds, 1)
            }
            
            # Write to JSONL
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f.flush()
            
            # Status update every 20 samples (~1 minute)
            if sample_count % 20 == 0:
                status = "IDLE" if idle_seconds > 5.0 else "ACTIVE"
                logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] {sample_count} events | {status} | {process_name or 'Unknown':<30} | {window_title[:50] if window_title else ''}")
            
    except KeyboardInterrupt:
        logging.info(f"Activity logger stopped ({sample_count} events)")

def setup_logging():
    """Configure logging output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Start activity monitoring"""
    setup_logging()
    
    try:
        collect_and_log()
    except KeyboardInterrupt:
        logging.info("Activity logger terminated")

if __name__ == "__main__":
    main()
