#!/usr/bin/env python3
"""
CompuCogLogger - Process Monitor Module
========================================

Monitors process spawns for suspicious activity.
Captures process metadata with intelligent filtering to reduce noise.

Schema: compucog_schema.py::ProcessEvent

Fields:
- timestamp: ISO 8601 timestamp
- pid: Process ID
- process_name: Process name (e.g., "chrome.exe")
- command_line: Command line arguments (truncated to 500 chars)
- parent_pid: Parent process ID
- origin: Classification (user_initiated, system_service, external_triggered, etc.)
- flagged: Boolean indicating suspicious activity
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import deque

# Add parent directory to path for logger_resilience module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
try:
    import psutil
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install psutil")
    sys.exit(1)

# Import resilience utilities
try:
    from logger_resilience import (
        enable_debug_privilege,
        is_anti_cheat_running,
        retry_on_failure,
        safe_process_access,
        create_protected_file,
        daemon_loop_with_recovery,
        check_cod_process_access
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    print("⚠️ logger_resilience not available, using basic error handling")
    RESILIENCE_AVAILABLE = False

# System processes to ignore (Windows noise)
SYSTEM_NOISE = {
    'System', 'Idle', 'Registry', 'smss.exe', 'csrss.exe', 'wininit.exe',
    'services.exe', 'lsass.exe', 'svchost.exe', 'dwm.exe', 'winlogon.exe',
    'fontdrvhost.exe', 'WUDFHost.exe', 'conhost.exe', 'sihost.exe',
    'taskhostw.exe', 'RuntimeBroker.exe', 'spoolsv.exe', 'SearchIndexer.exe',
    'MsMpEng.exe', 'SgrmBroker.exe', 'SecurityHealthService.exe',
    'audiodg.exe', 'dllhost.exe', 'WmiPrvSE.exe', 'CompPkgSrv.exe'
}

def get_log_file_path():
    """Get today's log file path"""
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "process"
    logs_dir = logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y%m%d')
    return logs_dir / f"process_activity_{today}.jsonl"

def classify_origin(process_name, parent_name, username, exe_path):
    """Determine if process is user-initiated, system, or external"""
    
    # System account processes
    if username and username.upper() in ('SYSTEM', 'LOCAL SERVICE', 'NETWORK SERVICE', 'NT AUTHORITY\\SYSTEM'):
        return 'system_service'
    
    # Explorer.exe parent = user clicked something
    if parent_name and 'explorer.exe' in parent_name.lower():
        return 'user_initiated'
    
    # Processes from user profile = likely user action
    if exe_path and 'Users\\' in exe_path and '\\AppData\\' in exe_path:
        return 'user_initiated'
    
    # Services.exe parent = Windows service
    if parent_name and 'services.exe' in parent_name.lower():
        return 'system_service'
    
    # Unknown parent = external
    if not parent_name:
        return 'external_triggered'
    
    return 'unknown'

def check_suspicious(process_name, parent_name, exe_path, command_line):
    """Check if process spawn pattern is suspicious"""
    
    # Office app spawning executable
    if parent_name:
        parent_upper = parent_name.upper()
        if any(office in parent_upper for office in ['WINWORD', 'EXCEL', 'POWERPNT', 'OUTLOOK']):
            if process_name.lower().endswith('.exe'):
                return True
    
    # Browser spawning shell
    if parent_name and any(browser in parent_name.lower() for browser in ['chrome', 'firefox', 'msedge', 'iexplore']):
        if process_name.lower() in {'powershell.exe', 'cmd.exe', 'wscript.exe', 'cscript.exe'}:
            return True
    
    # Executable from Temp or Downloads
    if exe_path:
        suspicious_paths = ['\\Temp\\', '\\Downloads\\', '\\AppData\\Local\\Temp\\']
        if any(path in exe_path for path in suspicious_paths):
            return True
    
    # PowerShell with encoded command
    if command_line and 'powershell' in process_name.lower():
        if '-enc' in command_line.lower() or '-encodedcommand' in command_line.lower():
            return True
    
    return False

class ProcessMonitor:
    """Monitor process creation using periodic snapshots"""
    
    def __init__(self):
        self.seen_pids = deque(maxlen=2000)  # Track recent PIDs
        self.parent_cache = {}
    
    def get_parent_info(self, ppid):
        """Get parent process name with anti-cheat protection"""
        if ppid == 0:
            return None
        
        if ppid in self.parent_cache:
            return self.parent_cache[ppid]
        
        # Use safe access if resilience available
        if RESILIENCE_AVAILABLE:
            parent = safe_process_access(ppid)
            if parent:
                try:
                    parent_name = parent.name()
                    self.parent_cache[ppid] = parent_name
                    return parent_name
                except:
                    return None
        else:
            try:
                parent = psutil.Process(ppid)
                parent_name = parent.name()
                self.parent_cache[ppid] = parent_name
                return parent_name
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None
        
        return None
    
    def capture_process_snapshot(self):
        """Capture snapshot of running processes"""
        events = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'ppid', 'exe']):
            try:
                pid = proc.info['pid']
                
                # Skip if already seen
                if pid in self.seen_pids:
                    continue
                
                self.seen_pids.append(pid)
                
                process_name = proc.info['name']
                
                # Filter system noise
                if process_name in SYSTEM_NOISE:
                    continue
                
                # Get details
                exe_path = proc.info.get('exe', '')
                cmdline = proc.info.get('cmdline')
                command_line = ' '.join(cmdline) if cmdline else ''
                command_line = command_line[:500]
                username = proc.info.get('username', 'Unknown')
                ppid = proc.info.get('ppid', 0)
                
                parent_name = self.get_parent_info(ppid)
                
                # Classify origin
                origin = classify_origin(process_name, parent_name, username, exe_path)
                
                # Check suspicious
                flagged = check_suspicious(process_name, parent_name, exe_path, command_line)
                
                # Build event (MUST match compucog_schema.py::ProcessEvent)
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'pid': pid,
                    'process_name': process_name,
                    'command_line': command_line,
                    'parent_pid': ppid,
                    'origin': origin,
                    'flagged': flagged
                }
                
                events.append(event)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return events
    
    def monitor(self):
        """Main monitoring loop"""
        log_file = get_log_file_path()
        
        logging.info(f"CompuCogLogger - Process Monitor Started")
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
                
                # Capture process snapshot
                events = self.capture_process_snapshot()
                
                # Write events to JSONL (with retry on file lock)
                if events:
                    if RESILIENCE_AVAILABLE:
                        for event in events:
                            event_line = json.dumps(event) + "\n"
                            create_protected_file(str(log_file), event_line)
                    else:
                        with open(log_file, "a", encoding="utf-8") as f:
                            for event in events:
                                f.write(json.dumps(event) + "\n")
                            f.flush()
                
                # Status update every 20 samples (~1 minute)
                if sample_count % 20 == 0:
                    flagged_count = sum(1 for e in events if e['flagged'])
                    status = f"[{datetime.now().strftime('%H:%M:%S')}] {sample_count} snapshots"
                    if events:
                        status += f" | {len(events)} new processes"
                    if flagged_count > 0:
                        status += f" | ⚠️ {flagged_count} flagged"
                    logging.info(status)
        
        except KeyboardInterrupt:
            logging.info(f"Process monitor stopped ({sample_count} snapshots)")

def setup_logging():
    """Configure logging output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Start process monitoring"""
    setup_logging()
    
    monitor = ProcessMonitor()
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        logging.info("Process monitor terminated")

if __name__ == "__main__":
    main()
