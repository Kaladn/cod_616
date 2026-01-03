"""
ProcessLoggerAdapter - Direct Process Monitoring
Monitors process spawns/exits and converts to Event objects per CONTRACT_ATLAS.md

Per Phase 2 Architecture:
- NO subprocess launches
- NO JSONL file writes
- Direct data collection via psutil
- Returns Event objects → EventManager → Forge
"""

from typing import Dict, Any, Optional, Set
import logging
import psutil
from datetime import datetime

from event_system.sensor_registry import SensorAdapter, SensorConfig
from event_system.event_manager import Event
from event_system.chronos_manager import ChronosManager


# System processes to ignore (Windows noise)
SYSTEM_NOISE = {
    'System', 'Idle', 'Registry', 'smss.exe', 'csrss.exe', 'wininit.exe',
    'services.exe', 'lsass.exe', 'svchost.exe', 'dwm.exe', 'winlogon.exe',
    'fontdrvhost.exe', 'WUDFHost.exe', 'conhost.exe', 'sihost.exe',
    'taskhostw.exe', 'RuntimeBroker.exe', 'spoolsv.exe', 'SearchIndexer.exe',
    'MsMpEng.exe', 'SgrmBroker.exe', 'SecurityHealthService.exe',
    'audiodg.exe', 'dllhost.exe', 'WmiPrvSE.exe', 'CompPkgSrv.exe'
}


class ProcessLoggerAdapter(SensorAdapter):
    """
    Direct process spawn/exit monitoring adapter.
    
    Per CONTRACT_ATLAS.md:
    - Monitors new process spawns and exits
    - Filters out system noise
    - Converts to Event objects (NEVER dicts or ForgeRecords)
    - EventManager writes Events → Forge via BinaryLog
    - source_id: "process_monitor"
    - tags: ["process", "spawn", "exit"]
    """
    
    def __init__(self, config: SensorConfig, chronos: ChronosManager, event_mgr):
        super().__init__(config, chronos, event_mgr)
        self.tracked_pids: Set[int] = set()
        self.last_scan_time = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(f"ProcessAdapter[{config.source_id}]")
        
    def initialize(self) -> bool:
        """Initialize process monitoring."""
        try:
            # Get initial process snapshot
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    self.tracked_pids.add(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            self.logger.info(f"ProcessLoggerAdapter initialized (tracking {len(self.tracked_pids)} processes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ProcessLoggerAdapter: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring (no subprocess needed)."""
        try:
            self.running = True
            self.last_scan_time = datetime.now().timestamp()
            self.logger.info("Process monitoring started (direct psutil mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start process monitoring: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop monitoring (no cleanup needed)."""
        try:
            self.running = False
            self.logger.info("Process monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop process monitoring: {e}")
            return False
    
    def _classify_origin(self, proc_info: Dict) -> str:
        """Classify process origin (user_initiated, system_service, etc.)."""
        try:
            username = proc_info.get('username', '')
            parent_name = proc_info.get('parent_name', '')
            exe_path = proc_info.get('exe', '')
            
            # System account processes
            if username and username.upper() in ('SYSTEM', 'LOCAL SERVICE', 'NETWORK SERVICE', 'NT AUTHORITY\\\\SYSTEM'):
                return 'system_service'
            
            # Explorer.exe parent = user clicked something
            if parent_name and 'explorer.exe' in parent_name.lower():
                return 'user_initiated'
            
            # Processes from user profile
            if exe_path and 'Users\\\\' in exe_path and '\\\\AppData\\\\' in exe_path:
                return 'user_initiated'
            
            # Services.exe parent = Windows service
            if parent_name and 'services.exe' in parent_name.lower():
                return 'system_service'
            
            return 'unknown'
        except Exception:
            return 'unknown'
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Scan for new process spawns or exits.
        
        Returns:
            Dict with process event data, or None if no changes
        """
        try:
            current_pids = set()
            new_processes = []
            
            # Scan current processes
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'username', 'ppid', 'create_time']):
                try:
                    proc_info = proc.info
                    pid = proc_info['pid']
                    name = proc_info['name']
                    
                    current_pids.add(pid)
                    
                    # Skip system noise
                    if name in SYSTEM_NOISE:
                        continue
                    
                    # Detect new process
                    if pid not in self.tracked_pids:
                        # Get parent info
                        parent_name = ""
                        try:
                            parent = proc.parent()
                            if parent:
                                parent_name = parent.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
                        proc_info['parent_name'] = parent_name
                        
                        new_processes.append({
                            'event_type': 'spawn',
                            'pid': pid,
                            'name': name,
                            'exe': proc_info.get('exe', ''),
                            'cmdline': ' '.join(proc_info.get('cmdline', []))[:500],  # Truncate
                            'username': proc_info.get('username', ''),
                            'ppid': proc_info.get('ppid', 0),
                            'parent_name': parent_name,
                            'create_time': proc_info.get('create_time', 0),
                            'origin': self._classify_origin(proc_info)
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Detect exited processes
            exited_pids = self.tracked_pids - current_pids
            
            # Update tracking
            self.tracked_pids = current_pids
            
            # Return first new event (if any)
            if new_processes:
                return new_processes[0]  # Report one at a time
            
            if exited_pids:
                # Return exit event for first exited PID
                return {
                    'event_type': 'exit',
                    'pid': next(iter(exited_pids)),
                    'name': 'unknown',
                    'origin': 'process_exit'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning processes: {e}")
            return None
    
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert process data to Event object per CONTRACT_ATLAS.md.
        
        Per CONTRACT_ATLAS.md Contract #2:
        - MUST return Event object (NEVER dict or ForgeRecord)
        - timestamp from ChronosManager (deterministic)
        - source_id from config
        - tags include sensor type
        - metadata is JSON-serializable
        
        Args:
            data: Dict from get_latest_data()
        
        Returns:
            Event object (validated by SensorAdapter.record_event)
        """
        # Get deterministic timestamp from ChronosManager
        timestamp = self.chronos.now()
        
        event_type = data.get('event_type', 'spawn')
        
        # Build Event object per CONTRACT_ATLAS.md
        return Event(
            event_id=f"process_{event_type}_{int(timestamp * 1000)}",
            timestamp=timestamp,
            source_id=self.config.source_id,
            tags=self.config.tags + ["process", event_type, data.get('origin', 'unknown')],
            metadata={
                "event_type": event_type,
                "pid": data.get("pid", 0),
                "process_name": data.get("name", "unknown"),
                "executable_path": data.get("exe", ""),
                "command_line": data.get("cmdline", ""),
                "parent_pid": data.get("ppid", 0),
                "parent_name": data.get("parent_name", ""),
                "username": data.get("username", ""),
                "origin": data.get("origin", "unknown")
            }
        )
