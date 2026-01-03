"""
ActivityLoggerAdapter - Direct Window Activity Monitoring
Collects active window data and converts to Event objects per CONTRACT_ATLAS.md

Per Phase 2 Architecture:
- NO subprocess launches
- NO JSONL file writes
- Direct data collection via Win32 APIs
- Returns Event objects → EventManager → Forge
"""

import ctypes
from ctypes import wintypes
from typing import Dict, Any, Optional
import psutil

from event_system.sensor_registry import SensorAdapter, SensorConfig
from event_system.event_manager import Event
from event_system.chronos_manager import ChronosManager


# Windows API structures for idle detection
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.UINT),
        ('dwTime', wintypes.DWORD),
    ]


class ActivityLoggerAdapter(SensorAdapter):
    """
    Direct window activity monitoring adapter.
    
    Per CONTRACT_ATLAS.md:
    - Collects active window title, process info, idle time
    - Converts to Event objects (NEVER dicts or ForgeRecords)
    - EventManager writes Events → Forge via BinaryLog
    - source_id: "activity_monitor"
    - tags: ["activity", "window"]
    """
    
    def __init__(self, config: SensorConfig, chronos: ChronosManager, event_mgr):
        super().__init__(config, chronos, event_mgr)
        self.last_window_title = None
        self.last_process_name = None
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(f"ActivityAdapter[{config.source_id}]")
        
    def initialize(self) -> bool:
        """Initialize Win32 API access."""
        try:
            # Test Win32 API availability
            try:
                import win32gui
                import win32process
                self.win32gui = win32gui
                self.win32process = win32process
            except ImportError:
                self.logger.error("pywin32 not installed - required for activity monitoring")
                return False
            
            self.logger.info("ActivityLoggerAdapter initialized (direct Win32 API mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ActivityLoggerAdapter: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring (no subprocess needed)."""
        try:
            self.running = True
            self.logger.info("Activity monitoring started (direct API mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start activity monitoring: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop monitoring (no cleanup needed)."""
        try:
            self.running = False
            self.logger.info("Activity monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop activity monitoring: {e}")
            return False
    
    def _get_idle_duration(self) -> float:
        """Get seconds since last user input (keyboard/mouse)."""
        lastInputInfo = LASTINPUTINFO()
        lastInputInfo.cbSize = ctypes.sizeof(lastInputInfo)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lastInputInfo))
        millis = ctypes.windll.kernel32.GetTickCount() - lastInputInfo.dwTime
        return millis / 1000.0
    
    def _get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """
        Get currently active window title and process info via Win32 API.
        
        Returns:
            Dict with window_title, process_name, executable_path, idle_seconds
            or None if unable to retrieve
        """
        try:
            # Get active window handle
            hwnd = self.win32gui.GetForegroundWindow()
            if hwnd == 0:
                return None
            
            # Get process ID and window title
            _, pid = self.win32process.GetWindowThreadProcessId(hwnd)
            window_title = self.win32gui.GetWindowText(hwnd)
            
            # Get process info
            try:
                process = psutil.Process(pid)
                process_name = process.name()
                executable_path = process.exe() if process.exe() else ""
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_name = "unknown"
                executable_path = ""
            
            # Get idle time
            idle_seconds = self._get_idle_duration()
            
            return {
                "window_title": window_title,
                "process_name": process_name,
                "executable_path": executable_path,
                "idle_seconds": idle_seconds
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting active window: {e}")
            return None
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Collect current active window data.
        
        Returns:
            Dict with activity data, or None if no change from last poll
        """
        try:
            data = self._get_active_window_info()
            
            if not data:
                return None
            
            # Only return data if window or process changed (reduce noise)
            if (data["window_title"] == self.last_window_title and 
                data["process_name"] == self.last_process_name):
                return None
            
            # Update tracking
            self.last_window_title = data["window_title"]
            self.last_process_name = data["process_name"]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting activity data: {e}")
            return None
    
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert activity data to Event object per CONTRACT_ATLAS.md.
        
        Per CONTRACT_ATLAS.md Contract #2:
        - MUST return Event object (NEVER dict or ForgeRecord)
        - timestamp from ChronosManager (deterministic)
        - source_id from config
        - tags include sensor type
        - metadata is JSON-serializable
        
        Args:
            data: Dict from _get_active_window_info()
        
        Returns:
            Event object (validated by SensorAdapter.record_event)
        """
        # Get deterministic timestamp from ChronosManager
        timestamp = self.chronos.now()
        
        # Build Event object per CONTRACT_ATLAS.md
        return Event(
            event_id=f"activity_{int(timestamp * 1000)}",
            timestamp=timestamp,
            source_id=self.config.source_id,
            tags=self.config.tags + ["activity", "window", "focus_change"],
            metadata={
                "window_title": data.get("window_title", "unknown"),
                "process_name": data.get("process_name", "unknown"),
                "executable_path": data.get("executable_path", ""),
                "idle_seconds": data.get("idle_seconds", 0)
            }
        )
