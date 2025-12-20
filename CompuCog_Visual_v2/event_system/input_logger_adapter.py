"""
InputLoggerAdapter - Direct Keyboard/Mouse Input Monitoring
Collects aggregate input metrics and converts to Event objects per CONTRACT_ATLAS.md

Per Phase 2 Architecture:
- NO subprocess launches
- NO JSONL file writes
- Direct data collection via Win32 APIs
- Returns Event objects → EventManager → Forge
"""

import ctypes
from ctypes import Structure, windll, c_uint, sizeof, byref
from typing import Dict, Any, Optional
import logging

from event_system.sensor_registry import SensorAdapter, SensorConfig
from event_system.event_manager import Event
from event_system.chronos_manager import ChronosManager


class LASTINPUTINFO(Structure):
    """Windows API structure for GetLastInputInfo"""
    _fields_ = [
        ('cbSize', c_uint),
        ('dwTime', c_uint),
    ]


class InputLoggerAdapter(SensorAdapter):
    """
    Direct keyboard/mouse input monitoring adapter.
    
    Per CONTRACT_ATLAS.md:
    - Collects aggregate input metrics (NO key content, NO coordinates)
    - Tracks: idle time, activity level, input event detection
    - Converts to Event objects (NEVER dicts or ForgeRecords)
    - EventManager writes Events → Forge via BinaryLog
    - source_id: "input_monitor"
    - tags: ["input", "keyboard", "mouse"]
    """
    
    def __init__(self, config: SensorConfig, chronos: ChronosManager, event_mgr):
        super().__init__(config, chronos, event_mgr)
        self.last_idle_time = 0.0
        self.input_events_this_period = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"InputAdapter[{config.source_id}]")
        
    def initialize(self) -> bool:
        """Initialize Win32 API access for input monitoring."""
        try:
            # Test Win32 API availability
            test_idle = self._get_idle_time_seconds()
            self.logger.info(f"InputLoggerAdapter initialized (current idle: {test_idle:.1f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InputLoggerAdapter: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring (no subprocess needed)."""
        try:
            self.running = True
            self.logger.info("Input monitoring started (direct API mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start input monitoring: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop monitoring (no cleanup needed)."""
        try:
            self.running = False
            self.logger.info("Input monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop input monitoring: {e}")
            return False
    
    def _get_idle_time_seconds(self) -> float:
        """Get seconds since last user input using Windows GetLastInputInfo API."""
        try:
            lastInputInfo = LASTINPUTINFO()
            lastInputInfo.cbSize = sizeof(lastInputInfo)
            windll.user32.GetLastInputInfo(byref(lastInputInfo))
            
            millis = windll.kernel32.GetTickCount() - lastInputInfo.dwTime
            return millis / 1000.0
        except Exception as e:
            self.logger.debug(f"Idle time check failed: {e}")
            return 0.0
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Collect current input activity metrics.
        
        Returns:
            Dict with input metrics, or None if no significant change
        """
        try:
            current_idle = self._get_idle_time_seconds()
            
            # Detect input event: idle time reset (went from higher to lower)
            input_detected = current_idle < self.last_idle_time
            if input_detected:
                self.input_events_this_period += 1
            
            # Return data if activity detected or threshold exceeded
            is_active = current_idle < 1.0
            
            data = {
                "idle_seconds": round(current_idle, 1),
                "is_idle": current_idle > 5.0,
                "is_active": is_active,
                "input_events": self.input_events_this_period
            }
            
            self.last_idle_time = current_idle
            
            # Only report if active or first data point
            if is_active or self.input_events_this_period == 0:
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error collecting input data: {e}")
            return None
    
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert input data to Event object per CONTRACT_ATLAS.md.
        
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
        
        # Build Event object per CONTRACT_ATLAS.md
        return Event(
            event_id=f"input_{int(timestamp * 1000)}",
            timestamp=timestamp,
            source_id=self.config.source_id,
            tags=self.config.tags + ["input", "activity", "user_interaction"],
            metadata={
                "idle_seconds": data.get("idle_seconds", 0.0),
                "is_idle": data.get("is_idle", False),
                "is_active": data.get("is_active", False),
                "input_events_detected": data.get("input_events", 0)
            }
        )
