"""
SensorRegistry v1 — Extensible Multi-Sensor Management

Central registry for all sensors (current + future). Provides standardized
interface for sensor registration, event routing, and extensibility.

Architecture:
- SensorType: 50 sensor types (30 defined + 20 custom expansion slots)
- SensorConfig: Configuration per sensor (sample rate, tags, metadata)
- SensorAdapter: Base class for all sensor adapters (abstract interface)
- SensorRegistry: Central coordinator (register, start, stop, poll)

Integration:
- ChronosManager: Deterministic timestamps
- EventManager: Event recording and 6-1-6 capsules

Author: Manus AI + GitHub Copilot
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

# Import cognitive foundation
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from event_system.chronos_manager import ChronosManager
from event_system.event_manager import EventManager, Event


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """
    Sensor type enumeration with 50 types (30 defined + 20 expansion slots).
    
    Categories:
    - Vision Sensors (1-5)
    - Input Sensors (6-10)
    - Network Sensors (11-15)
    - Audio Sensors (16-20)
    - System Sensors (21-25)
    - Activity Sensors (26-30)
    - Future Expansion (31-50)
    """
    
    # VISION SENSORS (1-5)
    TRUEVISION = "truevision"
    SCREEN_CAPTURE = "screen_capture"
    WEBCAM = "webcam"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"
    
    # INPUT SENSORS (6-10)
    KEYBOARD_INPUT = "keyboard_input"
    MOUSE = "mouse"
    GAMEPAD_INPUT = "gamepad_input"
    JOYSTICK = "joystick"
    TOUCH = "touch"
    
    # NETWORK SENSORS (11-15)
    NETWORK_TRAFFIC = "network_traffic"
    PACKET_ANALYZER = "packet_analyzer"
    LATENCY_MONITOR = "latency_monitor"
    BANDWIDTH_MONITOR = "bandwidth_monitor"
    CONNECTION_QUALITY = "connection_quality"
    
    # AUDIO SENSORS (16-20)
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"
    VOICE_ACTIVITY = "voice_activity"
    AUDIO_SPECTRUM = "audio_spectrum"
    ACOUSTIC_FINGERPRINT = "acoustic_fingerprint"
    
    # SYSTEM SENSORS (21-25)
    PROCESS_MONITOR = "process_monitor"
    CPU_MONITOR = "cpu_monitor"
    GPU_MONITOR = "gpu_monitor"
    MEMORY_MONITOR = "memory_monitor"
    DISK_IO = "disk_io"
    
    # ACTIVITY SENSORS (26-30)
    ACTIVITY_MONITOR = "activity_monitor"
    APPLICATION_FOCUS = "application_focus"
    USER_PRESENCE = "user_presence"
    IDLE_DETECTION = "idle_detection"
    SESSION_TRACKER = "session_tracker"
    
    # FUTURE EXPANSION SLOTS (31-50)
    BIOMETRIC_1 = "biometric_1"  # Heart rate, eye tracking, etc.
    BIOMETRIC_2 = "biometric_2"
    BIOMETRIC_3 = "biometric_3"
    ENVIRONMENTAL_1 = "environmental_1"  # Room temp, lighting, etc.
    ENVIRONMENTAL_2 = "environmental_2"
    CUSTOM_1 = "custom_1"
    CUSTOM_2 = "custom_2"
    CUSTOM_3 = "custom_3"
    CUSTOM_4 = "custom_4"
    CUSTOM_5 = "custom_5"
    CUSTOM_6 = "custom_6"
    CUSTOM_7 = "custom_7"
    CUSTOM_8 = "custom_8"
    CUSTOM_9 = "custom_9"
    CUSTOM_10 = "custom_10"
    CUSTOM_11 = "custom_11"
    CUSTOM_12 = "custom_12"
    CUSTOM_13 = "custom_13"
    CUSTOM_14 = "custom_14"
    CUSTOM_15 = "custom_15"


@dataclass
class SensorConfig:
    """
    Configuration for a sensor.
    
    Fields:
        sensor_type: Type of sensor (from SensorType enum)
        source_id: Unique identifier for EventManager (lowercase_snake_case)
        enabled: Whether sensor is active
        sample_rate_hz: Target sample rate (0 = event-driven, no polling)
        buffer_size: Event buffer size
        tags: Default tags for events
        metadata: Sensor-specific configuration
    """
    sensor_type: SensorType
    source_id: str
    enabled: bool = True
    sample_rate_hz: float = 0.0
    buffer_size: int = 100
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.sample_rate_hz >= 0.0, "sample_rate_hz must be >= 0.0"
        assert self.buffer_size > 0, "buffer_size must be > 0"
        assert isinstance(self.source_id, str), "source_id must be string"
        assert self.source_id == self.source_id.lower(), "source_id must be lowercase"


class SensorAdapter(ABC):
    """
    Base class for all sensor adapters.
    
    Provides standardized interface for converting sensor-specific data into
    EventManager events. All sensor adapters inherit from this.
    
    Abstract Methods:
        - initialize() -> bool: Initialize sensor hardware/connections
        - start() -> bool: Start sensor data collection
        - stop() -> bool: Stop sensor data collection
        - get_latest_data() -> Optional[Dict]: Get latest sensor reading
        - convert_to_event(data: Dict) -> Event: Convert sensor data to Event
    """
    
    def __init__(
        self,
        config: SensorConfig,
        chronos: ChronosManager,
        event_mgr: EventManager
    ):
        self.config = config
        self.chronos = chronos
        self.event_mgr = event_mgr
        self.running = False
        self.last_sample_time = 0.0
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize sensor hardware/connections.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start sensor data collection.
        
        Returns:
            True if start successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop sensor data collection.
        
        Returns:
            True if stop successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Get latest sensor reading.
        
        Returns:
            Dictionary with sensor data, or None if no data available
        """
        pass
    
    @abstractmethod
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert sensor data to EventManager Event.
        
        Args:
            data: Sensor-specific data dictionary
            
        Returns:
            Event object ready for EventManager
        """
        pass
    
    def should_sample(self) -> bool:
        """
        Check if it's time to sample (for polled sensors).
        
        Returns:
            True if sampling should occur, False otherwise
        """
        if self.config.sample_rate_hz == 0:
            return False  # Event-driven, no polling
        
        current_time = self.chronos.now()
        interval = 1.0 / self.config.sample_rate_hz
        
        if current_time - self.last_sample_time >= interval:
            self.last_sample_time = current_time
            return True
        
        return False
    
    def record_event(self, data: Dict[str, Any]) -> None:
        """
        Convert data to event and record in EventManager.
        
        Args:
            data: Sensor-specific data dictionary
        """
        try:
            event = self.convert_to_event(data)
            
            # ENFORCE CONTRACT: convert_to_event MUST return Event object, never dict
            if not isinstance(event, Event):
                raise TypeError(
                    f"Contract violation: {self.__class__.__name__}.convert_to_event() "
                    f"returned {type(event).__name__}, expected Event object. "
                    f"Dicts must be wrapped in Event.metadata or Event.payload, never returned directly."
                )
            
            # Record event through EventManager
            self.event_mgr.record_event(
                source_id=self.config.source_id,
                tags=self.config.tags + event.tags,
                metadata=event.metadata,
                pulse_id=event.pulse_id,
                nvme_ref=event.nvme_ref
            )
        except Exception as e:
            logger.error(f"Failed to record event from {self.config.source_id}: {e}")


class SensorRegistry:
    """
    Central registry for all sensors.
    
    Manages all sensors, routes events to EventManager, provides unified
    interface for sensor control and monitoring.
    
    Methods:
        - register_sensor(adapter: SensorAdapter): Register a sensor adapter
        - unregister_sensor(source_id: str): Unregister a sensor
        - start_all_sensors(): Start all registered sensors
        - stop_all_sensors(): Stop all running sensors
        - poll_sensors(): Poll all sensors that need polling (main loop)
        - get_sensor_stats() -> Dict: Get statistics about all sensors
        - get_active_sensors() -> List[str]: Get list of active sensor IDs
    """
    
    def __init__(self, chronos: ChronosManager, event_mgr: EventManager):
        self.chronos = chronos
        self.event_mgr = event_mgr
        self.sensors: Dict[str, SensorAdapter] = {}
        self.stats = {
            "total_sensors": 0,
            "active_sensors": 0,
            "total_events": 0,
            "events_per_sensor": {}
        }
    
    def register_sensor(self, adapter: SensorAdapter) -> bool:
        """
        Register a sensor adapter.
        
        Args:
            adapter: SensorAdapter instance to register
            
        Returns:
            True if registration successful, False otherwise
        """
        source_id = adapter.config.source_id
        
        if source_id in self.sensors:
            logger.warning(f"Sensor already registered: {source_id}")
            return False
        
        # Initialize sensor
        try:
            success = adapter.initialize()
            if not success:
                logger.error(f"Failed to initialize sensor: {source_id}")
                return False
        except Exception as e:
            logger.error(f"Exception during sensor initialization ({source_id}): {e}")
            return False
        
        # Register
        self.sensors[source_id] = adapter
        self.stats["total_sensors"] += 1
        self.stats["events_per_sensor"][source_id] = 0
        
        # Register source in EventManager
        sensor_kind = self._get_sensor_kind(adapter.config.sensor_type)
        self.event_mgr.register_source(source_id, kind=sensor_kind)
        
        logger.info(f"Registered sensor: {source_id} (type: {adapter.config.sensor_type.value})")
        return True
    
    def unregister_sensor(self, source_id: str) -> bool:
        """
        Unregister a sensor.
        
        Args:
            source_id: Source ID of sensor to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if source_id not in self.sensors:
            logger.warning(f"Sensor not registered: {source_id}")
            return False
        
        adapter = self.sensors[source_id]
        
        # Stop if running
        if adapter.running:
            try:
                adapter.stop()
                self.stats["active_sensors"] -= 1
            except Exception as e:
                logger.error(f"Error stopping sensor during unregister ({source_id}): {e}")
        
        # Remove from registry
        del self.sensors[source_id]
        self.stats["total_sensors"] -= 1
        
        logger.info(f"Unregistered sensor: {source_id}")
        return True
    
    def start_all_sensors(self) -> Dict[str, bool]:
        """
        Start all registered sensors.
        
        Returns:
            Dictionary mapping source_id -> success (bool)
        """
        results = {}
        
        for source_id, adapter in self.sensors.items():
            if not adapter.config.enabled:
                results[source_id] = False
                logger.info(f"Skipping disabled sensor: {source_id}")
                continue
            
            try:
                success = adapter.start()
                results[source_id] = success
                
                if success:
                    self.stats["active_sensors"] += 1
                    logger.info(f"Started sensor: {source_id}")
                else:
                    logger.error(f"Failed to start sensor: {source_id}")
            except Exception as e:
                logger.error(f"Exception starting sensor ({source_id}): {e}")
                results[source_id] = False
        
        return results
    
    def stop_all_sensors(self) -> Dict[str, bool]:
        """
        Stop all running sensors.
        
        Returns:
            Dictionary mapping source_id -> success (bool)
        """
        results = {}
        
        for source_id, adapter in self.sensors.items():
            if not adapter.running:
                results[source_id] = True
                continue
            
            try:
                success = adapter.stop()
                results[source_id] = success
                
                if success:
                    self.stats["active_sensors"] -= 1
                    logger.info(f"Stopped sensor: {source_id}")
                else:
                    logger.error(f"Failed to stop sensor: {source_id}")
            except Exception as e:
                logger.error(f"Exception stopping sensor ({source_id}): {e}")
                results[source_id] = False
        
        return results
    
    def poll_sensors(self) -> None:
        """
        Poll all sensors that need polling (called in main loop).
        
        For each sensor:
        1. Check if it's time to sample (based on sample_rate_hz)
        2. Get latest data from sensor
        3. Record event in EventManager
        """
        logger.debug(f"[DEBUG] poll_sensors() called, {len(self.sensors)} sensors registered")
        for source_id, adapter in self.sensors.items():
            logger.debug(f"[DEBUG] Checking sensor: {source_id}")
            if not adapter.running:
                logger.debug(f"[DEBUG] Sensor {source_id} not running, skipping")
                continue
            
            if not adapter.config.enabled:
                logger.debug(f"[DEBUG] Sensor {source_id} not enabled, skipping")
                continue
            
            # Check if it's time to sample
            logger.debug(f"[DEBUG] Checking should_sample() for {source_id}")
            if adapter.should_sample():
                logger.debug(f"[DEBUG] Sensor {source_id} should sample, getting data...")
                try:
                    data = adapter.get_latest_data()
                    logger.debug(f"[DEBUG] Got data from {source_id}: {data}")
                    
                    if data is not None:
                        logger.debug(f"[DEBUG] Recording event for {source_id}...")
                        adapter.record_event(data)
                        logger.debug(f"[DEBUG] Event recorded for {source_id}")
                        self.stats["total_events"] += 1
                        self.stats["events_per_sensor"][source_id] += 1
                except Exception as e:
                    logger.error(f"Error polling sensor ({source_id}): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.debug(f"[DEBUG] Sensor {source_id} should NOT sample yet")
    
    def get_sensor_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all sensors.
        
        Returns:
            Dictionary with sensor statistics
        """
        sensors_info = {}
        for source_id, adapter in self.sensors.items():
            sensors_info[source_id] = {
                "type": adapter.config.sensor_type.value,
                "enabled": adapter.config.enabled,
                "running": adapter.running,
                "sample_rate_hz": adapter.config.sample_rate_hz
            }
        
        return {
            "total_sensors": self.stats["total_sensors"],
            "active_sensors": self.stats["active_sensors"],
            "total_events": self.stats["total_events"],
            "events_per_sensor": self.stats["events_per_sensor"],
            "sensors": sensors_info
        }
    
    def get_active_sensors(self) -> List[str]:
        """
        Get list of active sensor source IDs.
        
        Returns:
            List of source IDs for running sensors
        """
        return [
            source_id
            for source_id, adapter in self.sensors.items()
            if adapter.running
        ]
    
    def _get_sensor_kind(self, sensor_type: SensorType) -> str:
        """
        Map sensor type to EventManager source kind.
        
        Args:
            sensor_type: SensorType enum value
            
        Returns:
            EventManager source kind string
        """
        # Vision sensors
        if sensor_type in [
            SensorType.TRUEVISION, SensorType.SCREEN_CAPTURE,
            SensorType.WEBCAM, SensorType.DEPTH_CAMERA, SensorType.THERMAL_CAMERA
        ]:
            return "sensor"
        
        # Input sensors
        elif sensor_type in [
            SensorType.KEYBOARD_INPUT, SensorType.MOUSE, SensorType.GAMEPAD_INPUT,
            SensorType.JOYSTICK, SensorType.TOUCH
        ]:
            return "sensor"
        
        # Network sensors
        elif sensor_type in [
            SensorType.NETWORK_TRAFFIC, SensorType.PACKET_ANALYZER,
            SensorType.LATENCY_MONITOR, SensorType.BANDWIDTH_MONITOR,
            SensorType.CONNECTION_QUALITY
        ]:
            return "sensor"
        
        # Audio sensors
        elif sensor_type in [
            SensorType.MICROPHONE, SensorType.SYSTEM_AUDIO,
            SensorType.VOICE_ACTIVITY, SensorType.AUDIO_SPECTRUM,
            SensorType.ACOUSTIC_FINGERPRINT
        ]:
            return "sensor"
        
        # System sensors
        elif sensor_type in [
            SensorType.PROCESS_MONITOR, SensorType.CPU_MONITOR,
            SensorType.GPU_MONITOR, SensorType.MEMORY_MONITOR, SensorType.DISK_IO
        ]:
            return "monitor"
        
        # Activity sensors
        elif sensor_type in [
            SensorType.ACTIVITY_MONITOR, SensorType.APPLICATION_FOCUS,
            SensorType.USER_PRESENCE, SensorType.IDLE_DETECTION,
            SensorType.SESSION_TRACKER
        ]:
            return "sensor"
        
        # Default
        else:
            return "sensor"


# Module-level test
if __name__ == "__main__":
    print("SensorRegistry v1 — Module Test")
    
    # Count only actual sensor type values (exclude enum internals)
    # Use __members__ which contains only user-defined enum members
    sensor_count = len(SensorType.__members__)
    print(f"Total sensor types defined: {sensor_count}")
    print(f"Expected: 50 (30 defined + 20 expansion slots)")
    
    # Verify enum count
    assert sensor_count == 50, f"Expected 50 sensor types, got {sensor_count}"
    
    print("✓ SensorType enum verified")
    print("\nSample sensor types:")
    for i, sensor_type in enumerate(list(SensorType)[:10], 1):
        print(f"  {i}. {sensor_type.value}")
    print("  ... (40 more types)")
    
    print("\n✓ sensor_registry.py module loaded successfully")
