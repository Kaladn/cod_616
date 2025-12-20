"""
Component 1 End-to-End Test: SensorRegistry
Tests enum structure, adapter contract, and event flow.
"""

import sys
import logging
from pathlib import Path

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_system.chronos_manager import ChronosManager
from event_system.event_manager import EventManager, Event
from event_system.sensor_registry import SensorRegistry, SensorType, SensorAdapter, SensorConfig


def test_enum_structure():
    """Test 1: Verify SensorType enum has exactly 50 members with correct structure."""
    print("=" * 60)
    print("TEST 1: Enum Structure Validation")
    print("=" * 60)
    
    sensor_types = list(SensorType)
    count = len(sensor_types)
    
    print(f"Total SensorType count: {count}")
    assert count == 50, f"Expected 50 sensor types, got {count}"
    print("✓ Count correct: 50 types")
    
    # Verify expansion slots exist
    expansion_names = [s.name for s in sensor_types]
    
    assert "BIOMETRIC_1" in expansion_names, "Missing BIOMETRIC_1"
    assert "BIOMETRIC_2" in expansion_names, "Missing BIOMETRIC_2"
    assert "BIOMETRIC_3" in expansion_names, "Missing BIOMETRIC_3"
    print("✓ Biometric slots present: 3")
    
    assert "ENVIRONMENTAL_1" in expansion_names, "Missing ENVIRONMENTAL_1"
    assert "ENVIRONMENTAL_2" in expansion_names, "Missing ENVIRONMENTAL_2"
    print("✓ Environmental slots present: 2")
    
    custom_slots = [s for s in expansion_names if s.startswith("CUSTOM_")]
    assert len(custom_slots) == 15, f"Expected 15 CUSTOM slots, got {len(custom_slots)}"
    print(f"✓ Custom slots present: {len(custom_slots)}")
    
    # Verify CUSTOM_16+ were removed
    assert "CUSTOM_16" not in expansion_names, "CUSTOM_16 should be removed"
    assert "CUSTOM_20" not in expansion_names, "CUSTOM_20 should be removed"
    print("✓ CUSTOM_16-20 correctly removed")
    
    print("\n✅ TEST 1 PASSED: Enum structure correct (30 defined + 20 expansion = 50)\n")


def test_registry_behavior():
    """Test 2: Verify SensorRegistry can register, start, poll, and stop sensors."""
    print("=" * 60)
    print("TEST 2: Registry Behavior Validation")
    print("=" * 60)
    
    # Initialize core components
    chronos = ChronosManager()
    event_mgr = EventManager(chronos)
    registry = SensorRegistry(chronos, event_mgr)
    print("✓ Core components initialized")
    
    # Create dummy sensor adapter
    class DummySensor(SensorAdapter):
        def __init__(self, config, chronos, event_mgr):
            super().__init__(config, chronos, event_mgr)
            self._counter = 0
            self._initialized = False
            self.running = False
        
        def initialize(self) -> bool:
            self._initialized = True
            return True
        
        def start(self) -> bool:
            if not self._initialized:
                return False
            self.running = True
            return True
        
        def stop(self) -> bool:
            self.running = False
            return True
        
        def get_latest_data(self):
            if not self.running:
                return None
            self._counter += 1
            return {
                "counter": self._counter,
                "sensor_id": self.config.source_id,
                "test_value": 42.0
            }
        
        def convert_to_event(self, data: dict) -> Event:
            """Convert sensor data to EventManager-compatible event."""
            if data is None:
                return None
            return Event(
                tags=self.config.tags,
                metadata={
                    "sensor_type": self.config.sensor_type.value,
                    "data": data
                },
                pulse_id=None,
                nvme_ref=None
            )
    
    # Create sensor config
    config = SensorConfig(
        sensor_type=SensorType.CUSTOM_1,
        source_id="test_dummy_sensor",
        enabled=True,
        sample_rate_hz=1.0,
        buffer_size=10,
        tags=["test", "component_1"],
        metadata={"purpose": "validation"}
    )
    print("✓ Test sensor config created")
    
    # Register sensor
    dummy = DummySensor(config, chronos, event_mgr)
    registry.register_sensor(dummy)
    print("✓ Dummy sensor registered")
    
    # Start all sensors
    start_results = registry.start_all_sensors()
    assert "test_dummy_sensor" in start_results, "Sensor not found in start results"
    assert start_results["test_dummy_sensor"] is True, "Sensor failed to start"
    print(f"✓ Sensors started: {start_results}")
    
    # Poll sensors (should generate events)
    initial_stats = event_mgr.get_stats()
    print(f"  Initial event count: {initial_stats.get('total_events', 0)}")
    
    # Poll ONCE to test basic flow
    print("  Attempting single poll...")
    registry.poll_sensors()
    print("✓ Single poll completed")
    
    post_poll_stats = event_mgr.get_stats()
    print(f"  Post-poll event count: {post_poll_stats.get('total_events', 0)}")
    
    # Verify events were generated
    total_events = post_poll_stats.get('total_events', 0)
    assert total_events > 0, "No events generated after polling"
    print(f"✓ Events generated: {total_events}")
    
    # Check sensor stats
    sensor_stats = registry.get_sensor_stats()
    assert "test_dummy_sensor" in sensor_stats, "Sensor stats not found"
    print(f"✓ Sensor stats: {sensor_stats['test_dummy_sensor']}")
    
    # Stop all sensors
    stop_results = registry.stop_all_sensors()
    assert stop_results["test_dummy_sensor"] is True, "Sensor failed to stop"
    print(f"✓ Sensors stopped: {stop_results}")
    
    print("\n✅ TEST 2 PASSED: Registry behavior correct (register/start/poll/stop)\n")


def main():
    """Run all Component 1 tests."""
    print("\n" + "=" * 60)
    print("COMPONENT 1 END-TO-END TEST: SensorRegistry")
    print("=" * 60 + "\n")
    
    try:
        test_enum_structure()
        test_registry_behavior()
        
        print("=" * 60)
        print("✅ ✅ ✅ COMPONENT 1: FULLY VALIDATED ✅ ✅ ✅")
        print("=" * 60)
        print("\nSensorRegistry is ready for production use.")
        print("Proceed to Component 2: Logger Adapters\n")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
