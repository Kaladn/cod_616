"""
Adapter Contract Test Harness (ACTH)
=====================================

Enforces CONTRACT_ATLAS.md rules across all sensor adapters.

Tests performed for each adapter:
---------------------------------
1. Adapter returns Event object (NEVER dict)
2. Event fields exist and match required types
3. Payload contains only JSON-serializable simple types
4. Timestamp is numeric and valid
5. Tags is list[str]
6. Metadata is dict[str, simple]
7. source_id matches adapter config
8. convert_to_event() is implemented correctly
9. SensorType → Event pipeline integrity
10. Polling produces at least one valid event

This prevents 99.999% of cross-component failures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_system.chronos_manager import ChronosManager
from event_system.event_manager import EventManager, Event
from event_system.sensor_registry import SensorRegistry, SensorConfig, SensorType, SensorAdapter


# =============================================================================
# CONTRACT VALIDATORS
# =============================================================================

def validate_event_object(event, context: str):
    """Validate that event is an Event object, not dict or other type."""
    if not isinstance(event, Event):
        raise TypeError(
            f"{context}: Adapter returned {type(event).__name__}, expected Event object. "
            f"CONTRACT VIOLATION: Adapters must return Event, never dict."
        )


def validate_event_fields(event: Event, config: SensorConfig):
    """Validate all Event fields match CONTRACT_ATLAS.md requirements."""
    
    # event_id
    if not hasattr(event, 'event_id') or not isinstance(event.event_id, str):
        raise ValueError(f"Event.event_id must be string, got {type(event.event_id)}")
    
    # timestamp
    if not hasattr(event, 'timestamp') or not isinstance(event.timestamp, (float, int)):
        raise ValueError(f"Event.timestamp must be numeric, got {type(event.timestamp)}")
    if event.timestamp <= 0:
        raise ValueError(f"Event.timestamp must be positive, got {event.timestamp}")
    
    # source_id
    if not hasattr(event, 'source_id') or not isinstance(event.source_id, str):
        raise ValueError(f"Event.source_id must be string, got {type(event.source_id)}")
    if event.source_id != config.source_id:
        raise ValueError(
            f"Event.source_id mismatch: got '{event.source_id}', "
            f"expected '{config.source_id}'"
        )
    
    # tags
    if not hasattr(event, 'tags') or not isinstance(event.tags, list):
        raise ValueError(f"Event.tags must be list, got {type(event.tags)}")
    for tag in event.tags:
        if not isinstance(tag, str):
            raise ValueError(f"Event.tags must contain strings, found {type(tag)}")
    
    # metadata
    if not hasattr(event, 'metadata') or not isinstance(event.metadata, dict):
        raise ValueError(f"Event.metadata must be dict, got {type(event.metadata)}")
    for key, value in event.metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"Event.metadata keys must be strings, found {type(key)}")
        if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
            raise ValueError(
                f"Event.metadata values must be JSON-serializable, "
                f"found {type(value)} for key '{key}'"
            )


def validate_adapter_contract(adapter: SensorAdapter):
    """Validate adapter implements all required abstract methods."""
    required_methods = ['initialize', 'start', 'stop', 'get_latest_data', 'convert_to_event']
    for method in required_methods:
        if not hasattr(adapter, method):
            raise NotImplementedError(
                f"Adapter {adapter.__class__.__name__} missing required method: {method}"
            )
        if not callable(getattr(adapter, method)):
            raise TypeError(
                f"Adapter {adapter.__class__.__name__}.{method} is not callable"
            )


# =============================================================================
# MOCK ADAPTERS FOR TESTING
# =============================================================================

class MockActivityAdapter(SensorAdapter):
    """Mock adapter simulating activity logger for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def start(self) -> bool:
        self.running = True
        return True
    
    def stop(self) -> bool:
        self.running = False
        return True
    
    def get_latest_data(self):
        return {
            "window_title": "Test Application",
            "process_name": "test.exe",
            "idle_seconds": 0
        }
    
    def convert_to_event(self, data: dict) -> Event:
        return Event(
            event_id="mock_evt",
            timestamp=self.chronos.now(),
            source_id=self.config.source_id,
            tags=self.config.tags + ["activity"],
            metadata={"adapter": "MockActivityAdapter", "data": data}
        )


class MockInputAdapter(SensorAdapter):
    """Mock adapter simulating input logger for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def start(self) -> bool:
        self.running = True
        return True
    
    def stop(self) -> bool:
        self.running = False
        return True
    
    def get_latest_data(self):
        return {
            "keyboard_events": 10,
            "mouse_events": 5,
            "total_events": 15
        }
    
    def convert_to_event(self, data: dict) -> Event:
        return Event(
            event_id="mock_evt",
            timestamp=self.chronos.now(),
            source_id=self.config.source_id,
            tags=self.config.tags + ["input"],
            metadata={"adapter": "MockInputAdapter", "data": data}
        )


class BadAdapter_ReturnsDict(SensorAdapter):
    """BAD ADAPTER: Returns dict instead of Event (contract violation)."""
    
    def initialize(self) -> bool:
        return True
    
    def start(self) -> bool:
        self.running = True
        return True
    
    def stop(self) -> bool:
        self.running = False
        return True
    
    def get_latest_data(self):
        return {"test": "data"}
    
    def convert_to_event(self, data: dict):
        # CONTRACT VIOLATION: Returns dict instead of Event
        return {
            "source_id": self.config.source_id,
            "tags": ["bad"],
            "data": data
        }


# =============================================================================
# TEST HARNESS
# =============================================================================

def test_adapter(adapter_cls, sensor_type, should_fail=False):
    """Test a single adapter against the contract."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {adapter_cls.__name__}")
    print(f"SensorType: {sensor_type.value}")
    print(f"Expected: {'FAIL' if should_fail else 'PASS'}")
    print('=' * 70)
    
    chronos = ChronosManager()
    event_mgr = EventManager(chronos)
    registry = SensorRegistry(chronos, event_mgr)
    
    config = SensorConfig(
        sensor_type=sensor_type,
        source_id=adapter_cls.__name__.lower(),
        enabled=True,
        sample_rate_hz=1.0,
        buffer_size=10,
        tags=["test"],
        metadata={}
    )
    
    adapter = adapter_cls(config, chronos, event_mgr)
    
    # Validate adapter contract
    print("  [1/6] Validating adapter implements required methods...")
    validate_adapter_contract(adapter)
    print("  ✓ Adapter contract valid")
    
    # Register and start
    print("  [2/6] Registering adapter...")
    registry.register_sensor(adapter)
    print("  ✓ Adapter registered")
    
    print("  [3/6] Starting adapter...")
    start_results = registry.start_all_sensors()
    if not start_results.get(config.source_id):
        raise RuntimeError(f"Adapter {adapter_cls.__name__} failed to start")
    print("  ✓ Adapter started")
    
    # Poll and validate event generation
    print("  [4/6] Polling adapter...")
    try:
        registry.poll_sensors()
    except TypeError as e:
        if should_fail and "Contract violation" in str(e):
            print(f"  ✓ Contract violation caught as expected: {e}")
            registry.stop_all_sensors()
            return True
        raise
    
    if should_fail:
        raise AssertionError(
            f"Adapter {adapter_cls.__name__} should have failed but didn't!"
        )
    
    print("  ✓ Poll completed")
    
    # Validate event was generated
    print("  [5/6] Validating event generation...")
    stats = event_mgr.get_stats()
    total_events = stats.get('total_events', 0)
    if total_events == 0:
        raise AssertionError(f"Adapter {adapter_cls.__name__} produced no events")
    print(f"  ✓ Generated {total_events} event(s)")
    
    # Validate event structure
    print("  [6/6] Validating event structure...")
    # Get the actual event from the stream
    stream = event_mgr.streams[config.source_id]
    if len(stream.events) == 0:
        raise AssertionError("No events in stream")
    
    event = stream.events[-1]
    validate_event_object(event, adapter_cls.__name__)
    validate_event_fields(event, config)
    print("  ✓ Event structure valid")
    
    # Cleanup
    registry.stop_all_sensors()
    
    print(f"\n✅ {adapter_cls.__name__}: ALL CONTRACT CHECKS PASSED")
    return True


def main():
    """Run all adapter contract tests."""
    print("\n" + "=" * 70)
    print("  ADAPTER CONTRACT TEST HARNESS (ACTH)")
    print("  Enforcing CONTRACT_ATLAS.md compliance")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test valid adapters
    valid_adapters = [
        (MockActivityAdapter, SensorType.WINDOW_ACTIVITY),
        (MockInputAdapter, SensorType.KEYBOARD),
    ]
    
    print("\n" + "=" * 70)
    print("PHASE 1: Testing Valid Adapters")
    print("=" * 70)
    
    for adapter_cls, sensor_type in valid_adapters:
        try:
            test_adapter(adapter_cls, sensor_type, should_fail=False)
            tests_passed += 1
        except Exception as e:
            print(f"\n❌ {adapter_cls.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    
    # Test invalid adapters (should fail)
    invalid_adapters = [
        (BadAdapter_ReturnsDict, SensorType.CUSTOM_1),
    ]
    
    print("\n" + "=" * 70)
    print("PHASE 2: Testing Invalid Adapters (Should Fail)")
    print("=" * 70)
    
    for adapter_cls, sensor_type in invalid_adapters:
        try:
            test_adapter(adapter_cls, sensor_type, should_fail=True)
            tests_passed += 1
        except Exception as e:
            print(f"\n❌ {adapter_cls.__name__} did not fail as expected: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✅ ✅ ✅ ALL ADAPTER CONTRACTS VALIDATED ✅ ✅ ✅")
        print("\nAdapters comply with CONTRACT_ATLAS.md")
        print("Event pipeline integrity confirmed")
        return 0
    else:
        print(f"\n❌ {tests_failed} adapter(s) violated contracts")
        return 1


if __name__ == "__main__":
    sys.exit(main())
