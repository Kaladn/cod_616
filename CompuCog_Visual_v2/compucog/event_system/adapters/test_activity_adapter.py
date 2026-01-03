"""
Test ActivityLoggerAdapter against CONTRACT_ATLAS.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_system.chronos_manager import ChronosManager
from event_system.event_manager import EventManager, Event
from event_system.sensor_registry import SensorRegistry, SensorConfig, SensorType
from event_system.activity_logger_adapter import ActivityLoggerAdapter


def main():
    print("=" * 70)
    print("Testing ActivityLoggerAdapter")
    print("=" * 70)
    
    # Initialize core components
    print("\n[1/7] Initializing ChronosManager...")
    chronos = ChronosManager()
    print(f"✓ Chronos time: {chronos.now()}")
    
    print("\n[2/7] Initializing EventManager...")
    event_mgr = EventManager(chronos)
    print("✓ EventManager ready")
    
    print("\n[3/7] Initializing SensorRegistry...")
    registry = SensorRegistry(chronos, event_mgr)
    print("✓ Registry ready")
    
    # Create adapter config
    print("\n[4/7] Creating ActivityLoggerAdapter config...")
    config = SensorConfig(
        sensor_type=SensorType.WINDOW_ACTIVITY,
        source_id="activity_monitor",
        enabled=True,
        sample_rate_hz=0.33,  # Every 3 seconds
        buffer_size=10,
        tags=["test"],
        metadata={}
    )
    print("✓ Config created")
    
    # Create and register adapter
    print("\n[5/7] Creating adapter...")
    adapter = ActivityLoggerAdapter(config, chronos, event_mgr)
    
    # Validate adapter has required methods
    required_methods = ['initialize', 'start', 'stop', 'get_latest_data', 'convert_to_event']
    for method in required_methods:
        if not hasattr(adapter, method):
            raise AssertionError(f"Missing required method: {method}")
    print("✓ Adapter implements all required methods")
    
    # Skip initialization (would launch subprocess)
    print("\n[6/7] Skipping adapter.initialize() (would launch daemon subprocess)")
    print("✓ Contract test does not require real subprocess")
    
    # Test convert_to_event with mock data
    print("\n[7/7] Testing convert_to_event() contract...")
    mock_data = {
        "timestamp": "2025-12-04T12:00:00",
        "active_window_title": "Test Window",
        "active_process": "test.exe",
        "idle_seconds": 0
    }
    
    event = adapter.convert_to_event(mock_data)
    
    # Validate event is Event object (not dict)
    if not isinstance(event, Event):
        raise TypeError(
            f"CONTRACT VIOLATION: convert_to_event returned {type(event).__name__}, "
            f"expected Event object"
        )
    print("✓ Returns Event object (not dict)")
    
    # Validate event structure per CONTRACT_ATLAS.md
    if not isinstance(event.event_id, str):
        raise ValueError("event_id must be string")
    print(f"✓ event_id: {event.event_id}")
    
    if not isinstance(event.timestamp, (float, int)):
        raise ValueError("timestamp must be numeric")
    print(f"✓ timestamp: {event.timestamp}")
    
    if event.source_id != "activity_monitor":
        raise ValueError(f"source_id should be 'activity_monitor', got '{event.source_id}'")
    print(f"✓ source_id: {event.source_id}")
    
    if not isinstance(event.tags, list) or not all(isinstance(t, str) for t in event.tags):
        raise ValueError("tags must be list[str]")
    print(f"✓ tags: {event.tags}")
    
    if not isinstance(event.metadata, dict):
        raise ValueError("metadata must be dict")
    print(f"✓ metadata keys: {list(event.metadata.keys())}")
    
    # Validate required metadata fields
    required_fields = ["adapter", "window_title", "process", "idle_seconds"]
    for field in required_fields:
        if field not in event.metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    print(f"✓ All required metadata fields present")
    
    print("\n" + "=" * 70)
    print("✅ ActivityLoggerAdapter: CONTRACT VALIDATED")
    print("=" * 70)
    print("\nAdapter complies with CONTRACT_ATLAS.md:")
    print("  - Returns Event object (not dict)")
    print("  - All required methods implemented")
    print("  - Event structure matches specification")
    print("  - source_id, tags, metadata correct")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
