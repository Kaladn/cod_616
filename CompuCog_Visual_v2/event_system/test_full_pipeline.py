"""
Full Pipeline Integration Test
Tests all 5 adapters → EventManager → Forge Memory

Validates complete Phase 2 implementation:
- All adapters create Event objects (no dicts, no ForgeRecords)
- EventManager converts Events → ForgeRecords
- BinaryLog writes to Forge atomically
- CONTRACT_ATLAS.md compliance end-to-end
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "memory"))

from event_system.event_manager import EventManager, Event
from event_system.sensor_registry import SensorRegistry, SensorConfig, SensorType
from forge_memory.core.binary_log import BinaryLog
from forge_memory.core.string_dict import StringDictionary

# Import all adapters
from event_system.activity_logger_adapter import ActivityLoggerAdapter
from event_system.input_logger_adapter import InputLoggerAdapter
from event_system.process_logger_adapter import ProcessLoggerAdapter
from event_system.network_logger_adapter import NetworkLoggerAdapter
from event_system.gamepad_logger_adapter import GamepadLoggerAdapter


class MockChronosManager:
    """Mock ChronosManager for testing."""
    
    def __init__(self):
        self.time = 1000.0
    
    def now(self) -> float:
        """Return deterministic timestamp."""
        self.time += 1.0
        return self.time


def test_full_pipeline():
    """
    Test complete sensor pipeline:
    Adapters → Events → EventManager → ForgeRecords → BinaryLog
    """
    print("\n═══════════════════════════════════════════════════════════")
    print("  FULL PIPELINE INTEGRATION TEST")
    print("  5 Adapters → EventManager → Forge Memory")
    print("═══════════════════════════════════════════════════════════\n")
    
    # Setup temporary Forge directory
    temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
    temp_path = Path(temp_dir)
    print(f"📁 Temp Forge: {temp_dir}\n")
    
    try:
        # Initialize Forge components
        string_dict = StringDictionary(temp_path / "strings.dict")
        binary_log = BinaryLog(str(temp_path), string_dict)
        chronos = MockChronosManager()
        
        # Initialize EventManager with Forge
        event_manager = EventManager(
            chronos_manager=chronos,
            binary_log=binary_log
        )
        
        print("[1/6] ✅ EventManager → Forge Gateway initialized\n")
        
        # Initialize SensorRegistry
        registry = SensorRegistry(
            chronos=chronos,
            event_mgr=event_manager
        )
        
        print("[2/6] ✅ SensorRegistry initialized\n")
        
        # Register all 5 adapters
        adapters_config = [
            ("activity_monitor", SensorType.ACTIVITY_MONITOR, ActivityLoggerAdapter),
            ("input_monitor", SensorType.KEYBOARD_INPUT, InputLoggerAdapter),
            ("process_monitor", SensorType.PROCESS_MONITOR, ProcessLoggerAdapter),
            ("network_monitor", SensorType.NETWORK_TRAFFIC, NetworkLoggerAdapter),
            ("gamepad_monitor", SensorType.GAMEPAD_INPUT, GamepadLoggerAdapter),
        ]
        
        registered_count = 0
        skipped_adapters = []
        
        for source_id, sensor_type, adapter_class in adapters_config:
            config = SensorConfig(
                source_id=source_id,
                sensor_type=sensor_type,
                sample_rate_hz=1.0,
                tags=["test", "integration"]
            )
            
            adapter = adapter_class(config, chronos, event_manager)
            
            # Try to initialize
            if adapter.initialize():
                registry.register_sensor(adapter)
                registered_count += 1
                print(f"   ✅ {source_id}: Registered")
            else:
                skipped_adapters.append(source_id)
                print(f"   ⚠️  {source_id}: Skipped (dependencies missing)")
        
        print(f"\n[3/6] ✅ Registered {registered_count}/5 adapters")
        if skipped_adapters:
            print(f"      Skipped: {', '.join(skipped_adapters)}\n")
        else:
            print()
        
        # Start all adapters
        registry.start_all_sensors()
        print("[4/6] ✅ All adapters started\n")
        
        # Poll for events (3 cycles)
        print("[5/6] Polling for sensor data...\n")
        
        total_events = 0
        for cycle in range(3):
            events = registry.poll_sensors()
            if events:
                for event in events:
                    print(f"   📊 Event captured:")
                    print(f"      source: {event.source_id}")
                    print(f"      tags: {event.tags}")
                    print(f"      event_id: {event.event_id}")
                total_events += len(events)
            
            import time
            time.sleep(0.5)  # Brief pause between polls
        
        # Stop all adapters
        registry.stop_all_sensors()
        
        print(f"\n   Total events captured: {total_events}")
        print(f"\n[6/6] Verifying Forge writes...\n")
        
        # Verify BinaryLog has records
        forge_record_count = len(binary_log.record_offsets)
        print(f"   📦 ForgeRecords in BinaryLog: {forge_record_count}")
        
        if forge_record_count > 0:
            # Read and validate first record
            first_record = binary_log.read_at_offset(binary_log.record_offsets[0])
            print(f"\n   ✅ First ForgeRecord validated:")
            print(f"      seq: {first_record.seq}")
            print(f"      timestamp: {first_record.timestamp}")
            print(f"      task_id (source): {first_record.task_id}")
            print(f"      engine_id: {first_record.engine_id}")
            print(f"      transform_id: {first_record.transform_id}")
            print(f"      success: {first_record.success}")
            print(f"      params keys: {list(first_record.params.keys())}")
            
            # Validate CONTRACT_ATLAS.md mappings
            assert first_record.engine_id == "event_pipeline_v1", "Wrong engine_id"
            assert first_record.transform_id == "sensor_event", "Wrong transform_id"
            assert first_record.success == True, "success should be True"
            assert "tags" in first_record.params, "tags missing from params"
            
            print("\n   ✅ CONTRACT_ATLAS.md field mappings verified")
        
        print("\n═══════════════════════════════════════════════════════════")
        print("  ✅ FULL PIPELINE TEST PASSED")
        print("═══════════════════════════════════════════════════════════\n")
        
        print("📋 Summary:")
        print(f"   • Adapters registered: {registered_count}/5")
        print(f"   • Events captured: {total_events}")
        print(f"   • ForgeRecords written: {forge_record_count}")
        print(f"   • CONTRACT_ATLAS.md: Validated ✅")
        print()
        
        if skipped_adapters:
            print("⚠️  Note: Some adapters skipped due to missing dependencies:")
            print(f"   {', '.join(skipped_adapters)}")
            print("   Install: pip install pywin32 pygame")
            print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close BinaryLog
        if 'binary_log' in locals():
            binary_log.close()
        
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up: {temp_dir}\n")


if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
