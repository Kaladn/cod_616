"""
Minimal test to validate EventManager → Forge Gateway integration.

Tests CONTRACT_ATLAS.md v1.1 Contract #12:
- Event → ForgeRecord conversion
- BinaryLog atomic append
- Field mapping correctness
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))

from event_system.event_manager import EventManager, Event
from forge_memory.core.binary_log import BinaryLog
from forge_memory.core.string_dict import StringDictionary


class MockChronosManager:
    """Mock ChronosManager for testing."""
    
    def __init__(self):
        self.time = 1000.0
    
    def now(self) -> float:
        """Return deterministic timestamp."""
        self.time += 1.0
        return self.time


def test_eventmanager_forge_integration():
    """
    Test EventManager writes Event → ForgeRecord → BinaryLog.
    
    Validates:
    1. Event created with correct structure
    2. ForgeRecord created with CONTRACT_ATLAS.md field mappings
    3. BinaryLog append succeeds
    4. ForgeRecord can be read back from BinaryLog
    5. Field values match expected mappings
    """
    print("\n=== TEST: EventManager → Forge Gateway ===\n")
    
    # Setup temporary Forge directory
    temp_dir = tempfile.mkdtemp(prefix="test_forge_")
    temp_path = Path(temp_dir)
    print(f"Created temp Forge directory: {temp_dir}")
    
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
        
        # Register test source
        event_manager.register_source(
            source_id="test_sensor",
            kind="sensor",
            metadata={"test": True}
        )
        
        print("✅ EventManager initialized with BinaryLog")
        
        # Record test event
        event = event_manager.record_event(
            source_id="test_sensor",
            tags=["test", "validation"],
            metadata={"frame": 123, "confidence": 0.95}
        )
        
        print(f"✅ Event recorded: {event.event_id}")
        print(f"   timestamp: {event.timestamp}")
        print(f"   source_id: {event.source_id}")
        print(f"   tags: {event.tags}")
        
        # Verify BinaryLog has exactly 1 record
        assert len(binary_log.record_offsets) == 1, f"Expected 1 record, got {len(binary_log.record_offsets)}"
        print(f"✅ BinaryLog contains 1 record at offset {binary_log.record_offsets[0]}")
        
        # Read ForgeRecord from BinaryLog
        forge_record = binary_log.read_at_offset(binary_log.record_offsets[0])
        print("\n=== ForgeRecord Field Validation ===")
        
        # Validate CONTRACT_ATLAS.md field mappings
        checks = []
        
        # Check 1: seq = event_counter
        expected_seq = 1  # First event
        checks.append(("seq", forge_record.seq, expected_seq, forge_record.seq == expected_seq))
        
        # Check 2: timestamp = event.timestamp
        checks.append(("timestamp", forge_record.timestamp, event.timestamp, forge_record.timestamp == event.timestamp))
        
        # Check 3: task_id = source_id
        checks.append(("task_id", forge_record.task_id, event.source_id, forge_record.task_id == event.source_id))
        
        # Check 4: engine_id = "event_pipeline_v1"
        checks.append(("engine_id", forge_record.engine_id, "event_pipeline_v1", forge_record.engine_id == "event_pipeline_v1"))
        
        # Check 5: transform_id = "sensor_event"
        checks.append(("transform_id", forge_record.transform_id, "sensor_event", forge_record.transform_id == "sensor_event"))
        
        # Check 6: success = True
        checks.append(("success", forge_record.success, True, forge_record.success == True))
        
        # Check 7: worker_id = hash(source_id) % 256
        expected_worker_id = hash(event.source_id) % 256
        checks.append(("worker_id", forge_record.worker_id, expected_worker_id, forge_record.worker_id == expected_worker_id))
        
        # Check 8: pulse_id = event_counter
        checks.append(("pulse_id", forge_record.pulse_id, 1, forge_record.pulse_id == 1))
        
        # Check 9: params contains tags
        has_tags = "tags" in forge_record.params and forge_record.params["tags"] == event.tags
        checks.append(("params['tags']", forge_record.params.get("tags"), event.tags, has_tags))
        
        # Check 10: context = metadata
        checks.append(("context", forge_record.context, event.metadata, forge_record.context == event.metadata))
        
        # Check 11: grid_shape_in = (0, 0)
        checks.append(("grid_shape_in", forge_record.grid_shape_in, (0, 0), forge_record.grid_shape_in == (0, 0)))
        
        # Check 12: failure_reason = None
        checks.append(("failure_reason", forge_record.failure_reason, None, forge_record.failure_reason is None))
        
        # Print results
        all_passed = True
        for field_name, actual, expected, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {field_name}: {actual} == {expected}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ ALL CONTRACT_ATLAS.md FIELD MAPPINGS VALIDATED")
        else:
            print("\n❌ SOME FIELD MAPPINGS FAILED")
            return False
        
        # Test multiple events to verify seq increments
        event2 = event_manager.record_event(
            source_id="test_sensor",
            tags=["second"],
            metadata={"frame": 124}
        )
        
        assert len(binary_log.record_offsets) == 2, f"Expected 2 records, got {len(binary_log.record_offsets)}"
        
        forge_record2 = binary_log.read_at_offset(binary_log.record_offsets[1])
        assert forge_record2.seq == 2, f"Expected seq=2, got {forge_record2.seq}"
        assert forge_record2.pulse_id == 2, f"Expected pulse_id=2, got {forge_record2.pulse_id}"
        
        print(f"\n✅ Sequential events validated (seq={forge_record2.seq}, pulse_id={forge_record2.pulse_id})")
        
        print("\n=== TEST PASSED ===")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close BinaryLog before cleanup (Windows requires file handles closed)
        if 'binary_log' in locals():
            binary_log.close()
        
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    success = test_eventmanager_forge_integration()
    sys.exit(0 if success else 1)
