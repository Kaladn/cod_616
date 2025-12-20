"""
MINIMAL Component 1 Test: SensorRegistry STRUCTURE ONLY
NO EventManager, NO event recording, JUST enum + config validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_system.sensor_registry import SensorType, SensorConfig

def main():
    print("=" * 60)
    print("MINIMAL TEST: SensorRegistry Structure")
    print("=" * 60)
    
    # TEST 1: Enum count
    sensor_types = list(SensorType)
    count = len(sensor_types)
    print(f"\nSensorType count: {count}")
    
    if count == 50:
        print("✅ PASS: 50 sensor types (30 defined + 20 expansion)")
    else:
        print(f"❌ FAIL: Expected 50, got {count}")
        return 1
    
    # TEST 2: Expansion slots
    names = [s.name for s in sensor_types]
    
    biometric = [n for n in names if n.startswith("BIOMETRIC_")]
    environmental = [n for n in names if n.startswith("ENVIRONMENTAL_")]
    custom = [n for n in names if n.startswith("CUSTOM_")]
    
    print(f"\nExpansion slots:")
    print(f"  BIOMETRIC: {len(biometric)} (expected 3)")
    print(f"  ENVIRONMENTAL: {len(environmental)} (expected 2)")
    print(f"  CUSTOM: {len(custom)} (expected 15)")
    
    if len(biometric) == 3 and len(environmental) == 2 and len(custom) == 15:
        print("✅ PASS: Expansion slots correct")
    else:
        print("❌ FAIL: Expansion slot count mismatch")
        return 1
    
    # TEST 3: SensorConfig creation
    try:
        config = SensorConfig(
            sensor_type=SensorType.KEYBOARD,
            source_id="test_keyboard",
            enabled=True,
            sample_rate_hz=10.0,
            buffer_size=100,
            tags=["input"],
            metadata={"device": "test"}
        )
        print(f"\n✅ PASS: SensorConfig created successfully")
        print(f"  Config: {config.source_id}, {config.sensor_type.value}, {config.sample_rate_hz}Hz")
    except Exception as e:
        print(f"\n❌ FAIL: SensorConfig creation failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ ✅ ✅ COMPONENT 1 STRUCTURE: VALIDATED ✅ ✅ ✅")
    print("=" * 60)
    print("\nSensorRegistry enum + config are correct.")
    print("EventManager integration will be tested separately.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
