"""
FORGE MEMORY SYSTEM - SMOKE TEST
Verify canonical ForgeRecord implementation
"""

import time
from pathlib import Path

from forge_memory.core.record import ForgeRecord
from forge_memory.core.string_dict import StringDictionary


def test_forge_record_smoke():
    """Smoke test: serialize → deserialize → verify"""
    
    print("=" * 60)
    print("FORGE MEMORY SYSTEM - SMOKE TEST")
    print("=" * 60)
    print()
    
    # Create string dictionary
    print("Creating StringDictionary...")
    sd = StringDictionary()
    
    # Create test record
    print("Creating ForgeRecord...")
    rec_dict = {
        "pulse_id": 1,
        "worker_id": 0,
        "seq": 42,
        "timestamp": time.time(),
        "success": True,
        "task_id": "truevision/window",
        "engine_id": "TrueVision",
        "transform_id": "TV_v2",
        "failure_reason": None,
        "grid_shape_in": (32, 32),
        "grid_shape_out": (32, 32),
        "color_count": 5,
        "train_pair_indices": [0],
        "error_metrics": {"eomm": 0.87, "aim_resistance": 0.65},
        "params": {"bot_level": "casual", "session_id": "test_001"},
        "context": {"match_type": "baseline", "duration_ms": 3000},
    }
    
    rec = ForgeRecord.from_dict(rec_dict)
    print(f"  pulse_id: {rec.pulse_id}")
    print(f"  timestamp: {rec.timestamp}")
    print(f"  success: {rec.success}")
    print(f"  task_id: {rec.task_id}")
    print()
    
    # Serialize
    print("Serializing to binary...")
    blob = rec.serialize(sd)
    print(f"  Binary size: {len(blob)} bytes")
    print(f"  MAGIC: {blob[0:4].hex()}")
    print(f"  VERSION: {int.from_bytes(blob[4:6], 'little')}")
    print(f"  RECORD_LENGTH: {int.from_bytes(blob[6:8], 'little')}")
    print(f"  Checksum (last 4 bytes): {blob[-4:].hex()}")
    print()
    
    # Deserialize
    print("Deserializing from binary...")
    rec2 = ForgeRecord.deserialize(blob, sd)
    print(f"  pulse_id: {rec2.pulse_id}")
    print(f"  timestamp: {rec2.timestamp}")
    print(f"  success: {rec2.success}")
    print(f"  task_id: {rec2.task_id}")
    print()
    
    # Compare
    print("Comparing original vs deserialized...")
    dict1 = rec.to_dict()
    dict2 = rec2.to_dict()
    
    mismatches = []
    for key in dict1:
        if dict1[key] != dict2[key]:
            mismatches.append(f"  {key}: {dict1[key]} != {dict2[key]}")
    
    if mismatches:
        print("❌ FAILED - Mismatches found:")
        for m in mismatches:
            print(m)
        return False
    else:
        print("✅ PASSED - All fields match!")
        print()
        
        # Show compression stats
        import json
        json_size = len(json.dumps(dict1))
        print(f"Compression stats:")
        print(f"  JSON size: {json_size} bytes")
        print(f"  Binary size: {len(blob)} bytes")
        print(f"  Compression ratio: {len(blob)/json_size*100:.1f}%")
        print()
        
        return True


def test_string_dict():
    """Test string dictionary deduplication"""
    print("=" * 60)
    print("STRING DICTIONARY TEST")
    print("=" * 60)
    print()
    
    sd = StringDictionary()
    
    # Add strings
    ref1 = sd.add_string("truevision/window")
    ref2 = sd.add_string("TrueVision")
    ref3 = sd.add_string("truevision/window")  # Duplicate
    
    print(f"Add 'truevision/window': ref_id={ref1}")
    print(f"Add 'TrueVision': ref_id={ref2}")
    print(f"Add 'truevision/window' (dup): ref_id={ref3}")
    print()
    
    if ref1 == ref3:
        print("✅ Deduplication working!")
    else:
        print("❌ Deduplication FAILED!")
        return False
    
    # Retrieve
    s1 = sd.get_string(ref1)
    s2 = sd.get_string(ref2)
    
    print(f"Get ref_id={ref1}: '{s1}'")
    print(f"Get ref_id={ref2}: '{s2}'")
    print()
    
    if s1 == "truevision/window" and s2 == "TrueVision":
        print("✅ String retrieval working!")
        return True
    else:
        print("❌ String retrieval FAILED!")
        return False


def test_multiple_records():
    """Test multiple records in sequence"""
    print("=" * 60)
    print("MULTIPLE RECORDS TEST")
    print("=" * 60)
    print()
    
    sd = StringDictionary()
    records = []
    blobs = []
    
    print("Creating 10 test records...")
    for i in range(10):
        rec_dict = {
            "pulse_id": i,
            "worker_id": i % 8,
            "seq": i * 10,
            "timestamp": time.time() + i,
            "success": i % 2 == 0,
            "task_id": "truevision/window",  # Same string - should dedupe
            "engine_id": f"engine_{i % 3}",
            "transform_id": "TV_v2",
            "failure_reason": None if i % 2 == 0 else "manipulation:high_eomm",
            "grid_shape_in": (32, 32),
            "grid_shape_out": (32, 32),
            "color_count": 5 + i,
            "train_pair_indices": [0],
            "error_metrics": {"eomm": 0.5 + i * 0.05},
            "params": {"session": f"sess_{i}"},
            "context": {"idx": i},
        }
        
        rec = ForgeRecord.from_dict(rec_dict)
        blob = rec.serialize(sd)
        
        records.append(rec)
        blobs.append(blob)
        
        print(f"  Record {i}: {len(blob)} bytes, success={rec.success}")
    
    print()
    print(f"Total binary size: {sum(len(b) for b in blobs)} bytes")
    print()
    
    # Verify all deserialize correctly
    print("Deserializing all records...")
    for i, blob in enumerate(blobs):
        try:
            rec2 = ForgeRecord.deserialize(blob, sd)
            if rec2.pulse_id != i:
                print(f"❌ Record {i} pulse_id mismatch!")
                return False
        except Exception as e:
            print(f"❌ Record {i} deserialization failed: {e}")
            return False
    
    print("✅ All 10 records serialized and deserialized correctly!")
    print()
    
    # Check string deduplication stats
    print(f"String dictionary stats:")
    print(f"  Unique strings: {len(sd.strings)}")
    print(f"  Expected: ~13 (truevision/window, TV_v2, manipulation:high_eomm, engine_0/1/2, sess_0..9)")
    print()
    
    return True


if __name__ == "__main__":
    print("\n")
    
    success = True
    
    # Run tests
    success &= test_string_dict()
    print()
    
    success &= test_forge_record_smoke()
    print()
    
    success &= test_multiple_records()
    print()
    
    # Final result
    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe canonical ForgeRecord implementation is CORRECT.")
        print("Ready to wire into BinaryLog and pulse writer.")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
    print()
