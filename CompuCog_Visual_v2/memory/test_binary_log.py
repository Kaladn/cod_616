"""
FORGE MEMORY SYSTEM - BINARY LOG TEST
Verify BinaryLog implementation
"""

import time
import tempfile
import shutil
from pathlib import Path

from forge_memory.core.binary_log import BinaryLog
from forge_memory.core.record import ForgeRecord
from forge_memory.core.string_dict import StringDictionary


def test_binary_log_basic():
    """Test basic append and read operations"""
    print("=" * 60)
    print("BINARY LOG - BASIC OPERATIONS TEST")
    print("=" * 60)
    print()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="forge_test_")
    print(f"Using temp directory: {temp_dir}")
    print()
    
    try:
        # Initialize
        print("Initializing BinaryLog...")
        sd = StringDictionary()
        blog = BinaryLog(temp_dir, sd, initial_size=10*1024*1024)  # 10 MB
        print(f"  Initial record count: {len(blog)}")
        print(f"  Current offset: {blog.get_current_offset()}")
        print()
        
        # Create test record
        print("Creating test record...")
        rec_dict = {
            "pulse_id": 1,
            "worker_id": 0,
            "seq": 0,
            "timestamp": time.time(),
            "success": True,
            "task_id": "truevision/window",
            "engine_id": "TrueVision",
            "transform_id": "TV_v2",
            "failure_reason": None,
            "grid_shape_in": (1080, 1920),  # Native resolution (H, W)
            "grid_shape_out": (1080, 1920),
            "color_count": 5,
            "train_pair_indices": [0],
            "error_metrics": {"eomm": 0.45},
            "params": {"session": "test_001"},
            "context": {"match_type": "baseline"},
        }
        
        rec = ForgeRecord.from_dict(rec_dict)
        print(f"  pulse_id: {rec.pulse_id}")
        print(f"  task_id: {rec.task_id}")
        print()
        
        # Append record
        print("Appending record...")
        offset = blog.append(rec)
        print(f"  Written at offset: {offset}")
        print(f"  Record count: {len(blog)}")
        print(f"  Current offset: {blog.get_current_offset()}")
        print()
        
        # Read back
        print("Reading record back...")
        rec2 = blog.read_at_offset(offset)
        print(f"  pulse_id: {rec2.pulse_id}")
        print(f"  task_id: {rec2.task_id}")
        print()
        
        # Compare
        if rec.to_dict() == rec2.to_dict():
            print("✅ Record matches!")
        else:
            print("❌ Record mismatch!")
            return False
        
        blog.close()
        return True
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory")
        print()


def test_binary_log_batch():
    """Test batch append operations"""
    print("=" * 60)
    print("BINARY LOG - BATCH APPEND TEST")
    print("=" * 60)
    print()
    
    temp_dir = tempfile.mkdtemp(prefix="forge_test_")
    print(f"Using temp directory: {temp_dir}")
    print()
    
    try:
        # Initialize
        sd = StringDictionary()
        blog = BinaryLog(temp_dir, sd)
        
        # Create 100 test records
        print("Creating 100 test records...")
        records = []
        for i in range(100):
            rec_dict = {
                "pulse_id": i,
                "worker_id": i % 8,
                "seq": i,
                "timestamp": time.time() + i,
                "success": i % 2 == 0,
                "task_id": "truevision/window",
                "engine_id": f"engine_{i % 3}",
                "transform_id": "TV_v2",
                "failure_reason": None if i % 2 == 0 else "manipulation",
                "grid_shape_in": (1080, 1920),  # Native resolution (H, W)
                "grid_shape_out": (1080, 1920),
                "color_count": 5 + i,
                "train_pair_indices": [0],
                "error_metrics": {"eomm": 0.5 + i * 0.001},
                "params": {"idx": i},
                "context": {"test": True},
            }
            records.append(ForgeRecord.from_dict(rec_dict))
        
        print(f"  Created {len(records)} records")
        print()
        
        # Batch append
        print("Batch appending 100 records...")
        start = time.time()
        offsets = blog.append_batch(records)
        duration = time.time() - start
        
        print(f"  Appended in {duration*1000:.2f} ms")
        print(f"  Record count: {len(blog)}")
        print(f"  Current offset: {blog.get_current_offset()}")
        print(f"  Throughput: {len(records)/duration:.1f} records/sec")
        print()
        
        # Verify all records
        print("Verifying all records...")
        for i, offset in enumerate(offsets):
            rec = blog.read_at_offset(offset)
            if rec.pulse_id != i:
                print(f"❌ Record {i} pulse_id mismatch!")
                return False
        
        print("✅ All records verified!")
        print()
        
        # Test read_all
        print("Testing read_all()...")
        all_records = blog.read_all()
        if len(all_records) == 100:
            print(f"✅ read_all() returned {len(all_records)} records")
        else:
            print(f"❌ read_all() returned {len(all_records)} (expected 100)")
            return False
        print()
        
        blog.close()
        return True
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory")
        print()


def test_binary_log_persistence():
    """Test persistence across close/reopen"""
    print("=" * 60)
    print("BINARY LOG - PERSISTENCE TEST")
    print("=" * 60)
    print()
    
    temp_dir = tempfile.mkdtemp(prefix="forge_test_")
    print(f"Using temp directory: {temp_dir}")
    print()
    
    try:
        # Phase 1: Write records
        print("Phase 1: Writing 50 records...")
        sd = StringDictionary(Path(temp_dir) / "string_dict.bin")
        blog = BinaryLog(temp_dir, sd)
        
        records = []
        for i in range(50):
            rec_dict = {
                "pulse_id": i,
                "worker_id": 0,
                "seq": i,
                "timestamp": time.time() + i,
                "success": True,
                "task_id": "test_task",
                "engine_id": "test_engine",
                "transform_id": "test_transform",
                "failure_reason": None,
                "grid_shape_in": (1080, 1920),  # Native resolution (H, W)
                "grid_shape_out": (1080, 1920),
                "color_count": 5,
                "train_pair_indices": [0],
                "error_metrics": {},
                "params": {},
                "context": {},
            }
            records.append(ForgeRecord.from_dict(rec_dict))
        
        blog.append_batch(records)
        print(f"  Written {len(blog)} records")
        print(f"  Current offset: {blog.get_current_offset()}")
        
        # Save string dict
        sd.save()
        blog.close()
        print("  Closed BinaryLog")
        print()
        
        # Phase 2: Reopen and verify
        print("Phase 2: Reopening and verifying...")
        sd2 = StringDictionary(Path(temp_dir) / "string_dict.bin")
        blog2 = BinaryLog(temp_dir, sd2)
        
        print(f"  Record count after reopen: {len(blog2)}")
        print(f"  Current offset: {blog2.get_current_offset()}")
        print()
        
        if len(blog2) != 50:
            print(f"❌ Expected 50 records, got {len(blog2)}")
            return False
        
        # Verify scan found all records
        print("Verifying all 50 records found by scan...")
        for i, offset in enumerate(blog2.get_offsets()):
            rec = blog2.read_at_offset(offset)
            if rec.pulse_id != i:
                print(f"❌ Record {i} pulse_id mismatch!")
                return False
        
        print("✅ All 50 records found and verified!")
        print()
        
        # Phase 3: Append more and verify
        print("Phase 3: Appending 25 more records...")
        more_records = []
        for i in range(50, 75):
            rec_dict = {
                "pulse_id": i,
                "worker_id": 0,
                "seq": i,
                "timestamp": time.time() + i,
                "success": True,
                "task_id": "test_task",
                "engine_id": "test_engine",
                "transform_id": "test_transform",
                "failure_reason": None,
                "grid_shape_in": (1080, 1920),  # Native resolution (H, W)
                "grid_shape_out": (1080, 1920),
                "color_count": 5,
                "train_pair_indices": [0],
                "error_metrics": {},
                "params": {},
                "context": {},
            }
            more_records.append(ForgeRecord.from_dict(rec_dict))
        
        blog2.append_batch(more_records)
        print(f"  Total records: {len(blog2)}")
        print()
        
        if len(blog2) == 75:
            print("✅ Persistence test passed!")
        else:
            print(f"❌ Expected 75 records, got {len(blog2)}")
            return False
        
        blog2.close()
        return True
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory")
        print()


if __name__ == "__main__":
    print("\n")
    
    success = True
    
    # Run tests
    success &= test_binary_log_basic()
    print()
    
    success &= test_binary_log_batch()
    print()
    
    success &= test_binary_log_persistence()
    print()
    
    # Final result
    print("=" * 60)
    if success:
        print("🎉 ALL BINARY LOG TESTS PASSED!")
        print("=" * 60)
        print("\nBinaryLog is production-ready.")
        print("ForgeRecord has a home.")
        print("TrueVision v2 can start pulsing windows into binary memory.")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
    print()
