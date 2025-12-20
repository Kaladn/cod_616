"""
Forge Memory Inspector - Read and display ForgeRecords from binary memory
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from forge_memory.core.string_dict import StringDictionary
from forge_memory.core.binary_log import BinaryLog


def inspect_forge_memory(data_dir: str, limit: int = 10):
    """Read and display ForgeRecords from Forge memory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Forge data directory not found: {data_dir}")
        return
    
    print(f"🧠 Inspecting Forge Memory: {data_dir}")
    print("=" * 80)
    
    # Load string dictionary
    strings_path = data_path / "strings.dict"
    if not strings_path.exists():
        print("❌ No strings.dict found")
        return
    
    print(f"[1/2] Loading string dictionary...")
    string_dict = StringDictionary(strings_path)
    print(f"      String dictionary loaded")
    
    # Load binary log
    print(f"[2/2] Loading binary log...")
    binary_log = BinaryLog(
        data_dir=str(data_path),
        string_dict=string_dict,
        filename="records.bin"
    )
    
    total_records = len(binary_log)
    print(f"      {total_records} ForgeRecords in memory")
    print()
    
    if total_records == 0:
        print("No records found.")
        return
    
    # Read and display records
    print(f"📋 Displaying first {min(limit, total_records)} records:")
    print("=" * 80)
    
    records = binary_log.read_all()
    
    for i, record in enumerate(records[:limit]):
        print(f"\n🔹 Record {i+1}/{total_records}")
        print(f"   Pulse ID: {record.pulse_id}")
        print(f"   Worker ID: {record.worker_id}")
        print(f"   Sequence: {record.seq}")
        print(f"   Timestamp: {record.timestamp}")
        print(f"   Success: {record.success}")
        print(f"   Task: {record.task_id}")
        print(f"   Engine: {record.engine_id}")
        print(f"   Transform: {record.transform_id}")
        if record.failure_reason:
            print(f"   Failure: {record.failure_reason}")
        print(f"   Grid In: {record.grid_shape_in}")
        print(f"   Grid Out: {record.grid_shape_out}")
        print(f"   Color Count: {record.color_count}")
        
        # Show error metrics
        if record.error_metrics:
            eomm = record.error_metrics.get("eomm_score")
            flags = record.error_metrics.get("flags", [])
            print(f"   EOMM Score: {eomm:.2f}")
            if flags:
                print(f"   Flags: {', '.join(flags)}")
        
        # Show session context
        if record.context and "session" in record.context:
            session = record.context["session"]
            print(f"   Session: {session.get('session_id', 'N/A')}")
            print(f"   Source: {session.get('source', 'N/A')}")
    
    print("\n" + "=" * 80)
    print(f"✅ Inspected {min(limit, total_records)}/{total_records} records")
    
    # Calculate stats
    if records:
        manipulation_count = sum(1 for r in records if not r.success)
        print(f"\n📊 Statistics:")
        print(f"   Total records: {total_records}")
        print(f"   Manipulation detected: {manipulation_count} ({manipulation_count/total_records*100:.1f}%)")
        print(f"   Normal windows: {total_records - manipulation_count} ({(total_records-manipulation_count)/total_records*100:.1f}%)")
        
        # Unique flags
        all_flags = set()
        for r in records:
            if r.error_metrics and "flags" in r.error_metrics:
                all_flags.update(r.error_metrics["flags"])
        if all_flags:
            print(f"   Unique flags: {', '.join(sorted(all_flags))}")
    
    binary_log.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect Forge memory files")
    parser.add_argument("--data-dir", "-d", 
                       default=str(Path(__file__).parent / "forge_data"),
                       help="Forge data directory")
    parser.add_argument("--limit", "-l", type=int, default=10,
                       help="Number of records to display (default: 10)")
    
    args = parser.parse_args()
    
    try:
        inspect_forge_memory(args.data_dir, args.limit)
    except Exception as e:
        print(f"\n❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
