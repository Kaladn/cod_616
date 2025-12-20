"""
Recognition Field v2 Demo — Temporal pattern detection
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from forge_memory.core.string_dict import StringDictionary
from forge_memory.core.binary_log import BinaryLog
from forge_memory.query.query_layer import QueryLayer
from forge_memory.recognition.recognition_field import RecognitionField


def demo_recognition(data_dir: str):
    """Demonstrate RecognitionField v2 temporal pattern detection"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Forge data directory not found: {data_dir}")
        return
    
    print(f"🧠 Recognition Field v2 Demo: {data_dir}")
    print("=" * 80)
    
    # Load components
    print(f"[1/4] Loading string dictionary...")
    strings_path = data_path / "strings.dict"
    string_dict = StringDictionary(strings_path)
    
    print(f"[2/4] Loading binary log...")
    binary_log = BinaryLog(
        data_dir=str(data_path),
        string_dict=string_dict,
        filename="records.bin"
    )
    
    print(f"[3/4] Initializing query layer...")
    q = QueryLayer(binary_log)
    
    print(f"[4/4] Initializing recognition field...")
    rf = RecognitionField(q)
    
    total_records = len(binary_log)
    print(f"      {total_records} ForgeRecords in memory")
    print()
    
    # Demo 1: Windows with INSTA_MELT flag
    print("=" * 80)
    print("💀 Demo 1: Windows with INSTA_MELT flag")
    print("=" * 80)
    insta_melt = rf.windows_with_flag("INSTA_MELT", limit=5)
    print(f"   Found {len(insta_melt)} INSTA_MELT windows (showing first 5)")
    for i, r in enumerate(insta_melt, 1):
        print(f"   [{i}] Pulse {r.pulse_id} | EOMM={r.error_metrics['eomm_score']:.2f} | ts={r.timestamp:.1f}")
    print()
    
    # Demo 2: Manipulation last N minutes (simulate "last 10 minutes")
    print("=" * 80)
    print("⚠️  Demo 2: Manipulation Events (simulated last 10 minutes)")
    print("=" * 80)
    # Note: With current data (2 sessions from same day), all records are "recent"
    recent_manip = rf.manipulation_last_minutes(10, threshold=0.5)
    print(f"   Found {len(recent_manip)} manipulation events in last 10 minutes")
    print(f"   First 3:")
    for i, r in enumerate(recent_manip[:3], 1):
        flags_str = ", ".join(r.context.get("flags", {}).get("operator_flags", []))
        print(f"      [{i}] EOMM={r.error_metrics['eomm_score']:.2f} | Flags=[{flags_str}]")
    print()
    
    # Demo 3: Temporal burst detection
    print("=" * 80)
    print("🔥 Demo 3: Temporal Burst Detection")
    print("=" * 80)
    bursts = rf.find_bursts(threshold=0.75, min_len=2, max_gap_sec=5.0)
    print(f"   Found {len(bursts)} temporal bursts (EOMM > 0.75, min_len=2, max_gap=5s)")
    for i, burst in enumerate(bursts, 1):
        print(f"\n   Burst {i}:")
        print(f"      Length: {burst.length} windows")
        print(f"      Duration: {burst.duration:.1f}s")
        print(f"      Peak EOMM: {burst.peak_eomm:.2f}")
        print(f"      Operators: {', '.join(sorted(burst.operators))}")
        print(f"      Time range: {burst.start_ts:.1f} → {burst.end_ts:.1f}")
    print()
    
    # Demo 4: Session summaries
    print("=" * 80)
    print("📊 Demo 4: Session Summaries")
    print("=" * 80)
    summaries = rf.summarize_all_sessions()
    print(f"   Found {len(summaries)} sessions:")
    for i, summary in enumerate(summaries, 1):
        print(f"\n   Session {i}: {summary.session_id}")
        print(f"      Windows: {summary.window_count}")
        print(f"      Manipulation rate: {summary.manipulation_rate:.1f}% ({summary.anomaly_count}/{summary.window_count})")
        print(f"      High-confidence rate: {summary.high_conf_rate:.1f}% ({summary.high_conf_count}/{summary.window_count})")
        print(f"      Avg EOMM: {summary.avg_eomm:.2f}")
        print(f"      Max EOMM: {summary.max_eomm:.2f}")
        print(f"      Operator histogram:")
        for op, count in sorted(summary.operator_histogram.items(), key=lambda x: x[1], reverse=True):
            print(f"         {op}: {count}")
    print()
    
    # Demo 5: High-confidence anomaly stream
    print("=" * 80)
    print("⚡ Demo 5: High-Confidence Anomaly Stream (EOMM > 0.85)")
    print("=" * 80)
    high_conf = list(rf.high_confidence_stream(0.85))
    print(f"   Found {len(high_conf)} high-confidence anomalies")
    print(f"   First 5:")
    for i, r in enumerate(high_conf[:5], 1):
        flags_str = ", ".join(r.context.get("flags", {}).get("operator_flags", []))
        print(f"      [{i}] EOMM={r.error_metrics['eomm_score']:.2f} | Flags=[{flags_str}] | Pulse={r.pulse_id}")
    print()
    
    # Demo 6: Operator frequency across all memory
    print("=" * 80)
    print("🎯 Demo 6: Operator Frequency (Global)")
    print("=" * 80)
    op_freq = rf.operator_frequency()
    print(f"   Operator usage across all {total_records} windows:")
    for op, count in sorted(op_freq.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_records) * 100
        print(f"      {op}: {count} ({pct:.1f}%)")
    print()
    
    # Demo 7: EOMM time series extraction
    print("=" * 80)
    print("📈 Demo 7: EOMM Time Series")
    print("=" * 80)
    time_series = rf.eomm_time_series()
    print(f"   {len(time_series)} data points ready for plotting")
    print(f"   First 5 points:")
    for ts, eomm in time_series[:5]:
        print(f"      ts={ts:.1f} → EOMM={eomm:.2f}")
    print(f"   Last 5 points:")
    for ts, eomm in time_series[-5:]:
        print(f"      ts={ts:.1f} → EOMM={eomm:.2f}")
    print()
    
    print("=" * 80)
    print("✅ Recognition Field v2 Demo Complete")
    print("=" * 80)
    print(f"\nThe brain now has temporal awareness:")
    print(f"  • Flag-based retrieval (INSTA_MELT, HITBOX_DRIFT, etc.)")
    print(f"  • Time-based queries (last N minutes)")
    print(f"  • Burst detection (clustered manipulation events)")
    print(f"  • Session summaries (per-session fingerprints)")
    print(f"  • Global operator frequency (what's happening most)")
    print(f"  • Time series extraction (ready for visualization)")
    print(f"\nThis is cognitive memory with pattern recognition.")
    
    binary_log.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognition Field v2 demo")
    parser.add_argument("--data-dir", "-d", 
                       default=str(Path(__file__).parent / "forge_data"),
                       help="Forge data directory")
    
    args = parser.parse_args()
    
    try:
        demo_recognition(args.data_dir)
    except Exception as e:
        print(f"\n❌ Recognition demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
