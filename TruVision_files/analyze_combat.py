#!/usr/bin/env python3
"""
Analyze combat session EOMM scores and detect manipulation patterns.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "memory"))

from memory.binary_log import BinaryLog
from memory.string_dict import StringDictionary
from memory.forge_schema import TrueVisionSchemaMap

def analyze_combat_session(data_dir: str = "forge_data"):
    """Analyze EOMM scores and manipulation patterns from combat session."""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Load Forge Memory
    print("🔍 Loading Forge Memory...")
    string_dict = StringDictionary(data_path / "strings.dict")
    binary_log = BinaryLog(data_path / "records.bin", string_dict)
    schema = TrueVisionSchemaMap(string_dict)
    
    records = binary_log.read_all()
    print(f"📊 Loaded {len(records)} records\n")
    
    if not records:
        print("❌ No records found!")
        return
    
    # Group by session
    sessions = defaultdict(list)
    for rec in records:
        window = schema.forge_to_window(rec)
        session_id = window.get("session_context", {}).get("session_id", "unknown")
        sessions[session_id].append(window)
    
    print(f"🎮 Found {len(sessions)} session(s):\n")
    
    for session_id, windows in sessions.items():
        print(f"{'='*70}")
        print(f"📍 SESSION: {session_id}")
        print(f"{'='*70}")
        
        # EOMM score analysis
        eomm_scores = [w["eomm_score"] for w in windows]
        eomm_min = min(eomm_scores)
        eomm_max = max(eomm_scores)
        eomm_avg = sum(eomm_scores) / len(eomm_scores)
        
        # Count manipulation windows (EOMM >= 0.7)
        high_eomm = [s for s in eomm_scores if s >= 0.7]
        medium_eomm = [s for s in eomm_scores if 0.4 <= s < 0.7]
        low_eomm = [s for s in eomm_scores if s < 0.4]
        
        print(f"\n📈 EOMM SCORE DISTRIBUTION:")
        print(f"   Total Windows: {len(windows)}")
        print(f"   Average EOMM: {eomm_avg:.3f}")
        print(f"   Min EOMM: {eomm_min:.3f}")
        print(f"   Max EOMM: {eomm_max:.3f}")
        print(f"\n   🔴 HIGH (>= 0.7):   {len(high_eomm):4d} ({100*len(high_eomm)/len(windows):5.1f}%) - MANIPULATION DETECTED")
        print(f"   🟡 MEDIUM (0.4-0.7): {len(medium_eomm):4d} ({100*len(medium_eomm)/len(windows):5.1f}%) - SUSPICIOUS")
        print(f"   🟢 LOW (< 0.4):      {len(low_eomm):4d} ({100*len(low_eomm)/len(windows):5.1f}%) - NORMAL")
        
        # Operator flag analysis
        print(f"\n🎯 OPERATOR TRIGGERS:")
        all_flags = []
        for w in windows:
            flags = w.get("operator_flags", [])
            if flags:
                all_flags.extend(flags)
        
        if all_flags:
            flag_counts = Counter(all_flags)
            for flag, count in flag_counts.most_common():
                pct = 100 * count / len(windows)
                print(f"   {flag:25s} {count:4d} windows ({pct:5.1f}%)")
        else:
            print(f"   No operator flags triggered")
        
        # Show top 10 highest EOMM windows
        print(f"\n🚨 TOP 10 HIGHEST EOMM WINDOWS:")
        sorted_windows = sorted(windows, key=lambda w: w["eomm_score"], reverse=True)[:10]
        
        for i, w in enumerate(sorted_windows, 1):
            ts = w["ts_start"]
            eomm = w["eomm_score"]
            flags = w.get("operator_flags", [])
            scores = w.get("operator_scores", {})
            
            print(f"\n   [{i:2d}] Timestamp: {ts:.2f}  |  EOMM: {eomm:.3f}")
            
            if flags:
                print(f"       Flags: {', '.join(flags)}")
            
            if scores:
                print(f"       Operator Scores:")
                for op_name, op_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"         {op_name:20s} {op_score:.3f}")
        
        # Time span
        ts_start = min(w["ts_start"] for w in windows)
        ts_end = max(w["ts_end"] for w in windows)
        duration = ts_end - ts_start
        
        print(f"\n⏱️  SESSION DURATION: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Capture Rate: {len(windows)/duration:.1f} windows/sec")
        print()

if __name__ == "__main__":
    analyze_combat_session()
