"""
Analyze fusion blocks that fall within the modality overlap window.
"""

import json
from pathlib import Path
from datetime import datetime


def main():
    blocks_file = Path(r"D:\cod_616\CompuCog_Visual_v2\gaming\telemetry\fusion_blocks.jsonl")
    
    # Overlap window: Dec 2, 22:44:14 -> 23:22:10
    overlap_start = datetime(2025, 12, 2, 22, 44, 14)
    overlap_end = datetime(2025, 12, 2, 23, 22, 10)
    
    print(f"Looking for blocks in overlap window: {overlap_start} -> {overlap_end}")
    print("="*70)
    
    in_window = []
    outside = 0
    
    with open(blocks_file, encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            block = json.loads(line)
            ts_str = block.get("anchor_timestamp", "")
            
            # Parse timestamp
            try:
                if isinstance(ts_str, (int, float)):
                    ts = datetime.fromtimestamp(ts_str)
                else:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    ts = ts.replace(tzinfo=None)
            except:
                continue
            
            if overlap_start <= ts <= overlap_end:
                in_window.append(block)
            else:
                outside += 1
    
    print(f"\nBlocks in overlap window: {len(in_window)}")
    print(f"Blocks outside overlap:   {outside}")
    
    if not in_window:
        print("\nNo blocks found in overlap window. Showing time distribution:")
        
        # Analyze what times we have
        times = []
        with open(blocks_file, encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():
                    continue
                block = json.loads(line)
                ts_str = block.get("anchor_timestamp", "")
                try:
                    if isinstance(ts_str, (int, float)):
                        ts = datetime.fromtimestamp(ts_str)
                    else:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        ts = ts.replace(tzinfo=None)
                    times.append(ts)
                except:
                    pass
        
        if times:
            print(f"\n  Earliest block: {min(times)}")
            print(f"  Latest block:   {max(times)}")
            print(f"  Total blocks:   {len(times)}")
            
            # Count by hour
            by_hour = {}
            for t in times:
                hour = t.hour
                by_hour[hour] = by_hour.get(hour, 0) + 1
            
            print("\n  Blocks by hour:")
            for hour in sorted(by_hour.keys()):
                print(f"    {hour:02d}:00 - {by_hour[hour]} blocks")
        return
    
    # Analyze blocks in window
    print("\n[BLOCKS IN FULL ALIGNMENT WINDOW]")
    
    event_types = {}
    eomm_scores = []
    modality_counts = {"truevision": 0, "gamepad": 0, "network": 0, "input": 0, "process": 0, "activity": 0}
    
    for block in in_window:
        event = block.get("anchor_event_type", "unknown")
        event_types[event] = event_types.get(event, 0) + 1
        
        anchor = block.get("anchor_frame", {})
        if anchor:
            tv = anchor.get("truevision", {})
            if tv:
                eomm_scores.append(tv.get("eomm_composite_score", 0))
                modality_counts["truevision"] += 1
            if anchor.get("gamepad"):
                modality_counts["gamepad"] += 1
            if anchor.get("network"):
                modality_counts["network"] += 1
            if anchor.get("input_event"):
                modality_counts["input"] += 1
            if anchor.get("process"):
                modality_counts["process"] += 1
            if anchor.get("activity"):
                modality_counts["activity"] += 1
    
    print(f"\n  Event distribution:")
    for e, c in sorted(event_types.items()):
        print(f"    {e}: {c}")
    
    print(f"\n  EOMM Score Stats:")
    if eomm_scores:
        print(f"    Mean: {sum(eomm_scores)/len(eomm_scores):.2f}")
        print(f"    Max:  {max(eomm_scores):.2f}")
        print(f"    Min:  {min(eomm_scores):.2f}")
    
    print(f"\n  Modality presence in anchor frames:")
    for mod, c in modality_counts.items():
        pct = (c / len(in_window) * 100) if in_window else 0
        print(f"    {mod}: {c}/{len(in_window)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
