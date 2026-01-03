"""
Timestamp alignment analysis for fusion blocks.
Diagnoses why modality alignment is low.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_ts(ts_str):
    """Parse various timestamp formats to datetime."""
    if not ts_str:
        return None
    
    # Try epoch
    if isinstance(ts_str, (int, float)):
        return datetime.fromtimestamp(ts_str)
    
    # Try ISO format
    try:
        if 'T' in str(ts_str):
            ts_str = str(ts_str).replace('Z', '+00:00')
            if '.' in ts_str:
                # Remove extra precision
                parts = ts_str.split('.')
                if len(parts) == 2:
                    frac_tz = parts[1]
                    if '+' in frac_tz:
                        frac, tz = frac_tz.split('+')
                        ts_str = f"{parts[0]}.{frac[:6]}+{tz}"
                    elif '-' in frac_tz and frac_tz.count('-') == 1:
                        frac, tz = frac_tz.rsplit('-', 1)
                        ts_str = f"{parts[0]}.{frac[:6]}-{tz}"
                    else:
                        ts_str = f"{parts[0]}.{frac_tz[:6]}"
            return datetime.fromisoformat(ts_str)
    except:
        pass
    
    return None


def analyze_log_time_ranges(base_dir: Path, logs_dir: Path):
    """Analyze time ranges of each modality."""
    ranges = {}
    
    # TrueVision
    tv_files = list(base_dir.glob("truevision_live_*.jsonl"))
    if tv_files:
        tv_times = []
        for f in tv_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        ts = data.get("window_start_epoch")
                        if ts:
                            tv_times.append(datetime.fromtimestamp(ts))
        if tv_times:
            ranges["truevision"] = (min(tv_times), max(tv_times))
    
    # Activity
    act_files = list(logs_dir.glob("activity/*.jsonl")) + list(logs_dir.glob("activity/user_activity_*.jsonl"))
    if act_files:
        act_times = []
        for f in act_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        ts = parse_ts(data.get("timestamp"))
                        if ts:
                            act_times.append(ts)
        if act_times:
            ranges["activity"] = (min(act_times), max(act_times))
    
    # Gamepad
    gp_files = list(logs_dir.glob("gamepad/*.jsonl")) + list(logs_dir.glob("gamepad/gamepad_stream_*.jsonl"))
    if gp_files:
        gp_times = []
        for f in gp_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            ts = parse_ts(data.get("timestamp"))
                            if ts:
                                gp_times.append(ts)
                        except:
                            pass
        if gp_times:
            ranges["gamepad"] = (min(gp_times), max(gp_times))
    
    # Input
    inp_files = list(logs_dir.glob("input/*.jsonl")) + list(logs_dir.glob("input/telemetry_*.jsonl"))
    if inp_files:
        inp_times = []
        for f in inp_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        ts = parse_ts(data.get("timestamp"))
                        if ts:
                            inp_times.append(ts)
        if inp_times:
            ranges["input"] = (min(inp_times), max(inp_times))
    
    # Network
    net_files = list(logs_dir.glob("network/*.jsonl")) + list(logs_dir.glob("network/network_state_*.jsonl"))
    if net_files:
        net_times = []
        for f in net_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        ts = parse_ts(data.get("Timestamp"))
                        if ts:
                            net_times.append(ts)
        if net_times:
            ranges["network"] = (min(net_times), max(net_times))
    
    # Process
    proc_files = list(logs_dir.glob("process/*.jsonl")) + list(logs_dir.glob("process/process_events_*.jsonl"))
    if proc_files:
        proc_times = []
        for f in proc_files:
            with open(f, encoding='utf-8-sig', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        ts = parse_ts(data.get("timestamp"))
                        if ts:
                            proc_times.append(ts)
        if proc_times:
            ranges["process"] = (min(proc_times), max(proc_times))
    
    return ranges


def main():
    session_dir = Path(r"D:\cod_616\CompuCog_Visual_v2\gaming\telemetry")
    logs_dir = Path(r"D:\cod_616\CompuCog_Visual_v2\logs")
    
    print("="*70)
    print("MODALITY TIMESTAMP ALIGNMENT ANALYSIS")
    print("="*70)
    
    ranges = analyze_log_time_ranges(session_dir, logs_dir)
    
    print("\n[TIME RANGES BY MODALITY]\n")
    
    for mod, (start, end) in sorted(ranges.items()):
        duration = end - start
        print(f"{mod:12}: {start.strftime('%Y-%m-%d %H:%M:%S')} -> {end.strftime('%H:%M:%S')}")
        print(f"             Duration: {duration.total_seconds()/60:.1f} min")
    
    # Find overlap
    if len(ranges) >= 2:
        # Normalize to naive datetimes for comparison
        all_starts = []
        all_ends = []
        for r in ranges.values():
            start, end = r
            # Remove timezone info if present
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if end.tzinfo is not None:
                end = end.replace(tzinfo=None)
            all_starts.append(start)
            all_ends.append(end)
        
        overlap_start = max(all_starts)
        overlap_end = min(all_ends)
        
        print("\n" + "="*70)
        print("OVERLAP ANALYSIS")
        print("="*70)
        
        if overlap_start < overlap_end:
            overlap = (overlap_end - overlap_start).total_seconds()
            print(f"\n✅ OVERLAP FOUND: {overlap_start.strftime('%H:%M:%S')} -> {overlap_end.strftime('%H:%M:%S')}")
            print(f"   Duration: {overlap/60:.1f} min ({overlap:.0f} seconds)")
        else:
            print("\n❌ NO OVERLAP - Logs are from different time periods!")
            print(f"   Latest start: {overlap_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Earliest end: {overlap_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Gap: {(overlap_start - overlap_end).total_seconds()/60:.1f} min")
    
    # TrueVision vs each modality
    if "truevision" in ranges:
        tv_start, tv_end = ranges["truevision"]
        # Normalize TrueVision timestamps
        if tv_start.tzinfo is not None:
            tv_start = tv_start.replace(tzinfo=None)
        if tv_end.tzinfo is not None:
            tv_end = tv_end.replace(tzinfo=None)
        
        print("\n[TRUEVISION ALIGNMENT WITH OTHER MODALITIES]\n")
        for mod, (start, end) in ranges.items():
            if mod == "truevision":
                continue
            
            # Normalize timestamps
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if end.tzinfo is not None:
                end = end.replace(tzinfo=None)
            
            # Check overlap with TrueVision
            ov_start = max(tv_start, start)
            ov_end = min(tv_end, end)
            
            if ov_start < ov_end:
                ov_sec = (ov_end - ov_start).total_seconds()
                print(f"  {mod:12}: ✅ {ov_sec/60:.1f} min overlap")
            else:
                gap = (ov_start - ov_end).total_seconds()
                print(f"  {mod:12}: ❌ No overlap (gap: {gap/60:.1f} min)")


if __name__ == "__main__":
    main()
