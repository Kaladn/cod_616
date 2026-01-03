#!/usr/bin/env python3

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog TrueVision — Mathematical Analysis                             ║
║     "Show me the math behind the detection"                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TrueVision sees shapes and grids - this tool shows the RAW MATH:
- Grid cell values (0-255 grayscale per 80×80px region)
- Operator detection scores (how each algorithm computed its result)
- Weighted EOMM composition (crosshair×0.30 + hit_reg×0.30 + death×0.25 + edge×0.15)
- Session baseline statistics (Welford's running variance)
- Temporal patterns across frames

Usage:
  python show_math.py telemetry/truevision_live_YYYYMMDD_HHMMSS.jsonl
  python show_math.py telemetry/truevision_live_YYYYMMDD_HHMMSS.jsonl --window 42
  python show_math.py telemetry/truevision_live_YYYYMMDD_HHMMSS.jsonl --high-eomm 0.8
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_telemetry(path: str) -> List[Dict]:
    """Load JSONL telemetry file"""
    entries = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_grid_math(grid: List[List[int]], max_rows: int = 8):
    """Display grid cell values (NATIVE resolution, show sample)"""
    print(f"    Grid Cell Values ({len(grid)}×{len(grid[0]) if grid else 0} native resolution, sample):")
    print("    " + "─" * 60)
    
    rows_to_show = min(len(grid), max_rows)
    for i, row in enumerate(grid[:rows_to_show]):
        # Show first 8 cells of row
        cells = [f"{val:3d}" for val in row[:8]]
        print(f"    Row {i:2d}: [{', '.join(cells)}...]")
    
    if len(grid) > max_rows:
        print(f"    ... ({len(grid) - max_rows} more rows)")
    print()


def print_operator_math(operators: Dict):
    """Display operator detection scores with computation details"""
    print("    Operator Detection Scores:")
    print("    " + "─" * 60)
    
    # Primary operators (weighted in EOMM)
    primary = ['crosshair_lock', 'hit_registration', 'death_event', 'edge_entry']
    weights = {'crosshair_lock': 0.30, 'hit_registration': 0.30, 'death_event': 0.25, 'edge_entry': 0.15}
    
    print("    PRIMARY (EOMM weighted):")
    for op in primary:
        if op in operators:
            score = operators[op]
            weight = weights.get(op, 0)
            contribution = score * weight
            print(f"      {op:20s}: {score:.3f}  (weight: {weight:.2f} → contribution: {contribution:.3f})")
    
    print()
    print("    AUXILIARY (detection only):")
    for op, score in operators.items():
        if op not in primary:
            print(f"      {op:20s}: {score:.3f}")
    print()


def print_eomm_math(eomm_score: float, operators: Dict):
    """Show EOMM composite calculation breakdown"""
    print("    EOMM Composite Calculation:")
    print("    " + "─" * 60)
    
    weights = {
        'crosshair_lock': 0.30,
        'hit_registration': 0.30,
        'death_event': 0.25,
        'edge_entry': 0.15
    }
    
    total = 0.0
    for op, weight in weights.items():
        score = operators.get(op, 0.0)
        contribution = score * weight
        total += contribution
        print(f"      {score:.3f} × {weight:.2f}  ({op:20s}) = {contribution:.3f}")
    
    print(f"      {'─' * 50}")
    print(f"      Final EOMM Score: {total:.3f}")
    
    if abs(total - eomm_score) > 0.001:
        print(f"      (Recorded: {eomm_score:.3f} - difference may be due to rounding)")
    print()


def print_baseline_math(baselines: Dict):
    """Display session baseline statistics (Welford's algorithm)"""
    print("    Session Baseline Statistics (Welford's running variance):")
    print("    " + "─" * 60)
    
    metrics = ['ttk', 'ttd', 'stk']
    for metric in metrics:
        data = baselines.get(metric, {})
        if data:
            mean = data.get('mean', 0)
            variance = data.get('variance', 0)
            std_dev = variance ** 0.5
            count = data.get('count', 0)
            
            print(f"      {metric.upper()}:")
            print(f"        Count:    {count}")
            print(f"        Mean:     {mean:.3f}")
            print(f"        Variance: {variance:.3f}")
            print(f"        Std Dev:  {std_dev:.3f}")
    print()


def print_flags(flags: List[str]):
    """Display active detection flags"""
    if flags:
        print("    Active Detection Flags:")
        print("    " + "─" * 60)
        for flag in flags:
            print(f"      • {flag}")
        print()


def analyze_window(entry: Dict, window_num: int):
    """Full mathematical breakdown of single detection window"""
    print("\n" + "═" * 80)
    print(f"  WINDOW {window_num} — Mathematical Analysis")
    print("═" * 80)
    
    print(f"\n  Timestamp: {entry.get('timestamp', 'N/A')}")
    print(f"  Frame Count: {entry.get('frame_count', 0)} frames buffered")
    print()
    
    # Grid math
    grid = entry.get('arc_grid', [])
    if grid:
        print_grid_math(grid, max_rows=6)
    
    # Operator math
    operators = entry.get('operators', {})
    if operators:
        print_operator_math(operators)
    
    # EOMM math
    eomm_score = entry.get('eomm_composite', 0.0)
    print_eomm_math(eomm_score, operators)
    
    # Baseline math
    baselines = entry.get('session_baselines', {})
    if baselines:
        print_baseline_math(baselines)
    
    # Flags
    flags = entry.get('flags', [])
    if flags:
        print_flags(flags)


def analyze_session_overview(entries: List[Dict]):
    """Statistical overview of entire session"""
    print("\n" + "═" * 80)
    print("  SESSION OVERVIEW — Aggregate Statistics")
    print("═" * 80)
    
    total_windows = len(entries)
    eomm_scores = [e.get('eomm_composite', 0.0) for e in entries]
    
    # EOMM distribution
    mean_eomm = sum(eomm_scores) / len(eomm_scores) if eomm_scores else 0
    min_eomm = min(eomm_scores) if eomm_scores else 0
    max_eomm = max(eomm_scores) if eomm_scores else 0
    
    high_eomm = [s for s in eomm_scores if s >= 0.8]
    med_eomm = [s for s in eomm_scores if 0.4 <= s < 0.8]
    low_eomm = [s for s in eomm_scores if s < 0.4]
    
    print(f"\n  Total Windows: {total_windows}")
    print(f"\n  EOMM Score Distribution:")
    print(f"    Mean:   {mean_eomm:.3f}")
    print(f"    Min:    {min_eomm:.3f}")
    print(f"    Max:    {max_eomm:.3f}")
    print(f"\n    High (≥0.8):    {len(high_eomm):3d} windows ({100*len(high_eomm)/total_windows:.1f}%)")
    print(f"    Medium (0.4-0.8): {len(med_eomm):3d} windows ({100*len(med_eomm)/total_windows:.1f}%)")
    print(f"    Low (<0.4):      {len(low_eomm):3d} windows ({100*len(low_eomm)/total_windows:.1f}%)")
    
    # Operator averages
    print(f"\n  Average Operator Scores:")
    operator_sums = {}
    operator_counts = {}
    
    for entry in entries:
        for op, score in entry.get('operators', {}).items():
            operator_sums[op] = operator_sums.get(op, 0.0) + score
            operator_counts[op] = operator_counts.get(op, 0) + 1
    
    for op in sorted(operator_sums.keys()):
        avg = operator_sums[op] / operator_counts[op]
        print(f"    {op:20s}: {avg:.3f}")
    
    # Flag frequency
    flag_counts = {}
    for entry in entries:
        for flag in entry.get('flags', []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    
    if flag_counts:
        print(f"\n  Flag Frequency:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_windows
            print(f"    {flag:20s}: {count:3d} windows ({pct:.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="TrueVision Mathematical Analysis - Show the raw computational data"
    )
    parser.add_argument(
        "telemetry_file",
        help="Path to TrueVision JSONL telemetry file"
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Analyze specific window number (1-indexed)"
    )
    parser.add_argument(
        "--high-eomm",
        type=float,
        help="Show only windows with EOMM >= threshold (e.g., 0.8)"
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Show only session overview, skip individual windows"
    )
    
    args = parser.parse_args()
    
    # Load telemetry
    if not Path(args.telemetry_file).exists():
        print(f"Error: File not found: {args.telemetry_file}")
        sys.exit(1)
    
    entries = load_telemetry(args.telemetry_file)
    print(f"\nLoaded {len(entries)} detection windows from {args.telemetry_file}")
    
    # Session overview
    if not args.window:
        analyze_session_overview(entries)
    
    if args.overview_only:
        return
    
    # Specific window
    if args.window:
        if 1 <= args.window <= len(entries):
            analyze_window(entries[args.window - 1], args.window)
        else:
            print(f"Error: Window {args.window} out of range (1-{len(entries)})")
            sys.exit(1)
    
    # High EOMM windows
    elif args.high_eomm:
        high_windows = [
            (i+1, e) for i, e in enumerate(entries)
            if e.get('eomm_composite', 0.0) >= args.high_eomm
        ]
        
        print(f"\n  Found {len(high_windows)} windows with EOMM >= {args.high_eomm}")
        
        for window_num, entry in high_windows[:10]:  # Show first 10
            analyze_window(entry, window_num)
        
        if len(high_windows) > 10:
            print(f"\n  (Showing first 10 of {len(high_windows)} high EOMM windows)")
            print(f"  Use --window <num> to see specific window details")


if __name__ == "__main__":
    main()
