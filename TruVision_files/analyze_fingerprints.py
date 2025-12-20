"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog — Sovereign Cognitive Defense System                           ║
║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
║                                                                              ║
║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
║                                                                              ║
║     "We use unconventional digital wisdom —                                  ║
║        because conventional digital wisdom doesn't protect anyone."         ║
║                                                                              ║
║     This software is proprietary and confidential.                           ║
║     Unauthorized access, copying, modification, or distribution             ║
║     is strictly prohibited and may violate applicable laws.                  ║
║                                                                              ║
║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

"""
Gaming Fingerprint Analyzer

Analyzes JSONL logs from gaming_sensor.py
Provides statistical summaries and pattern detection
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


def load_fingerprints(log_path):
    """Load all fingerprints from JSONL file"""
    fingerprints = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                fp = json.loads(line.strip())
                fingerprints.append(fp)
            except json.JSONDecodeError:
                continue
    return fingerprints


def analyze_session(fingerprints):
    """Compute session-level statistics"""
    if not fingerprints:
        return None
    
    # Overall anomaly scores
    overall_scores = [fp['features']['overall_anomaly_score'] for fp in fingerprints]
    
    # Individual feature scores
    flicker_scores = [fp['features']['flicker_score'] for fp in fingerprints]
    hud_scores = [fp['features']['hud_instability_score'] for fp in fingerprints]
    crosshair_scores = [fp['features']['crosshair_anomaly_score'] for fp in fingerprints]
    peripheral_scores = [fp['features']['peripheral_flash_score'] for fp in fingerprints]
    color_scores = [fp['features']['color_shift_score'] for fp in fingerprints]
    
    # Operator trigger counts
    all_operators = []
    for fp in fingerprints:
        for detection in fp['detections']:
            all_operators.append(detection['operator'])
    
    operator_counts = Counter(all_operators)
    
    # Temporal analysis
    duration = fingerprints[-1]['t_end'] - fingerprints[0]['t_start']
    
    return {
        'total_fingerprints': len(fingerprints),
        'duration_sec': duration,
        'overall': {
            'mean': sum(overall_scores) / len(overall_scores),
            'max': max(overall_scores),
            'min': min(overall_scores),
            'high_anomaly_count': sum(1 for s in overall_scores if s > 0.9)
        },
        'features': {
            'flicker': {
                'mean': sum(flicker_scores) / len(flicker_scores),
                'max': max(flicker_scores),
                'trigger_count': sum(1 for s in flicker_scores if s > 0)
            },
            'hud_instability': {
                'mean': sum(hud_scores) / len(hud_scores),
                'max': max(hud_scores),
                'trigger_count': sum(1 for s in hud_scores if s > 0)
            },
            'crosshair': {
                'mean': sum(crosshair_scores) / len(crosshair_scores),
                'max': max(crosshair_scores),
                'trigger_count': sum(1 for s in crosshair_scores if s > 0)
            },
            'peripheral_flash': {
                'mean': sum(peripheral_scores) / len(peripheral_scores),
                'max': max(peripheral_scores),
                'trigger_count': sum(1 for s in peripheral_scores if s > 0)
            },
            'color_shift': {
                'mean': sum(color_scores) / len(color_scores),
                'max': max(color_scores),
                'trigger_count': sum(1 for s in color_scores if s > 0)
            }
        },
        'operator_counts': operator_counts
    }


def print_summary(stats):
    """Pretty print analysis summary"""
    print("\n" + "="*60)
    print("GAMING FINGERPRINT ANALYSIS")
    print("="*60)
    
    print(f"\nSession Overview:")
    print(f"  Total Fingerprints: {stats['total_fingerprints']}")
    print(f"  Duration: {stats['duration_sec']:.1f}s ({stats['duration_sec']/60:.1f} min)")
    
    print(f"\nOverall Anomaly Score:")
    print(f"  Mean: {stats['overall']['mean']:.3f}")
    print(f"  Range: {stats['overall']['min']:.3f} - {stats['overall']['max']:.3f}")
    print(f"  High Anomaly (>0.9): {stats['overall']['high_anomaly_count']} ({stats['overall']['high_anomaly_count']/stats['total_fingerprints']*100:.1f}%)")
    
    print(f"\nFeature Breakdown:")
    for feature, data in stats['features'].items():
        trigger_pct = data['trigger_count'] / stats['total_fingerprints'] * 100
        print(f"  {feature}:")
        print(f"    Mean: {data['mean']:.3f} | Max: {data['max']:.3f}")
        print(f"    Triggered: {data['trigger_count']}/{stats['total_fingerprints']} ({trigger_pct:.1f}%)")
    
    print(f"\nOperator Trigger Counts:")
    for op, count in stats['operator_counts'].most_common():
        pct = count / stats['total_fingerprints'] * 100
        print(f"  {op}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Default to today's log
        date_str = datetime.now().strftime("%Y%m%d")
        log_path = Path(__file__).parent / "logs" / f"gaming_fingerprint_{date_str}.jsonl"
    
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return
    
    print(f"[*] Loading fingerprints from: {log_path}")
    fingerprints = load_fingerprints(log_path)
    print(f"[+] Loaded {len(fingerprints)} fingerprints")
    
    stats = analyze_session(fingerprints)
    print_summary(stats)


if __name__ == "__main__":
    main()
