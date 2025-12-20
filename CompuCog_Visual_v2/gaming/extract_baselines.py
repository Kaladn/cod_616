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
║     File automatically watermarked on: 2025-12-02 00:00:00                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TrueVision v1.0.0 - Baseline Fingerprint Extractor

Purpose:
  Analyze bot match telemetry to extract clean baseline fingerprints.
  
  Bot matches provide ground truth:
  - Easy bots = Normal gameplay baseline (what "fair" looks like)
  - Hard bots = Elevated difficulty signatures (mechanical stress patterns)
  - Bot spawns = Known AI behavior (not EOMM manipulation)
  
  Use these fingerprints to calibrate operators for real EOMM detection.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import statistics

def load_telemetry(jsonl_path: str) -> List[Dict]:
    """Load telemetry windows from JSONL file"""
    windows = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            windows.append(json.loads(line))
    return windows


def analyze_difficulty_transition(windows: List[Dict]) -> Dict:
    """
    Analyze where bot difficulty transitions from easy -> hard.
    Look for sustained EOMM score increases.
    """
    eomm_scores = [w['eomm_composite_score'] for w in windows]
    
    # Find transition point (where average shifts significantly)
    window_size = 10
    transition_idx = None
    
    for i in range(len(eomm_scores) - window_size):
        before_avg = statistics.mean(eomm_scores[max(0, i-window_size):i])
        after_avg = statistics.mean(eomm_scores[i:i+window_size])
        
        # Transition = sustained 0.3+ increase
        if after_avg - before_avg > 0.3:
            transition_idx = i
            break
    
    if transition_idx:
        easy_windows = windows[:transition_idx]
        hard_windows = windows[transition_idx:]
        
        return {
            'transition_window': transition_idx,
            'easy_bot_count': len(easy_windows),
            'hard_bot_count': len(hard_windows),
            'easy_bot_windows': easy_windows,
            'hard_bot_windows': hard_windows,
            'easy_avg_eomm': statistics.mean([w['eomm_composite_score'] for w in easy_windows]),
            'hard_avg_eomm': statistics.mean([w['eomm_composite_score'] for w in hard_windows])
        }
    else:
        # No clear transition - might be all one difficulty
        avg_eomm = statistics.mean(eomm_scores)
        if avg_eomm < 0.4:
            return {
                'transition_window': None,
                'easy_bot_count': len(windows),
                'hard_bot_count': 0,
                'easy_bot_windows': windows,
                'hard_bot_windows': [],
                'easy_avg_eomm': avg_eomm,
                'hard_avg_eomm': None
            }
        else:
            return {
                'transition_window': None,
                'easy_bot_count': 0,
                'hard_bot_count': len(windows),
                'easy_bot_windows': [],
                'hard_bot_windows': windows,
                'easy_avg_eomm': None,
                'hard_avg_eomm': avg_eomm
            }


def extract_operator_fingerprints(windows: List[Dict], label: str) -> Dict:
    """
    Extract operator-specific metrics for baseline fingerprinting.
    """
    fingerprints = {
        'label': label,
        'window_count': len(windows),
        'operators': {}
    }
    
    # Aggregate metrics per operator
    for window in windows:
        for op_result in window.get('operator_results', []):
            op_name = op_result['operator_name']
            
            if op_name not in fingerprints['operators']:
                fingerprints['operators'][op_name] = {
                    'detections': 0,
                    'avg_confidence': [],
                    'flags': Counter(),
                    'metrics': defaultdict(list)
                }
            
            op_fp = fingerprints['operators'][op_name]
            op_fp['detections'] += 1
            op_fp['avg_confidence'].append(op_result['confidence'])
            
            # Count flags
            for flag in op_result['flags']:
                op_fp['flags'][flag] += 1
            
            # Collect metrics
            for metric_name, metric_value in op_result['metrics'].items():
                op_fp['metrics'][metric_name].append(metric_value)
    
    # Compute averages
    for op_name, op_fp in fingerprints['operators'].items():
        if op_fp['avg_confidence']:
            op_fp['confidence_mean'] = statistics.mean(op_fp['avg_confidence'])
            op_fp['confidence_std'] = statistics.stdev(op_fp['avg_confidence']) if len(op_fp['avg_confidence']) > 1 else 0
        
        # Compute metric statistics
        metric_stats = {}
        for metric_name, values in op_fp['metrics'].items():
            if values:
                metric_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'samples': len(values)
                }
        op_fp['metric_stats'] = metric_stats
        
        # Remove raw lists (too large for output)
        del op_fp['avg_confidence']
        del op_fp['metrics']
    
    return fingerprints


def compute_threshold_recommendations(easy_fp: Dict, hard_fp: Dict) -> Dict:
    """
    Compute recommended thresholds to distinguish easy bots (normal) from hard bots.
    Real EOMM should be closer to easy bot baseline.
    """
    recommendations = {}
    
    for op_name in easy_fp['operators'].keys():
        if op_name not in hard_fp['operators']:
            continue
        
        easy_op = easy_fp['operators'][op_name]
        hard_op = hard_fp['operators'][op_name]
        
        recommendations[op_name] = {}
        
        # Confidence threshold (midpoint between easy and hard)
        easy_conf = easy_op.get('confidence_mean', 0)
        hard_conf = hard_op.get('confidence_mean', 0)
        threshold = (easy_conf + hard_conf) / 2
        recommendations[op_name]['confidence_threshold'] = round(threshold, 2)
        
        # Per-metric thresholds
        metric_thresholds = {}
        for metric_name in easy_op.get('metric_stats', {}).keys():
            if metric_name not in hard_op.get('metric_stats', {}):
                continue
            
            easy_metric = easy_op['metric_stats'][metric_name]
            hard_metric = hard_op['metric_stats'][metric_name]
            
            # Use mean + 1 std as threshold (catches hard bot behavior)
            easy_upper = easy_metric['mean'] + easy_metric['std']
            hard_lower = hard_metric['mean'] - hard_metric['std']
            
            # Threshold between easy upper bound and hard lower bound
            metric_threshold = (easy_upper + hard_lower) / 2
            metric_thresholds[metric_name] = round(metric_threshold, 3)
        
        recommendations[op_name]['metric_thresholds'] = metric_thresholds
    
    return recommendations


def generate_report(telemetry_path: str):
    """Generate baseline fingerprint report"""
    print("=" * 80)
    print("TrueVision v1.0.0 - Baseline Fingerprint Analysis")
    print("=" * 80)
    print()
    
    # Load telemetry
    print(f"[1/5] Loading telemetry: {Path(telemetry_path).name}")
    windows = load_telemetry(telemetry_path)
    print(f"      Loaded {len(windows)} detection windows")
    print()
    
    # Analyze difficulty transition
    print("[2/5] Analyzing bot difficulty transition...")
    transition = analyze_difficulty_transition(windows)
    
    if transition['transition_window']:
        print(f"      ✓ Transition detected at window {transition['transition_window']}")
        print(f"      Easy bots: {transition['easy_bot_count']} windows (avg EOMM={transition['easy_avg_eomm']:.2f})")
        print(f"      Hard bots: {transition['hard_bot_count']} windows (avg EOMM={transition['hard_avg_eomm']:.2f})")
    else:
        if transition['easy_bot_count'] > 0:
            print(f"      ✓ All easy bots ({transition['easy_bot_count']} windows, avg EOMM={transition['easy_avg_eomm']:.2f})")
        else:
            print(f"      ✓ All hard bots ({transition['hard_bot_count']} windows, avg EOMM={transition['hard_avg_eomm']:.2f})")
    print()
    
    # Extract fingerprints
    print("[3/5] Extracting operator fingerprints...")
    
    easy_fp = None
    hard_fp = None
    
    if transition['easy_bot_count'] > 0:
        easy_fp = extract_operator_fingerprints(transition['easy_bot_windows'], 'EASY_BOTS')
        print(f"      ✓ Easy bot fingerprints: {len(easy_fp['operators'])} operators")
    
    if transition['hard_bot_count'] > 0:
        hard_fp = extract_operator_fingerprints(transition['hard_bot_windows'], 'HARD_BOTS')
        print(f"      ✓ Hard bot fingerprints: {len(hard_fp['operators'])} operators")
    print()
    
    # Compute recommendations
    recommendations = None
    if easy_fp and hard_fp:
        print("[4/5] Computing threshold recommendations...")
        recommendations = compute_threshold_recommendations(easy_fp, hard_fp)
        print(f"      ✓ Thresholds computed for {len(recommendations)} operators")
        print()
    
    # Save fingerprints
    print("[5/5] Saving baseline fingerprints...")
    output_dir = Path(telemetry_path).parent / "baselines"
    output_dir.mkdir(exist_ok=True)
    
    session_name = Path(telemetry_path).stem
    
    if easy_fp:
        easy_path = output_dir / f"{session_name}_easy_bots.json"
        with open(easy_path, 'w') as f:
            json.dump(easy_fp, f, indent=2)
        print(f"      ✓ Easy bot baseline: {easy_path.name}")
    
    if hard_fp:
        hard_path = output_dir / f"{session_name}_hard_bots.json"
        with open(hard_path, 'w') as f:
            json.dump(hard_fp, f, indent=2)
        print(f"      ✓ Hard bot baseline: {hard_path.name}")
    
    if recommendations:
        rec_path = output_dir / f"{session_name}_thresholds.json"
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"      ✓ Threshold recommendations: {rec_path.name}")
    print()
    
    # Summary report
    print("=" * 80)
    print("BASELINE FINGERPRINT SUMMARY")
    print("=" * 80)
    print()
    
    if easy_fp:
        print("EASY BOT BASELINE (Normal Gameplay):")
        print("-" * 80)
        for op_name, op_data in easy_fp['operators'].items():
            print(f"  {op_name}:")
            print(f"    Confidence: {op_data['confidence_mean']:.2f} ± {op_data['confidence_std']:.2f}")
            print(f"    Top flags: {', '.join(f'{k}({v})' for k, v in op_data['flags'].most_common(3))}")
            if op_data['metric_stats']:
                print(f"    Key metrics:")
                for metric_name, stats in list(op_data['metric_stats'].items())[:3]:
                    print(f"      {metric_name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print()
    
    if hard_fp:
        print("HARD BOT BASELINE (Elevated Difficulty):")
        print("-" * 80)
        for op_name, op_data in hard_fp['operators'].items():
            print(f"  {op_name}:")
            print(f"    Confidence: {op_data['confidence_mean']:.2f} ± {op_data['confidence_std']:.2f}")
            print(f"    Top flags: {', '.join(f'{k}({v})' for k, v in op_data['flags'].most_common(3))}")
            if op_data['metric_stats']:
                print(f"    Key metrics:")
                for metric_name, stats in list(op_data['metric_stats'].items())[:3]:
                    print(f"      {metric_name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print()
    
    if recommendations:
        print("RECOMMENDED THRESHOLDS (for real EOMM detection):")
        print("-" * 80)
        print("Real human matches should stay below these thresholds.")
        print("Exceeding these = potential EOMM manipulation.")
        print()
        for op_name, thresholds in recommendations.items():
            print(f"  {op_name}:")
            print(f"    Confidence threshold: {thresholds['confidence_threshold']}")
            if thresholds['metric_thresholds']:
                print(f"    Metric thresholds:")
                for metric_name, threshold in thresholds['metric_thresholds'].items():
                    print(f"      {metric_name}: {threshold}")
        print()
    
    print("=" * 80)
    print("✓ Baseline fingerprints extracted")
    print("  Next: Run detection on HUMAN matches and compare to these baselines")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract baseline fingerprints from bot match telemetry")
    parser.add_argument("telemetry", help="Path to telemetry JSONL file")
    
    args = parser.parse_args()
    
    try:
        generate_report(args.telemetry)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
