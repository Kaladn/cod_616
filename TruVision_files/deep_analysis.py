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
Gaming Fingerprint Deep Aggregator

Extends analyze_fingerprints.py with:
- Temporal analysis (operator firing patterns over time)
- EOMM signature detection
- Multi-operator fusion analysis
- Statistical profiling
- JSON export for visualization

Uses same JSONL format as CompuCog telemetry logs
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import statistics


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


def temporal_analysis(fingerprints: List[Dict]) -> Dict:
    """
    Analyze operator firing patterns over time
    Returns time-series data for visualization
    """
    if not fingerprints:
        return {}
    
    # Build time series for each feature score
    time_series = {
        'timestamps': [],
        'overall_anomaly': [],
        'flicker': [],
        'hud_instability': [],
        'crosshair_anomaly': [],
        'peripheral_flash': [],
        'color_shift': [],
        'detection_counts': []
    }
    
    for fp in fingerprints:
        time_series['timestamps'].append(fp['t_start'])
        time_series['overall_anomaly'].append(fp['features']['overall_anomaly_score'])
        time_series['flicker'].append(fp['features']['flicker_score'])
        time_series['hud_instability'].append(fp['features']['hud_instability_score'])
        time_series['crosshair_anomaly'].append(fp['features']['crosshair_anomaly_score'])
        time_series['peripheral_flash'].append(fp['features']['peripheral_flash_score'])
        time_series['color_shift'].append(fp['features']['color_shift_score'])
        time_series['detection_counts'].append(len(fp['detections']))
    
    # Detect high-activity windows (multiple operators firing)
    high_activity = []
    for i, fp in enumerate(fingerprints):
        if len(fp['detections']) >= 3:
            high_activity.append({
                'index': i,
                'timestamp': fp['t_start'],
                'detection_count': len(fp['detections']),
                'operators': [d['operator'] for d in fp['detections']],
                'overall_score': fp['features']['overall_anomaly_score']
            })
    
    return {
        'time_series': time_series,
        'high_activity_windows': high_activity,
        'total_windows': len(fingerprints)
    }


def eomm_signature_detection(fingerprints: List[Dict]) -> Dict:
    """
    Detect EOMM manipulation patterns:
    - Visibility fogging (color shift events)
    - HUD manipulation timing
    - Crosshair behavior anomalies
    """
    if not fingerprints:
        return {}
    
    # Color shift analysis (fogging signature)
    color_shift_events = []
    for i, fp in enumerate(fingerprints):
        if fp['features']['color_shift_score'] > 0.5:
            # Find the actual detection
            for detection in fp['detections']:
                if detection['operator'] == 'color_shift':
                    color_shift_events.append({
                        'index': i,
                        'timestamp': fp['t_start'],
                        'confidence': detection['confidence'],
                        'temporal_emd': detection['features']['temporal_emd'],
                        'mean_frame_emd': detection['features']['mean_frame_emd']
                    })
    
    # HUD manipulation clusters
    hud_anomalies = []
    for i, fp in enumerate(fingerprints):
        if fp['features']['hud_instability_score'] > 0.5:
            for detection in fp['detections']:
                if detection['operator'] == 'hud_stability':
                    hud_anomalies.append({
                        'index': i,
                        'timestamp': fp['t_start'],
                        'confidence': detection['confidence'],
                        'max_instability': detection['features']['max_instability'],
                        'unstable_regions': detection['features']['unstable_regions']
                    })
    
    # Crosshair smoothness outliers (aim assist signature)
    crosshair_outliers = []
    crosshair_smoothness_values = []
    
    # First pass: collect all smoothness values
    for fp in fingerprints:
        for detection in fp['detections']:
            if detection['operator'] == 'crosshair_motion' and 'smoothness' in detection['features']:
                crosshair_smoothness_values.append(detection['features']['smoothness'])
    
    # Compute mean and stdev
    if crosshair_smoothness_values:
        mean_smoothness = statistics.mean(crosshair_smoothness_values)
        stdev_smoothness = statistics.stdev(crosshair_smoothness_values) if len(crosshair_smoothness_values) > 1 else 0
        
        # Second pass: find outliers (> 2 stdev above mean)
        for i, fp in enumerate(fingerprints):
            for detection in fp['detections']:
                if detection['operator'] == 'crosshair_motion' and 'smoothness' in detection['features']:
                    smoothness = detection['features']['smoothness']
                    if stdev_smoothness > 0 and smoothness > (mean_smoothness + 2 * stdev_smoothness):
                        crosshair_outliers.append({
                            'index': i,
                            'timestamp': fp['t_start'],
                            'smoothness': smoothness,
                            'snap_score': detection['features']['snap_score'],
                            'deviation_from_mean': smoothness - mean_smoothness
                        })
    
    return {
        'color_shift_events': color_shift_events,
        'color_shift_rate': len(color_shift_events) / len(fingerprints) if fingerprints else 0,
        'hud_anomalies': hud_anomalies,
        'hud_anomaly_rate': len(hud_anomalies) / len(fingerprints) if fingerprints else 0,
        'crosshair_outliers': crosshair_outliers,
        'crosshair_baseline': {
            'mean_smoothness': mean_smoothness if crosshair_smoothness_values else 0,
            'stdev_smoothness': stdev_smoothness if crosshair_smoothness_values else 0
        }
    }


def multi_operator_fusion(fingerprints: List[Dict]) -> Dict:
    """
    Analyze co-occurrence patterns of operators
    Returns correlation matrix and cascade events
    """
    if not fingerprints:
        return {}
    
    # Co-occurrence matrix
    operator_pairs = defaultdict(int)
    total_windows = len(fingerprints)
    
    for fp in fingerprints:
        operators = [d['operator'] for d in fp['detections']]
        # Count all pairs
        for i, op1 in enumerate(operators):
            for op2 in operators[i+1:]:
                pair = tuple(sorted([op1, op2]))
                operator_pairs[pair] += 1
    
    # Convert to percentages
    co_occurrence = {}
    for pair, count in operator_pairs.items():
        co_occurrence[f"{pair[0]}+{pair[1]}"] = {
            'count': count,
            'percentage': (count / total_windows) * 100
        }
    
    # Cascade events (3+ operators within 5 seconds)
    cascades = []
    for i in range(len(fingerprints) - 1):
        current = fingerprints[i]
        next_fp = fingerprints[i + 1]
        
        time_delta = next_fp['t_start'] - current['t_start']
        
        if time_delta <= 5.0:
            current_ops = [d['operator'] for d in current['detections']]
            next_ops = [d['operator'] for d in next_fp['detections']]
            
            if len(current_ops) >= 2 and len(next_ops) >= 1:
                cascades.append({
                    'start_index': i,
                    'start_time': current['t_start'],
                    'duration_sec': time_delta,
                    'operator_sequence': current_ops + next_ops,
                    'total_operators': len(current_ops) + len(next_ops)
                })
    
    return {
        'co_occurrence_matrix': co_occurrence,
        'cascade_events': cascades,
        'cascade_count': len(cascades)
    }


def statistical_profile(fingerprints: List[Dict]) -> Dict:
    """
    Compute comprehensive statistical profiles
    """
    if not fingerprints:
        return {}
    
    # Overall anomaly distribution
    overall_scores = [fp['features']['overall_anomaly_score'] for fp in fingerprints]
    
    profile = {
        'overall_anomaly': {
            'mean': statistics.mean(overall_scores),
            'median': statistics.median(overall_scores),
            'stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'min': min(overall_scores),
            'max': max(overall_scores),
            'percentiles': {
                'p25': statistics.quantiles(overall_scores, n=4)[0] if overall_scores else 0,
                'p50': statistics.median(overall_scores),
                'p75': statistics.quantiles(overall_scores, n=4)[2] if overall_scores else 0,
                'p90': statistics.quantiles(overall_scores, n=10)[8] if len(overall_scores) >= 10 else max(overall_scores)
            }
        }
    }
    
    # Per-feature distributions
    for feature in ['flicker_score', 'hud_instability_score', 'crosshair_anomaly_score', 'peripheral_flash_score', 'color_shift_score']:
        values = [fp['features'][feature] for fp in fingerprints]
        non_zero = [v for v in values if v > 0]
        
        profile[feature] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'non_zero_count': len(non_zero),
            'trigger_rate': len(non_zero) / len(values) if values else 0
        }
    
    return profile


def generate_report(fingerprints: List[Dict], output_json: bool = False) -> Dict:
    """
    Generate comprehensive analysis report
    """
    print("\n" + "="*80)
    print("GAMING FINGERPRINT DEEP ANALYSIS")
    print("="*80)
    
    # Basic stats
    duration = fingerprints[-1]['t_end'] - fingerprints[0]['t_start'] if fingerprints else 0
    print(f"\nSession Overview:")
    print(f"  Total Fingerprints: {len(fingerprints)}")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    # Temporal analysis
    print(f"\n[*] Running temporal analysis...")
    temporal = temporal_analysis(fingerprints)
    print(f"  High-activity windows (3+ operators): {len(temporal['high_activity_windows'])}")
    
    # EOMM signatures
    print(f"\n[*] Detecting EOMM signatures...")
    eomm = eomm_signature_detection(fingerprints)
    print(f"  Color shift events: {len(eomm['color_shift_events'])} ({eomm['color_shift_rate']*100:.1f}%)")
    print(f"  HUD anomalies: {len(eomm['hud_anomalies'])} ({eomm['hud_anomaly_rate']*100:.1f}%)")
    print(f"  Crosshair outliers: {len(eomm['crosshair_outliers'])}")
    if eomm['crosshair_baseline']['mean_smoothness'] > 0:
        print(f"  Crosshair baseline: {eomm['crosshair_baseline']['mean_smoothness']:.3f} ± {eomm['crosshair_baseline']['stdev_smoothness']:.3f}")
    
    # Multi-operator fusion
    print(f"\n[*] Analyzing multi-operator patterns...")
    fusion = multi_operator_fusion(fingerprints)
    print(f"  Operator cascade events: {fusion['cascade_count']}")
    print(f"  Top co-occurrences:")
    sorted_pairs = sorted(fusion['co_occurrence_matrix'].items(), key=lambda x: x[1]['count'], reverse=True)
    for pair, data in sorted_pairs[:5]:
        print(f"    {pair}: {data['count']} times ({data['percentage']:.1f}%)")
    
    # Statistical profile
    print(f"\n[*] Computing statistical profiles...")
    stats = statistical_profile(fingerprints)
    print(f"  Overall anomaly: μ={stats['overall_anomaly']['mean']:.3f}, σ={stats['overall_anomaly']['stdev']:.3f}")
    print(f"  Percentiles (25/50/75/90): {stats['overall_anomaly']['percentiles']['p25']:.2f} / {stats['overall_anomaly']['percentiles']['p50']:.2f} / {stats['overall_anomaly']['percentiles']['p75']:.2f} / {stats['overall_anomaly']['percentiles']['p90']:.2f}")
    
    print("\n" + "="*80)
    
    # Build full report
    report = {
        'session': {
            'total_fingerprints': len(fingerprints),
            'duration_sec': duration,
            'start_time': fingerprints[0]['t_start'] if fingerprints else 0,
            'end_time': fingerprints[-1]['t_end'] if fingerprints else 0
        },
        'temporal': temporal,
        'eomm_signatures': eomm,
        'multi_operator_fusion': fusion,
        'statistical_profile': stats
    }
    
    if output_json:
        # Write to JSON file for visualization
        output_path = Path(__file__).parent / "analysis" / f"deep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Deep analysis exported to: {output_path}")
    
    return report


def main():
    # Check for --json flag first
    output_json = '--json' in sys.argv
    
    # Filter out flags from argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if args:
        log_path = Path(args[0])
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
    
    # Generate deep analysis
    generate_report(fingerprints, output_json=output_json)


if __name__ == "__main__":
    main()
