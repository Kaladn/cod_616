"""
616 Fusion Block Analyzer

Analyzes fusion blocks for manipulation patterns and temporal anomalies.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any


def load_blocks(filepath: Path) -> List[Dict]:
    """Load fusion blocks from JSONL file."""
    blocks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))
    return blocks


def analyze_modality_coverage(blocks: List[Dict]) -> Dict[str, Any]:
    """Analyze which modalities are present across blocks."""
    coverage = defaultdict(int)
    total_slices = 0
    
    for block in blocks:
        # Count in all 13 frames
        all_frames = (
            block.get("precursor_frames", []) + 
            ([block.get("anchor_frame")] if block.get("anchor_frame") else []) +
            block.get("consequence_frames", [])
        )
        
        for frame in all_frames:
            if not frame:
                continue
            total_slices += 1
            if frame.get("truevision"):
                coverage["truevision"] += 1
            if frame.get("gamepad"):
                coverage["gamepad"] += 1
            if frame.get("network"):
                coverage["network"] += 1
            if frame.get("input_event"):
                coverage["input"] += 1
            if frame.get("process"):
                coverage["process"] += 1
            if frame.get("activity"):
                coverage["activity"] += 1
    
    # Convert to percentages
    pct = {k: (v / total_slices * 100) if total_slices > 0 else 0 
           for k, v in coverage.items()}
    
    return {
        "total_slices": total_slices,
        "coverage_counts": dict(coverage),
        "coverage_pct": pct
    }


def analyze_anchor_events(blocks: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of anchor event types."""
    event_types = defaultdict(int)
    eomm_scores = []
    flags_seen = defaultdict(int)
    
    for block in blocks:
        event_type = block.get("anchor_event_type", "unknown")
        event_types[event_type] += 1
        
        # Extract EOMM scores from anchor frame
        anchor = block.get("anchor_frame", {})
        tv = anchor.get("truevision", {}) if anchor else {}
        if tv:
            score = tv.get("eomm_composite_score", 0)
            eomm_scores.append(score)
            for flag in tv.get("eomm_flags", []):
                flags_seen[flag] += 1
    
    return {
        "event_distribution": dict(event_types),
        "eomm_scores": {
            "count": len(eomm_scores),
            "mean": sum(eomm_scores) / len(eomm_scores) if eomm_scores else 0,
            "max": max(eomm_scores) if eomm_scores else 0,
            "min": min(eomm_scores) if eomm_scores else 0,
        },
        "flags_seen": dict(flags_seen)
    }


def analyze_temporal_patterns(blocks: List[Dict]) -> Dict[str, Any]:
    """Analyze 6-1-6 temporal patterns for causality."""
    patterns = {
        "precursor_has_input": 0,
        "precursor_has_network": 0,
        "consequence_has_network": 0,
        "consequence_has_input": 0,
        "full_causality_chain": 0,  # input -> vision -> network
    }
    
    for block in blocks:
        precursor = block.get("precursor_frames", [])
        consequence = block.get("consequence_frames", [])
        
        # Check precursor (6 frames before)
        pre_has_input = any(
            f.get("gamepad") or f.get("input_event") 
            for f in precursor if f
        )
        pre_has_network = any(
            f.get("network") for f in precursor if f
        )
        
        # Check consequence (6 frames after)
        con_has_network = any(
            f.get("network") for f in consequence if f
        )
        con_has_input = any(
            f.get("gamepad") or f.get("input_event")
            for f in consequence if f
        )
        
        if pre_has_input:
            patterns["precursor_has_input"] += 1
        if pre_has_network:
            patterns["precursor_has_network"] += 1
        if con_has_network:
            patterns["consequence_has_network"] += 1
        if con_has_input:
            patterns["consequence_has_input"] += 1
        
        # Full causality: input before -> detection -> network after
        if pre_has_input and con_has_network:
            patterns["full_causality_chain"] += 1
    
    total = len(blocks)
    pct = {k: (v / total * 100) if total > 0 else 0 for k, v in patterns.items()}
    
    return {
        "counts": patterns,
        "percentages": pct,
        "total_blocks": total
    }


def analyze_network_activity(blocks: List[Dict]) -> Dict[str, Any]:
    """Analyze network patterns around anchor events."""
    network_stats = {
        "processes_seen": defaultdict(int),
        "states_seen": defaultdict(int),
        "protocols_seen": defaultdict(int),
        "remote_ports_seen": defaultdict(int),
    }
    
    for block in blocks:
        all_frames = (
            block.get("precursor_frames", []) + 
            ([block.get("anchor_frame")] if block.get("anchor_frame") else []) +
            block.get("consequence_frames", [])
        )
        
        for frame in all_frames:
            if not frame:
                continue
            net = frame.get("network")
            if net:
                process = net.get("ProcessName", "unknown")
                state = net.get("State", "unknown")
                proto = net.get("Protocol", "unknown")
                port = net.get("RemotePort", 0)
                
                network_stats["processes_seen"][process] += 1
                network_stats["states_seen"][state] += 1
                network_stats["protocols_seen"][proto] += 1
                if port:
                    network_stats["remote_ports_seen"][port] += 1
    
    # Convert to regular dicts and get top items
    return {
        "top_processes": dict(sorted(
            network_stats["processes_seen"].items(), 
            key=lambda x: x[1], reverse=True
        )[:10]),
        "connection_states": dict(network_stats["states_seen"]),
        "protocols": dict(network_stats["protocols_seen"]),
        "top_ports": dict(sorted(
            network_stats["remote_ports_seen"].items(),
            key=lambda x: x[1], reverse=True
        )[:10])
    }


def analyze_activity_context(blocks: List[Dict]) -> Dict[str, Any]:
    """Analyze what applications were in focus during events."""
    apps_seen = defaultdict(int)
    windows_seen = defaultdict(int)
    
    for block in blocks:
        anchor = block.get("anchor_frame", {})
        if not anchor:
            continue
        activity = anchor.get("activity")
        if activity:
            process = activity.get("processName", "unknown")
            window = activity.get("windowTitle", "")[:50]  # Truncate
            apps_seen[process] += 1
            if window:
                windows_seen[window] += 1
    
    return {
        "apps_in_focus": dict(sorted(apps_seen.items(), key=lambda x: x[1], reverse=True)),
        "window_titles": dict(sorted(windows_seen.items(), key=lambda x: x[1], reverse=True)[:10])
    }


def find_anomalies(blocks: List[Dict]) -> List[Dict]:
    """Find blocks with potential manipulation indicators."""
    anomalies = []
    
    for block in blocks:
        issues = []
        
        # High EOMM score
        anchor = block.get("anchor_frame", {})
        tv = anchor.get("truevision", {}) if anchor else {}
        if tv:
            eomm = tv.get("eomm_composite_score", 0)
            if eomm > 0.7:
                issues.append(f"HIGH_EOMM_SCORE:{eomm:.2f}")
            
            flags = tv.get("eomm_flags", [])
        # Low alignment (missing modalities)
        alignment = block.get("alignment_score", 0)
        if alignment < 0.1:
            issues.append(f"LOW_ALIGNMENT:{alignment:.2f}")
        
        # Low coherence
        coherence = block.get("resonance_coherence", 0)
        if coherence < 0.2:
            issues.append(f"LOW_COHERENCE:{coherence:.2f}")
        
        if issues:
            anomalies.append({
                "block_id": block.get("block_id"),
                "anchor_timestamp": block.get("anchor_timestamp"),
                "event_type": block.get("anchor_event_type"),
                "issues": issues
            })
    
    return anomalies


def main():
    parser = argparse.ArgumentParser(description="Analyze 616 Fusion Blocks")
    parser.add_argument("--blocks", "-b", type=str, required=True,
                        help="Path to fusion_blocks.jsonl")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON report path")
    
    args = parser.parse_args()
    
    blocks_path = Path(args.blocks)
    if not blocks_path.exists():
        print(f"[ERROR] Blocks file not found: {blocks_path}")
        return 1
    
    print(f"[ANALYZER] Loading blocks from {blocks_path}")
    blocks = load_blocks(blocks_path)
    print(f"[ANALYZER] Loaded {len(blocks)} blocks\n")
    
    # Run analyses
    print("="*60)
    print("616 FUSION BLOCK ANALYSIS")
    print("="*60)
    
    # 1. Modality Coverage
    print("\n[1] MODALITY COVERAGE")
    coverage = analyze_modality_coverage(blocks)
    print(f"    Total frame slices: {coverage['total_slices']}")
    for mod, pct in sorted(coverage['coverage_pct'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {mod:12}: {pct:5.1f}% ({coverage['coverage_counts'][mod]} slices)")
    
    # 2. Anchor Events
    print("\n[2] ANCHOR EVENT DISTRIBUTION")
    anchors = analyze_anchor_events(blocks)
    for event_type, count in sorted(anchors['event_distribution'].items()):
        print(f"    {event_type:20}: {count}")
    print(f"\n    EOMM Scores: mean={anchors['eomm_scores']['mean']:.2f}, "
          f"max={anchors['eomm_scores']['max']:.2f}")
    if anchors['flags_seen']:
        print("    Flags detected:")
        for flag, count in sorted(anchors['flags_seen'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {flag}: {count}")
    
    # 3. Temporal Patterns
    print("\n[3] TEMPORAL CAUSALITY PATTERNS")
    temporal = analyze_temporal_patterns(blocks)
    for pattern, pct in temporal['percentages'].items():
        print(f"    {pattern:30}: {pct:5.1f}%")
    
    # 4. Network Activity
    print("\n[4] NETWORK ACTIVITY")
    network = analyze_network_activity(blocks)
    print("    Top processes making connections:")
    for proc, count in list(network['top_processes'].items())[:5]:
        print(f"      {proc}: {count}")
    print("    Connection states:")
    for state, count in network['connection_states'].items():
        print(f"      {state}: {count}")
    
    # 5. Activity Context
    print("\n[5] APPLICATION CONTEXT")
    activity = analyze_activity_context(blocks)
    print("    Apps in focus during events:")
    for app, count in list(activity['apps_in_focus'].items())[:5]:
        print(f"      {app}: {count}")
    
    # 6. Anomalies
    print("\n[6] POTENTIAL ANOMALIES")
    anomalies = find_anomalies(blocks)
    if anomalies:
        print(f"    Found {len(anomalies)} blocks with issues:")
        for a in anomalies[:10]:  # Show first 10
            print(f"      {a['block_id']}: {', '.join(a['issues'])}")
    else:
        print("    No anomalies detected")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total blocks analyzed: {len(blocks)}")
    print(f"  Blocks with high EOMM (>0.7): {sum(1 for a in anomalies if any('HIGH_EOMM' in i for i in a['issues']))}")
    print(f"  Blocks with HITBOX_DRIFT: {anchors['flags_seen'].get('HITBOX_DRIFT', 0)}")
    print(f"  Full causality chains: {temporal['counts']['full_causality_chain']}/{len(blocks)}"))
    
    # Save report if requested
    if args.output:
        report = {
            "analysis_time": datetime.now().isoformat(),
            "blocks_file": str(blocks_path),
            "total_blocks": len(blocks),
            "modality_coverage": coverage,
            "anchor_events": anchors,
            "temporal_patterns": temporal,
            "network_activity": network,
            "activity_context": activity,
            "anomalies": anomalies
        }
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[ANALYZER] Report saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
