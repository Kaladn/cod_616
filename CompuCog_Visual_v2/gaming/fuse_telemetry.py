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

TrueVision v1.0.0 - Full Telemetry Fusion

Purpose:
  Merge ALL CompuCog telemetry sources into TrueVision detection windows:
  - Visual detection (screen-based operators)
  - Input metrics (mouse velocity, click patterns)
  - Network telemetry (latency, packet loss)
  - Process data (game hooks, memory)
  
  Output: Enriched JSONL with complete forensic context.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import glob

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CompuCogLogger"))


def load_latest_input_metrics(input_log_dir: Path, window_epoch: float) -> Optional[Dict]:
    """
    Load input metrics closest to detection window timestamp.
    Input logger writes to: CompuCogLogger/logs/input/input_activity_YYYYMMDD.jsonl
    """
    try:
        # Find today's input log
        date_str = datetime.fromtimestamp(window_epoch).strftime("%Y%m%d")
        input_file = input_log_dir / f"input_activity_{date_str}.jsonl"
        
        if not input_file.exists():
            return None
        
        # Read last ~100 lines (avoid loading entire file)
        with open(input_file, 'r') as f:
            lines = f.readlines()[-100:]
        
        # Find closest timestamp
        closest_entry = None
        min_delta = float('inf')
        
        for line in lines:
            try:
                entry = json.loads(line)
                entry_epoch = entry.get('epoch', 0)
                delta = abs(entry_epoch - window_epoch)
                
                if delta < min_delta and delta < 5.0:  # Within 5 seconds
                    min_delta = delta
                    closest_entry = entry
            except:
                continue
        
        return closest_entry
    except Exception as e:
        return None


def load_latest_network_metrics(network_log_dir: Path, window_epoch: float) -> Optional[Dict]:
    """
    Load network metrics closest to detection window timestamp.
    Network logger writes to: CompuCogLogger/logs/network/telemetry_YYYYMMDD.jsonl
    """
    try:
        date_str = datetime.fromtimestamp(window_epoch).strftime("%Y%m%d")
        network_file = network_log_dir / f"telemetry_{date_str}.jsonl"
        
        if not network_file.exists():
            return None
        
        with open(network_file, 'r') as f:
            lines = f.readlines()[-100:]
        
        closest_entry = None
        min_delta = float('inf')
        
        for line in lines:
            try:
                entry = json.loads(line)
                entry_epoch = entry.get('epoch', 0)
                delta = abs(entry_epoch - window_epoch)
                
                if delta < min_delta and delta < 5.0:
                    min_delta = delta
                    closest_entry = entry
            except:
                continue
        
        return closest_entry
    except Exception as e:
        return None


def load_latest_activity_snapshot(activity_log_dir: Path, window_epoch: float) -> Optional[Dict]:
    """
    Load user activity snapshot closest to detection window timestamp.
    Activity logger writes to: CompuCogLogger/logs/activity/user_activity_YYYYMMDD.jsonl
    """
    try:
        date_str = datetime.fromtimestamp(window_epoch).strftime("%Y%m%d")
        activity_file = activity_log_dir / f"user_activity_{date_str}.jsonl"
        
        if not activity_file.exists():
            return None
        
        with open(activity_file, 'r') as f:
            lines = f.readlines()[-50:]
        
        closest_entry = None
        min_delta = float('inf')
        
        for line in lines:
            try:
                entry = json.loads(line)
                entry_epoch = entry.get('epoch', 0)
                delta = abs(entry_epoch - window_epoch)
                
                if delta < min_delta and delta < 10.0:  # Within 10 seconds
                    min_delta = delta
                    closest_entry = entry
            except:
                continue
        
        return closest_entry
    except Exception as e:
        return None


def fuse_telemetry(truevision_jsonl: str, output_path: Optional[str] = None):
    """
    Fuse TrueVision detection windows with all CompuCog logger telemetry.
    """
    print("=" * 80)
    print("TrueVision v1.0.0 - Full Telemetry Fusion")
    print("=" * 80)
    print()
    
    # Paths
    truevision_path = Path(truevision_jsonl)
    logger_root = Path(__file__).parent.parent.parent / "CompuCogLogger" / "logs"
    
    input_log_dir = logger_root / "input"
    network_log_dir = logger_root / "network"
    activity_log_dir = logger_root / "activity"
    
    print(f"[1/3] Loading TrueVision detection windows...")
    print(f"      Source: {truevision_path.name}")
    
    # Load TrueVision windows
    windows = []
    with open(truevision_path, 'r') as f:
        for line in f:
            windows.append(json.loads(line))
    
    print(f"      Loaded {len(windows)} detection windows")
    print()
    
    print(f"[2/3] Fusing with CompuCog logger telemetry...")
    print(f"      Input logs: {input_log_dir}")
    print(f"      Network logs: {network_log_dir}")
    print(f"      Activity logs: {activity_log_dir}")
    print()
    
    fused_windows = []
    input_fused = 0
    network_fused = 0
    activity_fused = 0
    
    for i, window in enumerate(windows):
        fused_window = window.copy()
        window_epoch = window['window_start_epoch']
        
        # Fuse input metrics
        input_metrics = load_latest_input_metrics(input_log_dir, window_epoch)
        if input_metrics:
            fused_window['input_telemetry'] = {
                'idle_seconds': input_metrics.get('idle_seconds'),
                'is_active': input_metrics.get('is_active'),
                'audio_active': input_metrics.get('audio_active'),
                'camera_active': input_metrics.get('camera_active')
            }
            input_fused += 1
        
        # Fuse network metrics
        network_metrics = load_latest_network_metrics(network_log_dir, window_epoch)
        if network_metrics:
            fused_window['network_telemetry'] = {
                'bytes_sent': network_metrics.get('bytes_sent'),
                'bytes_recv': network_metrics.get('bytes_recv'),
                'packets_sent': network_metrics.get('packets_sent'),
                'packets_recv': network_metrics.get('packets_recv'),
                'connections_active': network_metrics.get('connections_active')
            }
            network_fused += 1
        
        # Fuse activity snapshot
        activity_snapshot = load_latest_activity_snapshot(activity_log_dir, window_epoch)
        if activity_snapshot:
            fused_window['activity_telemetry'] = {
                'active_window_title': activity_snapshot.get('active_window_title'),
                'active_process_name': activity_snapshot.get('active_process_name'),
                'cpu_percent': activity_snapshot.get('cpu_percent'),
                'memory_percent': activity_snapshot.get('memory_percent')
            }
            activity_fused += 1
        
        fused_windows.append(fused_window)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"      Processed {i+1}/{len(windows)} windows...")
    
    print()
    print(f"      ✓ Input telemetry fused: {input_fused}/{len(windows)} windows ({input_fused/len(windows)*100:.1f}%)")
    print(f"      ✓ Network telemetry fused: {network_fused}/{len(windows)} windows ({network_fused/len(windows)*100:.1f}%)")
    print(f"      ✓ Activity telemetry fused: {activity_fused}/{len(windows)} windows ({activity_fused/len(windows)*100:.1f}%)")
    print()
    
    # Save fused telemetry
    print(f"[3/3] Saving fused telemetry...")
    
    if output_path is None:
        output_path = truevision_path.parent / f"{truevision_path.stem}_FUSED.jsonl"
    else:
        output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        for window in fused_windows:
            f.write(json.dumps(window) + "\n")
    
    print(f"      Output: {output_path.name}")
    print(f"      Size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    
    # Summary
    print("=" * 80)
    print("FUSION SUMMARY")
    print("=" * 80)
    print(f"Original windows: {len(windows)}")
    print(f"Fused windows: {len(fused_windows)}")
    print(f"New fields per window:")
    print(f"  - input_telemetry: {input_fused} windows")
    print(f"  - network_telemetry: {network_fused} windows")
    print(f"  - activity_telemetry: {activity_fused} windows")
    print()
    print("✓ Full telemetry fusion complete")
    print(f"  Use fused file for comprehensive EOMM analysis")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fuse TrueVision detection with all CompuCog logger telemetry")
    parser.add_argument("truevision_jsonl", help="Path to TrueVision detection JSONL file")
    parser.add_argument("--output", "-o", help="Output path for fused telemetry (default: *_FUSED.jsonl)")
    
    args = parser.parse_args()
    
    try:
        output_path = fuse_telemetry(args.truevision_jsonl, args.output)
        print(f"\n✅ Fused telemetry saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Fusion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
