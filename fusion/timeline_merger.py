"""
Timeline Merger - Merges all event streams into one canonical timeline

This is the FIRST step in the fusion pipeline:
  1. timeline_merger.py  → fused_events.jsonl (flat timeline)
  2. fusion.py           → fusion_blocks.jsonl (6-1-6 windows)

Based on: telemetry_fusion_worker.py (One Shot build)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_events_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load all events from a JSONL file."""
    events = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    print(f"[MERGER] Warning: Skipping malformed line in {file_path}")
    return events


def merge_event_streams(session_dir: Path) -> List[Dict[str, Any]]:
    """
    Merge all event streams from a session into a single timeline.
    
    Args:
        session_dir: Path to session directory
    
    Returns:
        List of events sorted by monotonic_ts
    """
    print(f"[MERGER] Merging event streams from {session_dir}")
    
    # All possible event files (superset)
    event_files = [
        # Video/Audio indexes
        "video_index.jsonl",
        "audio_game_index.jsonl",
        "audio_comms_index.jsonl",
        # Input streams
        "gamepad_events.jsonl",
        "input_events.jsonl",
        # System streams
        "activity_events.jsonl",
        "network_events.jsonl",
        "process_events.jsonl",
        "system_perf_events.jsonl",
        # TrueVision detection
        "truevision_events.jsonl"
    ]
    
    all_events = []
    loaded_sources = []
    
    for filename in event_files:
        file_path = session_dir / filename
        if file_path.exists():
            events = load_events_from_file(file_path)
            # Tag each event with its source file
            source_name = filename.replace("_events.jsonl", "").replace(".jsonl", "")
            for event in events:
                if "source" not in event:
                    event["source"] = source_name
            all_events.extend(events)
            loaded_sources.append(filename)
            print(f"  [{source_name}] {len(events)} events")
    
    # Sort all events by monotonic timestamp
    all_events.sort(key=lambda e: e.get("monotonic_ts", 0))
    
    print(f"[MERGER] Total: {len(all_events)} events from {len(loaded_sources)} sources")
    
    return all_events


def write_fused_events(events: List[Dict[str, Any]], output_path: Path):
    """Write fused events to output file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    print(f"[MERGER] Wrote {len(events)} events to {output_path}")


def generate_fusion_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics about the fused event stream."""
    if not events:
        return {
            "total_events": 0,
            "sources": {},
            "time_range": None
        }
    
    # Count events by source
    sources = {}
    for event in events:
        source = event.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    # Calculate time range
    first_ts = events[0].get("monotonic_ts", 0)
    last_ts = events[-1].get("monotonic_ts", 0)
    duration = last_ts - first_ts
    
    stats = {
        "total_events": len(events),
        "sources": sources,
        "time_range": {
            "start_ts": first_ts,
            "end_ts": last_ts,
            "duration_seconds": duration
        },
        "events_per_second": len(events) / duration if duration > 0 else 0
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Timeline Merger - Merge all event streams into one timeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline:
  1. python timeline_merger.py --session D:\\sessions\\game1
     → Creates fused_events.jsonl (flat timeline)
  
  2. python fusion.py --session D:\\sessions\\game1
     → Creates fusion_blocks.jsonl (6-1-6 windows)
        """
    )
    parser.add_argument("--session", "-s", type=str, required=True, 
                        help="Path to session directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: session/fused_events.jsonl)")
    
    args = parser.parse_args()
    
    session_dir = Path(args.session)
    
    if not session_dir.exists():
        print(f"[MERGER] Error: Session directory not found: {session_dir}")
        return 1
    
    # Merge all event streams
    fused_events = merge_event_streams(session_dir)
    
    if not fused_events:
        print("[MERGER] Warning: No events found")
        return 1
    
    # Write fused events
    output_path = Path(args.output) if args.output else session_dir / "fused_events.jsonl"
    write_fused_events(fused_events, output_path)
    
    # Generate statistics
    stats = generate_fusion_stats(fused_events)
    stats_path = session_dir / "fusion_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "="*50)
    print("MERGE SUMMARY")
    print("="*50)
    print(f"  Total events: {stats['total_events']}")
    if stats['time_range']:
        print(f"  Duration: {stats['time_range']['duration_seconds']:.2f}s")
        print(f"  Events/sec: {stats['events_per_second']:.2f}")
    print(f"  Sources: {', '.join(sorted(stats['sources'].keys()))}")
    print(f"\n  Output: {output_path}")
    print(f"  Stats:  {stats_path}")
    print("\n[DONE] Ready for fusion.py block building")
    
    return 0


if __name__ == "__main__":
    exit(main())
