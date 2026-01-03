"""
6-1-6 Temporal Fusion Block Builder

Stand-alone module for ML training data generation.
NOT part of real-time detection - runs on-demand only.

Architecture:
    616 = 6 frames (precursor) + 1 frame (anchor) + 6 frames (consequence)
    Total: 13-frame multi-modal aligned windows

Input: Forge Memory JSONL logs (truevision, gamepad, network, input)
Output: FusionBlocks for ML training

Resonance Coherence: Extracted from old 616 frequency-domain analysis.
The 6-1-6 Hz pattern becomes 6-1-6 frame causality windows.

Built: January 3, 2026
Lineage: CompuCogVision Phase 1 → CompuCog_Visual_v2 → TrueVision → 616 Temporal Fusion
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import bisect


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModalitySlice:
    """Single timestamp slice across all modalities."""
    timestamp: float
    frame_index: int
    
    # TrueVision (from truevision_events.jsonl)
    truevision: Optional[Dict[str, Any]] = None
    
    # Gamepad (from gamepad_events.jsonl)
    gamepad: Optional[Dict[str, Any]] = None
    
    # Network (from network_events.jsonl)
    network: Optional[Dict[str, Any]] = None
    
    # Input (from input_events.jsonl)
    input_event: Optional[Dict[str, Any]] = None
    
    # Process (from process_events.jsonl)
    process: Optional[Dict[str, Any]] = None
    
    # Activity (from activity_events.jsonl) - window focus/context
    activity: Optional[Dict[str, Any]] = None


@dataclass
class FusionBlock:
    """
    13-frame multi-modal aligned window centered on anchor event.
    
    Structure: [precursor_6] + [anchor_1] + [consequence_6]
    """
    block_id: str
    anchor_timestamp: float
    anchor_event_type: str  # "crosshair_lock", "hit_registration", "death", etc.
    
    # 6-1-6 temporal structure
    precursor_frames: List[ModalitySlice] = field(default_factory=list)   # 6 frames before
    anchor_frame: Optional[ModalitySlice] = None                           # The event
    consequence_frames: List[ModalitySlice] = field(default_factory=list)  # 6 frames after
    
    # Cross-modal alignment quality
    alignment_score: float = 0.0
    modality_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Resonance coherence (from old 616 frequency-domain concept)
    # Measures how well modalities correlate across the window
    resonance_coherence: float = 0.0
    
    # Labels for ML training
    labels: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "block_id": self.block_id,
            "anchor_timestamp": self.anchor_timestamp,
            "anchor_event_type": self.anchor_event_type,
            "precursor_frames": [self._slice_to_dict(s) for s in self.precursor_frames],
            "anchor_frame": self._slice_to_dict(self.anchor_frame) if self.anchor_frame else None,
            "consequence_frames": [self._slice_to_dict(s) for s in self.consequence_frames],
            "alignment_score": self.alignment_score,
            "modality_coverage": self.modality_coverage,
            "resonance_coherence": self.resonance_coherence,
            "labels": self.labels
        }
    
    @staticmethod
    def _slice_to_dict(s: ModalitySlice) -> Dict[str, Any]:
        return {
            "timestamp": s.timestamp,
            "frame_index": s.frame_index,
            "truevision": s.truevision,
            "gamepad": s.gamepad,
            "network": s.network,
            "input_event": s.input_event,
            "process": s.process,
            "activity": s.activity
        }


# ============================================================================
# FUSION BLOCK BUILDER
# ============================================================================

class FusionBlockBuilder:
    """
    Builds 6-1-6 temporal fusion blocks from Forge Memory logs.
    
    Reads JSONL logs, aligns modalities by timestamp, and produces
    13-frame windows centered on detected anchor events.
    """
    
    # Anchor event types that trigger block creation
    ANCHOR_EVENTS = [
        "crosshair_lock",
        "hit_registration", 
        "death_event",
        "edge_entry",
        "eomm_peak"  # High EOMM score detected
    ]
    
    # Frame timing (16.67ms = 60 FPS)
    FRAME_DURATION_MS = 16.67
    FRAME_DURATION_S = FRAME_DURATION_MS / 1000.0
    
    def __init__(self, session_dir: Path, frame_rate: float = 60.0, logs_dir: Path = None):
        """
        Args:
            session_dir: Path to session directory with JSONL logs
            frame_rate: Expected frame rate (default 60 FPS)
            logs_dir: Optional separate logs directory (for CompuCog_Visual_v2 structure)
        """
        self.session_dir = Path(session_dir)
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        
        # Loaded event streams (6 modalities)
        self.truevision_events: List[Dict] = []
        self.gamepad_events: List[Dict] = []
        self.network_events: List[Dict] = []
        self.input_events: List[Dict] = []
        self.process_events: List[Dict] = []
        self.activity_events: List[Dict] = []  # Window focus context
        
        # Timestamp index for fast lookups
        self.truevision_ts: List[float] = []
        self.gamepad_ts: List[float] = []
        self.network_ts: List[float] = []
        self.input_ts: List[float] = []
        self.process_ts: List[float] = []
        self.activity_ts: List[float] = []
        
        print(f"[FusionBlockBuilder] Session: {session_dir}")
        if logs_dir:
            print(f"[FusionBlockBuilder] Logs dir: {logs_dir}")
        print(f"[FusionBlockBuilder] Frame rate: {frame_rate} FPS ({self.frame_duration*1000:.2f}ms)")
    
    def load_all_streams(self) -> bool:
        """Load all event streams from session directory and logs directory."""
        print(f"[FusionBlockBuilder] Loading event streams...")
        
        any_loaded = False
        
        # Load TrueVision files (may have timestamp suffixes like truevision_live_*.jsonl)
        tv_patterns = ["truevision_events.jsonl", "truevision_live_*.jsonl", "truevision_smoke_*.jsonl"]
        for pattern in tv_patterns:
            for filepath in self.session_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.truevision_events, self.truevision_ts)
                if loaded > 0:
                    print(f"  [truevision] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # Determine logs directory (CompuCog_Visual_v2 structure or flat)
        logs_base = self.logs_dir if self.logs_dir else self.session_dir
        
        # ==== ACTIVITY LOGS (user_activity_*.jsonl) ====
        # Schema: {timestamp, windowTitle, processName, executablePath, idleSeconds}
        activity_dir = logs_base / "activity" if (logs_base / "activity").exists() else logs_base
        for pattern in ["user_activity_*.jsonl", "activity_events.jsonl", "activity_*.jsonl"]:
            for filepath in activity_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.activity_events, self.activity_ts)
                if loaded > 0:
                    print(f"  [activity] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # ==== GAMEPAD LOGS (gamepad_stream_*.jsonl) ====
        # Schema: {timestamp, event, button/axis/trigger, state/value}
        gamepad_dir = logs_base / "gamepad" if (logs_base / "gamepad").exists() else logs_base
        for pattern in ["gamepad_stream_*.jsonl", "gamepad_events.jsonl", "gamepad_*.jsonl"]:
            for filepath in gamepad_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.gamepad_events, self.gamepad_ts)
                if loaded > 0:
                    print(f"  [gamepad] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # ==== INPUT LOGS (input_activity_*.jsonl) ====
        # Schema: {timestamp, keystroke_count, mouse_click_count, mouse_movement_distance, idle_seconds, audio_active, camera_active}
        input_dir = logs_base / "input" if (logs_base / "input").exists() else logs_base
        for pattern in ["input_activity_*.jsonl", "input_events.jsonl", "input_*.jsonl"]:
            for filepath in input_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.input_events, self.input_ts)
                if loaded > 0:
                    print(f"  [input] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # ==== NETWORK LOGS (telemetry_*.jsonl or network_capture_*.jsonl) ====
        # Schema: {Timestamp, LocalAddress, LocalPort, RemoteAddress, RemotePort, State, Protocol, PID, ProcessName}
        network_dir = logs_base / "network" if (logs_base / "network").exists() else logs_base
        for pattern in ["telemetry_*.jsonl", "network_capture_*.jsonl", "network_events.jsonl", "network_*.jsonl"]:
            for filepath in network_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.network_events, self.network_ts)
                if loaded > 0:
                    print(f"  [network] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # ==== PROCESS LOGS (process_activity_*.jsonl) ====
        # Schema: {timestamp, pid, process_name, command_line, parent_pid, origin, flagged}
        process_dir = logs_base / "process" if (logs_base / "process").exists() else logs_base
        for pattern in ["process_activity_*.jsonl", "process_events.jsonl", "process_*.jsonl"]:
            for filepath in process_dir.glob(pattern):
                loaded = self._load_jsonl(filepath, self.process_events, self.process_ts)
                if loaded > 0:
                    print(f"  [process] Loaded {loaded} events from {filepath.name}")
                    any_loaded = True
        
        # Summary
        print(f"[FusionBlockBuilder] Stream totals:")
        print(f"  TrueVision: {len(self.truevision_events)} events")
        print(f"  Activity:   {len(self.activity_events)} events")
        print(f"  Gamepad:    {len(self.gamepad_events)} events")
        print(f"  Input:      {len(self.input_events)} events")
        print(f"  Network:    {len(self.network_events)} events")
        print(f"  Process:    {len(self.process_events)} events")
        
        return any_loaded
    
    def _load_jsonl(self, filepath: Path, events_list: List, ts_list: List) -> int:
        """Load JSONL file into events list and timestamp index."""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    # Extract timestamp (try multiple fields for different schemas)
                    # TrueVision uses window_start_epoch
                    # Gamepad uses timestamp (ISO string)
                    # Others use monotonic_ts
                    ts = (
                        event.get("window_start_epoch") or  # TrueVision TelemetryWindow
                        event.get("monotonic_ts") or        # Standard format
                        event.get("ts") or                  # Abbreviated
                        0
                    )
                    
                    # Handle ISO timestamp strings
                    if isinstance(ts, str):
                        try:
                            from datetime import datetime
                            # Parse ISO format: 2025-12-02T16:52:23.001234Z
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except:
                            ts = count * self.frame_duration  # Fallback to sequential
                        ts = count * self.frame_duration
                    events_list.append(event)
                    ts_list.append(float(ts))
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count
    
    def find_anchor_events(self) -> List[Tuple[float, str, Dict]]:
        """
        Find all anchor events in truevision stream.
        
        Returns:
            List of (timestamp, event_type, event_data) tuples
        """
        anchors = []
        
        for event, ts in zip(self.truevision_events, self.truevision_ts):
            # TrueVision TelemetryWindow format has operator_results array
            operator_results = event.get("operator_results", [])
            
            # Check each operator result for anchor events
            for op_result in operator_results:
                op_name = op_result.get("operator_name", "")
                confidence = op_result.get("confidence", 0)
                flags = op_result.get("flags", [])
                
                # Map operator names to anchor event types
                if op_name == "crosshair_lock" and confidence > 0.5:
                    anchors.append((ts, "crosshair_lock", event))
                elif op_name == "hit_registration" and flags:  # Has ghost hits
                    anchors.append((ts, "hit_registration", event))
                elif op_name == "death_event" and confidence > 0.5:
                    anchors.append((ts, "death_event", event))
                elif op_name == "edge_entry" and confidence > 0.5:
                    anchors.append((ts, "edge_entry", event))
            
            # Also check for high EOMM composite scores
            eomm_score = event.get("eomm_composite_score", 0)
            if eomm_score > 0.7:
                anchors.append((ts, "eomm_peak", event))
        
        # Deduplicate by timestamp (keep first occurrence)
        seen_ts = set()
        unique_anchors = []
        for ts, event_type, event in anchors:
            ts_key = round(ts, 3)  # Round to millisecond
            if ts_key not in seen_ts:
                seen_ts.add(ts_key)
                unique_anchors.append((ts, event_type, event))
        
        # Sort by timestamp
        unique_anchors.sort(key=lambda x: x[0])
        
        print(f"[FusionBlockBuilder] Found {len(unique_anchors)} anchor events")
        return unique_anchors
    
    def get_nearest_event(
        self, 
        ts_list: List[float], 
        events_list: List[Dict], 
        target_ts: float,
        max_delta: float = 0.1  # 100ms max
    ) -> Optional[Dict]:
        """Find nearest event to target timestamp using binary search."""
        if not ts_list:
            return None
        
        # Binary search for insertion point
        idx = bisect.bisect_left(ts_list, target_ts)
        
        # Check neighbors
        candidates = []
        if idx > 0:
            candidates.append((abs(ts_list[idx-1] - target_ts), events_list[idx-1]))
        if idx < len(ts_list):
            candidates.append((abs(ts_list[idx] - target_ts), events_list[idx]))
        
        if not candidates:
            return None
        
        # Return closest within max_delta
        best_delta, best_event = min(candidates, key=lambda x: x[0])
        if best_delta <= max_delta:
            return best_event
        return None
    
    def build_slice_at_timestamp(self, target_ts: float, frame_idx: int) -> ModalitySlice:
        """Build a ModalitySlice by finding nearest events in each stream."""
        return ModalitySlice(
            timestamp=target_ts,
            frame_index=frame_idx,
            truevision=self.get_nearest_event(self.truevision_ts, self.truevision_events, target_ts),
            gamepad=self.get_nearest_event(self.gamepad_ts, self.gamepad_events, target_ts),
            network=self.get_nearest_event(self.network_ts, self.network_events, target_ts),
            input_event=self.get_nearest_event(self.input_ts, self.input_events, target_ts),
            process=self.get_nearest_event(self.process_ts, self.process_events, target_ts),
            activity=self.get_nearest_event(self.activity_ts, self.activity_events, target_ts),
        )
    
    def compute_alignment_score(self, slices: List[ModalitySlice]) -> Tuple[float, Dict[str, float]]:
        """
        Compute alignment quality across modalities.
        
        Returns:
            (overall_score, modality_coverage_dict)
        """
        if not slices:
            return 0.0, {}
        
        modality_counts = defaultdict(int)
        total_slots = len(slices)
        
        for s in slices:
            if s.truevision:
                modality_counts["truevision"] += 1
            if s.gamepad:
                modality_counts["gamepad"] += 1
            if s.network:
                modality_counts["network"] += 1
            if s.input_event:
                modality_counts["input"] += 1
            if s.process:
                modality_counts["process"] += 1
            if s.activity:
                modality_counts["activity"] += 1
        
        coverage = {k: v / total_slots for k, v in modality_counts.items()}
        overall = sum(coverage.values()) / 6.0  # 6 modalities now
        
        return overall, coverage
    
    def compute_resonance_coherence(self, block: FusionBlock) -> float:
        """
        Compute resonance coherence across the 6-1-6 window.
        
        This is inspired by the old 616 frequency-domain analysis:
        - Measures cross-modal correlation
        - High coherence = modalities move together (suspicious)
        - Low coherence = normal independent behavior
        
        Returns:
            Coherence score 0.0 - 1.0
        """
        # Simple implementation: check if modalities are present in each phase
        # More sophisticated analysis would use actual correlation
        
        precursor_has_input = any(
            (s.gamepad is not None) or (s.input_event is not None) 
            for s in block.precursor_frames
        )
        anchor_has_vision = (
            block.anchor_frame is not None and 
            block.anchor_frame.truevision is not None
        )
        consequence_has_network = any(
            s.network is not None 
            for s in block.consequence_frames
        )
        
        # Convert to int for summing
        signals = [
            1 if precursor_has_input else 0,
            1 if anchor_has_vision else 0,
            1 if consequence_has_network else 0
        ]
        coherence = sum(signals) / len(signals)
        
        return coherence
    
    def build_block(self, anchor_ts: float, event_type: str, event_data: Dict, block_idx: int) -> FusionBlock:
        """
        Build a FusionBlock centered on an anchor event.
        
        6 frames before + 1 anchor + 6 frames after = 13 frames total
        """
        block_id = f"block_{block_idx:06d}_{event_type}"
        
        # Build precursor frames (6 frames before anchor)
        precursor_frames = []
        for i in range(-6, 0):
            ts = anchor_ts + (i * self.frame_duration)
            slice_data = self.build_slice_at_timestamp(ts, i + 6)  # Frame 0-5
            precursor_frames.append(slice_data)
        
        # Build anchor frame
        anchor_frame = self.build_slice_at_timestamp(anchor_ts, 6)  # Frame 6 (center)
        # Inject the actual anchor event
        anchor_frame.truevision = event_data
        
        # Build consequence frames (6 frames after anchor)
        consequence_frames = []
        for i in range(1, 7):
            ts = anchor_ts + (i * self.frame_duration)
            slice_data = self.build_slice_at_timestamp(ts, i + 6)  # Frame 7-12
            consequence_frames.append(slice_data)
        
        # Compute quality metrics
        all_slices = precursor_frames + [anchor_frame] + consequence_frames
        alignment_score, modality_coverage = self.compute_alignment_score(all_slices)
        
        block = FusionBlock(
            block_id=block_id,
            anchor_timestamp=anchor_ts,
            anchor_event_type=event_type,
            precursor_frames=precursor_frames,
            anchor_frame=anchor_frame,
            consequence_frames=consequence_frames,
            alignment_score=alignment_score,
            modality_coverage=modality_coverage,
            labels={"source_event": event_type}
        )
        
        # Compute resonance coherence
        block.resonance_coherence = self.compute_resonance_coherence(block)
        
        return block
    
    def build_all_blocks(self) -> List[FusionBlock]:
        """
        Build FusionBlocks for all anchor events.
        
        Returns:
            List of FusionBlock objects
        """
        # Load streams
        if not self.load_all_streams():
            print("[FusionBlockBuilder] ERROR: No event streams loaded")
            return []
        
        # Find anchors
        anchors = self.find_anchor_events()
        if not anchors:
            print("[FusionBlockBuilder] WARNING: No anchor events found")
            return []
        
        # Build blocks
        blocks = []
        for idx, (ts, event_type, event_data) in enumerate(anchors):
            block = self.build_block(ts, event_type, event_data, idx)
            blocks.append(block)
            
            if (idx + 1) % 100 == 0:
                print(f"[FusionBlockBuilder] Built {idx + 1}/{len(anchors)} blocks...")
        
        print(f"[FusionBlockBuilder] Built {len(blocks)} fusion blocks")
        return blocks
    
    def export_blocks(self, blocks: List[FusionBlock], output_path: Path):
        """Export blocks to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for block in blocks:
                f.write(json.dumps(block.to_dict()) + '\n')
        
        print(f"[FusionBlockBuilder] Exported {len(blocks)} blocks to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6-1-6 Temporal Fusion Block Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fusion.py --session D:\\sessions\\2025-01-03_game1
  python fusion.py --session D:\\sessions\\2025-01-03_game1 --output fusion_blocks.jsonl
  python fusion.py --session D:\\sessions\\2025-01-03_game1 --fps 120
  python fusion.py --session D:\\gaming\\telemetry --logs-dir D:\\CompuCog_Visual_v2\\logs
        """
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        required=True,
        help="Path to session directory with TrueVision JSONL logs"
    )
    
    parser.add_argument(
        "--logs-dir", "-l",
        type=str,
        default=None,
        help="Separate logs directory for other modalities (activity, gamepad, input, network, process)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file (default: session_dir/fusion_blocks.jsonl)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Frame rate for temporal alignment (default: 60)"
    )
    
    args = parser.parse_args()
    
    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}")
        return 1
    
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    if logs_dir and not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}")
        return 1
    
    # Build fusion blocks
    builder = FusionBlockBuilder(session_dir, frame_rate=args.fps, logs_dir=logs_dir)
    blocks = builder.build_all_blocks()
    
    if not blocks:
        print("[WARNING] No blocks built - check session data")
        return 1
    
    # Export
    output_path = Path(args.output) if args.output else session_dir / "fusion_blocks.jsonl"
    builder.export_blocks(blocks, output_path)
    
    # Summary
    print("\n" + "="*60)
    print("FUSION SUMMARY")
    print("="*60)
    print(f"  Session: {session_dir}")
    print(f"  Blocks built: {len(blocks)}")
    print(f"  Output: {output_path}")
    
    # Event type distribution
    event_types = defaultdict(int)
    for block in blocks:
        event_types[block.anchor_event_type] += 1
    
    print("\n  Anchor event distribution:")
    for event_type, count in sorted(event_types.items()):
        print(f"    {event_type}: {count}")
    
    # Alignment quality
    avg_alignment = sum(b.alignment_score for b in blocks) / len(blocks)
    avg_coherence = sum(b.resonance_coherence for b in blocks) / len(blocks)
    print(f"\n  Average alignment score: {avg_alignment:.2%}")
    print(f"  Average resonance coherence: {avg_coherence:.2%}")
    
    print("\n[DONE] Fusion blocks ready for ML training")
    return 0


if __name__ == "__main__":
    exit(main())
