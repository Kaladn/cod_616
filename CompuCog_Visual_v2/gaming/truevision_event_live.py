"""
TrueVision + EventManager + Forge Memory — Live Integration

Captures TrueVision windows, records events for significant detections,
writes to Forge Memory, and builds 6-1-6 capsules for analysis.

This is the full cognitive stack in action:
- Layer 0: Forge Memory (storage)
- Layer 1: ChronosManager (deterministic time)
- Layer 2: EventManager (6-1-6 capsules, chains)
- Layer 3: TrueVision (vision operators)

Author: Manus AI
Date: December 2025
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add parent and memory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "memory"))
sys.path.insert(0, str(parent_dir / "core"))
sys.path.insert(0, str(parent_dir / "operators"))
sys.path.insert(0, str(parent_dir / "compositor"))
sys.path.insert(0, str(parent_dir / "baselines"))

# Import cognitive stack
from event_system.chronos_manager import ChronosManager, ChronosMode
from event_system.event_manager import EventManager, Event
from forge_memory.live_forge_harness import build_truevision_forge_pipeline
from forge_memory.wal.pulse_writer import PulseConfig

# Import TrueVision components
from frame_to_grid import FrameCapture, FrameToGrid
from crosshair_lock import CrosshairLockOperator, FrameSequence
from hit_registration import HitRegistrationOperator
from death_event import DeathEventOperator
from edge_entry import EdgeEntryOperator
from eomm_compositor import EommCompositor
from session_baseline import SessionBaselineTracker
import yaml


class CognitiveHarness:
    """
    Full cognitive stack integration.
    
    Responsibilities:
    1. Capture TrueVision windows (NATIVE resolution grids, operators, EOMM)
    2. Write to Forge Memory (ForgeRecords with durability)
    3. Record significant events (high EOMM, operator triggers, anomalies)
    4. Build 6-1-6 capsules for event context
    5. Track chains (gaming sessions, matches)
    """
    
    def __init__(
        self,
        data_dir: str = "./forge_data",
        enable_events: bool = True,
        event_threshold: float = 0.7,
        session_id: str = None
    ):
        """
        Initialize cognitive harness.
        
        Parameters:
        - data_dir: Directory for Forge Memory files
        - enable_events: Whether to record events (True recommended)
        - event_threshold: EOMM threshold for event recording (0.0-1.0)
        - session_id: Optional session identifier for chain tracking
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_events = enable_events
        self.event_threshold = event_threshold
        self.session_id = session_id or f"session_{int(time.time())}"
        
        print("\n" + "="*80)
        print("Cognitive Harness — Initializing Full Stack")
        print("="*80)
        
        # Layer 1: ChronosManager (deterministic time)
        print("\n[Layer 1] Initializing ChronosManager...")
        self.chronos = ChronosManager()
        self.chronos.initialize(ChronosMode.LIVE)
        print(f"  ✓ ChronosManager initialized in LIVE mode")
        print(f"  ✓ Current timestamp: {self.chronos.now():.3f}")
        
        # Layer 2: EventManager (events + capsules)
        if self.enable_events:
            print("\n[Layer 2] Initializing EventManager...")
            self.event_mgr = EventManager(chronos_manager=self.chronos)
            
            # Register sources
            self.event_mgr.register_source("vision", "organ", {"type": "TrueVision"})
            self.event_mgr.register_source("operators", "detector", {"type": "TrueVisionOperators"})
            self.event_mgr.register_source("session", "tracker", {"type": "SessionTracker"})
            
            print(f"  ✓ EventManager initialized with 3 sources")
            
            # Create session chain
            self.session_chain = self.event_mgr.create_chain(
                chain_id=self.session_id,
                metadata={
                    "start_time": self.chronos.now(),
                    "type": "gaming_session"
                }
            )
            print(f"  ✓ Session chain created: {self.session_id}")
        else:
            self.event_mgr = None
            print("\n[Layer 2] EventManager disabled (--no-events)")
        
        # Layer 3: TrueVision (vision capture)
        print("\n[Layer 3] Initializing TrueVision...")
        
        # Load main integration config FIRST (needed for Forge config)
        config_path = Path(__file__).parent / "config" / "truevision_integration.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Layer 0: Forge Memory (storage) - using config values
        print("\n[Layer 0] Initializing Forge Memory...")
        forge_config = self.config.get("forge", {})
        pulse_cfg = forge_config.get("pulse_config", {})
        
        pulse_config = PulseConfig(
            max_records_per_pulse=pulse_cfg.get("max_records_per_pulse", 128),
            max_bytes_per_pulse=pulse_cfg.get("max_bytes_per_pulse", 512 * 1024),
            max_age_ms_per_pulse=pulse_cfg.get("max_age_ms_per_pulse", 250)
        )
        
        self.pulse_writer = build_truevision_forge_pipeline(
            data_dir=str(self.data_dir),
            pulse_config=pulse_config
        )
        
        # Access BinaryLog through PulseWriter
        self.binary_log = self.pulse_writer.binary_log
        
        print(f"  ✓ Forge Memory initialized at {self.data_dir}")
        print(f"  ✓ PulseWriter: batching at {pulse_cfg.get('max_records_per_pulse', 128)} records / {pulse_cfg.get('max_bytes_per_pulse', 512*1024)} bytes / {pulse_cfg.get('max_age_ms_per_pulse', 250)}ms")
        
        # Frame capture config
        capture_config = {
            "capture": self.config.get("capture", {
                "source": "auto",
                "region": "full"
            }),
            "grid": self.config.get("grid", {
                "target_height": 32,
                "target_width": 32,
                "enable_downsampling": True
            })
        }
        
        # Frame capture and grid conversion
        self.frame_capture = FrameCapture(capture_config)
        self.frame_to_grid = FrameToGrid(capture_config)
        
        # Initialize operators from config
        operators_config = self.config.get("operators", {})
        config_base = Path(__file__).parent / "config" / "operators"
        
        if operators_config.get("crosshair_lock", {}).get("enabled", True):
            crosshair_config = str(config_base / "crosshair_lock.yaml")
            self.crosshair_op = CrosshairLockOperator(crosshair_config)
            print(f"  ✓ Crosshair Lock operator loaded")
        else:
            self.crosshair_op = None
        
        if operators_config.get("hit_registration", {}).get("enabled", True):
            hit_config = str(config_base / "hit_registration.yaml")
            self.hit_op = HitRegistrationOperator(hit_config)
            print(f"  ✓ Hit Registration operator loaded")
        else:
            self.hit_op = None
        
        if operators_config.get("death_event", {}).get("enabled", True):
            death_config = str(config_base / "death_event.yaml")
            self.death_op = DeathEventOperator(death_config)
            print(f"  ✓ Death Event operator loaded")
        else:
            self.death_op = None
        
        if operators_config.get("edge_entry", {}).get("enabled", True):
            edge_config = str(config_base / "edge_entry.yaml")
            self.edge_op = EdgeEntryOperator(edge_config)
            print(f"  ✓ Edge Entry operator loaded")
        else:
            self.edge_op = None
        
        # EOMM compositor
        eomm_config = str(Path(__file__).parent / "config" / "eomm_compositor.yaml")
        self.eomm = EommCompositor(eomm_config)
        print(f"  ✓ EOMM Compositor loaded")
        
        # Session baseline tracker
        baseline_config = self.config.get("session_baseline", {})
        if baseline_config.get("enabled", True):
            min_samples = baseline_config.get("min_samples_for_warmup", 5)
            self.baseline = SessionBaselineTracker(min_samples_for_warmup=min_samples)
            print(f"  ✓ Session Baseline Tracker loaded")
        else:
            self.baseline = None
        
        # Frame buffer for operators (they need sequences)
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        operator_count = sum(1 for op in [self.crosshair_op, self.hit_op, self.death_op, self.edge_op] if op is not None)
        print(f"  ✓ TrueVision initialized with {operator_count} operators")
        
        # Statistics
        self.windows_captured = 0
        self.events_recorded = 0
        self.high_eomm_count = 0
        self.operator_triggers = 0
        self.start_time = time.time()
        
        print("\n" + "="*80)
        print("🔥 Cognitive Stack Operational")
        print("="*80)
        print(f"\nSession ID: {self.session_id}")
        print(f"Event Recording: {'ENABLED' if self.enable_events else 'DISABLED'}")
        if self.enable_events:
            print(f"Event Threshold: EOMM >= {self.event_threshold:.2f}")
        print(f"Data Directory: {self.data_dir}")
        print("")
    
    def process_window(self, window: Dict[str, Any]) -> None:
        """
        Process a single TrueVision window.
        
        Steps:
        1. Write to Forge Memory (via PulseWriter)
        2. Check for significant events
        3. Record events if thresholds met
        4. Build capsules for high-priority events
        
        Parameters:
        - window: TrueVision window dictionary
        """
        self.windows_captured += 1
        
        # Step 1: Write to Forge Memory
        try:
            self.pulse_writer.submit_window(window)
        except Exception as e:
            print(f"\n❌ ERROR submitting window to Forge: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 2: Check for significant events
        if not self.enable_events:
            return
        
        eomm_score = window.get("eomm_composite", 0.0)
        operator_flags = window.get("operator_flags", [])
        
        # High EOMM event
        if eomm_score >= self.event_threshold:
            self._record_high_eomm_event(window, eomm_score)
        
        # Operator trigger events
        if operator_flags:
            self._record_operator_events(window, operator_flags)
    
    def _record_high_eomm_event(self, window: Dict[str, Any], eomm_score: float) -> None:
        """Record high EOMM detection event."""
        self.high_eomm_count += 1
        
        # Determine severity
        if eomm_score >= 0.9:
            severity = "critical"
        elif eomm_score >= 0.8:
            severity = "high"
        else:
            severity = "medium"
        
        event = self.event_mgr.record_event(
            source_id="vision",
            tags=["high_eomm", "manipulation", severity],
            metadata={
                "eomm_score": eomm_score,
                "window_id": window.get("window_id"),
                "epoch": window.get("epoch"),
                "operator_flags": window.get("operator_flags", []),
                "grid_shape": window.get("grid", {}).get("shape"),
                "severity": severity
            }
        )
        
        self.events_recorded += 1
        
        # Attach to session chain
        self.event_mgr.attach_event_to_chain(event.event_id, self.session_id)
        
        # Print alert for critical events
        if severity == "critical":
            print(f"\n🚨 CRITICAL EOMM: {eomm_score:.3f} | Event: {event.event_id}")
    
    def _record_operator_events(self, window: Dict[str, Any], operator_flags: List[str]) -> None:
        """Record operator trigger events."""
        self.operator_triggers += len(operator_flags)
        
        for flag in operator_flags:
            event = self.event_mgr.record_event(
                source_id="operators",
                tags=["operator_trigger", flag.lower().replace(" ", "_")],
                metadata={
                    "operator": flag,
                    "eomm_score": window.get("eomm_composite", 0.0),
                    "window_id": window.get("window_id"),
                    "epoch": window.get("epoch"),
                    "all_flags": operator_flags
                }
            )
            
            self.events_recorded += 1
            
            # Attach to session chain
            self.event_mgr.attach_event_to_chain(event.event_id, self.session_id)
    
    def run(self, duration: int = 30, verbose: bool = True) -> None:
        """
        Run live capture loop.
        
        Parameters:
        - duration: Capture duration in seconds
        - verbose: Print progress updates
        """
        print(f"Starting capture for {duration} seconds...\n")
        
        last_stats_time = time.time()
        stats_interval = 5.0  # Print stats every 5 seconds
        
        start = time.time()
        window_counter = 0
        
        try:
            while time.time() - start < duration:
                # Capture frame
                frame = self.frame_capture.capture()
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Convert to grid
                frame_timestamp = time.time()
                grid_result = self.frame_to_grid.convert(
                    frame=frame,
                    frame_id=len(self.frame_buffer),
                    t_sec=frame_timestamp,
                    source="truevision_live"
                )
                
                # Add to buffer
                self.frame_buffer.append(grid_result)
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)
                
                # Run operators (need at least 3 frames)
                if len(self.frame_buffer) >= 3:
                    window = self._build_detection_window(window_counter)
                    
                    if window:
                        # Process window
                        self.process_window(window)
                        window_counter += 1
                
                # Print stats periodically
                if verbose and time.time() - last_stats_time >= stats_interval:
                    self._print_progress()
                    last_stats_time = time.time()
                
                # Brief sleep to avoid overwhelming system
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Capture interrupted by user")
        
        finally:
            # Cleanup
            print("\n\nShutting down...")
            self._shutdown()
    
    def _build_detection_window(self, window_id: int) -> Dict[str, Any]:
        """
        Build TrueVision detection window from frame buffer.
        
        Parameters:
        - window_id: Unique window identifier
        
        Returns:
        - Detection window dictionary compatible with Forge schema
        """
        if len(self.frame_buffer) < 3:
            return None
        
        # Create frame sequence for operators (FrameGrid objects)
        recent_grids = self.frame_buffer[-3:]
        seq = FrameSequence(
            frames=recent_grids,
            t_start=recent_grids[0].t_sec,
            t_end=recent_grids[-1].t_sec,
            src="truevision_live"
        )
        
        # Run operators (if available)
        operator_results = []
        operator_flags = []
        
        if self.crosshair_op:
            crosshair_result = self.crosshair_op.analyze(seq)
            if crosshair_result and crosshair_result.detected:
                operator_results.append(crosshair_result)
                operator_flags.append(crosshair_result.flag.value if hasattr(crosshair_result, 'flag') else "AIM_SUPPRESSION")
        
        if self.hit_op:
            hit_result = self.hit_op.analyze(seq)
            if hit_result and hit_result.detected:
                operator_results.append(hit_result)
                operator_flags.append(hit_result.flag.value if hasattr(hit_result, 'flag') else "GHOST_BULLETS")
        
        if self.death_op:
            death_result = self.death_op.analyze(seq)
            if death_result and death_result.detected:
                operator_results.append(death_result)
                operator_flags.append(death_result.flag.value if hasattr(death_result, 'flag') else "INSTA_MELT")
        
        if self.edge_op:
            edge_result = self.edge_op.analyze(seq)
            if edge_result and edge_result.detected:
                operator_results.append(edge_result)
                operator_flags.append(edge_result.flag.value if hasattr(edge_result, 'flag') else "REAR_SPAWN")
        
        # Compute EOMM composite using compositor
        latest_grid = self.frame_buffer[-1]
        window_start = recent_grids[0].t_sec
        window_end = recent_grids[-1].t_sec
        
        telemetry_window = self.eomm.compose_window(
            operator_results=operator_results,
            window_start_epoch=window_start,
            window_end_epoch=window_end,
            session_id=self.session_id,
            frame_count=len(recent_grids),
            session_tracker=self.baseline if self.baseline else SessionBaselineTracker(min_samples_for_warmup=5)
        )
        
        eomm_score = telemetry_window.eomm_composite_score
        
        # Build operator scores dict
        operator_scores = {}
        for op in operator_results:
            operator_scores[op.operator_name] = op.confidence
        
        # Build window dict compatible with Forge schema (TRUEVISION_SCHEMA_MAP.md)
        window = {
            # Timing
            "ts_start": window_start,
            "ts_end": window_end,
            
            # Grid metadata - NATIVE resolution
            # Native: use actual dimensions, fallback to 1920×1080 (not 32×32)
            "grid_shape": [latest_grid.grid.shape[0], latest_grid.grid.shape[1]] if hasattr(latest_grid.grid, 'shape') else [latest_grid.h, latest_grid.w] if hasattr(latest_grid, 'h') else [1080, 1920],
            "grid_color_count": 10,  # 0-9 palette
            
            # EOMM scoring
            "eomm_score": eomm_score,
            "operator_scores": operator_scores,
            "operator_flags": operator_flags,
            
            # Telemetry (from TelemetryWindow)
            "telemetry": {
                "frame_count": len(recent_grids),
                "window_duration_ms": int((window_end - window_start) * 1000),
            },
            
            # Session context
            "session_context": {
                "session_id": self.session_id,
                "window_id": window_id,
                "source": "truevision_live",
                "render_scale": 1.0,
                "roi": None
            }
        }
        
        return window
    
    def _print_progress(self) -> None:
        """Print progress statistics."""
        elapsed = time.time() - self.start_time
        rate = self.windows_captured / elapsed if elapsed > 0 else 0
        
        print(f"\n[Progress @ {elapsed:.1f}s]")
        print(f"  Windows: {self.windows_captured} ({rate:.1f}/sec)")
        
        if self.enable_events:
            print(f"  Events: {self.events_recorded}")
            print(f"  High EOMM: {self.high_eomm_count}")
            print(f"  Operators: {self.operator_triggers}")
        
        # Forge stats
        print(f"  Forge: {self.binary_log.get_record_count()} records")
    
    def _shutdown(self) -> None:
        """Clean shutdown of all systems."""
        print("  Flushing PulseWriter...")
        try:
            self.pulse_writer.flush(reason="shutdown")
        except Exception as e:
            print(f"    ❌ Flush error: {e}")
        
        print("  Closing Forge Memory...")
        try:
            self.pulse_writer.close()
            self.binary_log.close()  # Close memory-mapped file
        except Exception as e:
            print(f"    ❌ Close error: {e}")
            import traceback
            traceback.print_exc()
        
        # Final statistics
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("Session Complete")
        print("="*80)
        print(f"\nDuration: {elapsed:.1f} seconds")
        print(f"Windows Captured: {self.windows_captured}")
        print(f"Forge Records: {self.binary_log.get_record_count()}")
        
        if self.enable_events:
            print(f"\nEvents Recorded: {self.events_recorded}")
            print(f"  High EOMM: {self.high_eomm_count}")
            print(f"  Operator Triggers: {self.operator_triggers}")
            
            # Session chain summary
            chain = self.event_mgr.get_chain(self.session_id)
            print(f"\nSession Chain: {self.session_id}")
            print(f"  Events: {len(chain.event_ids)}")
            print(f"  Duration: {chain.get_duration():.1f} seconds")
            
            # Print EventManager stats
            self.event_mgr.print_stats()
            
            # Show recent high-priority events
            print("\n" + "-"*80)
            print("Recent High-Priority Events:")
            print("-"*80)
            
            recent = self.event_mgr.get_recent_events(limit=5)
            for event in recent:
                print(f"\n  [{event.event_id}] {event.source_id}")
                print(f"    Tags: {', '.join(event.tags)}")
                print(f"    Timestamp: {event.timestamp:.3f}")
                if "eomm_score" in event.metadata:
                    print(f"    EOMM: {event.metadata['eomm_score']:.3f}")
                if "operator" in event.metadata:
                    print(f"    Operator: {event.metadata['operator']}")
            
            # Demonstrate capsule retrieval
            if recent:
                print("\n" + "-"*80)
                print("Sample Capsule (Most Recent Event):")
                print("-"*80)
                
                capsule = self.event_mgr.get_capsule(recent[0].event_id)
                print(f"\n  Anchor: {capsule.anchor_event.event_id}")
                print(f"  Before: {len(capsule.events_before)} events")
                print(f"  After: {len(capsule.events_after)} events")
                print(f"  Time Span: {capsule.get_time_span():.3f} seconds")
        
        print("\n" + "="*80)
        print(f"Data saved to: {self.data_dir}")
        print("="*80)
        print("\nNext steps:")
        print("  1. python forge_inspect.py --limit 10")
        print("  2. python forge_query_demo.py")
        print("  3. python recognition_demo.py")
        print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TrueVision + EventManager + Forge Memory — Live Integration"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./forge_data",
        help="Data directory for Forge Memory (default: ./forge_data)"
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable event recording (Forge only)"
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.7,
        help="EOMM threshold for event recording (default: 0.7)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session identifier (default: auto-generated)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates"
    )
    
    args = parser.parse_args()
    
    # Create harness
    harness = CognitiveHarness(
        data_dir=args.data_dir,
        enable_events=not args.no_events,
        event_threshold=args.event_threshold,
        session_id=args.session_id
    )
    
    # Run capture
    harness.run(
        duration=args.duration,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
