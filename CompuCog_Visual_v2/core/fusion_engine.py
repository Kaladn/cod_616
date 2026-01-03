"""
FusionEngine - Cross-Modal Event Fusion

PURPOSE:
    Fuses events from multiple sensor streams into unified multi-modal records.
    Enables cross-modal correlation (e.g., aim lock + no mouse input = aimbot).

CONTRACT (from CONTRACT_ATLAS.md):
    Input: Events from EventManager grouped by time window
    Output: Fused event dict with vision, input, network, gamepad, process streams

ARCHITECTURE:
    FusionEngine receives events from multiple sources within a time window
    and produces a single FusedEvent containing all sensor data aligned.

Author: TrueVision System
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import time


@dataclass
class FusedEvent:
    """
    Multi-modal fused event containing all sensor streams.
    
    At least ONE sensor stream must be present.
    """
    timestamp: float
    window_start: float
    window_end: float
    
    # Sensor streams (None if not available)
    vision: Optional[Dict[str, Any]] = None
    input: Optional[Dict[str, Any]] = None
    network: Optional[Dict[str, Any]] = None
    gamepad: Optional[Dict[str, Any]] = None
    process: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None  # Future
    
    # Fusion metadata
    source_count: int = 0
    sources_present: List[str] = field(default_factory=list)
    fusion_latency_ms: float = 0.0
    
    # Cross-modal correlations detected
    correlations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "vision": self.vision,
            "input": self.input,
            "network": self.network,
            "gamepad": self.gamepad,
            "process": self.process,
            "audio": self.audio,
            "source_count": self.source_count,
            "sources_present": self.sources_present,
            "fusion_latency_ms": self.fusion_latency_ms,
            "correlations": self.correlations,
        }


class FusionEngine:
    """
    Cross-modal event fusion engine.
    
    Takes events from multiple sensors in a time window and produces
    unified FusedEvent records for downstream analysis.
    
    Key Features:
    - Time-window alignment (configurable window size)
    - Cross-modal correlation detection
    - Missing sensor tolerance (works with partial data)
    - Bounded memory (drop-oldest for old windows)
    """
    
    # Bounded memory limits
    MAX_PENDING_WINDOWS = 100  # Max windows in buffer
    MAX_EVENTS_PER_WINDOW = 1000  # Max events per window
    
    def __init__(
        self,
        window_duration_ms: float = 1000.0,
        correlation_threshold: float = 0.7,
        enable_correlations: bool = True
    ):
        """
        Initialize FusionEngine.
        
        Args:
            window_duration_ms: Time window for grouping events (default 1 second)
            correlation_threshold: Minimum confidence for correlation detection
            enable_correlations: Whether to run cross-modal correlation analysis
        """
        self.window_duration_ms = window_duration_ms
        self.window_duration_sec = window_duration_ms / 1000.0
        self.correlation_threshold = correlation_threshold
        self.enable_correlations = enable_correlations
        
        # Pending events by window key (epoch // window_duration)
        self._pending_windows: Dict[int, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Statistics
        self._stats = {
            "events_received": 0,
            "windows_fused": 0,
            "correlations_detected": 0,
            "windows_dropped": 0,
        }
        
        print(f"[+] FusionEngine initialized")
        print(f"    Window: {window_duration_ms}ms")
        print(f"    Correlations: {'enabled' if enable_correlations else 'disabled'}")
        print(f"    Max pending windows: {self.MAX_PENDING_WINDOWS}")
    
    def _get_window_key(self, timestamp: float) -> int:
        """Get window key for a timestamp."""
        return int(timestamp / self.window_duration_sec)
    
    def _enforce_window_bounds(self):
        """Drop oldest windows if we exceed MAX_PENDING_WINDOWS."""
        while len(self._pending_windows) > self.MAX_PENDING_WINDOWS:
            oldest_key = min(self._pending_windows.keys())
            del self._pending_windows[oldest_key]
            self._stats["windows_dropped"] += 1
    
    def ingest_event(self, event: Dict[str, Any], source: str):
        """
        Ingest an event from a sensor source.
        
        Args:
            event: Event dict with at least 'timestamp' field
            source: Source identifier ('vision', 'input', 'network', 'gamepad', 'process')
        """
        timestamp = event.get("timestamp", time.time())
        window_key = self._get_window_key(timestamp)
        
        # Add to pending window
        events_in_window = self._pending_windows[window_key][source]
        
        # Enforce per-window limit
        if len(events_in_window) < self.MAX_EVENTS_PER_WINDOW:
            events_in_window.append(event)
            self._stats["events_received"] += 1
        
        # Enforce window count limit
        self._enforce_window_bounds()
    
    def ingest_vision_window(self, window: Dict[str, Any]):
        """Ingest a TrueVision telemetry window."""
        self.ingest_event(window, "vision")
    
    def ingest_input_event(self, event: Dict[str, Any]):
        """Ingest an input (keyboard/mouse) event."""
        self.ingest_event(event, "input")
    
    def ingest_network_event(self, event: Dict[str, Any]):
        """Ingest a network telemetry event."""
        self.ingest_event(event, "network")
    
    def ingest_gamepad_event(self, event: Dict[str, Any]):
        """Ingest a gamepad event."""
        self.ingest_event(event, "gamepad")
    
    def ingest_process_event(self, event: Dict[str, Any]):
        """Ingest a process monitoring event."""
        self.ingest_event(event, "process")
    
    def fuse_window(self, window_key: int) -> Optional[FusedEvent]:
        """
        Fuse all events in a completed window.
        
        Args:
            window_key: The window key to fuse
            
        Returns:
            FusedEvent if window exists, None otherwise
        """
        if window_key not in self._pending_windows:
            return None
        
        start_time = time.time()
        window_data = self._pending_windows[window_key]
        
        # Calculate window boundaries
        window_start = window_key * self.window_duration_sec
        window_end = window_start + self.window_duration_sec
        
        # Aggregate each source
        sources_present = []
        
        vision_data = self._aggregate_vision(window_data.get("vision", []))
        if vision_data:
            sources_present.append("vision")
        
        input_data = self._aggregate_input(window_data.get("input", []))
        if input_data:
            sources_present.append("input")
        
        network_data = self._aggregate_network(window_data.get("network", []))
        if network_data:
            sources_present.append("network")
        
        gamepad_data = self._aggregate_gamepad(window_data.get("gamepad", []))
        if gamepad_data:
            sources_present.append("gamepad")
        
        process_data = self._aggregate_process(window_data.get("process", []))
        if process_data:
            sources_present.append("process")
        
        # Detect cross-modal correlations
        correlations = []
        if self.enable_correlations and len(sources_present) >= 2:
            correlations = self._detect_correlations(
                vision_data, input_data, network_data, gamepad_data
            )
            self._stats["correlations_detected"] += len(correlations)
        
        # Build fused event
        fusion_latency = (time.time() - start_time) * 1000
        
        fused = FusedEvent(
            timestamp=window_start,
            window_start=window_start,
            window_end=window_end,
            vision=vision_data,
            input=input_data,
            network=network_data,
            gamepad=gamepad_data,
            process=process_data,
            source_count=len(sources_present),
            sources_present=sources_present,
            fusion_latency_ms=fusion_latency,
            correlations=correlations,
        )
        
        # Remove processed window
        del self._pending_windows[window_key]
        self._stats["windows_fused"] += 1
        
        return fused
    
    def flush_completed_windows(self, current_time: float = None) -> List[FusedEvent]:
        """
        Fuse all windows older than current time.
        
        Args:
            current_time: Current epoch time (default: now)
            
        Returns:
            List of FusedEvent objects
        """
        if current_time is None:
            current_time = time.time()
        
        current_key = self._get_window_key(current_time)
        
        # Find windows that are complete (older than current)
        completed_keys = [k for k in self._pending_windows.keys() if k < current_key]
        
        fused_events = []
        for key in sorted(completed_keys):
            fused = self.fuse_window(key)
            if fused and fused.source_count > 0:
                fused_events.append(fused)
        
        return fused_events
    
    # -------------------------------------------------------------------------
    # Aggregation methods (combine multiple events per source)
    # -------------------------------------------------------------------------
    
    def _aggregate_vision(self, events: List[Dict]) -> Optional[Dict]:
        """Aggregate vision events in window."""
        if not events:
            return None
        
        # Take the most recent or highest EOMM score
        best_event = max(events, key=lambda e: e.get("eomm_score", e.get("eomm_composite_score", 0)))
        
        return {
            "event_count": len(events),
            "eomm_score": best_event.get("eomm_score", best_event.get("eomm_composite_score", 0)),
            "eomm_flags": best_event.get("eomm_flags", best_event.get("operator_flags", [])),
            "operator_scores": best_event.get("operator_scores", {}),
            "frame_count": sum(e.get("frame_count", 0) for e in events),
        }
    
    def _aggregate_input(self, events: List[Dict]) -> Optional[Dict]:
        """Aggregate input events in window."""
        if not events:
            return None
        
        # Count input events
        key_presses = sum(1 for e in events if e.get("type") == "key_press")
        mouse_moves = sum(1 for e in events if e.get("type") == "mouse_move")
        mouse_clicks = sum(1 for e in events if e.get("type") == "mouse_click")
        
        return {
            "event_count": len(events),
            "key_presses": key_presses,
            "mouse_moves": mouse_moves,
            "mouse_clicks": mouse_clicks,
            "idle": len(events) == 0,
        }
    
    def _aggregate_network(self, events: List[Dict]) -> Optional[Dict]:
        """Aggregate network events in window."""
        if not events:
            return None
        
        bytes_sent = sum(e.get("bytes_sent", 0) for e in events)
        bytes_recv = sum(e.get("bytes_recv", 0) for e in events)
        
        return {
            "event_count": len(events),
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv,
            "packets": sum(e.get("packets", 0) for e in events),
        }
    
    def _aggregate_gamepad(self, events: List[Dict]) -> Optional[Dict]:
        """Aggregate gamepad events in window."""
        if not events:
            return None
        
        # Analyze stick movement
        stick_movements = [e for e in events if e.get("type") == "stick"]
        button_presses = [e for e in events if e.get("type") == "button"]
        
        return {
            "event_count": len(events),
            "stick_events": len(stick_movements),
            "button_events": len(button_presses),
            "active": len(events) > 0,
        }
    
    def _aggregate_process(self, events: List[Dict]) -> Optional[Dict]:
        """Aggregate process events in window."""
        if not events:
            return None
        
        # Get unique processes
        processes = set(e.get("process_name", "") for e in events)
        
        return {
            "event_count": len(events),
            "unique_processes": len(processes),
            "cpu_avg": sum(e.get("cpu_percent", 0) for e in events) / len(events) if events else 0,
            "memory_avg": sum(e.get("memory_mb", 0) for e in events) / len(events) if events else 0,
        }
    
    # -------------------------------------------------------------------------
    # Cross-modal correlation detection
    # -------------------------------------------------------------------------
    
    def _detect_correlations(
        self,
        vision: Optional[Dict],
        input: Optional[Dict],
        network: Optional[Dict],
        gamepad: Optional[Dict]
    ) -> List[str]:
        """
        Detect cross-modal correlations that may indicate manipulation.
        
        Key correlations:
        - AIM_LOCK + NO_INPUT = Potential aimbot
        - HIGH_EOMM + LOW_NETWORK = Local manipulation (not lag)
        - GAMEPAD_IDLE + AIM_MOVEMENT = Potential auto-aim
        """
        correlations = []
        
        # Correlation 1: Aim lock without input
        if vision and input:
            eomm = vision.get("eomm_score", 0)
            flags = vision.get("eomm_flags", [])
            mouse_moves = input.get("mouse_moves", 0)
            
            # High EOMM (aim manipulation) but no mouse movement
            if eomm > self.correlation_threshold and "AIM_RESISTANCE" in flags:
                if mouse_moves < 5:  # Very few mouse movements
                    correlations.append("AIM_LOCK_NO_INPUT")
        
        # Correlation 2: High EOMM but stable network (not lag)
        if vision and network:
            eomm = vision.get("eomm_score", 0)
            bytes_recv = network.get("bytes_recv", 0)
            
            if eomm > self.correlation_threshold and bytes_recv > 10000:
                # High EOMM but network is active (not a disconnect)
                correlations.append("MANIPULATION_STABLE_NETWORK")
        
        # Correlation 3: Gamepad idle but vision shows aim movement
        if vision and gamepad:
            flags = vision.get("eomm_flags", [])
            gamepad_active = gamepad.get("active", False)
            stick_events = gamepad.get("stick_events", 0)
            
            if not gamepad_active or stick_events < 3:
                if "AIM_RESISTANCE" in flags or "CROSSHAIR_LOCK" in flags:
                    correlations.append("AIM_WITHOUT_GAMEPAD")
        
        # Correlation 4: HITBOX_DRIFT without reaction time
        # (INSTA_MELT removed as unreliable TTD-based metric)
        if vision and input:
            flags = vision.get("eomm_flags", [])
            key_presses = input.get("key_presses", 0)
            
            if "HITBOX_DRIFT" in flags and key_presses < 2:
                correlations.append("HITBOX_DRIFT_NO_REACTION")
        
        return correlations
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics."""
        return {
            **self._stats,
            "pending_windows": len(self._pending_windows),
            "window_duration_ms": self.window_duration_ms,
        }
    
    def print_stats(self):
        """Print fusion engine statistics."""
        stats = self.get_stats()
        print("\n[FusionEngine Stats]")
        print(f"  Events received: {stats['events_received']}")
        print(f"  Windows fused: {stats['windows_fused']}")
        print(f"  Correlations detected: {stats['correlations_detected']}")
        print(f"  Windows dropped: {stats['windows_dropped']}")
        print(f"  Pending windows: {stats['pending_windows']}")


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FUSION ENGINE - SELF TEST")
    print("=" * 60)
    print()
    
    engine = FusionEngine(window_duration_ms=1000)
    
    base_time = time.time()
    
    # Simulate vision events
    print("[1/4] Ingesting vision events...")
    for i in range(5):
        engine.ingest_vision_window({
            "timestamp": base_time + i * 0.2,
            "eomm_score": 0.3 + i * 0.1,
            "eomm_flags": ["HITBOX_DRIFT"] if i > 2 else [],
            "operator_scores": {"crosshair": 0.4, "hit_reg": 0.3},
            "frame_count": 30,
        })
    print(f"    Ingested 5 vision windows")
    
    # Simulate input events
    print("[2/4] Ingesting input events...")
    for i in range(20):
        engine.ingest_input_event({
            "timestamp": base_time + i * 0.05,
            "type": "mouse_move" if i % 3 != 0 else "key_press",
        })
    print(f"    Ingested 20 input events")
    
    # Simulate network events
    print("[3/4] Ingesting network events...")
    for i in range(3):
        engine.ingest_network_event({
            "timestamp": base_time + i * 0.3,
            "bytes_sent": 1000,
            "bytes_recv": 5000,
            "packets": 50,
        })
    print(f"    Ingested 3 network events")
    
    # Flush and fuse
    print("[4/4] Fusing completed windows...")
    fused = engine.flush_completed_windows(base_time + 2.0)
    
    print(f"\n    Fused windows: {len(fused)}")
    for f in fused:
        print(f"    - Window {f.window_start:.1f}-{f.window_end:.1f}")
        print(f"      Sources: {f.sources_present}")
        print(f"      Vision EOMM: {f.vision.get('eomm_score', 'N/A') if f.vision else 'N/A'}")
        print(f"      Input events: {f.input.get('event_count', 0) if f.input else 0}")
        print(f"      Correlations: {f.correlations}")
    
    engine.print_stats()
    
    print()
    print("=" * 60)
    print("[OK] FusionEngine self-test complete")
    print("=" * 60)
