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
║     File automatically watermarked on: 2025-11-29 00:00:00                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import List, Dict, Any
import yaml
from pathlib import Path
import sys

# Import unified schema
sys.path.insert(0, str(Path(__file__).parent.parent))
from truevision_schema import OperatorResult, TelemetryWindow, ManipulationFlags
from session_baseline import SessionBaselineTracker


class EommCompositor:
    """
    EOMM Signature Compositor - combines all operator results into unified TelemetryWindow.
    
    Responsibilities:
    - Aggregate confidence scores from all operators
    - Weight by operator reliability and detection quality
    - Generate composite EOMM manipulation score
    - Deduplicate flags across operators
    - Package results in TelemetryWindow format
    """
    
    def __init__(self, config_path: str = None):
        # Make config optional and tolerant for test contexts
        self.config = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception:
                # Fall back to defaults
                print(f"[WARN] EOMM config {config_path} not found or unreadable; using defaults")
                self.config = {}

        eomm_config = self.config.get("eomm_composite", {})
        self.operator_weights = eomm_config.get("operator_weights", {
            "crosshair_lock": 0.3,
            "hit_registration": 0.3,
            "death_event": 0.25,
            "edge_entry": 0.15
        })

        self.manipulation_threshold = eomm_config.get("manipulation_threshold", 0.5)
        self.high_confidence_threshold = eomm_config.get("high_confidence_threshold", 0.75)

        print(f"[+] EOMM Compositor initialized")
        print(f"    Operator weights: {self.operator_weights}")
        print(f"    Manipulation threshold: {self.manipulation_threshold}")
    
    def compose_window(
        self,
        operator_results: List[OperatorResult],
        window_start_epoch: float,
        window_end_epoch: float,
        session_id: str,
        frame_count: int,
        session_tracker: SessionBaselineTracker
    ) -> TelemetryWindow:
        """
        Compose TelemetryWindow from multiple operator results.
        
        Args:
            operator_results: List of results from all operators
            window_start_epoch: Window start timestamp (Unix epoch)
            window_end_epoch: Window end timestamp
            session_id: Unique session identifier
            frame_count: Number of frames analyzed
            session_tracker: Session baseline tracker for metadata
        
        Returns:
            TelemetryWindow with composite EOMM scoring
        """
        # Compute weighted average confidence score
        composite_score = self._compute_composite_score(operator_results)
        
        # Aggregate flags from all operators (deduplicated)
        all_flags = self._aggregate_flags(operator_results)
        
        # Build metadata with session baselines
        if session_tracker is not None and hasattr(session_tracker, 'to_dict'):
            metadata = session_tracker.to_dict()
        else:
            metadata = {}

        metadata["manipulation_detected"] = composite_score >= self.manipulation_threshold
        metadata["high_confidence"] = composite_score >= self.high_confidence_threshold
        
        # Package into TelemetryWindow
        window = TelemetryWindow(
            window_start_epoch=window_start_epoch,
            window_end_epoch=window_end_epoch,
            window_duration_ms=int((window_end_epoch - window_start_epoch) * 1000),
            operator_results=operator_results,
            eomm_composite_score=composite_score,
            eomm_flags=all_flags,
            session_id=session_id,
            frame_count=frame_count,
            metadata=metadata
        )
        
        return window
    
    def _compute_composite_score(self, operator_results: List[OperatorResult]) -> float:
        """
        Compute weighted average confidence score.
        
        Score = Σ(operator_confidence * operator_weight) / Σ(operator_weight)
        
        Only operators that produced results contribute to the score.
        """
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for result in operator_results:
            weight = self.operator_weights.get(result.operator_name, 0.0)
            weighted_sum += result.confidence * weight
            weight_sum += weight
        
        if weight_sum == 0:
            return 0.0
        
        composite = weighted_sum / weight_sum
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, composite))
    
    def _aggregate_flags(self, operator_results: List[OperatorResult]) -> List[str]:
        """
        Aggregate flags from all operators, deduplicated.
        
        Returns sorted list of unique flags.
        """
        all_flags = set()
        
        for result in operator_results:
            all_flags.update(result.flags)
        
        return sorted(list(all_flags))
    
    def generate_summary_stats(self, window: TelemetryWindow) -> Dict[str, Any]:
        """
        Generate human-readable summary statistics for telemetry window.
        
        Useful for logging and debugging.
        """
        stats = {
            "timestamp": window.get_timestamp_iso(),
            "composite_score": round(window.eomm_composite_score, 3),
            "manipulation_detected": window.has_manipulation_detected(),
            "flag_count": len(window.eomm_flags),
            "flags": window.eomm_flags,
            "operator_count": len(window.operator_results),
            "operators": {}
        }
        
        # Per-operator breakdown
        for op_result in window.operator_results:
            stats["operators"][op_result.operator_name] = {
                "confidence": round(op_result.confidence, 3),
                "flags": op_result.flags,
                "key_metrics": self._extract_key_metrics(op_result)
            }
        
        return stats
    
    def _extract_key_metrics(self, op_result: OperatorResult) -> Dict[str, Any]:
        """Extract most important metrics per operator for summary"""
        op_name = op_result.operator_name
        metrics = op_result.metrics
        
        if op_name == "crosshair_lock":
            return {
                "on_target_frames": metrics.get("on_target_frames", 0),
                "hit_efficiency": round(metrics.get("hit_efficiency", 0.0), 2),
                "aim_resistance_score": round(metrics.get("aim_resistance_score", 0.0), 2)
            }
        
        elif op_name == "hit_registration":
            return {
                "hit_marker_count": metrics.get("hit_marker_count", 0),
                "ghost_ratio": round(metrics.get("ghost_ratio", 0.0), 2),
                "ttk_zscore": round(metrics.get("ttk_zscore", 0.0), 2)
            }
        
        elif op_name == "death_event":
            return {
                "time_to_death_frames": metrics.get("time_to_death_frames", 0),
                "is_instamelt": metrics.get("is_instamelt", False),
                "ttd_zscore": round(metrics.get("ttd_zscore", 0.0), 2)
            }
        
        elif op_name == "edge_entry":
            return {
                "total_entries": metrics.get("total_entries", 0),
                "rear_spawn_ratio": round(metrics.get("rear_spawn_ratio", 0.0), 2),
                "entry_rate_per_sec": round(metrics.get("entry_rate_per_sec", 0.0), 2)
            }
        
        return {}


# Example usage demonstrating full pipeline
def example_full_detection_pipeline():
    """
    Example showing how EOMM compositor integrates all operators.
    """
    import time
    from datetime import datetime
    
    # Initialize compositor with config
    compositor = EommCompositor("gaming/truevision_config.yaml")
    
    # Initialize session tracker
    session_tracker = SessionBaselineTracker(min_samples_for_warmup=5)
    
    # Simulate operator results (normally from actual operators)
    operator_results = [
        OperatorResult(
            operator_name="crosshair_lock",
            confidence=0.7,
            flags=[ManipulationFlags.AIM_RESISTANCE, ManipulationFlags.HITBOX_DRIFT],
            metrics={
                "on_target_frames": 12,
                "hit_marker_frames": 2,
                "intersection_ratio": 0.4,
                "hit_efficiency": 0.17,
                "aim_resistance_score": 0.65
            },
            metadata={"frames_analyzed": 30}
        ),
        OperatorResult(
            operator_name="hit_registration",
            confidence=0.8,
            flags=[ManipulationFlags.GHOST_HITS, ManipulationFlags.TTK_OUTLIER],
            metrics={
                "hit_marker_count": 8,
                "ghosted_hits": 3,
                "ghost_ratio": 0.375,
                "ttk_zscore": 2.3
            },
            metadata={"engagements_detected": 2}
        ),
        OperatorResult(
            operator_name="death_event",
            confidence=0.3,
            flags=[],
            metrics={
                "time_to_death_frames": 45,
                "is_instamelt": False,
                "ttd_zscore": 0.8
            },
            metadata={"death_detected": False}
        ),
        OperatorResult(
            operator_name="edge_entry",
            confidence=0.5,
            flags=[ManipulationFlags.SPAWN_BIAS],
            metrics={
                "total_entries": 5,
                "rear_spawn_ratio": 0.6,
                "entry_rate_per_sec": 5.0
            },
            metadata={"direction_counts": {"rear": 3, "front": 1, "side": 1}}
        )
    ]
    
    # Compose telemetry window
    now = time.time()
    window = compositor.compose_window(
        operator_results=operator_results,
        window_start_epoch=now - 1.0,
        window_end_epoch=now,
        session_id="session_20251129_001",
        frame_count=30,
        session_tracker=session_tracker
    )
    
    # Generate summary
    summary = compositor.generate_summary_stats(window)
    
    print("\n" + "="*80)
    print("EOMM DETECTION SUMMARY")
    print("="*80)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Composite Score: {summary['composite_score']} {'⚠️  MANIPULATION DETECTED' if summary['manipulation_detected'] else '✓ Clean'}")
    print(f"\nFlags Detected: {summary['flag_count']}")
    for flag in summary['flags']:
        print(f"  - {flag}")
    
    print(f"\nOperator Breakdown:")
    for op_name, op_stats in summary['operators'].items():
        print(f"\n  {op_name.upper()}:")
        print(f"    Confidence: {op_stats['confidence']}")
        print(f"    Flags: {', '.join(op_stats['flags']) if op_stats['flags'] else 'None'}")
        print(f"    Key Metrics: {op_stats['key_metrics']}")
    
    print("\n" + "="*80)
    
    return window


if __name__ == "__main__":
    example_full_detection_pipeline()
