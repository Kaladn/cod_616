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

TrueVision v1.0.0 - Unified Telemetry Schema

Purpose:
  Standardized schema for all manipulation detection operators.
  All operators emit OperatorResult, compositor aggregates into TelemetryWindow.
  
  This ensures:
  - Consistent output format across operators
  - Clean JSON serialization for JSONL export
  - Type safety for downstream analysis
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ManipulationFlags:
    """
    Standardized manipulation flag constants.
    Used across all operators to indicate specific manipulation patterns.
    """
    # Aim manipulation
    AIM_RESISTANCE = "AIM_RESISTANCE"           # Crosshair drag near enemies
    HITBOX_DRIFT = "HITBOX_DRIFT"               # Hitbox shrinking/shifting
    AIM_ASSIST_VARIANCE = "AIM_ASSIST_VARIANCE" # Inconsistent aim assist strength
    
    # Damage output manipulation
    GHOST_HITS = "GHOST_HITS"                   # Hit markers without damage
    DAMAGE_SUPPRESSION = "DAMAGE_SUPPRESSION"   # Reduced damage output
    TTK_OUTLIER = "TTK_OUTLIER"                 # Time-to-kill above baseline
    
    # Damage input manipulation (TTD-based - INSTA_MELT removed as unreliable)
    INCOMING_DAMAGE_SPIKE = "INCOMING_DAMAGE_SPIKE"  # Sudden damage amplification
    FLINCH_ANOMALY = "FLINCH_ANOMALY"           # Excessive screen shake
    
    # Spawn manipulation
    SPAWN_PRESSURE = "SPAWN_PRESSURE"           # High enemy spawn rate
    SPAWN_BIAS = "SPAWN_BIAS"                   # Rear/side spawn clustering
    POST_KILL_SPAWN = "POST_KILL_SPAWN"         # Immediate replacement spawns
    
    @classmethod
    def all_flags(cls) -> List[str]:
        """Get list of all valid flags"""
        return [
            cls.AIM_RESISTANCE, cls.HITBOX_DRIFT, cls.AIM_ASSIST_VARIANCE,
            cls.GHOST_HITS, cls.DAMAGE_SUPPRESSION, cls.TTK_OUTLIER,
            cls.INCOMING_DAMAGE_SPIKE, cls.FLINCH_ANOMALY,
            cls.SPAWN_PRESSURE, cls.SPAWN_BIAS, cls.POST_KILL_SPAWN
        ]


@dataclass
class OperatorResult:
    """
    Standardized output from a detection operator.
    
    Fields:
      operator_name: Unique operator identifier (e.g., "crosshair_lock")
      confidence: Manipulation confidence score [0.0, 1.0]
      flags: List of ManipulationFlags constants indicating detected patterns
      metrics: Numeric metrics specific to operator (e.g., hit_efficiency, ghost_ratio)
      metadata: Optional diagnostic info (e.g., frames_analyzed, debug data)
    """
    operator_name: str
    confidence: float
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class TelemetryWindow:
    """
    1-second detection envelope containing all operator results + EOMM composite score.
    
    Fields:
      window_start_epoch: Start time (Unix epoch seconds)
      window_end_epoch: End time (Unix epoch seconds)
      window_duration_ms: Duration in milliseconds (typically 1000)
      operator_results: List of OperatorResult from all operators
      eomm_composite_score: Weighted aggregate score [0.0, 1.0]
      eomm_flags: Deduplicated flags from all operators
      session_id: Unique session identifier
      frame_count: Number of frames analyzed in window
      metadata: Session baselines, timestamps, etc.
    """
    window_start_epoch: float
    window_end_epoch: float
    window_duration_ms: int
    operator_results: List[OperatorResult]
    eomm_composite_score: float
    eomm_flags: List[str]
    session_id: str
    frame_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "window_start_epoch": self.window_start_epoch,
            "window_end_epoch": self.window_end_epoch,
            "window_duration_ms": self.window_duration_ms,
            "operator_results": [op.to_dict() for op in self.operator_results],
            "eomm_composite_score": self.eomm_composite_score,
            "eomm_flags": self.eomm_flags,
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "metadata": self.metadata
        }


def validate_operator_result(result: OperatorResult) -> bool:
    """
    Validate OperatorResult schema compliance.
    
    Checks:
    - confidence in [0, 1]
    - flags are valid ManipulationFlags
    - metrics are numeric
    """
    if not 0.0 <= result.confidence <= 1.0:
        raise ValueError(f"Confidence {result.confidence} out of range [0, 1]")
    
    valid_flags = ManipulationFlags.all_flags()
    for flag in result.flags:
        if flag not in valid_flags:
            raise ValueError(f"Invalid flag: {flag}")
    
    for key, value in result.metrics.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric {key} must be numeric, got {type(value)}")
    
    return True


def validate_telemetry_window(window: TelemetryWindow) -> bool:
    """
    Validate TelemetryWindow schema compliance.
    
    Checks:
    - window duration matches epoch delta
    - eomm_composite_score in [0, 1]
    - all operator results are valid
    """
    epoch_delta_ms = int((window.window_end_epoch - window.window_start_epoch) * 1000)
    if abs(epoch_delta_ms - window.window_duration_ms) > 10:  # Allow 10ms tolerance
        raise ValueError(f"Window duration mismatch: {epoch_delta_ms}ms != {window.window_duration_ms}ms")
    
    if not 0.0 <= window.eomm_composite_score <= 1.0:
        raise ValueError(f"EOMM score {window.eomm_composite_score} out of range [0, 1]")
    
    for op_result in window.operator_results:
        validate_operator_result(op_result)
    
    return True


# Example usage
if __name__ == "__main__":
    # Create sample operator result
    op_result = OperatorResult(
        operator_name="crosshair_lock",
        confidence=0.7,
        flags=[ManipulationFlags.AIM_RESISTANCE, ManipulationFlags.HITBOX_DRIFT],
        metrics={
            "on_target_frames": 12,
            "hit_efficiency": 0.17,
            "aim_resistance_score": 0.65
        },
        metadata={"frames_analyzed": 30}
    )
    
    # Validate
    validate_operator_result(op_result)
    print("✓ OperatorResult valid")
    
    # Create sample telemetry window
    window = TelemetryWindow(
        window_start_epoch=1701475200.0,
        window_end_epoch=1701475201.0,
        window_duration_ms=1000,
        operator_results=[op_result],
        eomm_composite_score=0.7,
        eomm_flags=[ManipulationFlags.AIM_RESISTANCE, ManipulationFlags.HITBOX_DRIFT],
        session_id="test_session_001",
        frame_count=30,
        metadata={"test": True}
    )
    
    # Validate
    validate_telemetry_window(window)
    print("✓ TelemetryWindow valid")
    
    # Serialize
    import json
    print("\nJSON output:")
    print(json.dumps(window.to_dict(), indent=2))
