from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

class ManipulationFlags:
    AIM_RESISTANCE = "AIM_RESISTANCE"
    HITBOX_DRIFT = "HITBOX_DRIFT"

    @classmethod
    def all_flags(cls) -> List[str]:
        return [cls.AIM_RESISTANCE, cls.HITBOX_DRIFT]

@dataclass
class OperatorResult:
    operator_name: str
    confidence: float
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TelemetryWindow:
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
