from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class CoreBlock:
    avg_intensity: float = 0.0
    peak_intensity: float = 0.0

@dataclass
class SectorDefinition:
    name: str
    weighted_avg_intensity: float = 0.0
    raw_count: int = 0

@dataclass
class DirectionalVector:
    angle_deg: float
    gradient_change: float = 0.0

@dataclass
class AnomalyMetrics:
    entropy: float = 0.0
    symmetry: float = 0.0
    flags: List[str] = field(default_factory=list)

@dataclass
class ScreenVectorState:
    """Minimal ScreenVectorState compatible shape used by operators.
    This is intentionally small — it allows integrating telemetry-only inputs
    into operator pipelines while the full SVE is implemented.
    """
    core_block: CoreBlock = field(default_factory=CoreBlock)
    sectors: Dict[str, SectorDefinition] = field(default_factory=dict)
    directional_vectors: Optional[List[DirectionalVector]] = None
    anomaly_metrics: Optional[AnomalyMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
