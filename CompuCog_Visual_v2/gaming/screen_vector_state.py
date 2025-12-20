from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple

@dataclass
class ScreenInfo:
    screen_width_px: int
    screen_height_px: int
    aspect_ratio: float

@dataclass
class GridConfig:
    cells_x: int
    cells_y: int
    cell_width_px: float
    cell_height_px: float

@dataclass
class CoreBlock:
    # core_cells: list of (row, col) tuples (numpy indexing: row, col)
    core_cells: List[Tuple[int, int]]
    core_bounds_px: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    avg_intensity: float
    motion_delta: float

@dataclass
class SectorDefinition:
    name: str  # "UP", "DOWN", "LEFT", "RIGHT"
    cell_indices: List[Tuple[int, int]]  # list of (row, col)
    pixel_bounds: Tuple[int, int, int, int]
    weighted_avg_intensity: float
    weighted_motion_delta: float

@dataclass
class DirectionalVector:
    direction: str
    gradient_change: float
    entropy: float

@dataclass
class AnomalyMetrics:
    global_entropy: float
    sector_entropies: Dict[str, float]
    horizontal_symmetry_score: float
    vertical_symmetry_score: float
    diagonal_symmetry_score: float
    anomaly_flags: List[str] = field(default_factory=list)

@dataclass
class ScreenVectorState:
    screen_info: ScreenInfo
    grid_config: GridConfig
    core_block: CoreBlock
    sectors: Dict[str, SectorDefinition]
    directional_vectors: List[DirectionalVector]
    anomaly_metrics: AnomalyMetrics
    weights_matrix: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        # Use asdict but ensure nested dataclasses are converted
        return asdict(self)
