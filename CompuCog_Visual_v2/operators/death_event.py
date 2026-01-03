"""
Gaming Domain: Death Event Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects damage input manipulation by tracking time-to-death patterns.
  Measures:
    - Time from first hit to death
    - Incoming fire rate and clustering
    - Flinch intensity and frequency
  
  EOMM Manipulation Vector: DAMAGE INPUT AMPLIFICATION
    - Insta-melt (TTD below weapon baseline)
    - Enemy accuracy spikes
    - Damage multiplier on incoming fire

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (damage indicators, death screen signatures)

Outputs:
  - VideoOpResult with:
    - first_hit_frame: When first incoming damage detected
    - death_frame: When death screen transition detected
    - time_to_death: Frames between first hit and death
    - flinch_count: Number of flinch events
    - flinch_intensity: Average flinch magnitude
    
Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns death event timeline)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))
from frame_to_grid import FrameGrid

# Import TrueVision unified schema
sys.path.insert(0, str(Path(__file__).parent.parent))
from truevision_schema import OperatorResult, ManipulationFlags


@dataclass
class FrameSequence:
    """Contiguous sequence of frames for temporal analysis"""
    frames: List[FrameGrid]
    t_start: float
    t_end: float
    src: str


class DeathEventOperator:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Tracks incoming damage and death events to detect input manipulation.
    """
    
    def __init__(self, config_path: str):
        self.name = "death_event"
        
        # Load config from YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        op_config = self.config.get("operators", {}).get("death_event", {})
        
        # Incoming damage indicators
        # - Screen edge red vignette (damage indicator) - palette 7-8 at edges
        # - Hit flinch (sudden screen shake / palette shift)
        self.damage_indicator_palette_min = op_config.get("damage_indicator_palette_min", 7)
        self.damage_indicator_palette_max = op_config.get("damage_indicator_palette_max", 8)
        self.edge_width = op_config.get("edge_width", 0.1)  # 10% edge region
        
        # Death screen detection
        # - Grayscale transition (palette values collapse toward middle)
        # - Specific UI element (death screen overlay) - palette 9 in corners
        self.death_screen_palette = op_config.get("death_screen_palette", 9)
        
        # TTD (time-to-death) thresholds
        # Normal TTD: 0.5-1.5 seconds (15-45 frames at 30fps)
        # Insta-melt: <0.5 seconds (<15 frames)
        self.normal_ttd_min = op_config.get("normal_ttd_min", 15)  # frames
        self.instamelt_threshold = op_config.get("instamelt_threshold", 15)  # frames
        
        print(f"[+] DeathEventOperator initialized")
        print(f"    Damage indicator palette: {self.damage_indicator_palette_min}-{self.damage_indicator_palette_max}")
        print(f"    Normal TTD: >{self.normal_ttd_min} frames")
        print(f"    Insta-melt threshold: <{self.instamelt_threshold} frames")
    
    def _get_edge_cells(self, h: int, w: int) -> Dict[str, List[tuple]]:
        """Get cells in edge regions (top, bottom, left, right)"""
        edge_h = int(h * self.edge_width)
        edge_w = int(w * self.edge_width)
        
        edges = {
            "top": [(y, x) for y in range(edge_h) for x in range(w)],
            "bottom": [(y, x) for y in range(h - edge_h, h) for x in range(w)],
            "left": [(y, x) for y in range(h) for x in range(edge_w)],
            "right": [(y, x) for y in range(h) for x in range(w - edge_w, w)]
        }
        
        return edges
    
    def _detect_damage_indicator(self, grid: List[List[int]], edges: Dict[str, List[tuple]]) -> tuple:
        """
        Detect damage indicator (red vignette at screen edges).
        Returns (has_damage, intensity, direction)
        """
        edge_scores = {}
        
        for edge_name, cells in edges.items():
            damage_pixels = 0
            total_pixels = len(cells)
            
            for y, x in cells:
                val = grid[y][x]
                if self.damage_indicator_palette_min <= val <= self.damage_indicator_palette_max:
                    damage_pixels += 1
            
            edge_scores[edge_name] = damage_pixels / total_pixels if total_pixels > 0 else 0
        
        # Has damage if any edge > 5% red
        max_edge = max(edge_scores.values())
        has_damage = max_edge > 0.05
        
        # Direction is edge with highest score
        direction = max(edge_scores, key=edge_scores.get) if has_damage else None
        
        return has_damage, max_edge, direction
    
    def _detect_flinch(self, prev_grid: List[List[int]], curr_grid: List[List[int]], 
                      h: int, w: int) -> float:
        """
        Detect screen flinch (sudden palette shift across entire frame).
        Returns flinch magnitude (0-1).
        """
        total_change = 0
        total_cells = h * w
        
        for y in range(h):
            for x in range(w):
                total_change += abs(curr_grid[y][x] - prev_grid[y][x])
        
        # Normalize by max possible change
        avg_change = total_change / total_cells
        flinch_magnitude = avg_change / 9.0  # Max palette value is 9
        
        return flinch_magnitude
    
    def _detect_death_screen(self, grid: List[List[int]], h: int, w: int) -> bool:
        """
        Detect death screen transition.
        - Grayscale (palette variance drops)
        - Specific UI elements in corners
        """
        # Check for grayscale (low palette variance)
        palette_values = []
        for y in range(h):
            for x in range(w):
                palette_values.append(grid[y][x])
        
        # Compute variance
        mean = sum(palette_values) / len(palette_values)
        variance = sum((x - mean) ** 2 for x in palette_values) / len(palette_values)
        
        # Death screen typically has low variance (grayscale) and specific palette in corners
        is_grayscale = variance < 2.0  # Low color variance
        
        # Check corners for death screen UI (palette 9 typically)
        corner_size = int(min(h, w) * 0.1)
        corners = [
            (0, 0), (0, w - corner_size),
            (h - corner_size, 0), (h - corner_size, w - corner_size)
        ]
        
        corner_ui_pixels = 0
        for corner_y, corner_x in corners:
            for dy in range(corner_size):
                for dx in range(corner_size):
                    y = corner_y + dy
                    x = corner_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        if grid[y][x] == self.death_screen_palette:
                            corner_ui_pixels += 1
        
        has_death_ui = corner_ui_pixels > (corner_size * corner_size * 0.1)  # 10% of corners
        
        return is_grayscale or has_death_ui
    
    def analyze(self, seq: FrameSequence) -> Optional[OperatorResult]:
        """
        APPLY stage: Detect death event and compute TTD.
        
        Returns:
          OperatorResult with death event metrics and manipulation flags
        """
        if len(seq.frames) < 5:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        edges = self._get_edge_cells(h, w)
        
        # Track damage indicators, flinch, and death
        first_hit_frame = None
        death_frame = None
        flinch_events = []
        damage_events = []
        
        for i, frame in enumerate(seq.frames):
            # Detect incoming damage
            has_damage, intensity, direction = self._detect_damage_indicator(frame.grid, edges)
            
            if has_damage:
                damage_events.append({
                    "frame": i,
                    "intensity": intensity,
                    "direction": direction
                })
                if first_hit_frame is None:
                    first_hit_frame = i
            
            # Detect flinch
            if i > 0:
                flinch_mag = self._detect_flinch(seq.frames[i - 1].grid, frame.grid, h, w)
                if flinch_mag > 0.1:  # Significant flinch
                    flinch_events.append({
                        "frame": i,
                        "magnitude": flinch_mag
                    })
            
            # Detect death screen
            if self._detect_death_screen(frame.grid, h, w):
                death_frame = i
                break  # Stop once death detected
        
        # Compute metrics
        if first_hit_frame is None:
            return None  # No damage in this window
        
        time_to_death = None
        is_instamelt = False
        
        if death_frame is not None:
            time_to_death = death_frame - first_hit_frame
            is_instamelt = time_to_death < self.instamelt_threshold
        
        flinch_count = len(flinch_events)
        avg_flinch_intensity = sum(e["magnitude"] for e in flinch_events) / flinch_count if flinch_count > 0 else 0
        
        # Confidence: based on flinch patterns (TTD/insta-melt removed as unreliable)
        confidence = 0.0
        if flinch_count > 10:  # Excessive flinch in 1 second
            confidence = min(1.0, flinch_count / 20.0)
        
        # Determine manipulation flags
        flags = []
        if flinch_count > 10:
            flags.append(ManipulationFlags.INCOMING_DAMAGE_SPIKE)
        
        return OperatorResult(
            operator_name=self.name,
            confidence=min(1.0, confidence),
            flags=flags,
            metrics={
                "time_to_death_frames": time_to_death,
                "flinch_count": flinch_count,
                "avg_flinch_intensity": avg_flinch_intensity,
                "damage_event_count": len(damage_events)
            },
            metadata={
                "frames_analyzed": len(seq.frames),
                "first_hit_frame": first_hit_frame,
                "death_frame": death_frame
            }
        )
