"""
Gaming Domain: Crosshair Lock Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects aim manipulation by tracking crosshair-to-enemy intersection physics.
  Measures:
    - Crosshair on-target frames vs hit markers
    - Aim resistance (velocity changes near enemies)
    - Hitbox intersection anomalies
  
  EOMM Manipulation Vector: AIM SUPPRESSION
    - Hitbox shrinking (crosshair on target, no hit)
    - Aim drag/resistance (slowdown without magnetism)
    - Variable aim assist strength

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (crosshair region, enemy palette signatures)

Outputs:
  - VideoOpResult with:
    - on_target_frames: How many frames had enemy in crosshair
    - hit_marker_frames: How many frames showed hit marker flash
    - intersection_ratio: on_target / total_frames
    - hit_efficiency: hit_markers / on_target_frames (should be ~0.3-0.5 for good aim)
    - aim_resistance_score: Velocity changes near enemies
    
Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 15ms for 30 frames)
  - Explainable (returns frame-by-frame intersection data)
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


class CrosshairLockOperator:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Tracks crosshair-to-enemy physics to detect aim manipulation.
    """
    
    def __init__(self, config_path: str):
        self.name = "crosshair_lock"
        
        # Load config from YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        op_config = self.config.get("operators", {}).get("crosshair_lock", {})
        
        # Crosshair region (center % of screen)
        self.crosshair_radius = op_config.get("crosshair_radius", 0.05)  # 5% of screen
        
        # Enemy detection (palette values that indicate enemies vs environment)
        # Will need tuning per game - COD enemies are typically palette 6-9 range
        self.enemy_palette_min = op_config.get("enemy_palette_min", 5)
        self.enemy_palette_max = op_config.get("enemy_palette_max", 9)
        
        # Hit marker detection (white flash in center, typically palette 9)
        self.hit_marker_palette = op_config.get("hit_marker_palette", 9)
        
        # Aim resistance thresholds
        self.velocity_window = op_config.get("velocity_window", 5)  # frames
        
        print(f"[+] CrosshairLockOperator initialized")
        print(f"    Crosshair radius: {self.crosshair_radius * 100}%")
        print(f"    Enemy palette range: {self.enemy_palette_min}-{self.enemy_palette_max}")
        print(f"    Hit marker palette: {self.hit_marker_palette}")
    
    def _get_crosshair_cells(self, h: int, w: int) -> List[tuple]:
        """Get cell coordinates for crosshair region (center circle)"""
        center_y = h // 2
        center_x = w // 2
        radius = int(min(h, w) * self.crosshair_radius)
        
        cells = []
        for y in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                # Circular region
                dy = y - center_y
                dx = x - center_x
                if (dx * dx + dy * dy) <= (radius * radius):
                    cells.append((y, x))
        
        return cells
    
    def _detect_enemy_in_crosshair(self, grid: List[List[int]], cells: List[tuple]) -> bool:
        """Check if enemy palette signature is in crosshair region"""
        enemy_pixels = 0
        total_pixels = len(cells)
        
        for y, x in cells:
            val = grid[y][x]
            if self.enemy_palette_min <= val <= self.enemy_palette_max:
                enemy_pixels += 1
        
        # Require >20% of crosshair region to have enemy signature
        return (enemy_pixels / total_pixels) > 0.2 if total_pixels > 0 else False
    
    def _detect_hit_marker(self, grid: List[List[int]], cells: List[tuple]) -> bool:
        """Check if hit marker flash is visible (white flash in center)"""
        hit_pixels = 0
        total_pixels = len(cells)
        
        for y, x in cells:
            if grid[y][x] == self.hit_marker_palette:
                hit_pixels += 1
        
        # Hit marker is a brief flash - even 10% coverage is significant
        return (hit_pixels / total_pixels) > 0.1 if total_pixels > 0 else False
    
    def _compute_crosshair_motion(self, seq: FrameSequence) -> List[float]:
        """
        Compute frame-to-frame crosshair motion magnitude.
        Uses center of mass of high-contrast regions as proxy for crosshair movement.
        """
        h = seq.frames[0].h
        w = seq.frames[0].w
        center_region_size = int(min(h, w) * 0.3)  # 30% center region
        
        motions = []
        
        for i in range(1, len(seq.frames)):
            prev_grid = seq.frames[i - 1].grid
            curr_grid = seq.frames[i].grid
            
            # Compute center of mass shift (simple motion proxy)
            prev_com_y, prev_com_x = self._compute_center_of_mass(prev_grid, h, w)
            curr_com_y, curr_com_x = self._compute_center_of_mass(curr_grid, h, w)
            
            # Euclidean distance
            dy = curr_com_y - prev_com_y
            dx = curr_com_x - prev_com_x
            motion = (dy * dy + dx * dx) ** 0.5
            
            motions.append(motion)
        
        return motions
    
    def _compute_center_of_mass(self, grid: List[List[int]], h: int, w: int) -> tuple:
        """Compute center of mass of high-contrast regions (proxy for scene motion)"""
        total_weight = 0
        weighted_y = 0
        weighted_x = 0
        
        for y in range(h):
            for x in range(w):
                val = grid[y][x]
                # Use palette value as weight (higher = more salient)
                total_weight += val
                weighted_y += y * val
                weighted_x += x * val
        
        if total_weight > 0:
            return weighted_y / total_weight, weighted_x / total_weight
        return h // 2, w // 2
    
    def _compute_aim_resistance(self, motions: List[float], on_target_frames: List[bool]) -> float:
        """
        Detect aim resistance: slowdown when crosshair passes over enemies.
        Normal aim assist = slight slowdown.
        Manipulation = excessive slowdown or erratic behavior.
        """
        if len(motions) < 5 or sum(on_target_frames) < 3:
            return 0.0
        
        # Motion is frame-to-frame, so skip first frame
        # Compare velocity on-target vs off-target (align indices by skipping frame 0)
        on_target_velocities = [motions[i] for i in range(len(motions)) if on_target_frames[i+1]]
        off_target_velocities = [motions[i] for i in range(len(motions)) if not on_target_frames[i+1]]
        
        if not on_target_velocities or not off_target_velocities:
            return 0.0
        
        avg_on = sum(on_target_velocities) / len(on_target_velocities)
        avg_off = sum(off_target_velocities) / len(off_target_velocities)
        
        # Resistance score: ratio of slowdown
        # Normal aim assist: 0.7-0.9 (slight slowdown)
        # Manipulation: <0.5 (excessive drag) or >1.0 (no assist when expected)
        if avg_off > 0:
            resistance = avg_on / avg_off
            # Convert to anomaly score (0 = normal, 1 = manipulated)
            if resistance < 0.7:
                return (0.7 - resistance) / 0.7  # Excessive drag
            elif resistance > 1.0:
                return min(1.0, (resistance - 1.0))  # Assist removed
        
        return 0.0
    
    def analyze(self, seq: FrameSequence) -> Optional[OperatorResult]:
        """
        APPLY stage: Detect aim manipulation in sequence.
        
        Returns:
          OperatorResult with aim physics metrics and manipulation flags
        """
        if len(seq.frames) < 5:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        crosshair_cells = self._get_crosshair_cells(h, w)
        
        # Frame-by-frame analysis
        on_target_frames = []
        hit_marker_frames = []
        
        for frame in seq.frames:
            enemy_in_crosshair = self._detect_enemy_in_crosshair(frame.grid, crosshair_cells)
            hit_marker_visible = self._detect_hit_marker(frame.grid, crosshair_cells)
            
            on_target_frames.append(enemy_in_crosshair)
            hit_marker_frames.append(hit_marker_visible)
        
        # Compute metrics
        total_frames = len(seq.frames)
        on_target_count = sum(on_target_frames)
        hit_marker_count = sum(hit_marker_frames)
        
        intersection_ratio = on_target_count / total_frames
        hit_efficiency = hit_marker_count / on_target_count if on_target_count > 0 else 0.0
        
        # Compute aim resistance
        motions = self._compute_crosshair_motion(seq)
        aim_resistance_score = self._compute_aim_resistance(motions, on_target_frames)
        
        # Detect anomalies
        # Normal hit efficiency: 0.3-0.5 (30-50% of on-target frames result in hits)
        # Manipulation: <0.2 (hitbox shrink) or >0.6 (aim assist boost)
        hit_efficiency_anomaly = 0.0
        if on_target_count > 5:
            if hit_efficiency < 0.2:
                hit_efficiency_anomaly = (0.2 - hit_efficiency) / 0.2
            elif hit_efficiency > 0.6:
                hit_efficiency_anomaly = (hit_efficiency - 0.6) / 0.4
        
        # Overall confidence: max of resistance and efficiency anomalies
        confidence = max(aim_resistance_score, hit_efficiency_anomaly)
        
        # Only return result if we have meaningful data
        if on_target_count < 3:
            return None
        
        # Determine manipulation flags
        flags = []
        if aim_resistance_score > 0.3:
            flags.append(ManipulationFlags.AIM_RESISTANCE)
        if hit_efficiency_anomaly > 0.5:
            flags.append(ManipulationFlags.HITBOX_DRIFT)
        
        return OperatorResult(
            operator_name=self.name,
            confidence=min(1.0, confidence),  # Clamp to [0, 1]
            flags=flags,
            metrics={
                "on_target_frames": on_target_count,
                "hit_marker_frames": hit_marker_count,
                "intersection_ratio": intersection_ratio,
                "hit_efficiency": hit_efficiency,
                "aim_resistance_score": aim_resistance_score
            },
            metadata={
                "frames_analyzed": total_frames,
                "avg_motion": sum(motions) / len(motions) if motions else 0.0
            }
        )
