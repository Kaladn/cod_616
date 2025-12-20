"""
Gaming Domain: Hit Registration Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects damage output manipulation by tracking hit marker ghosting.
  Measures:
    - Hit markers vs damage confirmation
    - Shots-to-kill observed vs expected
    - Hit marker clustering (burst analysis)
  
  EOMM Manipulation Vector: DAMAGE OUTPUT SUPPRESSION
    - Hit markers without damage (ghosting)
    - Elevated STK (shots-to-kill) vs weapon baseline
    - Damage scalar reduction

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (hit marker signatures, damage indicators)

Outputs:
  - VideoOpResult with:
    - hit_marker_count: Number of hit marker flashes detected
    - damage_confirm_count: Number of damage confirmations (health drops, flinch)
    - ghost_ratio: hit_markers without confirmation / total_hit_markers
    - burst_pattern: Hit marker timing within bursts
    
Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns frame-level hit marker data)
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


class HitRegistrationOperator:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Tracks hit markers vs damage confirmation to detect output suppression.
    """
    
    def __init__(self, config_path: str):
        self.name = "hit_registration"
        
        # Load config from YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        op_config = self.config.get("operators", {}).get("hit_registration", {})
        
        # Hit marker detection (white flash, usually center screen)
        self.hit_marker_palette = op_config.get("hit_marker_palette", 9)
        self.hit_marker_region_size = op_config.get("hit_marker_region_size", 0.15)  # 15% center
        
        # Damage confirmation indicators
        # - Red/orange flash (blood splatter) - palette 7-8
        # - Enemy flinch (sudden palette shift in enemy region)
        self.blood_palette_min = op_config.get("blood_palette_min", 7)
        self.blood_palette_max = op_config.get("blood_palette_max", 8)
        
        # Kill confirmation (specific UI element or screen transition)
        self.kill_confirm_palette = op_config.get("kill_confirm_palette", 9)
        
        # Ghosting thresholds
        self.ghost_threshold = op_config.get("ghost_threshold", 0.3)  # 30% ghost rate = anomaly
        
        print(f"[+] HitRegistrationOperator initialized")
        print(f"    Hit marker palette: {self.hit_marker_palette}")
        print(f"    Blood palette range: {self.blood_palette_min}-{self.blood_palette_max}")
        print(f"    Ghost threshold: {self.ghost_threshold * 100}%")
    
    def _get_center_cells(self, h: int, w: int, radius_pct: float) -> List[tuple]:
        """Get cells in center region of screen"""
        center_y = h // 2
        center_x = w // 2
        radius = int(min(h, w) * radius_pct)
        
        cells = []
        for y in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                dy = y - center_y
                dx = x - center_x
                if (dx * dx + dy * dy) <= (radius * radius):
                    cells.append((y, x))
        
        return cells
    
    def _detect_hit_marker(self, grid: List[List[int]], cells: List[tuple]) -> bool:
        """Detect hit marker flash in center region"""
        hit_pixels = 0
        total_pixels = len(cells)
        
        for y, x in cells:
            if grid[y][x] == self.hit_marker_palette:
                hit_pixels += 1
        
        # Hit marker is bright and distinct - even 5% coverage is signal
        return (hit_pixels / total_pixels) > 0.05 if total_pixels > 0 else False
    
    def _detect_blood_splatter(self, grid: List[List[int]], h: int, w: int) -> bool:
        """
        Detect blood splatter (damage confirmation).
        Blood appears as red/orange flash, typically near center but can be anywhere.
        """
        blood_pixels = 0
        total_pixels = h * w
        
        for y in range(h):
            for x in range(w):
                val = grid[y][x]
                if self.blood_palette_min <= val <= self.blood_palette_max:
                    blood_pixels += 1
        
        # Blood is more diffuse than hit marker - require 2% of screen
        return (blood_pixels / total_pixels) > 0.02 if total_pixels > 0 else False
    
    def _detect_enemy_flinch(self, prev_grid: List[List[int]], curr_grid: List[List[int]], 
                            h: int, w: int) -> bool:
        """
        Detect enemy flinch animation (sudden change in enemy region).
        This is a proxy for damage confirmation.
        """
        # Look for sudden palette shifts in outer regions (where enemies typically are)
        # Compare frame-to-frame difference in non-center regions
        
        changes = 0
        total_cells = 0
        
        center_y = h // 2
        center_x = w // 2
        center_radius = int(min(h, w) * 0.2)  # Exclude center 20%
        
        for y in range(h):
            for x in range(w):
                dy = y - center_y
                dx = x - center_x
                if (dx * dx + dy * dy) > (center_radius * center_radius):
                    # Outer region
                    if prev_grid[y][x] != curr_grid[y][x]:
                        changes += 1
                    total_cells += 1
        
        # Flinch = sudden 10%+ change in outer regions
        change_ratio = changes / total_cells if total_cells > 0 else 0
        return change_ratio > 0.1
    
    def analyze(self, seq: FrameSequence) -> Optional[OperatorResult]:
        """
        APPLY stage: Detect hit registration anomalies.
        
        Returns:
          OperatorResult with hit registration metrics and manipulation flags
        """
        if len(seq.frames) < 5:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        hit_marker_cells = self._get_center_cells(h, w, self.hit_marker_region_size)
        
        # Track hit markers and damage confirmations
        hit_marker_frames = []
        damage_confirm_frames = []
        
        for i, frame in enumerate(seq.frames):
            # Detect hit marker
            has_hit_marker = self._detect_hit_marker(frame.grid, hit_marker_cells)
            hit_marker_frames.append(has_hit_marker)
            
            # Detect damage confirmation (blood or flinch)
            has_blood = self._detect_blood_splatter(frame.grid, h, w)
            has_flinch = False
            if i > 0:
                has_flinch = self._detect_enemy_flinch(
                    seq.frames[i - 1].grid, frame.grid, h, w
                )
            
            damage_confirm = has_blood or has_flinch
            damage_confirm_frames.append(damage_confirm)
        
        # Compute metrics
        hit_marker_count = sum(hit_marker_frames)
        damage_confirm_count = sum(damage_confirm_frames)
        
        if hit_marker_count == 0:
            return None  # No hits in this window
        
        # Ghost detection: hit markers without subsequent damage confirmation
        # Allow 1-2 frame delay for blood/flinch to appear
        ghosted_hits = 0
        for i in range(len(hit_marker_frames)):
            if hit_marker_frames[i]:
                # Check next 2-3 frames for damage confirmation
                confirmed = False
                for j in range(i, min(i + 3, len(damage_confirm_frames))):
                    if damage_confirm_frames[j]:
                        confirmed = True
                        break
                if not confirmed:
                    ghosted_hits += 1
        
        ghost_ratio = ghosted_hits / hit_marker_count
        
        # Confidence: ghost ratio above threshold indicates manipulation
        confidence = 0.0
        if ghost_ratio > self.ghost_threshold:
            confidence = min(1.0, (ghost_ratio - self.ghost_threshold) / (1.0 - self.ghost_threshold))
        
        # Burst analysis: hit markers clustered together (full-auto fire)
        hit_marker_indices = [i for i, x in enumerate(hit_marker_frames) if x]
        burst_gaps = []
        if len(hit_marker_indices) > 1:
            for i in range(1, len(hit_marker_indices)):
                gap = hit_marker_indices[i] - hit_marker_indices[i - 1]
                burst_gaps.append(gap)
        
        avg_burst_gap = sum(burst_gaps) / len(burst_gaps) if burst_gaps else 0
        
        # Determine manipulation flags
        flags = []
        if ghost_ratio > self.ghost_threshold:
            flags.append(ManipulationFlags.GHOST_HITS)
        if ghost_ratio > 0.5:  # Severe ghosting
            flags.append(ManipulationFlags.DAMAGE_SUPPRESSION)
        
        return OperatorResult(
            operator_name=self.name,
            confidence=min(1.0, confidence),
            flags=flags,
            metrics={
                "hit_marker_count": hit_marker_count,
                "ghosted_hits": ghosted_hits,
                "ghost_ratio": ghost_ratio,
                "avg_burst_gap_frames": avg_burst_gap
            },
            metadata={
                "frames_analyzed": len(seq.frames),
                "damage_confirm_count": damage_confirm_count
            }
        )
