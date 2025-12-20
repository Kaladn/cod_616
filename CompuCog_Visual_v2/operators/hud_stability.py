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
║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

"""
Gaming Domain: HUD Stability Detector Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects unexpected changes in HUD regions that should be static.
  May indicate:
    - Cheating overlays
    - UI injection
    - Screen tampering

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (HUD region definitions, thresholds)

Outputs:
  - VideoOpResult with confidence score and features
  - None if HUD is stable

Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns which HUD regions changed)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))
from frame_to_grid import FrameGrid


@dataclass
class FrameSequence:
    """Contiguous sequence of frames for temporal analysis"""
    frames: List[FrameGrid]
    t_start: float
    t_end: float
    src: str


@dataclass
class VideoOpResult:
    """Result of a video operator's pattern detection"""
    op_name: str
    config: Dict
    confidence: float
    features: Dict


class HUDStabilityDetector:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Monitors HUD regions for unexpected changes.
    """
    
    def __init__(self, config: dict):
        self.name = "hud_stability"
        self.config = config
        
        # Extract operator-specific config
        op_config = config.get("operators", {}).get("hud_stability", {})
        self.hud_regions = op_config.get("hud_regions", {})
        self.max_change_ratio = op_config.get("max_change_ratio", 0.02)
        
        print(f"[+] HUDStabilityDetector initialized")
        print(f"    HUD regions: {list(self.hud_regions.keys())}")
        print(f"    Max change ratio: {self.max_change_ratio}")
    
    def _get_region_cells(self, h: int, w: int, region_def: dict) -> List[tuple]:
        """Convert region definition to cell coordinates"""
        cells = []
        
        if "y" in region_def and "height" in region_def:
            # Horizontal bar (top_bar, bottom_bar)
            y_start = int(region_def["y"] * h)
            y_end = int((region_def["y"] + region_def["height"]) * h)
            for y in range(y_start, min(y_end, h)):
                for x in range(w):
                    cells.append((y, x))
        
        elif "x" in region_def and "y" in region_def and "width" in region_def and "height" in region_def:
            # Rectangular region (minimap, etc.)
            y_start = int(region_def["y"] * h)
            y_end = int((region_def["y"] + region_def["height"]) * h)
            x_start = int(region_def["x"] * w)
            x_end = int((region_def["x"] + region_def["width"]) * w)
            
            for y in range(y_start, min(y_end, h)):
                for x in range(x_start, min(x_end, w)):
                    cells.append((y, x))
        
        return cells
    
    def analyze(self, seq: FrameSequence) -> Optional[VideoOpResult]:
        """
        APPLY stage: Detect HUD instability in sequence.
        
        Returns:
          VideoOpResult if HUD is unstable, else None
        """
        if len(seq.frames) < 2:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        
        # Check each HUD region
        region_instabilities = {}
        max_instability = 0.0
        
        for region_name, region_def in self.hud_regions.items():
            cells = self._get_region_cells(h, w, region_def)
            
            if not cells:
                continue
            
            # Count changes in this region across frames
            changes = 0
            total_comparisons = 0
            
            for i in range(1, len(seq.frames)):
                prev_grid = seq.frames[i - 1].grid
                curr_grid = seq.frames[i].grid
                
                for y, x in cells:
                    if prev_grid[y][x] != curr_grid[y][x]:
                        changes += 1
                    total_comparisons += 1
            
            if total_comparisons > 0:
                change_ratio = changes / total_comparisons
                region_instabilities[region_name] = change_ratio
                max_instability = max(max_instability, change_ratio)
        
        # EVALUATE stage (internal): Check if instability is significant
        if max_instability > self.max_change_ratio:
            # Compute confidence (0-1)
            confidence = min(1.0, max_instability / (self.max_change_ratio * 5))
            
            return VideoOpResult(
                op_name=self.name,
                config={
                    "hud_regions": list(self.hud_regions.keys()),
                    "max_change_ratio": self.max_change_ratio
                },
                confidence=confidence,
                features={
                    "max_instability": max_instability,
                    "region_instabilities": region_instabilities,
                    "unstable_regions": [r for r, ratio in region_instabilities.items() if ratio > self.max_change_ratio],
                    "frames_analyzed": len(seq.frames)
                }
            )
        
        return None


# Test harness
if __name__ == "__main__":
    import time
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("[*] Testing HUDStabilityDetector with synthetic data...\n")
    
    detector = HUDStabilityDetector(config)
    
    # Create synthetic frame sequence with unstable HUD
    frames = []
    base_time = time.time()
    
    for i in range(30):
        # Create grid with stable HUD initially
        grid = [[1 for _ in range(32)] for _ in range(32)]
        
        # Top bar (HUD) - should be stable
        for y in range(3):  # Top 10% of grid
            for x in range(32):
                grid[y][x] = 3
        
        # Add instability in top bar after frame 15
        if i > 15:
            for y in range(3):
                for x in range(10):
                    grid[y][x] = 7  # Changed color
        
        frame = FrameGrid(
            frame_id=i,
            t_sec=base_time + i * 0.033,
            grid=grid,
            source="Test",
            capture_region="full",
            h=32,
            w=32
        )
        frames.append(frame)
    
    # Analyze sequence
    seq = FrameSequence(
        frames=frames,
        t_start=frames[0].t_sec,
        t_end=frames[-1].t_sec,
        src="Test"
    )
    
    result = detector.analyze(seq)
    
    if result:
        print(f"[DETECTION] HUD instability detected!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Features: {result.features}")
    else:
        print(f"[OK] HUD is stable")
