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
Gaming Domain: Flicker Detector Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects rapid brightness or color changes (flicker/pulse) that may indicate:
    - EOMM subliminal cues
    - Cheating overlays
    - Visual manipulation

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (thresholds, min_flips)

Outputs:
  - VideoOpResult with confidence score and features
  - None if no flicker detected

Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic (same input → same output)
  - Fast (< 10ms for 30 frames)
  - Explainable (returns which cells flickered, how many times)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add core to path
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
    confidence: float  # 0.0 to 1.0
    features: Dict


class FlickerDetector:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Detects rapid brightness/color oscillations in frame sequences.
    """
    
    def __init__(self, config: dict):
        self.name = "flicker_detector"
        self.config = config
        
        # Extract operator-specific config
        op_config = config.get("operators", {}).get("flicker_detector", {})
        self.flicker_threshold = op_config.get("threshold", 2)
        self.min_flips = op_config.get("min_flips", 3)
        self.alert_ratio = op_config.get("alert_ratio", 0.05)
        
        print(f"[+] FlickerDetector initialized")
        print(f"    Threshold: {self.flicker_threshold}")
        print(f"    Min flips: {self.min_flips}")
        print(f"    Alert ratio: {self.alert_ratio}")
    
    def analyze(self, seq: FrameSequence) -> Optional[VideoOpResult]:
        """
        APPLY stage: Detect flicker patterns in sequence.
        
        Returns:
          VideoOpResult if significant flicker detected, else None
        """
        if len(seq.frames) < 3:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        
        # Track flips per cell
        flip_counts = [[0 for _ in range(w)] for _ in range(h)]
        
        # Count rapid changes for each cell across frames
        for i in range(1, len(seq.frames)):
            prev_grid = seq.frames[i - 1].grid
            curr_grid = seq.frames[i].grid
            
            for y in range(h):
                for x in range(w):
                    delta = abs(curr_grid[y][x] - prev_grid[y][x])
                    if delta >= self.flicker_threshold:
                        flip_counts[y][x] += 1
        
        # Count cells with significant flicker
        flickering_cells = 0
        total_cells = h * w
        max_flips = 0
        
        for y in range(h):
            for x in range(w):
                if flip_counts[y][x] >= self.min_flips:
                    flickering_cells += 1
                    max_flips = max(max_flips, flip_counts[y][x])
        
        flicker_ratio = flickering_cells / total_cells
        
        # EVALUATE stage (internal): Check if detection is significant
        if flicker_ratio > self.alert_ratio:
            # Compute confidence (0-1)
            confidence = min(1.0, flicker_ratio / (self.alert_ratio * 10))
            
            return VideoOpResult(
                op_name=self.name,
                config={
                    "flicker_threshold": self.flicker_threshold,
                    "min_flips": self.min_flips,
                    "alert_ratio": self.alert_ratio
                },
                confidence=confidence,
                features={
                    "flickering_cells": flickering_cells,
                    "flicker_ratio": flicker_ratio,
                    "total_cells": total_cells,
                    "max_flips": max_flips,
                    "frames_analyzed": len(seq.frames)
                }
            )
        
        return None


# Test harness
if __name__ == "__main__":
    import time
    import yaml
    from collections import deque
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with synthetic flickering grid
    print("[*] Testing FlickerDetector with synthetic data...\n")
    
    detector = FlickerDetector(config)
    
    # Create synthetic frame sequence with flicker
    frames = []
    base_time = time.time()
    
    for i in range(30):
        # Create grid with some flickering cells
        grid = [[0 for _ in range(32)] for _ in range(32)]
        
        # Add flicker in top-left quadrant (every other frame)
        if i % 2 == 0:
            for y in range(8):
                for x in range(8):
                    grid[y][x] = 5  # Bright
        else:
            for y in range(8):
                for x in range(8):
                    grid[y][x] = 0  # Dark
        
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
        print(f"[DETECTION] Flicker detected!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Features: {result.features}")
    else:
        print(f"[OK] No flicker detected")
