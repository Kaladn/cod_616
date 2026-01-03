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
Gaming Domain: Crosshair Motion Analyzer Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Analyzes motion patterns in center region (where crosshair typically is).
  Detects:
    - Aim-assist (unnatural smoothness)
    - Snap-to-target signatures
    - Mechanical aiming patterns

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (center region definition, thresholds)

Outputs:
  - VideoOpResult with confidence score and features
  - None if motion is normal

Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns motion metrics)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import sys
from pathlib import Path
import math

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


class CrosshairMotionAnalyzer:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Analyzes center region motion for aim-assist signatures.
    """
    
    def __init__(self, config: dict):
        self.name = "crosshair_motion"
        self.config = config
        
        # Extract operator-specific config
        op_config = config.get("operators", {}).get("crosshair_motion", {})
        self.center_region = op_config.get("center_region", {"x": 0.35, "y": 0.35, "width": 0.30, "height": 0.30})
        self.smoothness_threshold = op_config.get("smoothness_threshold", 0.95)
        self.snap_threshold = op_config.get("snap_threshold", 5)
        
        print(f"[+] CrosshairMotionAnalyzer initialized")
        print(f"    Center region: {self.center_region}")
        print(f"    Smoothness threshold: {self.smoothness_threshold}")
        print(f"    Snap threshold: {self.snap_threshold}")
    
    def _get_center_cells(self, h: int, w: int) -> List[tuple]:
        """Get cells in center region"""
        cells = []
        y_start = int(self.center_region["y"] * h)
        y_end = int((self.center_region["y"] + self.center_region["height"]) * h)
        x_start = int(self.center_region["x"] * w)
        x_end = int((self.center_region["x"] + self.center_region["width"]) * w)
        
        for y in range(y_start, min(y_end, h)):
            for x in range(x_start, min(x_end, w)):
                cells.append((y, x))
        
        return cells
    
    def _compute_center_of_mass(self, grid: List[List[int]], cells: List[tuple]) -> tuple:
        """Compute center of mass of changes in region"""
        total_weight = 0
        weighted_y = 0
        weighted_x = 0
        
        for y, x in cells:
            weight = grid[y][x]
            total_weight += weight
            weighted_y += y * weight
            weighted_x += x * weight
        
        if total_weight > 0:
            return (weighted_y / total_weight, weighted_x / total_weight)
        else:
            return (len(cells) // 2, len(cells) // 2)
    
    def analyze(self, seq: FrameSequence) -> Optional[VideoOpResult]:
        """
        APPLY stage: Analyze crosshair motion in sequence.
        
        Returns:
          VideoOpResult if suspicious motion detected, else None
        """
        if len(seq.frames) < 5:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        
        cells = self._get_center_cells(h, w)
        
        # Track center of mass over time
        centroids = []
        for frame in seq.frames:
            centroid = self._compute_center_of_mass(frame.grid, cells)
            centroids.append(centroid)
        
        # Compute motion deltas
        deltas = []
        for i in range(1, len(centroids)):
            dy = centroids[i][0] - centroids[i-1][0]
            dx = centroids[i][1] - centroids[i-1][1]
            magnitude = math.sqrt(dy**2 + dx**2)
            deltas.append(magnitude)
        
        if not deltas:
            return None
        
        # Compute smoothness (variance of deltas)
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta)**2 for d in deltas) / len(deltas)
        smoothness = 1.0 / (1.0 + variance) if variance > 0 else 1.0
        
        # Detect snaps (large sudden changes)
        snaps = [d for d in deltas if d > self.snap_threshold]
        snap_score = len(snaps) / len(deltas) if deltas else 0.0
        
        # Compute stutter (rapid direction changes)
        direction_changes = 0
        for i in range(1, len(deltas)):
            if deltas[i] > 0.5 and deltas[i-1] > 0.5:
                # Both moving, check if direction changed significantly
                if abs(deltas[i] - deltas[i-1]) > 1.0:
                    direction_changes += 1
        
        stutter_score = direction_changes / max(1, len(deltas) - 1)
        
        # EVALUATE stage (internal): Check if motion is suspicious
        suspicious = False
        confidence = 0.0
        
        # Aim-assist: too smooth (unnatural)
        if smoothness > self.smoothness_threshold:
            suspicious = True
            confidence = max(confidence, (smoothness - self.smoothness_threshold) / (1.0 - self.smoothness_threshold))
        
        # Snap-to-target: sudden large movements
        if snap_score > 0.1:  # More than 10% snaps
            suspicious = True
            confidence = max(confidence, min(1.0, snap_score * 5))
        
        if suspicious:
            return VideoOpResult(
                op_name=self.name,
                config={
                    "center_region": self.center_region,
                    "smoothness_threshold": self.smoothness_threshold,
                    "snap_threshold": self.snap_threshold
                },
                confidence=confidence,
                features={
                    "smoothness": smoothness,
                    "snap_score": snap_score,
                    "stutter_score": stutter_score,
                    "mean_motion": mean_delta,
                    "snap_count": len(snaps),
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
    
    print("[*] Testing CrosshairMotionAnalyzer with synthetic data...\n")
    
    analyzer = CrosshairMotionAnalyzer(config)
    
    # Create synthetic frame sequence with suspicious motion
    frames = []
    base_time = time.time()
    
    for i in range(30):
        # Create grid with moving center-of-mass
        grid = [[1 for _ in range(32)] for _ in range(32)]
        
        # Simulate unnaturally smooth aim-assist motion
        center_y = 16 + int(5 * math.sin(i * 0.1))  # Perfectly smooth sine wave
        center_x = 16 + int(5 * math.cos(i * 0.1))
        
        # Bright spot at center (simulating crosshair)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y = min(31, max(0, center_y + dy))
                x = min(31, max(0, center_x + dx))
                grid[y][x] = 8
        
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
    
    result = analyzer.analyze(seq)
    
    if result:
        print(f"[DETECTION] Suspicious crosshair motion detected!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Features: {result.features}")
    else:
        print(f"[OK] Motion appears normal")
