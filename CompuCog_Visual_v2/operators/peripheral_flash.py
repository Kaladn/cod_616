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
Gaming Domain: Peripheral Flash Detector Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects rapid brightness changes at screen edges.
  These can indicate:
    - Stealth aim-magnetism cues (edge flashes guide micro-corrections)
    - EOMM subliminal directional hints
    - Attention manipulation patterns

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (edge width, flash threshold)

Outputs:
  - VideoOpResult with confidence score and features
  - None if peripheral activity is normal

Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns flash locations and metrics)
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


class PeripheralFlashDetector:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Detects rapid brightness changes in outer edges of screen.
    """
    
    def __init__(self, config: dict):
        self.name = "peripheral_flash"
        self.config = config
        
        # Extract operator-specific config
        op_config = config.get("operators", {}).get("peripheral_flash", {})
        self.edge_width = op_config.get("edge_width", 0.15)
        self.flash_threshold = op_config.get("flash_threshold", 3)
        self.alert_ratio = op_config.get("alert_ratio", 0.03)
        
        print(f"[+] PeripheralFlashDetector initialized")
        print(f"    Edge width: {self.edge_width}")
        print(f"    Flash threshold: {self.flash_threshold} palette units")
        print(f"    Alert ratio: {self.alert_ratio}")
    
    def _get_edge_cells(self, h: int, w: int) -> Dict[str, List[tuple]]:
        """Get cells in each edge region"""
        edge_h = int(self.edge_width * h)
        edge_w = int(self.edge_width * w)
        
        edges = {
            "top": [],
            "bottom": [],
            "left": [],
            "right": []
        }
        
        # Top edge
        for y in range(edge_h):
            for x in range(w):
                edges["top"].append((y, x))
        
        # Bottom edge
        for y in range(h - edge_h, h):
            for x in range(w):
                edges["bottom"].append((y, x))
        
        # Left edge (excluding corners already in top/bottom)
        for y in range(edge_h, h - edge_h):
            for x in range(edge_w):
                edges["left"].append((y, x))
        
        # Right edge (excluding corners already in top/bottom)
        for y in range(edge_h, h - edge_h):
            for x in range(w - edge_w, w):
                edges["right"].append((y, x))
        
        return edges
    
    def analyze(self, seq: FrameSequence) -> Optional[VideoOpResult]:
        """
        APPLY stage: Detect peripheral flashes in sequence.
        
        Returns:
          VideoOpResult if suspicious flashes detected, else None
        """
        if len(seq.frames) < 2:
            return None
        
        h = seq.frames[0].h
        w = seq.frames[0].w
        
        edges = self._get_edge_cells(h, w)
        
        # Track flashes per edge
        edge_flash_counts = {edge: 0 for edge in edges.keys()}
        edge_total_comparisons = {edge: 0 for edge in edges.keys()}
        
        # Compare consecutive frames
        for i in range(1, len(seq.frames)):
            prev_grid = seq.frames[i-1].grid
            curr_grid = seq.frames[i].grid
            
            for edge_name, cells in edges.items():
                for y, x in cells:
                    delta = abs(curr_grid[y][x] - prev_grid[y][x])
                    if delta >= self.flash_threshold:
                        edge_flash_counts[edge_name] += 1
                    edge_total_comparisons[edge_name] += 1
        
        # Compute flash ratios per edge
        edge_flash_ratios = {}
        for edge_name in edges.keys():
            total = edge_total_comparisons[edge_name]
            if total > 0:
                edge_flash_ratios[edge_name] = edge_flash_counts[edge_name] / total
            else:
                edge_flash_ratios[edge_name] = 0.0
        
        # Overall flash metrics
        total_flash_count = sum(edge_flash_counts.values())
        total_comparisons = sum(edge_total_comparisons.values())
        overall_flash_ratio = total_flash_count / total_comparisons if total_comparisons > 0 else 0.0
        
        # Find max flash ratio across edges
        max_flash_ratio = max(edge_flash_ratios.values()) if edge_flash_ratios else 0.0
        max_flash_edge = max(edge_flash_ratios, key=edge_flash_ratios.get) if edge_flash_ratios else None
        
        # EVALUATE stage (internal): Check if flashes are suspicious
        if max_flash_ratio > self.alert_ratio:
            confidence = min(1.0, max_flash_ratio / self.alert_ratio)
            
            return VideoOpResult(
                op_name=self.name,
                config={
                    "edge_width": self.edge_width,
                    "flash_threshold": self.flash_threshold,
                    "alert_ratio": self.alert_ratio
                },
                confidence=confidence,
                features={
                    "overall_flash_ratio": overall_flash_ratio,
                    "max_flash_ratio": max_flash_ratio,
                    "max_flash_edge": max_flash_edge,
                    "edge_flash_ratios": edge_flash_ratios,
                    "edge_flash_counts": edge_flash_counts,
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
    
    print("[*] Testing PeripheralFlashDetector with synthetic data...\n")
    
    detector = PeripheralFlashDetector(config)
    
    # Create synthetic frame sequence with peripheral flashes
    frames = []
    base_time = time.time()
    
    for i in range(30):
        # Create base grid (dark)
        grid = [[1 for _ in range(32)] for _ in range(32)]
        
        # Every 5 frames, flash right edge (simulating aim-magnetism cue)
        if i % 5 == 0:
            for y in range(32):
                for x in range(28, 32):  # Right edge (outer 15%)
                    grid[y][x] = 8  # Bright flash
        
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
        print(f"[DETECTION] Suspicious peripheral flashes detected!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Max flash edge: {result.features['max_flash_edge']}")
        print(f"  Max flash ratio: {result.features['max_flash_ratio']:.4f}")
        print(f"  Edge ratios: {result.features['edge_flash_ratios']}")
    else:
        print(f"[OK] Peripheral activity appears normal")
