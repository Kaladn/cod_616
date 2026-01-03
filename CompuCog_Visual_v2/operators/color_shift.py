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
Gaming Domain: Color Shift Detector Operator

Core Loop Stage: APPLY (Detector type)

Purpose:
  Detects palette distribution changes across frame sequence.
  These can indicate:
    - EOMM "fogging" (reduced visibility during skill-based play)
    - Dynamic resolution scaling (visual degradation)
    - Color grading manipulation (mood/performance effects)

Inputs:
  - FrameSequence (1-second window of ARC grids)
  - Config (histogram bins, EMD threshold)

Outputs:
  - VideoOpResult with confidence score and features
  - None if color distribution is stable

Core Loop Compliance:
  - Pure detector (no state modification)
  - Deterministic
  - Fast (< 10ms for 30 frames)
  - Explainable (returns histogram shifts and EMD score)
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


class ColorShiftDetector:
    """
    CORE LOOP STAGE: APPLY (Detector)
    
    Detects palette distribution changes using histogram comparison.
    """
    
    def __init__(self, config: dict):
        self.name = "color_shift"
        self.config = config
        
        # Extract operator-specific config
        op_config = config.get("operators", {}).get("color_shift", {})
        self.histogram_bins = op_config.get("histogram_bins", 10)
        self.emd_threshold = op_config.get("emd_threshold", 0.3)
        
        print(f"[+] ColorShiftDetector initialized")
        print(f"    Histogram bins: {self.histogram_bins}")
        print(f"    EMD threshold: {self.emd_threshold}")
    
    def _compute_histogram(self, grid: List[List[int]]) -> List[float]:
        """Compute normalized histogram of palette values"""
        counts = [0] * self.histogram_bins
        total = 0
        
        for row in grid:
            for val in row:
                bin_idx = min(val, self.histogram_bins - 1)
                counts[bin_idx] += 1
                total += 1
        
        # Normalize
        if total > 0:
            return [c / total for c in counts]
        else:
            return [0.0] * self.histogram_bins
    
    def _earth_movers_distance(self, hist1: List[float], hist2: List[float]) -> float:
        """
        Compute Earth Mover's Distance (Wasserstein) between two histograms.
        Simplified 1D implementation.
        """
        cumsum1 = 0.0
        cumsum2 = 0.0
        emd = 0.0
        
        for i in range(len(hist1)):
            cumsum1 += hist1[i]
            cumsum2 += hist2[i]
            emd += abs(cumsum1 - cumsum2)
        
        return emd
    
    def analyze(self, seq: FrameSequence) -> Optional[VideoOpResult]:
        """
        APPLY stage: Detect color distribution shifts in sequence.
        
        Returns:
          VideoOpResult if significant shift detected, else None
        """
        if len(seq.frames) < 10:
            return None
        
        # Compute histograms for all frames
        histograms = []
        for frame in seq.frames:
            hist = self._compute_histogram(frame.grid)
            histograms.append(hist)
        
        # Compare first half vs second half (detect temporal shift)
        mid_point = len(histograms) // 2
        first_half_avg = [0.0] * self.histogram_bins
        second_half_avg = [0.0] * self.histogram_bins
        
        for i in range(mid_point):
            for j in range(self.histogram_bins):
                first_half_avg[j] += histograms[i][j]
        
        for i in range(mid_point, len(histograms)):
            for j in range(self.histogram_bins):
                second_half_avg[j] += histograms[i][j]
        
        # Normalize averages
        first_half_avg = [v / mid_point for v in first_half_avg]
        second_half_avg = [v / (len(histograms) - mid_point) for v in second_half_avg]
        
        # Compute EMD between halves
        emd_score = self._earth_movers_distance(first_half_avg, second_half_avg)
        
        # Also track frame-to-frame EMD variance
        frame_emds = []
        for i in range(1, len(histograms)):
            emd = self._earth_movers_distance(histograms[i-1], histograms[i])
            frame_emds.append(emd)
        
        mean_frame_emd = sum(frame_emds) / len(frame_emds) if frame_emds else 0.0
        max_frame_emd = max(frame_emds) if frame_emds else 0.0
        
        # EVALUATE stage (internal): Check if shift is suspicious
        if emd_score > self.emd_threshold:
            confidence = min(1.0, emd_score / self.emd_threshold)
            
            return VideoOpResult(
                op_name=self.name,
                config={
                    "histogram_bins": self.histogram_bins,
                    "emd_threshold": self.emd_threshold
                },
                confidence=confidence,
                features={
                    "temporal_emd": emd_score,
                    "mean_frame_emd": mean_frame_emd,
                    "max_frame_emd": max_frame_emd,
                    "first_half_hist": first_half_avg,
                    "second_half_hist": second_half_avg,
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
    
    print("[*] Testing ColorShiftDetector with synthetic data...\n")
    
    detector = ColorShiftDetector(config)
    
    # Create synthetic frame sequence with color shift
    frames = []
    base_time = time.time()
    
    for i in range(30):
        grid = [[0 for _ in range(32)] for _ in range(32)]
        
        # First half: bright palette (values 6-9)
        # Second half: dark palette (values 0-3) - simulating EOMM fogging
        if i < 15:
            base_val = 7
        else:
            base_val = 2
        
        for y in range(32):
            for x in range(32):
                # Add some variation
                variation = (x + y) % 3 - 1
                grid[y][x] = max(0, min(9, base_val + variation))
        
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
        print(f"[DETECTION] Suspicious color shift detected!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Temporal EMD: {result.features['temporal_emd']:.4f}")
        print(f"  Mean frame EMD: {result.features['mean_frame_emd']:.4f}")
        print(f"  First half histogram: {[f'{v:.3f}' for v in result.features['first_half_hist']]}")
        print(f"  Second half histogram: {[f'{v:.3f}' for v in result.features['second_half_hist']]}")
    else:
        print(f"[OK] Color distribution appears stable")
