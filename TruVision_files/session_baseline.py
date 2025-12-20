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
║     File automatically watermarked on: 2025-12-02 00:00:00                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TrueVision v1.0.0 - Session Baseline Tracker

Purpose:
  Session-based baseline tracker for TTK/TTD/STK using Welford's online algorithm.
  Enables z-score anomaly detection without storing full sample history.
  
  No weapon database required - baselines adapt per gaming session.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class WeaponBaseline:
    """
    Online statistics tracker for a weapon using Welford's algorithm.
    Computes mean and variance incrementally without storing all samples.
    
    Tracks:
    - TTK (Time-To-Kill): Milliseconds from first hit to kill confirmation
    - TTD (Time-To-Death): Milliseconds from first incoming hit to death
    - STK (Shots-To-Kill): Number of hits required for kill
    """
    weapon_name: str = "global"  # v1.0: single global baseline, v2.0: per-weapon
    
    # TTK stats
    ttk_mean: float = 0.0
    ttk_m2: float = 0.0  # Sum of squared differences from mean (for variance)
    ttk_count: int = 0
    
    # TTD stats
    ttd_mean: float = 0.0
    ttd_m2: float = 0.0
    ttd_count: int = 0
    
    # STK stats
    stk_mean: float = 0.0
    stk_m2: float = 0.0
    stk_count: int = 0
    
    def add_ttk_sample(self, ttk_ms: float, outlier_threshold: float = 3.0) -> bool:
        """
        Add TTK sample using Welford's algorithm.
        
        Args:
          ttk_ms: Time-to-kill in milliseconds
          outlier_threshold: Z-score threshold for outlier rejection (default 3.0)
        
        Returns:
          True if sample accepted, False if rejected as outlier
        """
        # Outlier rejection (after warmup)
        if self.ttk_count >= 5:
            z_score = self.compute_ttk_zscore(ttk_ms)
            if abs(z_score) > outlier_threshold:
                return False  # Reject extreme outlier
        
        # Welford's algorithm
        self.ttk_count += 1
        delta = ttk_ms - self.ttk_mean
        self.ttk_mean += delta / self.ttk_count
        delta2 = ttk_ms - self.ttk_mean
        self.ttk_m2 += delta * delta2
        
        return True
    
    def add_ttd_sample(self, ttd_ms: float, outlier_threshold: float = 2.5) -> bool:
        """Add TTD sample (more permissive outlier threshold for death events)"""
        if self.ttd_count >= 3:
            z_score = self.compute_ttd_zscore(ttd_ms)
            if abs(z_score) > outlier_threshold:
                return False
        
        self.ttd_count += 1
        delta = ttd_ms - self.ttd_mean
        self.ttd_mean += delta / self.ttd_count
        delta2 = ttd_ms - self.ttd_mean
        self.ttd_m2 += delta * delta2
        
        return True
    
    def add_stk_sample(self, stk: int, outlier_threshold: float = 3.0) -> bool:
        """Add STK sample"""
        if self.stk_count >= 5:
            z_score = self.compute_stk_zscore(stk)
            if abs(z_score) > outlier_threshold:
                return False
        
        self.stk_count += 1
        delta = stk - self.stk_mean
        self.stk_mean += delta / self.stk_count
        delta2 = stk - self.stk_mean
        self.stk_m2 += delta * delta2
        
        return True
    
    def get_ttk_variance(self) -> float:
        """Get TTK variance"""
        if self.ttk_count < 2:
            return 0.0
        return self.ttk_m2 / (self.ttk_count - 1)
    
    def get_ttd_variance(self) -> float:
        """Get TTD variance"""
        if self.ttd_count < 2:
            return 0.0
        return self.ttd_m2 / (self.ttd_count - 1)
    
    def get_stk_variance(self) -> float:
        """Get STK variance"""
        if self.stk_count < 2:
            return 0.0
        return self.stk_m2 / (self.stk_count - 1)
    
    def compute_ttk_zscore(self, ttk_ms: float) -> float:
        """
        Compute z-score for TTK value.
        Z-score = (value - mean) / std_dev
        
        Used for anomaly detection:
        - z > 2.0: TTK above normal (damage suppression)
        - z < -2.0: TTK below normal (damage boost)
        """
        if self.ttk_count < 2:
            return 0.0
        
        variance = self.get_ttk_variance()
        if variance == 0:
            return 0.0
        
        std_dev = math.sqrt(variance)
        return (ttk_ms - self.ttk_mean) / std_dev
    
    def compute_ttd_zscore(self, ttd_ms: float) -> float:
        """
        Compute z-score for TTD value.
        
        Used for insta-melt detection:
        - z < -2.5: TTD below normal (incoming damage spike)
        """
        if self.ttd_count < 2:
            return 0.0
        
        variance = self.get_ttd_variance()
        if variance == 0:
            return 0.0
        
        std_dev = math.sqrt(variance)
        return (ttd_ms - self.ttd_mean) / std_dev
    
    def compute_stk_zscore(self, stk: int) -> float:
        """Compute z-score for STK value"""
        if self.stk_count < 2:
            return 0.0
        
        variance = self.get_stk_variance()
        if variance == 0:
            return 0.0
        
        std_dev = math.sqrt(variance)
        return (stk - self.stk_mean) / std_dev
    
    def is_ttk_outlier(self, ttk_ms: float, threshold: float = 2.0) -> bool:
        """Check if TTK is outlier (>2σ = manipulation)"""
        z = self.compute_ttk_zscore(ttk_ms)
        return abs(z) > threshold
    
    def is_ttd_outlier(self, ttd_ms: float, threshold: float = 2.5) -> bool:
        """Check if TTD is outlier (<-2.5σ = insta-melt)"""
        z = self.compute_ttd_zscore(ttd_ms)
        return z < -threshold  # Only flag low TTD (insta-melt)
    
    def is_stk_outlier(self, stk: int, threshold: float = 2.0) -> bool:
        """Check if STK is outlier"""
        z = self.compute_stk_zscore(stk)
        return abs(z) > threshold
    
    def to_dict(self) -> Dict:
        """Export baseline stats for metadata"""
        return {
            "weapon_name": self.weapon_name,
            "ttk": {
                "mean_ms": self.ttk_mean,
                "std_ms": math.sqrt(self.get_ttk_variance()) if self.ttk_count >= 2 else 0.0,
                "samples": self.ttk_count
            },
            "ttd": {
                "mean_ms": self.ttd_mean,
                "std_ms": math.sqrt(self.get_ttd_variance()) if self.ttd_count >= 2 else 0.0,
                "samples": self.ttd_count
            },
            "stk": {
                "mean": self.stk_mean,
                "std": math.sqrt(self.get_stk_variance()) if self.stk_count >= 2 else 0.0,
                "samples": self.stk_count
            }
        }


class SessionBaselineTracker:
    """
    Unified baseline tracker for gaming session.
    
    v1.0: Single global baseline for all weapons
    v2.0: Per-weapon baselines with weapon detection
    """
    
    def __init__(self, min_samples_for_warmup: int = 5):
        self.global_baseline = WeaponBaseline(weapon_name="global")
        self.min_samples = min_samples_for_warmup
        self.warmup_complete = False
    
    def add_kill_event(self, ttk_ms: float, stk: int) -> None:
        """Add kill event (player killed enemy)"""
        self.global_baseline.add_ttk_sample(ttk_ms)
        self.global_baseline.add_stk_sample(stk)
        
        # Check warmup completion
        if not self.warmup_complete:
            if self.global_baseline.ttk_count >= self.min_samples:
                self.warmup_complete = True
    
    def add_death_event(self, ttd_ms: float) -> None:
        """Add death event (player died)"""
        self.global_baseline.add_ttd_sample(ttd_ms)
    
    def get_baseline(self) -> WeaponBaseline:
        """Get global baseline"""
        return self.global_baseline
    
    def is_warmed_up(self) -> bool:
        """Check if baseline has enough samples"""
        return self.warmup_complete
    
    def to_dict(self) -> Dict:
        """Export for TelemetryWindow metadata"""
        return {
            "warmed_up": self.warmup_complete,
            "min_samples": self.min_samples,
            "baseline": self.global_baseline.to_dict()
        }


# Example usage
if __name__ == "__main__":
    tracker = SessionBaselineTracker(min_samples_for_warmup=5)
    
    # Simulate kill events
    print("Adding kill events...")
    for i in range(10):
        ttk = 1200 + (i * 100)  # Simulate TTK variance
        stk = 5 + (i % 2)       # 5-6 shots
        tracker.add_kill_event(ttk, stk)
        print(f"  Kill {i+1}: TTK={ttk}ms, STK={stk}")
    
    baseline = tracker.get_baseline()
    print(f"\nBaseline after 10 kills:")
    print(f"  TTK: mean={baseline.ttk_mean:.1f}ms, std={math.sqrt(baseline.get_ttk_variance()):.1f}ms")
    print(f"  STK: mean={baseline.stk_mean:.2f}, std={math.sqrt(baseline.get_stk_variance()):.2f}")
    
    # Test anomaly detection
    print("\nAnomaly detection:")
    test_ttk = 2500  # Very high TTK (damage suppression)
    z_score = baseline.compute_ttk_zscore(test_ttk)
    print(f"  TTK={test_ttk}ms -> z-score={z_score:.2f} (outlier={baseline.is_ttk_outlier(test_ttk)})")
    
    test_ttd = 300  # Very low TTD (insta-melt)
    tracker.add_death_event(300)
    z_score = baseline.compute_ttd_zscore(test_ttd)
    print(f"  TTD={test_ttd}ms -> z-score={z_score:.2f} (insta-melt={baseline.is_ttd_outlier(test_ttd)})")
