"""
ScreenResonanceState - Per-Frame Visual Feature Extraction

Pure math signal processing for screen motion analysis.
Extracts 20 visual features from 10×10 grid temporal patterns.

No ML. No YOLO. Just derivatives, EMAs, and spatial analysis.
"""

import numpy as np
from typing import Optional, Dict


class ScreenResonanceState:
    """
    Maintains temporal state for screen grid analysis.
    Computes 20 visual features per frame from motion patterns.
    """
    
    def __init__(self, 
                 grid_size: int = 10,
                 ema_alpha_fast: float = 0.3,
                 ema_alpha_slow: float = 0.1,
                 flash_threshold: float = 0.15,
                 static_threshold: float = 0.01):
        """
        Initialize resonance state tracker.
        
        Args:
            grid_size: Grid dimensions (assumed square, e.g., 10×10)
            ema_alpha_fast: Alpha for fast EMA (higher = more responsive)
            ema_alpha_slow: Alpha for slow EMA (lower = smoother)
            flash_threshold: Delta threshold for flash detection
            static_threshold: Delta threshold for static cell detection
        """
        self.grid_size = grid_size
        self.alpha_fast = ema_alpha_fast
        self.alpha_slow = ema_alpha_slow
        self.flash_threshold = flash_threshold
        self.static_threshold = static_threshold
        
        # Temporal state
        self.prev_grid: Optional[np.ndarray] = None
        self.ema_fast: Optional[np.ndarray] = None
        self.ema_slow: Optional[np.ndarray] = None
        self.prev_energy_total: float = 0.0
        
        # Precompute spatial masks for efficiency
        self._setup_spatial_masks()
    
    def _setup_spatial_masks(self):
        """Precompute masks for center/edge regions."""
        n = self.grid_size
        
        # Center mask (central 4×4 for 10×10 grid)
        center_start = (n - 4) // 2
        center_end = center_start + 4
        self.center_mask = np.zeros((n, n), dtype=bool)
        self.center_mask[center_start:center_end, center_start:center_end] = True
        
        # Edge mask (border cells)
        self.edge_mask = np.zeros((n, n), dtype=bool)
        self.edge_mask[0, :] = True
        self.edge_mask[-1, :] = True
        self.edge_mask[:, 0] = True
        self.edge_mask[:, -1] = True
        
        # Upper/lower half masks
        mid = n // 2
        self.upper_mask = np.zeros((n, n), dtype=bool)
        self.upper_mask[:mid, :] = True
        self.lower_mask = np.zeros((n, n), dtype=bool)
        self.lower_mask[mid:, :] = True
    
    def update(self, grid_t: np.ndarray) -> Dict[str, float]:
        """
        Process new frame grid and extract all 20 visual features.
        
        Args:
            grid_t: Current frame's 10×10 normalized intensity grid (0-1)
        
        Returns:
            Dictionary with 20 visual features
        """
        # Initialize temporal state on first frame
        if self.prev_grid is None:
            self.prev_grid = grid_t.copy()
            self.ema_fast = grid_t.copy()
            self.ema_slow = grid_t.copy()
            return self._zero_features()
        
        # Update EMAs
        self.ema_fast = self.alpha_fast * grid_t + (1 - self.alpha_fast) * self.ema_fast
        self.ema_slow = self.alpha_slow * grid_t + (1 - self.alpha_slow) * self.ema_slow
        
        # Compute delta (frame-to-frame change)
        delta = np.abs(grid_t - self.prev_grid)
        
        # Frequency decomposition
        high_freq = np.abs(grid_t - self.ema_slow)
        low_freq = np.abs(self.ema_fast - self.ema_slow)
        
        # Extract all 20 features
        features = {}
        
        # A. Core Motion & Energy (5 features)
        features.update(self._compute_core_energy(delta))
        
        # B. Spatial Structure (5 features)
        features.update(self._compute_spatial_structure(delta, grid_t))
        
        # C. Temporal & Frequency (5 features)
        features.update(self._compute_frequency_features(high_freq, low_freq, delta))
        
        # D. Event-Focused (5 features)
        features.update(self._compute_event_features(delta, grid_t, high_freq))
        
        # Update state for next frame
        self.prev_grid = grid_t.copy()
        self.prev_energy_total = features['vis_energy_total']
        
        return features
    
    def _zero_features(self) -> Dict[str, float]:
        """Return zero-valued features for first frame."""
        return {
            # Core energy
            'vis_energy_total': 0.0,
            'vis_energy_mean': 0.0,
            'vis_energy_std': 0.0,
            'vis_motion_concentration_50': 0.0,
            'vis_static_ratio': 1.0,  # All static on first frame
            
            # Spatial structure
            'vis_center_energy_ratio': 0.0,
            'vis_edge_energy_ratio': 0.0,
            'vis_horizontal_vs_vertical_ratio': 1.0,
            'vis_recoil_vertical_bias': 0.0,
            'vis_scope_tunnel_index': 0.0,
            
            # Frequency
            'vis_highfreq_energy': 0.0,
            'vis_lowfreq_energy': 0.0,
            'vis_high_to_low_ratio': 0.0,
            'vis_smoothness_index': 1.0,
            'vis_stutter_score': 0.0,
            
            # Event-focused
            'vis_flash_intensity': 0.0,
            'vis_firefight_focus_ratio': 0.0,
            'vis_jitter_band_energy': 0.0,
            'vis_contrast_shift_score': 0.0,
            'vis_aim_lock_score': 0.0,
        }
    
    def _compute_core_energy(self, delta: np.ndarray) -> Dict[str, float]:
        """Compute 5 core motion/energy features."""
        # 1. Total energy
        energy_total = np.sum(delta)
        
        # 2. Mean energy per cell
        energy_mean = np.mean(delta)
        
        # 3. Standard deviation of energy
        energy_std = np.std(delta)
        
        # 4. Motion concentration (fraction of cells for 50% energy)
        flat_delta = delta.flatten()
        sorted_delta = np.sort(flat_delta)[::-1]  # Descending
        cumsum = np.cumsum(sorted_delta)
        total = cumsum[-1]
        if total > 1e-8:
            idx_50 = np.searchsorted(cumsum, 0.5 * total)
            concentration_50 = (idx_50 + 1) / len(flat_delta)
        else:
            concentration_50 = 0.0
        
        # 5. Static ratio (cells below threshold)
        static_count = np.sum(delta < self.static_threshold)
        static_ratio = static_count / delta.size
        
        return {
            'vis_energy_total': float(energy_total),
            'vis_energy_mean': float(energy_mean),
            'vis_energy_std': float(energy_std),
            'vis_motion_concentration_50': float(concentration_50),
            'vis_static_ratio': float(static_ratio),
        }
    
    def _compute_spatial_structure(self, delta: np.ndarray, grid_t: np.ndarray) -> Dict[str, float]:
        """Compute 5 spatial structure features."""
        epsilon = 1e-8
        total_delta = np.sum(delta) + epsilon
        
        # 6. Center energy ratio
        center_energy = np.sum(delta[self.center_mask])
        center_ratio = center_energy / total_delta
        
        # 7. Edge energy ratio
        edge_energy = np.sum(delta[self.edge_mask])
        edge_ratio = edge_energy / total_delta
        
        # 8. Horizontal vs vertical gradient ratio
        # Approximate gradients using adjacent cell differences
        horiz_grad = np.abs(delta[:, 1:] - delta[:, :-1]).sum()
        vert_grad = np.abs(delta[1:, :] - delta[:-1, :]).sum()
        hv_ratio = horiz_grad / (vert_grad + epsilon)
        
        # 9. Recoil vertical bias (upper half - lower half energy)
        upper_energy = np.sum(delta[self.upper_mask])
        lower_energy = np.sum(delta[self.lower_mask])
        vertical_bias = (upper_energy - lower_energy) / total_delta
        
        # 10. Scope tunnel index (edge variance - center variance) / (sum)
        edge_var = np.var(grid_t[self.edge_mask])
        center_var = np.var(grid_t[self.center_mask])
        tunnel_index = (edge_var - center_var) / (edge_var + center_var + epsilon)
        
        return {
            'vis_center_energy_ratio': float(center_ratio),
            'vis_edge_energy_ratio': float(edge_ratio),
            'vis_horizontal_vs_vertical_ratio': float(hv_ratio),
            'vis_recoil_vertical_bias': float(vertical_bias),
            'vis_scope_tunnel_index': float(tunnel_index),
        }
    
    def _compute_frequency_features(self, high_freq: np.ndarray, low_freq: np.ndarray, delta: np.ndarray) -> Dict[str, float]:
        """Compute 5 temporal/frequency features."""
        epsilon = 1e-8
        
        # 11. High frequency energy (recoil, jitter, flash)
        highfreq_energy = np.sum(high_freq)
        
        # 12. Low frequency energy (sweeps, tracking)
        lowfreq_energy = np.sum(low_freq)
        
        # 13. High to low ratio
        high_to_low = highfreq_energy / (lowfreq_energy + epsilon)
        
        # 14. Smoothness index (inverse of energy std)
        energy_std = np.std(delta)
        smoothness = 1.0 / (1.0 + energy_std)
        
        # 15. Stutter score (frame-to-frame energy swing)
        energy_total = np.sum(delta)
        if abs(self.prev_energy_total) > epsilon:
            stutter = abs(energy_total - self.prev_energy_total) / (self.prev_energy_total + epsilon)
        else:
            stutter = 0.0
        
        return {
            'vis_highfreq_energy': float(highfreq_energy),
            'vis_lowfreq_energy': float(lowfreq_energy),
            'vis_high_to_low_ratio': float(high_to_low),
            'vis_smoothness_index': float(smoothness),
            'vis_stutter_score': float(stutter),
        }
    
    def _compute_event_features(self, delta: np.ndarray, grid_t: np.ndarray, high_freq: np.ndarray) -> Dict[str, float]:
        """Compute 5 event-focused features."""
        epsilon = 1e-8
        
        # 16. Flash intensity (max cell delta)
        flash_intensity = np.max(delta)
        
        # 17. Firefight focus ratio (center ratio when flash is high)
        if flash_intensity > self.flash_threshold:
            center_energy = np.sum(delta[self.center_mask])
            total_energy = np.sum(delta) + epsilon
            firefight_focus = center_energy / total_energy
        else:
            firefight_focus = 0.0
        
        # 18. Jitter band energy (mid-band metric for central region)
        # Use high_freq energy minus raw delta for central 4×4
        center_high_freq = np.sum(high_freq[self.center_mask])
        center_delta = np.sum(delta[self.center_mask])
        jitter_band = abs(center_high_freq - center_delta)
        
        # 19. Contrast shift score (variance change vs EMA)
        current_var = np.var(grid_t)
        ema_var = np.var(self.ema_slow)
        contrast_shift = abs(current_var - ema_var)
        
        # 20. Aim lock score (composite: high center, high smoothness, low highfreq)
        # Normalize components to 0-1 range using thresholds
        center_ratio = np.sum(delta[self.center_mask]) / (np.sum(delta) + epsilon)
        smoothness = 1.0 / (1.0 + np.std(delta))
        highfreq_normalized = 1.0 - min(1.0, np.sum(high_freq) / 10.0)  # Inverse normalized
        
        aim_lock = center_ratio * smoothness * highfreq_normalized
        
        return {
            'vis_flash_intensity': float(flash_intensity),
            'vis_firefight_focus_ratio': float(firefight_focus),
            'vis_jitter_band_energy': float(jitter_band),
            'vis_contrast_shift_score': float(contrast_shift),
            'vis_aim_lock_score': float(aim_lock),
        }
    
    def reset(self):
        """Reset temporal state (use between matches)."""
        self.prev_grid = None
        self.ema_fast = None
        self.ema_slow = None
        self.prev_energy_total = 0.0


# Quick test harness
if __name__ == "__main__":
    print("ScreenResonanceState - Visual Feature Extraction Test")
    print("=" * 60)
    
    # Create synthetic test frames
    resonance = ScreenResonanceState()
    
    # Frame 1: Static (baseline)
    grid1 = np.random.rand(10, 10) * 0.3
    features1 = resonance.update(grid1)
    print("\nFrame 1 (static baseline):")
    print(f"  Energy total: {features1['vis_energy_total']:.4f}")
    print(f"  Static ratio: {features1['vis_static_ratio']:.2f}")
    
    # Frame 2: Small motion
    grid2 = grid1 + np.random.rand(10, 10) * 0.05
    features2 = resonance.update(grid2)
    print("\nFrame 2 (small motion):")
    print(f"  Energy total: {features2['vis_energy_total']:.4f}")
    print(f"  Smoothness: {features2['vis_smoothness_index']:.4f}")
    
    # Frame 3: Center-focused motion (aim-like)
    grid3 = grid2.copy()
    grid3[3:7, 3:7] += 0.2  # Boost center
    features3 = resonance.update(grid3)
    print("\nFrame 3 (center-focused):")
    print(f"  Center ratio: {features3['vis_center_energy_ratio']:.4f}")
    print(f"  Aim lock score: {features3['vis_aim_lock_score']:.4f}")
    
    # Frame 4: High-frequency spike (recoil/flash)
    grid4 = grid3 + np.random.rand(10, 10) * 0.3
    features4 = resonance.update(grid4)
    print("\nFrame 4 (high-freq spike):")
    print(f"  Highfreq energy: {features4['vis_highfreq_energy']:.4f}")
    print(f"  Flash intensity: {features4['vis_flash_intensity']:.4f}")
    print(f"  Stutter score: {features4['vis_stutter_score']:.4f}")
    
    # Frame 5: Vertical motion (recoil pattern)
    grid5 = grid4.copy()
    grid5[:5, :] += 0.15  # Upper half boost
    features5 = resonance.update(grid5)
    print("\nFrame 5 (vertical bias):")
    print(f"  Recoil vertical bias: {features5['vis_recoil_vertical_bias']:.4f}")
    print(f"  High/low ratio: {features5['vis_high_to_low_ratio']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All 20 features extracted successfully")
    print(f"✓ Feature keys: {len(features5)}")
    print(f"✓ Expected: 20, Got: {len(features5)}")
