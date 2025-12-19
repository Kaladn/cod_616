"""
616 Multimodal Fusion Engine + CompuCogVision
CompuCog Multimodal Game Intelligence Engine

Fuses all modalities into coherent 616 anchor signature:
- Screen grid (100 features - spatial blocks)
- Visual Resonance (20 features - CompuCogVision Phase 1, YOLO-FREE)
- Gamepad input (54 features)
- Network telemetry (8 features)

Total: 182 multimodal features → 616 resonance signature

YOLO REMOVED: Nov 26, 2025 - Replaced with pure math signal processing.

Built: November 25, 2025
Updated: November 26, 2025 - Vision Resonance Integration
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque


class Fusion616Engine:
    """
    Multimodal fusion engine with 616 resonance anchors.
    
    Combines screen, visual resonance (CompuCogVision), gamepad, network into coherent signature.
    Applies 6-1-6 Hz resonance pattern for manipulation detection.
    """
    
    def __init__(
        self,
        anchor_frequencies: List[float] = [6.0, 1.0, 6.0],
        window_size_ms: int = 1000,
        overlap_ms: int = 500
    ):
        """
        Args:
            anchor_frequencies: 616 resonance pattern (Hz)
            window_size_ms: Rolling window size (ms)
            overlap_ms: Window overlap (ms)
        """
        self.anchor_frequencies = anchor_frequencies
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
        
        # Feature dimensions (YOLO REMOVED - replaced with Vision Resonance)
        self.screen_dim = 100  # 10×10 blocks (spatial)
        self.visual_resonance_dim = 20  # CompuCogVision Phase 1 (temporal/freq features)
        self.audio_dim = 20  # Audio resonance features (parallel to visual)
        self.gamepad_dim = 54
        self.network_dim = 8
        self.total_dim = self.screen_dim + self.visual_resonance_dim + self.audio_dim + self.gamepad_dim + self.network_dim
        
        # Feature history (for temporal analysis)
        self.feature_history = deque(maxlen=int(window_size_ms / 16.67))  # ~60 frames at 60 FPS
        
        # 616 resonance state
        self.resonance_state = np.zeros(3, dtype=np.float32)  # [6Hz, 1Hz, 6Hz] phases
        self.resonance_amplitudes = np.ones(3, dtype=np.float32)
        
        # Statistics
        self.frames_fused = 0
        self.total_fusion_time = 0.0
        
        print(f"[616 Fusion Engine + CompuCogVision]")
        print(f"  Anchor frequencies: {anchor_frequencies} Hz")
        print(f"  Window size: {window_size_ms}ms")
        print(f"  Feature dims: Screen({self.screen_dim}) + VisionResonance({self.visual_resonance_dim}) + "
              f"Gamepad({self.gamepad_dim}) + Network({self.network_dim}) = {self.total_dim}")
        print(f"  YOLO REMOVED - Using pure math signal processing")
    
    def fuse(
        self,
        screen_features: Optional[np.ndarray] = None,
        visual_resonance: Optional[Dict[str, float]] = None,
        audio_resonance: Optional[Dict[str, float]] = None,
        gamepad_features: Optional[np.ndarray] = None,
        network_features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Fuse multimodal features into 616 signature.
        
        Args:
            screen_features: Screen grid features (100,)
            visual_resonance: CompuCogVision features (20,) - dict of feature name → value
            gamepad_features: Gamepad input features (54,)
            network_features: Network telemetry features (8,)
            timestamp: Capture timestamp (uses current time if None)
        
        Returns:
            Dict with:
                - 'fused_vector': Combined feature vector (182,)
                - 'resonance_vector': 616 resonance features (12,)
                - 'full_signature': fused_vector + resonance_vector (194,)
                - 'manipulation_score': Anomaly detection score (0-1)
                - 'timestamp': Fusion timestamp
        """
        start_time = time.perf_counter()
        if timestamp is None:
            timestamp = start_time
        
        # Default to zeros if modality not provided
        if screen_features is None:
            screen_features = np.zeros(self.screen_dim, dtype=np.float32)
        
        # Convert visual_resonance dict to array (CompuCogVision Phase 1)
        if visual_resonance is None:
            visual_resonance_array = np.zeros(self.visual_resonance_dim, dtype=np.float32)
        else:
            # Extract values in consistent order (sorted by key)
            visual_resonance_array = np.array([
                visual_resonance[k] for k in sorted(visual_resonance.keys())
            ], dtype=np.float32)
        
        if audio_resonance is None:
            audio_resonance_array = np.zeros(self.audio_dim, dtype=np.float32)
        else:
            audio_resonance_array = np.array([
                audio_resonance[k] for k in sorted(audio_resonance.keys())
            ], dtype=np.float32)

        if gamepad_features is None:
            gamepad_features = np.zeros(self.gamepad_dim, dtype=np.float32)
        if network_features is None:
            network_features = np.zeros(self.network_dim, dtype=np.float32)

        # Validate dimensions
        assert len(screen_features) == self.screen_dim
        assert len(visual_resonance_array) == self.visual_resonance_dim
        assert len(audio_resonance_array) == self.audio_dim
        assert len(gamepad_features) == self.gamepad_dim
        assert len(network_features) == self.network_dim

        # Concatenate all modalities (visual_resonance + audio_resonance included)
        fused_vector = np.concatenate([
            screen_features,
            visual_resonance_array,
            audio_resonance_array,
            gamepad_features,
            network_features
        ])
        
        # Update resonance state (6-1-6 Hz oscillators)
        dt = 0.0167  # ~60 FPS (will be overridden by actual timing later)
        if len(self.feature_history) > 0:
            dt = timestamp - self.feature_history[-1]['timestamp']
        
        for i, freq in enumerate(self.anchor_frequencies):
            # Update phase
            self.resonance_state[i] += 2 * np.pi * freq * dt
            self.resonance_state[i] %= (2 * np.pi)
            
            # Update amplitude based on feature energy
            if i == 0:  # 6 Hz - screen motion energy
                energy = np.sum(screen_features**2)
            elif i == 1:  # 1 Hz - gamepad input rhythm
                energy = np.sum(gamepad_features[:16])  # Button presses
            else:  # 6 Hz - network jitter (also influenced by audio energy)
                audio_energy = 0.0
                try:
                    audio_energy = float(audio_resonance.get('aud_energy_total', 0.0)) if audio_resonance is not None else 0.0
                except Exception:
                    audio_energy = 0.0
                energy = network_features[2] + 0.01 * audio_energy  # small audio contribution
            
            # Smooth amplitude update
            self.resonance_amplitudes[i] = 0.9 * self.resonance_amplitudes[i] + 0.1 * energy
        
        # Build resonance vector
        resonance_vector = np.array([
            # 6-1-6 phases
            np.sin(self.resonance_state[0]),
            np.cos(self.resonance_state[0]),
            np.sin(self.resonance_state[1]),
            np.cos(self.resonance_state[1]),
            np.sin(self.resonance_state[2]),
            np.cos(self.resonance_state[2]),
            # 6-1-6 amplitudes
            self.resonance_amplitudes[0],
            self.resonance_amplitudes[1],
            self.resonance_amplitudes[2],
            # Phase coherence (cross-correlation)
            np.cos(self.resonance_state[0] - self.resonance_state[1]),
            np.cos(self.resonance_state[1] - self.resonance_state[2]),
            np.cos(self.resonance_state[0] - self.resonance_state[2])
        ], dtype=np.float32)
        
        # Combine into full signature
        full_signature = np.concatenate([fused_vector, resonance_vector])
        
        # Detect manipulation (anomaly detection)
        manipulation_score = self.compute_manipulation_score(
            fused_vector, resonance_vector
        )
        
        # Store in history
        self.feature_history.append({
            'fused_vector': fused_vector,
            'resonance_vector': resonance_vector,
            'timestamp': timestamp
        })
        
        # Update statistics
        fusion_time = time.perf_counter() - start_time
        self.frames_fused += 1
        self.total_fusion_time += fusion_time
        
        return {
            'fused_vector': fused_vector,
            'resonance_vector': resonance_vector,
            'full_signature': full_signature,
            'manipulation_score': manipulation_score,
            'timestamp': timestamp
        }
    
    def compute_manipulation_score(
        self,
        fused_vector: np.ndarray,
        resonance_vector: np.ndarray
    ) -> float:
        """
        Detect manipulation using resonance coherence analysis.
        
        Args:
            fused_vector: Combined features (353,)
            resonance_vector: 616 resonance features (12,)
        
        Returns:
            float: Manipulation probability (0-1)
        """
        # Check if enough history
        if len(self.feature_history) < 10:
            return 0.0
        
        # Compute z-scores for anomaly detection
        recent_features = np.array([h['fused_vector'] for h in self.feature_history])
        mean_features = np.mean(recent_features, axis=0)
        std_features = np.std(recent_features, axis=0) + 1e-6
        
        z_scores = np.abs((fused_vector - mean_features) / std_features)
        max_z_score = np.max(z_scores)
        
        # Resonance coherence (should be ~1.0 for normal play)
        phase_coherence = resonance_vector[9:12]  # Phase correlations
        coherence_loss = 1.0 - np.mean(np.abs(phase_coherence))
        
        # Network anomaly (sudden RTT spike)
        network_anomaly = fused_vector[-1]  # is_spike feature
        
        # Combine signals
        manipulation_score = np.tanh(
            0.3 * (max_z_score / 3.0) +  # Feature anomaly
            0.4 * coherence_loss +  # Resonance desync
            0.3 * network_anomaly  # Network manipulation
        )
        
        return float(manipulation_score)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get fusion statistics.
        
        Returns:
            Dict with frame count, average FPS, average fusion time
        """
        if self.frames_fused == 0:
            return {
                'frames_fused': 0,
                'avg_fps': 0.0,
                'avg_fusion_time_ms': 0.0
            }
        
        avg_fusion_time = self.total_fusion_time / self.frames_fused
        avg_fps = 1.0 / avg_fusion_time if avg_fusion_time > 0 else 0.0
        
        return {
            'frames_fused': self.frames_fused,
            'avg_fps': avg_fps,
            'avg_fusion_time_ms': avg_fusion_time * 1000
        }
    
    def reset(self):
        """Reset fusion state (for new match)."""
        self.feature_history.clear()
        self.resonance_state = np.zeros(3, dtype=np.float32)
        self.resonance_amplitudes = np.ones(3, dtype=np.float32)
        self.frames_fused = 0
        self.total_fusion_time = 0.0


if __name__ == "__main__":
    """Test 616 fusion engine."""
    
    print("Testing 616 Fusion Engine...")
    print("Generating synthetic multimodal data\n")
    
    # Initialize engine
    engine = Fusion616Engine(
        anchor_frequencies=[6.0, 1.0, 6.0],
        window_size_ms=1000
    )
    
    # Simulate 300 frames (5 seconds at 60 FPS)
    for frame in range(300):
        # Generate synthetic features
        t = frame / 60.0  # Time in seconds
        
        # Screen features (sinusoidal motion)
        screen_features = np.abs(np.sin(2 * np.pi * 6.0 * t + np.random.randn(100) * 0.1))
        
        # Visual Resonance features (CompuCogVision - 20 dim)
        visual_resonance = {
            'vis_energy_total': np.abs(np.sin(t * 6.0)),
            'vis_energy_mean': 0.1 + np.random.rand() * 0.05,
            'vis_energy_std': 0.02 + np.random.rand() * 0.01,
            'vis_motion_concentration_50': 0.2 + np.random.rand() * 0.1,
            'vis_static_ratio': 0.5 + np.random.rand() * 0.2,
            'vis_center_energy_ratio': 0.4 + np.random.rand() * 0.1,
            'vis_edge_energy_ratio': 0.1 + np.random.rand() * 0.05,
            'vis_horizontal_vs_vertical_ratio': 1.0 + np.random.rand() * 0.2,
            'vis_recoil_vertical_bias': -0.1 + np.random.rand() * 0.05,
            'vis_scope_tunnel_index': 0.0 + np.random.rand() * 0.1,
            'vis_highfreq_energy': np.abs(np.sin(t * 12.0)),
            'vis_lowfreq_energy': np.abs(np.cos(t * 1.0)),
            'vis_high_to_low_ratio': 2.0 + np.random.rand() * 0.5,
            'vis_smoothness_index': 0.7 + np.random.rand() * 0.2,
            'vis_stutter_score': np.random.rand() * 0.1,
            'vis_flash_intensity': np.random.rand() * 0.2,
            'vis_firefight_focus_ratio': 0.3 + np.random.rand() * 0.1,
            'vis_jitter_band_energy': np.random.rand() * 0.05,
            'vis_contrast_shift_score': np.random.rand() * 0.02,
            'vis_aim_lock_score': np.random.rand() * 0.1
        }
        
        # Gamepad features (periodic button presses)
        gamepad_features = np.zeros(54)
        if frame % 60 < 5:  # Press button every second
            gamepad_features[0] = 1.0
        
        # Network features (stable with occasional spike)
        network_features = np.array([20.0, 0.0, 2.0, 20.0, 2.0, 25.0, 15.0, 0.0])
        if frame == 150:  # Spike at 2.5 seconds
            network_features[0] = 100.0  # RTT spike
            network_features[7] = 1.0  # is_spike flag
        
        # Fuse (YOLO removed, visual_resonance added)
        result = engine.fuse(
            screen_features=screen_features,
            visual_resonance=visual_resonance,
            gamepad_features=gamepad_features,
            network_features=network_features
        )
        
        # Print every 60 frames
        if frame % 60 == 0 or frame == 150:
            print(f"[Frame {frame:>3}] "
                  f"Manipulation score: {result['manipulation_score']:.3f} | "
                  f"Resonance phases: [{result['resonance_vector'][0]:.2f}, "
                  f"{result['resonance_vector'][2]:.2f}, "
                  f"{result['resonance_vector'][4]:.2f}]")
        
        time.sleep(0.0167)  # 60 FPS
    
    stats = engine.get_statistics()
    print(f"\n[Final Stats]")
    print(f"  Frames fused: {stats['frames_fused']}")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Average fusion time: {stats['avg_fusion_time_ms']:.2f}ms")
