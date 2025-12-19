"""
MatchFingerprintBuilder - Phase 2 Implementation
CompuCog 616 Vision System

Accumulates per-frame features into 365-dimensional match signature.
Pure bookkeeping - no ML, just stats on stats.

Built: November 26, 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from scipy.stats import pearsonr


class MatchFingerprintBuilder:
    """
    Accumulates thousands of frames into one 525-dim match fingerprint.
    
    Layout (Extended for Audio):
    - Dims 0-127: Visual summary (20 features × 4 stats + segmentation)
    - Dims 128-223: Gamepad summary (16 features × 4 stats + events)
    - Dims 224-255: Network summary (8 features × 4 stats)
    - Dims 256-335: Cross-modal correlations (20 pairs × 4 stats)
    - Dims 336-364: Meta + anomaly flags (29 dims)
    - Dims 365-444: Audio summary (20 features × 4 stats)
    - Dims 445-524: Audio cross-modal (20 pairs × 4 stats)
    """
    
    # Feature names for consistent ordering
    VISUAL_FEATURES = [
        'vis_energy_total', 'vis_energy_mean', 'vis_energy_std',
        'vis_motion_concentration_50', 'vis_static_ratio',
        'vis_center_energy_ratio', 'vis_edge_energy_ratio',
        'vis_horizontal_vs_vertical_ratio', 'vis_recoil_vertical_bias',
        'vis_scope_tunnel_index', 'vis_highfreq_energy', 'vis_lowfreq_energy',
        'vis_high_to_low_ratio', 'vis_smoothness_index', 'vis_stutter_score',
        'vis_flash_intensity', 'vis_firefight_focus_ratio',
        'vis_jitter_band_energy', 'vis_contrast_shift_score', 'vis_aim_lock_score'
    ]
    
    GAMEPAD_FEATURES = [
        'gp_button_press_count', 'gp_stick_magnitude', 'gp_stick_velocity',
        'gp_button_press_rate', 'gp_left_stick_mag', 'gp_right_stick_mag',
        'gp_left_stick_x', 'gp_left_stick_y', 'gp_right_stick_x', 'gp_right_stick_y',
        'gp_trigger_left', 'gp_trigger_right',
        'gp_button_0', 'gp_button_1', 'gp_button_2', 'gp_button_3'
    ]
    
    NETWORK_FEATURES = [
        'net_rtt', 'net_packet_loss', 'net_jitter',
        'net_rtt_mean', 'net_rtt_std', 'net_rtt_max', 'net_rtt_min', 'net_is_spike'
    ]
    
    AUDIO_FEATURES = [
        'aud_energy_total', 'aud_energy_var', 'aud_spectral_centroid_mean',
        'aud_spectral_centroid_std', 'aud_low_mid_high_balance_low',
        'aud_low_mid_high_balance_high', 'aud_transient_rate',
        'aud_transient_to_total_ratio', 'aud_gunshot_peak_index',
        'aud_impact_burst_score', 'aud_sustain_ratio', 'aud_harmonicity_mean',
        'aud_harmonicity_std', 'aud_voice_activity_ratio',
        'aud_noise_dominance_ratio', 'aud_stereo_width', 'aud_left_right_bias',
        'aud_occlusion_index', 'aud_clarity_index', 'aud_ui_presence_score'
    ]
    
    # Cross-modal correlation pairs (20 key relationships)
    CORRELATION_PAIRS = [
        ('vis_energy_total', 'gp_stick_magnitude'),
        ('vis_center_energy_ratio', 'gp_stick_magnitude'),
        ('vis_highfreq_energy', 'gp_button_press_rate'),
        ('vis_smoothness_index', 'gp_stick_magnitude'),
        ('vis_aim_lock_score', 'gp_stick_magnitude'),
        ('vis_stutter_score', 'net_rtt'),
        ('vis_stutter_score', 'net_is_spike'),
        ('vis_contrast_shift_score', 'net_rtt'),
        ('vis_high_to_low_ratio', 'gp_stick_velocity'),
        ('vis_highfreq_energy', 'net_rtt'),
        ('vis_energy_total', 'net_rtt'),
        ('gp_button_press_rate', 'net_rtt'),
        ('gp_stick_magnitude', 'net_jitter'),
        ('vis_center_energy_ratio', 'net_is_spike'),
        ('vis_aim_lock_score', 'net_is_spike'),
        ('vis_highfreq_energy', 'gp_stick_velocity'),
        ('vis_lowfreq_energy', 'gp_stick_magnitude'),
        ('vis_stutter_score', 'gp_stick_magnitude'),
        ('vis_contrast_shift_score', 'gp_button_press_rate'),
        ('vis_energy_total', 'gp_button_press_rate')
    ]
    
    # Audio cross-modal correlation pairs (20 key relationships with visual/gamepad/network)
    AUDIO_CORRELATION_PAIRS = [
        ('aud_energy_total', 'vis_energy_total'),           # Audio loudness vs visual chaos
        ('aud_transient_rate', 'vis_highfreq_energy'),      # Gunshots audio vs visual
        ('aud_transient_rate', 'gp_button_press_rate'),     # Gunshots vs trigger pulls
        ('aud_gunshot_peak_index', 'vis_flash_intensity'),  # Audio peaks vs visual flashes
        ('aud_energy_total', 'gp_stick_magnitude'),         # Audio intensity vs movement
        ('aud_low_mid_high_balance_low', 'vis_lowfreq_energy'),  # Low freq audio/visual
        ('aud_clarity_index', 'net_rtt'),                   # Audio clarity vs network lag
        ('aud_occlusion_index', 'net_is_spike'),            # Muffled audio vs lag spikes
        ('aud_voice_activity_ratio', 'gp_button_press_rate'), # Voice comms vs activity
        ('aud_harmonicity_mean', 'vis_smoothness_index'),   # Harmonic audio vs smooth visual
        ('aud_transient_rate', 'vis_contrast_shift_score'), # Transients vs visual changes
        ('aud_energy_var', 'vis_stutter_score'),            # Audio choppiness vs visual stutter
        ('aud_stereo_width', 'vis_center_energy_ratio'),    # Stereo field vs visual center
        ('aud_left_right_bias', 'gp_stick_magnitude'),      # Audio panning vs movement
        ('aud_ui_presence_score', 'vis_highfreq_energy'),   # UI audio vs visual detail
        ('aud_sustain_ratio', 'vis_static_ratio'),          # Sustained audio vs static visual
        ('aud_impact_burst_score', 'vis_firefight_focus_ratio'), # Audio bursts vs firefight
        ('aud_noise_dominance_ratio', 'vis_jitter_band_energy'), # Noisy audio vs jittery visual
        ('aud_spectral_centroid_mean', 'vis_aim_lock_score'), # Audio brightness vs aim lock
        ('aud_transient_rate', 'net_jitter')                # Audio events vs network jitter
    ]
    
    def __init__(self, max_frames: Optional[int] = 6000, sample_size: int = 1000):
        """Initialize accumulator state.

        Args:
            max_frames: Optional soft cap for total frames to keep in memory-aware operations (informational).
            sample_size: Number of recent samples to keep for percentile/correlation computations (bounded memory).
        """
        # Configuration
        self.max_frames = max_frames
        self.sample_size = sample_size

        # Primary counters
        self.frame_count = 0
        self.match_start_time = None
        self.match_duration = 0.0

        # Per-feature statistics (sum, sum_sq, min, max for mean/std/max/p90)
        self.visual_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'min': np.inf, 'max': -np.inf, 'values': deque(maxlen=self.sample_size)}
                            for name in self.VISUAL_FEATURES}
        self.gamepad_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'min': np.inf, 'max': -np.inf, 'values': deque(maxlen=self.sample_size)}
                             for name in self.GAMEPAD_FEATURES}
        self.network_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'min': np.inf, 'max': -np.inf, 'values': deque(maxlen=self.sample_size)}
                             for name in self.NETWORK_FEATURES}
        self.audio_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'min': np.inf, 'max': -np.inf, 'values': deque(maxlen=self.sample_size)}
                           for name in self.AUDIO_FEATURES}

        # Cross-modal correlation accumulators (bounded sample deques)
        self.correlation_stats = {pair: {'sum_x': 0.0, 'sum_y': 0.0, 'sum_xy': 0.0,
                                         'sum_x2': 0.0, 'sum_y2': 0.0,
                                         'count_x_high': 0, 'count_y_high': 0, 'count_both_high': 0,
                                         'sum_x_when_y_high': 0.0, 'sum_y_when_x_high': 0.0,
                                         'x_vals': deque(maxlen=self.sample_size), 'y_vals': deque(maxlen=self.sample_size)}
                                 for pair in self.CORRELATION_PAIRS}

        # Audio cross-modal correlation accumulators
        self.audio_correlation_stats = {pair: {'sum_x': 0.0, 'sum_y': 0.0, 'sum_xy': 0.0,
                                               'sum_x2': 0.0, 'sum_y2': 0.0,
                                               'count_x_high': 0, 'count_y_high': 0, 'count_both_high': 0,
                                               'sum_x_when_y_high': 0.0, 'sum_y_when_x_high': 0.0,
                                               'x_vals': deque(maxlen=self.sample_size), 'y_vals': deque(maxlen=self.sample_size)}
                                       for pair in self.AUDIO_CORRELATION_PAIRS}

        # Event counters for gamepad summary (dims 192-223)
        self.total_button_presses = 0
        self.frames_with_stick_input = 0
        self.frames_with_button_input = 0

        # Segmentation counters (firefight, ADS-like, anomalies)
        self.firefight_frame_count = 0
        self.high_flash_frame_count = 0
        self.high_center_ratio_frame_count = 0
        self.visual_anomaly_frame_count = 0
        self.input_anomaly_frame_count = 0
        self.network_anomaly_frame_count = 0

        # Temporal tracking for early/mid/late analysis
        self.early_third_frames = []
        self.middle_third_frames = []
        self.late_third_frames = []

        # Meta
        self.profile_id = 0  # 0=play_nice, 1=forensic, 2=overwatch
        self.user_label = 0  # 0=unknown, 1=fair, 2=rigged

        print("[MatchFingerprintBuilder] Initialized")
        print(f"  Visual features: {len(self.VISUAL_FEATURES)}")
        print(f"  Gamepad features: {len(self.GAMEPAD_FEATURES)}")
        print(f"  Network features: {len(self.NETWORK_FEATURES)}")
        print(f"  Correlation pairs: {len(self.CORRELATION_PAIRS)}")
    
    def _flatten_features(self, fused_frame: Dict) -> Dict[str, float]:
        """
        Flatten nested feature dict to flat name → value mapping.
        
        Args:
            fused_frame: Dict with 'visual_resonance', 'gamepad', 'network' keys
        
        Returns:
            Flat dict like {'vis_energy_total': 0.42, 'gp_stick_magnitude': 0.67, ...}
        """
        flat = {}
        
        # Visual resonance (already dict of str → float)
        if 'visual_resonance' in fused_frame:
            for k, v in fused_frame['visual_resonance'].items():
                flat[k] = float(v)
        
        # Gamepad (may be dict or have 'feature_vector' key)
        if 'gamepad' in fused_frame:
            gp = fused_frame['gamepad']
            if isinstance(gp, dict):
                # Map known keys
                flat['gp_button_press_count'] = float(gp.get('button_press_count', 0.0))
                flat['gp_stick_magnitude'] = float(gp.get('stick_magnitude', 0.0))
                flat['gp_stick_velocity'] = float(gp.get('stick_velocity', 0.0))  # May not exist
                flat['gp_button_press_rate'] = float(gp.get('button_press_rate', 0.0))  # May not exist
                
                # Extract stick components
                if 'sticks' in gp:
                    sticks = gp['sticks']
                    flat['gp_left_stick_x'] = float(sticks[0]) if len(sticks) > 0 else 0.0
                    flat['gp_left_stick_y'] = float(sticks[1]) if len(sticks) > 1 else 0.0
                    flat['gp_right_stick_x'] = float(sticks[2]) if len(sticks) > 2 else 0.0
                    flat['gp_right_stick_y'] = float(sticks[3]) if len(sticks) > 3 else 0.0
                    flat['gp_trigger_left'] = float(sticks[4]) if len(sticks) > 4 else 0.0
                    flat['gp_trigger_right'] = float(sticks[5]) if len(sticks) > 5 else 0.0
                    flat['gp_left_stick_mag'] = np.sqrt(flat['gp_left_stick_x']**2 + flat['gp_left_stick_y']**2)
                    flat['gp_right_stick_mag'] = np.sqrt(flat['gp_right_stick_x']**2 + flat['gp_right_stick_y']**2)
                
                # Extract button states
                if 'buttons' in gp:
                    buttons = gp['buttons']
                    for i in range(min(4, len(buttons))):  # First 4 buttons
                        flat[f'gp_button_{i}'] = float(buttons[i])
        
        # Network (may be dict or have 'feature_vector' key)
        if 'network' in fused_frame:
            net = fused_frame['network']
            if isinstance(net, dict):
                flat['net_rtt'] = float(net.get('rtt', 0.0))
                flat['net_packet_loss'] = float(net.get('packet_loss', 0.0))
                flat['net_jitter'] = float(net.get('jitter', 0.0))
                flat['net_rtt_mean'] = float(net.get('rtt_mean', 0.0))
                flat['net_rtt_std'] = float(net.get('rtt_std', 0.0))
                flat['net_rtt_max'] = float(net.get('rtt_max', 0.0))
                flat['net_rtt_min'] = float(net.get('rtt_min', 0.0))
                flat['net_is_spike'] = float(net.get('is_spike', 0.0))
        
        # Audio (AudioResonanceState output)
        if 'audio_resonance' in fused_frame:
            for k, v in fused_frame['audio_resonance'].items():
                flat[k] = float(v)
        
        return flat
    
    def update(self, fused_frame: Dict):
        """
        Process one frame and update all accumulators.
        
        Args:
            fused_frame: Dict with 'visual_resonance', 'gamepad', 'network', 'timestamp'
        """
        # Track timing
        if self.match_start_time is None:
            self.match_start_time = fused_frame.get('timestamp', 0.0)
        
        self.match_duration = fused_frame.get('timestamp', 0.0) - self.match_start_time
        
        # Flatten to scalar features
        flat = self._flatten_features(fused_frame)
        
        # Update visual stats
        for name in self.VISUAL_FEATURES:
            if name in flat:
                val = flat[name]
                stats = self.visual_stats[name]
                stats['sum'] += val
                stats['sum_sq'] += val * val
                stats['min'] = min(stats['min'], val)
                stats['max'] = max(stats['max'], val)
                stats['values'].append(val)  # For p90 calculation
        
        # Update gamepad stats
        for name in self.GAMEPAD_FEATURES:
            if name in flat:
                val = flat[name]
                stats = self.gamepad_stats[name]
                stats['sum'] += val
                stats['sum_sq'] += val * val
                stats['min'] = min(stats['min'], val)
                stats['max'] = max(stats['max'], val)
                stats['values'].append(val)
        
        # Update network stats
        for name in self.NETWORK_FEATURES:
            if name in flat:
                val = flat[name]
                stats = self.network_stats[name]
                stats['sum'] += val
                stats['sum_sq'] += val * val
                stats['min'] = min(stats['min'], val)
                stats['max'] = max(stats['max'], val)
                stats['values'].append(val)
        
        # Update audio stats
        for name in self.AUDIO_FEATURES:
            if name in flat:
                val = flat[name]
                stats = self.audio_stats[name]
                stats['sum'] += val
                stats['sum_sq'] += val * val
                stats['min'] = min(stats['min'], val)
                stats['max'] = max(stats['max'], val)
                stats['values'].append(val)
        
        # Update correlation accumulators
        for (x_name, y_name) in self.CORRELATION_PAIRS:
            if x_name in flat and y_name in flat:
                x = flat[x_name]
                y = flat[y_name]
                stats = self.correlation_stats[(x_name, y_name)]
                
                stats['sum_x'] += x
                stats['sum_y'] += y
                stats['sum_xy'] += x * y
                stats['sum_x2'] += x * x
                stats['sum_y2'] += y * y
                stats['x_vals'].append(x)
                stats['y_vals'].append(y)
                
                # Dynamic thresholds (will compute at build time for simplicity)
                # For now just store values
        
        # Update audio correlation accumulators
        for (x_name, y_name) in self.AUDIO_CORRELATION_PAIRS:
            if x_name in flat and y_name in flat:
                x = flat[x_name]
                y = flat[y_name]
                stats = self.audio_correlation_stats[(x_name, y_name)]
                
                stats['sum_x'] += x
                stats['sum_y'] += y
                stats['sum_xy'] += x * y
                stats['sum_x2'] += x * x
                stats['sum_y2'] += y * y
                stats['x_vals'].append(x)
                stats['y_vals'].append(y)
        
        # Segmentation flags
        if 'vis_flash_intensity' in flat and flat['vis_flash_intensity'] > 0.15:
            self.firefight_frame_count += 1
            self.high_flash_frame_count += 1
        
        if 'vis_center_energy_ratio' in flat and flat['vis_center_energy_ratio'] > 0.5:
            self.high_center_ratio_frame_count += 1
        
        # Anomaly flags (simple thresholds for v1)
        if 'vis_aim_lock_score' in flat and flat['vis_aim_lock_score'] > 0.3:
            self.visual_anomaly_frame_count += 1
        
        if 'gp_stick_magnitude' in flat and flat['gp_stick_magnitude'] > 0.8:
            self.input_anomaly_frame_count += 1
        
        if 'net_is_spike' in flat and flat['net_is_spike'] > 0.5:
            self.network_anomaly_frame_count += 1
        
        # Event counters
        if 'gp_button_press_count' in flat and flat['gp_button_press_count'] > 0:
            self.frames_with_button_input += 1
        
        if 'gp_stick_magnitude' in flat and flat['gp_stick_magnitude'] > 0.1:
            self.frames_with_stick_input += 1
        
        # Temporal tracking (divide into thirds at build time)
        self.frame_count += 1
    
    def _compute_stats(self, feature_dict: Dict, feature_name: str) -> Tuple[float, float, float, float]:
        """
        Compute mean, std, max, p90 for a feature.
        
        Returns:
            (mean, std, max, p90)
        """
        stats = feature_dict[feature_name]
        n = self.frame_count
        
        if n == 0:
            return (0.0, 0.0, 0.0, 0.0)
        
        mean = stats['sum'] / n
        variance = max(0.0, stats['sum_sq'] / n - mean * mean)
        std = np.sqrt(variance)
        max_val = stats['max'] if stats['max'] != -np.inf else 0.0
        
        # p90 from stored values (approx)
        if len(stats['values']) > 0:
            p90 = np.percentile(stats['values'], 90)
        else:
            p90 = max_val
        
        return (mean, std, max_val, p90)
    
    def build(self) -> Dict:
        """
        Construct 525-dim fingerprint vector from accumulated stats (extended for audio).
        
        Layout:
        - Dims 0-127: Visual summary
        - Dims 128-223: Gamepad summary
        - Dims 224-255: Network summary
        - Dims 256-335: Cross-modal correlations
        - Dims 336-364: Meta + anomaly flags
        - Dims 365-444: Audio summary (20 features × 4 stats)
        - Dims 445-524: Audio cross-modal correlations (20 pairs × 4 stats)
        
        Returns:
            Dict with:
                - 'vector': np.ndarray (525,)
                - 'layout_version': str ('v2_audio')
                - 'meta': Dict with metadata
                - 'frame_count': int
                - 'duration': float
        """
        fp = np.zeros(525, dtype=np.float32)
        
        # Block A: Visual Summary (0-127)
        idx = 0
        for name in self.VISUAL_FEATURES:
            mean, std, max_val, p90 = self._compute_stats(self.visual_stats, name)
            fp[idx] = mean
            fp[idx+1] = std
            fp[idx+2] = max_val
            fp[idx+3] = p90
            idx += 4
        
        # Dims 80-89: Firefight segmentation
        fp[80] = self.firefight_frame_count
        fp[81] = self.firefight_frame_count / max(1, self.frame_count)  # Fraction
        fp[82] = self.high_flash_frame_count
        fp[83] = self.high_flash_frame_count / max(1, self.frame_count)
        fp[84] = self.high_center_ratio_frame_count
        fp[85] = self.high_center_ratio_frame_count / max(1, self.frame_count)
        fp[86:90] = 0  # Reserved
        
        # Dims 90-99: ADS-like segmentation (placeholder)
        fp[90:100] = 0
        
        # Dims 100-127: Visual anomaly distribution (placeholder)
        fp[100] = self.visual_anomaly_frame_count / max(1, self.frame_count)
        fp[101:128] = 0
        
        # Block B: Gamepad Summary (128-223)
        idx = 128
        for name in self.GAMEPAD_FEATURES:
            mean, std, max_val, p90 = self._compute_stats(self.gamepad_stats, name)
            fp[idx] = mean
            fp[idx+1] = std
            fp[idx+2] = max_val
            fp[idx+3] = p90
            idx += 4
        
        # Dims 192-223: Event counts and patterns
        fp[192] = self.total_button_presses
        fp[193] = self.frames_with_button_input
        fp[194] = self.frames_with_stick_input
        fp[195] = self.frames_with_button_input / max(1, self.frame_count)
        fp[196] = self.frames_with_stick_input / max(1, self.frame_count)
        fp[197:224] = 0  # Reserved
        
        # Block C: Network Summary (224-255)
        idx = 224
        for name in self.NETWORK_FEATURES:
            mean, std, max_val, p90 = self._compute_stats(self.network_stats, name)
            fp[idx] = mean
            fp[idx+1] = std
            fp[idx+2] = max_val
            fp[idx+3] = p90
            idx += 4
        
        # Block D: Cross-Modal Correlations (256-335)
        idx = 256
        for (x_name, y_name) in self.CORRELATION_PAIRS:
            stats = self.correlation_stats[(x_name, y_name)]
            n = self.frame_count
            
            if n > 1 and len(stats['x_vals']) > 1:
                # Pearson r
                x_vals = np.array(stats['x_vals'])
                y_vals = np.array(stats['y_vals'])
                if np.std(x_vals) > 1e-6 and np.std(y_vals) > 1e-6:
                    r, _ = pearsonr(x_vals, y_vals)
                else:
                    r = 0.0
                
                # E[X | Y high] (top 25%)
                y_threshold = np.percentile(y_vals, 75)
                x_when_y_high = x_vals[y_vals >= y_threshold]
                mean_x_given_y_high = np.mean(x_when_y_high) if len(x_when_y_high) > 0 else 0.0
                
                # E[Y | X high]
                x_threshold = np.percentile(x_vals, 75)
                y_when_x_high = y_vals[x_vals >= x_threshold]
                mean_y_given_x_high = np.mean(y_when_x_high) if len(y_when_x_high) > 0 else 0.0
                
                # Fraction both high
                frac_both_high = np.sum((x_vals >= x_threshold) & (y_vals >= y_threshold)) / n
                
                fp[idx] = r
                fp[idx+1] = mean_x_given_y_high
                fp[idx+2] = mean_y_given_x_high
                fp[idx+3] = frac_both_high
            else:
                fp[idx:idx+4] = 0.0
            
            idx += 4
        
        # Block E: Meta + Anomaly (336-364)
        fp[336] = self.frame_count
        fp[337] = self.match_duration
        fp[338] = self.frame_count / max(1.0, self.match_duration)  # Avg FPS
        fp[339] = self.profile_id
        fp[340] = self.user_label
        
        # Global anomaly fractions
        fp[341] = self.visual_anomaly_frame_count / max(1, self.frame_count)
        fp[342] = self.input_anomaly_frame_count / max(1, self.frame_count)
        fp[343] = self.network_anomaly_frame_count / max(1, self.frame_count)
        fp[344] = (fp[341] + fp[342] + fp[343]) / 3  # Avg anomaly score
        
        # Composite suspect scores (simple weighted combos for v1)
        # Aim assist: high aim_lock_score, low input correlation
        fp[353] = self.visual_stats['vis_aim_lock_score']['sum'] / max(1, self.frame_count)  # Placeholder
        
        # Recoil compensation: low highfreq during action
        fp[354] = 1.0 - (self.visual_stats['vis_highfreq_energy']['sum'] / max(1.0, self.frame_count * 10))  # Placeholder
        
        # EOMM lag: stutter correlated with network
        fp[355] = self.network_anomaly_frame_count / max(1, self.frame_count)  # Placeholder
        
        # Network priority bias: high ping during high input
        fp[356] = (self.network_stats['net_rtt']['sum'] / max(1, self.frame_count)) / 100.0  # Placeholder
        
        fp[357:365] = 0  # Reserved
        
        # Block F: Audio Summary (365-444) - 20 features × 4 stats
        idx = 365
        for name in self.AUDIO_FEATURES:
            mean, std, max_val, p90 = self._compute_stats(self.audio_stats, name)
            fp[idx] = mean
            fp[idx+1] = std
            fp[idx+2] = max_val
            fp[idx+3] = p90
            idx += 4
        
        # Block G: Audio Cross-Modal Correlations (445-524) - 20 pairs × 4 stats
        idx = 445
        for (x_name, y_name) in self.AUDIO_CORRELATION_PAIRS:
            stats = self.audio_correlation_stats[(x_name, y_name)]
            n = self.frame_count
            
            if n > 1 and len(stats['x_vals']) > 1:
                # Pearson r
                x_vals = np.array(stats['x_vals'])
                y_vals = np.array(stats['y_vals'])
                if np.std(x_vals) > 1e-6 and np.std(y_vals) > 1e-6:
                    r, _ = pearsonr(x_vals, y_vals)
                else:
                    r = 0.0
                
                # E[X | Y high] (top 25%)
                y_threshold = np.percentile(y_vals, 75)
                x_when_y_high = x_vals[y_vals >= y_threshold]
                mean_x_given_y_high = np.mean(x_when_y_high) if len(x_when_y_high) > 0 else 0.0
                
                # E[Y | X high]
                x_threshold = np.percentile(x_vals, 75)
                y_when_x_high = y_vals[x_vals >= x_threshold]
                mean_y_given_x_high = np.mean(y_when_x_high) if len(y_when_x_high) > 0 else 0.0
                
                # Fraction both high
                frac_both_high = np.sum((x_vals >= x_threshold) & (y_vals >= y_threshold)) / n
                
                fp[idx] = r
                fp[idx+1] = mean_x_given_y_high
                fp[idx+2] = mean_y_given_x_high
                fp[idx+3] = frac_both_high
            else:
                fp[idx:idx+4] = 0.0
            
            idx += 4
        
        return {
            'vector': fp,
            'layout_version': 'v2_audio',
            'frame_count': self.frame_count,
            'duration': self.match_duration,
            'meta': {
                'profile_id': self.profile_id,
                'user_label': self.user_label,
                'visual_anomaly_fraction': fp[341],
                'input_anomaly_fraction': fp[342],
                'network_anomaly_fraction': fp[343],
                'aim_assist_suspect_score': fp[353],
                'recoil_compensation_suspect_score': fp[354],
                'eomm_lag_suspect_score': fp[355],
                'network_priority_bias_score': fp[356]
            }
        }
    
    def reset(self):
        """Reset for new match (preserves configured max_frames and sample_size)."""
        self.__init__(self.max_frames, self.sample_size)


# Quick test
if __name__ == "__main__":
    print("MatchFingerprintBuilder - Test Harness")
    print("=" * 60)
    
    builder = MatchFingerprintBuilder()
    
    # Simulate 300 frames (5 seconds at 60 FPS)
    for frame_num in range(300):
        t = frame_num / 60.0
        
        # Synthetic fused frame
        fused_frame = {
            'timestamp': t,
            'visual_resonance': {
                'vis_energy_total': np.abs(np.sin(t * 6.0)) + np.random.rand() * 0.1,
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
            },
            'gamepad': {
                'button_press_count': 1.0 if frame_num % 60 < 5 else 0.0,
                'stick_magnitude': 0.5 + 0.3 * np.sin(t * 2.0),
                'stick_velocity': 0.1 + np.random.rand() * 0.05,
                'button_press_rate': 0.2,
                'sticks': [0.3, 0.4, 0.2, 0.1, 0.0, 0.0],
                'buttons': [1, 0, 0, 0]
            },
            'network': {
                'rtt': 20.0 + np.random.rand() * 5.0,
                'packet_loss': 0.0,
                'jitter': 2.0 + np.random.rand(),
                'rtt_mean': 22.0,
                'rtt_std': 3.0,
                'rtt_max': 30.0,
                'rtt_min': 18.0,
                'is_spike': 1.0 if frame_num == 150 else 0.0
            },
            'audio_resonance': {
                'aud_energy_total': 0.3 + 0.2 * np.sin(t * 4.0),
                'aud_energy_var': 0.05 + np.random.rand() * 0.02,
                'aud_spectral_centroid_mean': 2500.0 + np.random.rand() * 500,
                'aud_spectral_centroid_std': 300.0 + np.random.rand() * 100,
                'aud_low_mid_high_balance_low': 0.3 + np.random.rand() * 0.1,
                'aud_low_mid_high_balance_high': 0.35 + np.random.rand() * 0.1,
                'aud_transient_rate': 10.0 + np.random.rand() * 5.0,
                'aud_transient_to_total_ratio': 0.5 + np.random.rand() * 0.2,
                'aud_gunshot_peak_index': np.random.rand() * 0.5,
                'aud_impact_burst_score': np.random.rand() * 0.3,
                'aud_sustain_ratio': 0.4 + np.random.rand() * 0.2,
                'aud_harmonicity_mean': 0.3 + np.random.rand() * 0.2,
                'aud_harmonicity_std': 0.1 + np.random.rand() * 0.05,
                'aud_voice_activity_ratio': np.random.rand() * 0.1,
                'aud_noise_dominance_ratio': 0.6 + np.random.rand() * 0.2,
                'aud_stereo_width': 0.5 + np.random.rand() * 0.3,
                'aud_left_right_bias': (np.random.rand() - 0.5) * 0.4,
                'aud_occlusion_index': np.random.rand() * 0.2,
                'aud_clarity_index': 0.6 + np.random.rand() * 0.2,
                'aud_ui_presence_score': np.random.rand() * 0.1
            }
        }
        
        builder.update(fused_frame)
    
    # Build fingerprint
    result = builder.build()
    
    print(f"\n[Fingerprint Built]")
    print(f"  Frames processed: {result['frame_count']}")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Vector shape: {result['vector'].shape}")
    print(f"  Vector dims: {len(result['vector'])}")
    print(f"  Layout version: {result['layout_version']}")
    print(f"\n[Sample Values]")
    print(f"  Visual energy mean (dim 0): {result['vector'][0]:.4f}")
    print(f"  Gamepad stick magnitude mean (dim 128): {result['vector'][128]:.4f}")
    print(f"  Network RTT mean (dim 224): {result['vector'][224]:.4f}")
    print(f"  Correlation 0 (dim 256): {result['vector'][256]:.4f}")
    print(f"  Frame count (dim 336): {result['vector'][336]:.0f}")
    print(f"  Audio energy mean (dim 365): {result['vector'][365]:.4f}")
    print(f"  Audio correlation 0 (dim 445): {result['vector'][445]:.4f}")
    print(f"\n[Suspect Scores]")
    print(f"  Aim assist: {result['meta']['aim_assist_suspect_score']:.4f}")
    print(f"  Recoil compensation: {result['meta']['recoil_compensation_suspect_score']:.4f}")
    print(f"  EOMM lag: {result['meta']['eomm_lag_suspect_score']:.4f}")
    print(f"  Network bias: {result['meta']['network_priority_bias_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Phase 2 Extended - MatchFingerprintBuilder ready (525 dims)")
    print("  Visual (0-127), Gamepad (128-223), Network (224-255)")
    print("  Cross-modal (256-335), Meta (336-364)")
    print("  Audio (365-444), Audio Cross-modal (445-524)")
