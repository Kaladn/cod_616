"""
Audio Resonance State - COD 616 Audio Organ
CompuCog Acoustic Cognition Engine

Extracts 20-dim audio resonance state from PCM audio buffer.
No ML. Just signal processing.

Architecture: 6 Acoustic Channels → 20-dim AudioResonanceState → 6 Temporal Ops

Built: November 26, 2025
"""

import numpy as np
from typing import Dict, Optional, Deque
from collections import deque
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class AudioResonanceState:
    """
    Phase 1: Audio Organ for COD 616
    
    Extracts 20-dim acoustic resonance state from audio buffer.
    Parallel architecture to ScreenResonanceState (visual organ).
    """
    
    # Audio configuration
    SAMPLE_RATE = 48000
    FRAME_SIZE = 2048  # ~43ms @ 48kHz
    HOP_SIZE = 512     # ~11ms hop, 75% overlap
    
    # Frequency bands
    LOW_BAND = (0, 500)        # Footsteps, bass
    MID_BAND = (500, 4000)     # Voice, gunfire
    HIGH_BAND = (4000, 20000)  # UI, detail
    
    # Detection thresholds
    ONSET_THRESHOLD = 0.05
    VOICE_HARMONICITY_THRESHOLD = 0.6
    SILENCE_THRESHOLD = 0.01
    
    def __init__(self, sample_rate: int = 48000, block_duration: float = 0.5,
                 ema_alpha_fast: float = 0.3, ema_alpha_slow: float = 0.05):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            block_duration: Processing block duration (seconds)
            ema_alpha_fast: Fast EMA decay for micro-events
            ema_alpha_slow: Slow EMA decay for long-term trends
        """
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_samples = int(sample_rate * block_duration)
        
        # EMA parameters
        self.alpha_fast = ema_alpha_fast
        self.alpha_slow = ema_alpha_slow
        
        # State history (15 blocks = 7.5 seconds)
        self.history: Deque[Dict[str, float]] = deque(maxlen=15)
        
        # EMA state
        self.fast_ema: Optional[Dict[str, float]] = None
        self.slow_ema: Optional[Dict[str, float]] = None
        
        # Previous spectrum for onset detection
        self.prev_spectrum: Optional[np.ndarray] = None
        
        # FFT frequencies
        self.fft_freqs = np.fft.rfftfreq(self.FRAME_SIZE, 1.0 / sample_rate)
        
        # Hanning window
        self.window = np.hanning(self.FRAME_SIZE)
        
        print(f"[AudioResonanceState] Initialized")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Block duration: {block_duration}s ({self.block_samples} samples)")
        print(f"  Frame size: {self.FRAME_SIZE} samples (~{self.FRAME_SIZE/sample_rate*1000:.1f}ms)")
        print(f"  Hop size: {self.HOP_SIZE} samples (~{self.HOP_SIZE/sample_rate*1000:.1f}ms)")
    
    def _extract_frames(self, audio_buffer: np.ndarray) -> np.ndarray:
        """
        Split audio buffer into overlapping frames.
        
        Args:
            audio_buffer: Audio samples (mono or stereo)
        
        Returns:
            Frame array (num_frames, FRAME_SIZE)
        """
        # Convert stereo to mono if needed
        if audio_buffer.ndim == 2:
            audio_buffer = np.mean(audio_buffer, axis=1)
        
        num_frames = 1 + (len(audio_buffer) - self.FRAME_SIZE) // self.HOP_SIZE
        frames = np.zeros((num_frames, self.FRAME_SIZE))
        
        for i in range(num_frames):
            start = i * self.HOP_SIZE
            end = start + self.FRAME_SIZE
            if end <= len(audio_buffer):
                frames[i] = audio_buffer[start:end]
        
        return frames
    
    def _compute_spectrum(self, frame: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum via FFT."""
        windowed = frame * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        return spectrum
    
    def _compute_energy(self, frame: np.ndarray) -> float:
        """Compute RMS energy."""
        return np.sqrt(np.mean(frame ** 2))
    
    def _compute_spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Compute spectral centroid (brightness)."""
        total_energy = np.sum(spectrum)
        if total_energy < 1e-10:
            return 0.0
        centroid = np.sum(self.fft_freqs * spectrum) / total_energy
        return float(centroid)
    
    def _compute_band_energies(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Compute energy in low/mid/high bands."""
        low_mask = (self.fft_freqs >= self.LOW_BAND[0]) & (self.fft_freqs < self.LOW_BAND[1])
        mid_mask = (self.fft_freqs >= self.MID_BAND[0]) & (self.fft_freqs < self.MID_BAND[1])
        high_mask = (self.fft_freqs >= self.HIGH_BAND[0]) & (self.fft_freqs < self.HIGH_BAND[1])
        
        energy_low = np.sum(spectrum[low_mask] ** 2)
        energy_mid = np.sum(spectrum[mid_mask] ** 2)
        energy_high = np.sum(spectrum[high_mask] ** 2)
        
        total = energy_low + energy_mid + energy_high + 1e-10
        
        return {
            'low': float(energy_low),
            'mid': float(energy_mid),
            'high': float(energy_high),
            'low_ratio': float(energy_low / total),
            'mid_ratio': float(energy_mid / total),
            'high_ratio': float(energy_high / total)
        }
    
    def _compute_onset_strength(self, spectrum: np.ndarray) -> float:
        """Compute spectral flux (onset strength)."""
        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum
            return 0.0
        
        flux = spectrum - self.prev_spectrum
        onset_strength = np.sum(np.maximum(0, flux))
        self.prev_spectrum = spectrum.copy()
        
        return float(onset_strength)
    
    def _compute_harmonicity(self, spectrum: np.ndarray) -> float:
        """
        Compute harmonicity via spectral flatness.
        Flat spectrum = noise (0), Peaked = harmonic (1)
        """
        spectrum = spectrum + 1e-10
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / arith_mean
        harmonicity = 1.0 - flatness
        return float(np.clip(harmonicity, 0.0, 1.0))
    
    def _compute_stereo_features(self, audio_buffer: np.ndarray) -> Dict[str, float]:
        """Compute stereo width and L/R bias."""
        if audio_buffer.ndim != 2:
            # Mono signal - no stereo features
            return {
                'stereo_width': 0.0,
                'left_right_bias': 0.0,
                'stereo_correlation': 1.0
            }
        
        left = audio_buffer[:, 0]
        right = audio_buffer[:, 1]
        
        # L/R energy
        left_energy = np.sqrt(np.mean(left ** 2))
        right_energy = np.sqrt(np.mean(right ** 2))
        
        # L/R balance
        total_energy = left_energy + right_energy + 1e-10
        lr_bias = (left_energy - right_energy) / total_energy
        
        # Stereo correlation
        if len(left) > 1 and len(right) > 1:
            correlation = np.corrcoef(left, right)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        stereo_width = 1.0 - abs(correlation)
        
        return {
            'stereo_width': float(np.clip(stereo_width, 0.0, 1.0)),
            'left_right_bias': float(np.clip(lr_bias, -1.0, 1.0)),
            'stereo_correlation': float(correlation)
        }
    
    def update(self, audio_buffer: np.ndarray) -> Dict[str, float]:
        """
        Process audio buffer and extract 20-dim resonance state.
        
        Args:
            audio_buffer: Audio samples (mono or stereo), shape (samples,) or (samples, 2)
        
        Returns:
            Dict with 20 audio features
        """
        # Extract frames
        frames = self._extract_frames(audio_buffer)
        num_frames = len(frames)
        
        if num_frames == 0:
            # Return silence state
            return self._get_silence_state()
        
        # Per-frame features
        energies = []
        centroids = []
        onsets = []
        harmonicities = []
        band_energies_list = []
        
        for frame in frames:
            # Energy
            energy = self._compute_energy(frame)
            energies.append(energy)
            
            # Spectrum
            spectrum = self._compute_spectrum(frame)
            
            # Spectral features
            centroid = self._compute_spectral_centroid(spectrum)
            centroids.append(centroid)
            
            # Band energies
            band_energies = self._compute_band_energies(spectrum)
            band_energies_list.append(band_energies)
            
            # Onset detection
            onset = self._compute_onset_strength(spectrum)
            onsets.append(onset)
            
            # Harmonicity
            harmonicity = self._compute_harmonicity(spectrum)
            harmonicities.append(harmonicity)
        
        # Convert to arrays
        energies = np.array(energies)
        centroids = np.array(centroids)
        onsets = np.array(onsets)
        harmonicities = np.array(harmonicities)
        
        # Aggregate band energies
        low_ratios = [b['low_ratio'] for b in band_energies_list]
        high_ratios = [b['high_ratio'] for b in band_energies_list]
        
        # Stereo features
        stereo_features = self._compute_stereo_features(audio_buffer)
        
        # Build 20-dim state
        state = {}
        
        # A. Energy & Spectral Shape (6 features)
        state['aud_energy_total'] = float(np.mean(energies))
        state['aud_energy_var'] = float(np.var(energies))
        state['aud_spectral_centroid_mean'] = float(np.mean(centroids))
        state['aud_spectral_centroid_std'] = float(np.std(centroids))
        state['aud_low_mid_high_balance_low'] = float(np.mean(low_ratios))
        state['aud_low_mid_high_balance_high'] = float(np.mean(high_ratios))
        
        # B. Transient / Event Structure (5 features)
        onset_peaks = np.sum(onsets > self.ONSET_THRESHOLD)
        transient_rate = onset_peaks / self.block_duration
        transient_energy = np.sum(onsets)
        total_energy = np.sum(energies) + 1e-10
        transient_ratio = transient_energy / total_energy
        
        state['aud_transient_rate'] = float(transient_rate)
        state['aud_transient_to_total_ratio'] = float(np.clip(transient_ratio, 0.0, 1.0))
        state['aud_gunshot_peak_index'] = float(np.max(onsets) / (np.mean(onsets) * 10 + 1e-10))
        
        # Impact burst score (consecutive high onsets)
        burst_count = 0
        in_burst = False
        for onset in onsets:
            if onset > self.ONSET_THRESHOLD:
                if not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                in_burst = False
        state['aud_impact_burst_score'] = float(np.clip(burst_count / (num_frames / 10 + 1), 0.0, 1.0))
        
        # Sustain ratio (low variance = sustained)
        sustain_variance = np.var(energies)
        max_variance = 0.1  # Normalize
        state['aud_sustain_ratio'] = float(1.0 - np.clip(sustain_variance / max_variance, 0.0, 1.0))
        
        # C. Harmonicity & Voice/Noise Split (4 features)
        state['aud_harmonicity_mean'] = float(np.mean(harmonicities))
        state['aud_harmonicity_std'] = float(np.std(harmonicities))
        
        voice_frames = np.sum(harmonicities > self.VOICE_HARMONICITY_THRESHOLD)
        state['aud_voice_activity_ratio'] = float(voice_frames / num_frames)
        
        noise_ratios = 1.0 - harmonicities
        state['aud_noise_dominance_ratio'] = float(np.mean(noise_ratios))
        
        # D. Spatial & Occlusion (5 features)
        state['aud_stereo_width'] = stereo_features['stereo_width']
        state['aud_left_right_bias'] = stereo_features['left_right_bias']
        
        # Occlusion index (high energy + low centroid)
        mean_centroid_norm = np.mean(centroids) / 8000.0  # Normalize to ~8kHz
        occlusion_index = state['aud_energy_total'] * (1.0 - mean_centroid_norm)
        state['aud_occlusion_index'] = float(np.clip(occlusion_index, 0.0, 1.0))
        
        # Clarity index (high centroid + wide stereo)
        clarity_index = mean_centroid_norm * stereo_features['stereo_width']
        state['aud_clarity_index'] = float(np.clip(clarity_index, 0.0, 1.0))
        
        # UI presence (high-freq + low-energy + stable)
        is_high_freq = state['aud_low_mid_high_balance_high'] > 0.5
        is_low_energy = state['aud_energy_total'] < 0.1
        is_stable = state['aud_energy_var'] < 0.01
        ui_score = float(is_high_freq) * 0.5 + float(is_low_energy) * 0.3 + float(is_stable) * 0.2
        state['aud_ui_presence_score'] = float(ui_score)
        
        # Store in history
        self.history.append(state)
        
        # Update EMAs
        self._update_emas(state)
        
        return state
    
    def _get_silence_state(self) -> Dict[str, float]:
        """Return zero state for silence."""
        return {
            'aud_energy_total': 0.0,
            'aud_energy_var': 0.0,
            'aud_spectral_centroid_mean': 0.0,
            'aud_spectral_centroid_std': 0.0,
            'aud_low_mid_high_balance_low': 0.33,
            'aud_low_mid_high_balance_high': 0.33,
            'aud_transient_rate': 0.0,
            'aud_transient_to_total_ratio': 0.0,
            'aud_gunshot_peak_index': 0.0,
            'aud_impact_burst_score': 0.0,
            'aud_sustain_ratio': 1.0,
            'aud_harmonicity_mean': 0.0,
            'aud_harmonicity_std': 0.0,
            'aud_voice_activity_ratio': 0.0,
            'aud_noise_dominance_ratio': 0.0,
            'aud_stereo_width': 0.0,
            'aud_left_right_bias': 0.0,
            'aud_occlusion_index': 0.0,
            'aud_clarity_index': 0.0,
            'aud_ui_presence_score': 0.0
        }
    
    def _update_emas(self, state: Dict[str, float]):
        """Update fast and slow EMAs."""
        if self.fast_ema is None:
            self.fast_ema = state.copy()
            self.slow_ema = state.copy()
        else:
            for key in state.keys():
                self.fast_ema[key] = self.alpha_fast * state[key] + (1 - self.alpha_fast) * self.fast_ema[key]
                self.slow_ema[key] = self.alpha_slow * state[key] + (1 - self.alpha_slow) * self.slow_ema[key]
    
    def get_temporal_features(self) -> Dict:
        """Get temporal operations over history."""
        if len(self.history) < 2:
            return {}
        
        # Convert history to arrays
        keys = list(self.history[0].keys())
        history_arrays = {k: np.array([h[k] for h in self.history]) for k in keys}
        
        temporal = {}
        
        # Fast/slow EMAs
        temporal['fast_ema'] = self.fast_ema.copy() if self.fast_ema else {}
        temporal['slow_ema'] = self.slow_ema.copy() if self.slow_ema else {}
        
        # Derivatives (current - previous)
        current = self.history[-1]
        previous = self.history[-2]
        temporal['derivatives'] = {k: current[k] - previous[k] for k in keys}
        
        # Segment classification
        recent = list(self.history)[-5:]  # Last 2.5s
        avg_energy = np.mean([h['aud_energy_total'] for h in recent])
        avg_transient = np.mean([h['aud_transient_rate'] for h in recent])
        avg_voice = np.mean([h['aud_voice_activity_ratio'] for h in recent])
        
        if avg_energy < 0.05:
            segment = "silence"
        elif avg_transient > 10.0 and avg_energy > 0.2:
            segment = "firefight"
        elif avg_voice > 0.3:
            segment = "voice_comm"
        else:
            segment = "ambience"
        
        temporal['acoustic_segment'] = {
            'current_segment': segment,
            'avg_energy': float(avg_energy),
            'avg_transient': float(avg_transient),
            'avg_voice': float(avg_voice)
        }
        
        # Anomaly detection
        anomalies = {}
        
        # Sudden silence
        energy_drop = history_arrays['aud_energy_total'][0] - history_arrays['aud_energy_total'][-1]
        anomalies['sudden_silence_anomaly'] = float(np.maximum(0.0, energy_drop) * 2.0)
        
        # Muffling onset
        clarity_drop = history_arrays['aud_clarity_index'][0] - history_arrays['aud_clarity_index'][-1]
        anomalies['muffling_onset_anomaly'] = float(np.maximum(0.0, clarity_drop) * 2.0)
        
        # Voice tilt
        anomalies['voice_tilt_anomaly'] = float(avg_voice if avg_voice > 0.5 else 0.0)
        
        # UI spam
        avg_ui = np.mean([h['aud_ui_presence_score'] for h in recent])
        anomalies['ui_spam_anomaly'] = float(avg_ui if avg_ui > 0.5 else 0.0)
        
        temporal['anomaly_scores'] = anomalies
        
        return temporal


# Quick test
if __name__ == "__main__":
    print("AudioResonanceState - Test Harness")
    print("=" * 60)
    
    audio = AudioResonanceState()
    
    # Test 1: Silence
    print("\n[Test 1] Silence")
    silence = np.zeros(24000)
    state = audio.update(silence)
    print(f"  Energy: {state['aud_energy_total']:.3f}")
    print(f"  Transient rate: {state['aud_transient_rate']:.1f}/sec")
    print(f"  Segment: silence" if state['aud_energy_total'] < 0.05 else "  Segment: active")
    
    # Test 2: Pink noise (ambience)
    print("\n[Test 2] Pink noise (ambience)")
    noise = np.random.randn(24000) * 0.1
    state = audio.update(noise)
    print(f"  Energy: {state['aud_energy_total']:.3f}")
    print(f"  Harmonicity: {state['aud_harmonicity_mean']:.3f}")
    print(f"  Noise dominance: {state['aud_noise_dominance_ratio']:.3f}")
    
    # Test 3: Sine tone (voice simulation)
    print("\n[Test 3] Sine tone (voice)")
    t = np.arange(24000) / 48000.0
    tone = 0.3 * np.sin(2 * np.pi * 300 * t)
    state = audio.update(tone)
    print(f"  Energy: {state['aud_energy_total']:.3f}")
    print(f"  Harmonicity: {state['aud_harmonicity_mean']:.3f}")
    print(f"  Voice activity: {state['aud_voice_activity_ratio']:.3f}")
    
    # Test 4: Burst noise (gunfire)
    print("\n[Test 4] Burst noise (gunfire)")
    gunfire = np.zeros(24000)
    gunfire[5000:5200] = np.random.randn(200) * 0.8
    state = audio.update(gunfire)
    print(f"  Energy: {state['aud_energy_total']:.3f}")
    print(f"  Transient rate: {state['aud_transient_rate']:.1f}/sec")
    print(f"  Gunshot peak: {state['aud_gunshot_peak_index']:.3f}")
    print(f"  Impact burst: {state['aud_impact_burst_score']:.3f}")
    
    # Test 5: Temporal features
    print("\n[Test 5] Temporal features")
    temporal = audio.get_temporal_features()
    if 'acoustic_segment' in temporal:
        seg = temporal['acoustic_segment']
        print(f"  Segment: {seg['current_segment']}")
        print(f"  Avg energy: {seg['avg_energy']:.3f}")
        print(f"  Anomalies:")
        for name, score in temporal['anomaly_scores'].items():
            if score > 0.01:
                print(f"    {name}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ AudioResonanceState test complete")
    print(f"  20-dim state extracted per 0.5s block")
    print(f"  Temporal features tracked over 7.5s history")
    print(f"  Ready for integration with COD 616")
