"""
Recognition Field - Phase 7 Implementation
CompuCog 616 COD Telemetry Engine

Baseline deviation analysis for manipulation detection.
No ML. Just stats, distances, and physics.

Layer A: Z-scores & block deviations
Layer B: Pattern classification (suspect channels)
Layer C: Match verdict + narrative

Built: November 26, 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BaselineIndex:
    """Statistical profile of 'fair' matches."""
    layout_version: str
    count: int
    mean: np.ndarray  # (365,)
    std: np.ndarray   # (365,)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'layout_version': self.layout_version,
            'count': self.count,
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BaselineIndex':
        """Load from JSON dict."""
        return cls(
            layout_version=d['layout_version'],
            count=d['count'],
            mean=np.array(d['mean']),
            std=np.array(d['std'])
        )


@dataclass
class ChannelScore:
    """Suspect channel score."""
    score: float  # 0-1
    level: str    # 'none', 'low', 'medium', 'high', 'critical'
    contributing_dims: List[int]  # Dims that drove this score


@dataclass
class RecognitionReport:
    """Full match analysis report."""
    match_id: str
    profile: str
    duration_seconds: float
    frame_count: int
    
    # Block-level z-scores
    global_anomaly_score: float
    visual_block_z: float
    gamepad_block_z: float
    network_block_z: float
    crossmodal_block_z: float
    meta_block_z: float
    
    # Suspect channels
    aim_assist: ChannelScore
    recoil_compensation: ChannelScore
    eomm_lag: ChannelScore
    network_priority_bias: ChannelScore
    
    # Verdict
    verdict: str  # 'normal', 'suspicious', 'manipulated_likely', 'manipulated_certain'
    confidence: float  # 0-1
    explanation: List[str]
    
    # Metadata
    analysis_timestamp: str
    baseline_count: int
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'match_id': self.match_id,
            'profile': self.profile,
            'duration_seconds': self.duration_seconds,
            'frame_count': self.frame_count,
            'global_anomaly_score': float(self.global_anomaly_score),
            'visual_block_z': float(self.visual_block_z),
            'gamepad_block_z': float(self.gamepad_block_z),
            'network_block_z': float(self.network_block_z),
            'crossmodal_block_z': float(self.crossmodal_block_z),
            'meta_block_z': float(self.meta_block_z),
            'channels': {
                'aim_assist': {
                    'score': float(self.aim_assist.score),
                    'level': self.aim_assist.level,
                    'contributing_dims': self.aim_assist.contributing_dims
                },
                'recoil_compensation': {
                    'score': float(self.recoil_compensation.score),
                    'level': self.recoil_compensation.level,
                    'contributing_dims': self.recoil_compensation.contributing_dims
                },
                'eomm_lag': {
                    'score': float(self.eomm_lag.score),
                    'level': self.eomm_lag.level,
                    'contributing_dims': self.eomm_lag.contributing_dims
                },
                'network_priority_bias': {
                    'score': float(self.network_priority_bias.score),
                    'level': self.network_priority_bias.level,
                    'contributing_dims': self.network_priority_bias.contributing_dims
                }
            },
            'verdict': self.verdict,
            'confidence': float(self.confidence),
            'explanation': self.explanation,
            'analysis_timestamp': self.analysis_timestamp,
            'baseline_count': self.baseline_count
        }


class RecognitionField:
    """
    Phase 7: Recognition Field for COD 616 Telemetry Engine.
    
    Analyzes match fingerprints against baseline to detect manipulation.
    """
    
    # Block boundaries (dims)
    VISUAL_BLOCK = (0, 128)
    GAMEPAD_BLOCK = (128, 224)
    NETWORK_BLOCK = (224, 256)
    CROSSMODAL_BLOCK = (256, 336)
    META_BLOCK = (336, 365)
    
    # Channel-specific dimensions (key features for each suspect pattern)
    AIM_ASSIST_DIMS = [
        19,  # vis_aim_lock_score (mean)
        23,  # vis_aim_lock_score (p90)
        # Cross-modal: visual center vs gamepad stick (correlation block)
        256, 257, 258, 259,  # First correlation pair
        100, 101,  # Visual anomaly distribution
    ]
    
    RECOIL_COMP_DIMS = [
        10,  # vis_highfreq_energy (mean)
        14,  # vis_smoothness_index (mean)
        8,   # vis_recoil_vertical_bias (mean)
        354, # recoil_compensation_suspect_score (meta)
    ]
    
    EOMM_LAG_DIMS = [
        14,  # vis_stutter_score (mean)
        224, # net_rtt (mean)
        228, # net_jitter (mean)
        232, # net_is_spike (mean)
        343, # network_anomaly_fraction (meta)
        355, # eomm_lag_suspect_score (meta)
    ]
    
    NETWORK_BIAS_DIMS = [
        224, 225, 226, 227,  # net_rtt stats
        228, 229, 230, 231,  # net_jitter stats
        # Cross-modal: input intensity vs network
        280, 281, 282, 283,  # Correlation pairs
        356, # network_priority_bias_score (meta)
    ]
    
    def __init__(self, baseline_index: Optional[BaselineIndex] = None):
        """
        Args:
            baseline_index: Pre-loaded baseline. If None, must call load_baseline().
        """
        self.baseline = baseline_index
        self.eps = 1e-6  # Floor for std to avoid division by zero
        
        print("[RecognitionField] Initialized")
        if self.baseline is not None:
            print(f"  Baseline: {self.baseline.count} matches")
    
    def load_baseline(self, path: str):
        """Load baseline index from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.baseline = BaselineIndex.from_dict(data)
        print(f"[RecognitionField] Loaded baseline: {self.baseline.count} matches")
    
    def build_baseline(self, fingerprint_paths: List[str]) -> BaselineIndex:
        """
        Build baseline index from list of fingerprint files.
        
        Args:
            fingerprint_paths: Paths to baseline fingerprint JSONs
        
        Returns:
            BaselineIndex with mean/std for all 365 dims
        """
        vectors = []
        
        for path in fingerprint_paths:
            with open(path, 'r') as f:
                fp = json.load(f)
            vectors.append(np.array(fp['vector']))
        
        vectors = np.array(vectors)  # (N, 365)
        
        mean = np.mean(vectors, axis=0)
        std = np.std(vectors, axis=0)
        std = np.maximum(std, self.eps)  # Floor std
        
        baseline = BaselineIndex(
            layout_version='v1',
            count=len(vectors),
            mean=mean,
            std=std
        )
        
        print(f"[RecognitionField] Built baseline from {len(vectors)} fingerprints")
        
        return baseline
    
    def save_baseline(self, baseline: BaselineIndex, path: str):
        """Save baseline index to JSON."""
        with open(path, 'w') as f:
            json.dump(baseline.to_dict(), f, indent=2)
        
        print(f"[RecognitionField] Saved baseline: {path}")
    
    def _compute_z_scores(self, vector: np.ndarray) -> np.ndarray:
        """
        Compute z-scores for all 365 dims.
        
        Args:
            vector: Match fingerprint (365,)
        
        Returns:
            z-scores (365,)
        """
        if self.baseline is None:
            raise ValueError("No baseline loaded. Call load_baseline() first.")
        
        z = (vector - self.baseline.mean) / (self.baseline.std + self.eps)
        return z
    
    def _compute_block_z(self, z_scores: np.ndarray, start: int, end: int) -> float:
        """
        Compute RMS z-score for a block of dims.
        
        Args:
            z_scores: Full z-score vector (365,)
            start, end: Block boundaries
        
        Returns:
            RMS of |z| in block
        """
        block_z = z_scores[start:end]
        rms_z = np.sqrt(np.mean(block_z ** 2))
        return float(rms_z)
    
    def _compute_channel_score(self, z_scores: np.ndarray, dims: List[int]) -> Tuple[float, List[int]]:
        """
        Compute suspect channel score from key dimensions.
        
        Args:
            z_scores: Full z-score vector (365,)
            dims: List of dimension indices for this channel
        
        Returns:
            (score, contributing_dims) where score is 0-1 and contributing_dims are top offenders
        """
        channel_z = np.abs(z_scores[dims])
        
        # Score = sigmoid(weighted RMS)
        rms_z = np.sqrt(np.mean(channel_z ** 2))
        score = 1.0 / (1.0 + np.exp(-1.5 * (rms_z - 1.5)))  # Sigmoid centered at z=1.5
        
        # Find top 3 contributing dims
        sorted_indices = np.argsort(-channel_z)[:3]
        contributing_dims = [dims[i] for i in sorted_indices]
        
        return float(score), contributing_dims
    
    def _score_to_level(self, score: float) -> str:
        """Convert 0-1 score to severity level."""
        if score < 0.2:
            return 'none'
        elif score < 0.4:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _compute_verdict(self, global_z: float, channel_scores: Dict[str, float]) -> Tuple[str, float, List[str]]:
        """
        Compute match verdict and explanation.
        
        Args:
            global_z: Global anomaly score
            channel_scores: Dict of channel name → score
        
        Returns:
            (verdict, confidence, explanation_lines)
        """
        explanation = []
        
        # Count high/critical channels
        high_channels = [name for name, score in channel_scores.items() if score >= 0.6]
        critical_channels = [name for name, score in channel_scores.items() if score >= 0.8]
        
        # Verdict logic
        if global_z < 1.5 and len(high_channels) == 0:
            verdict = 'normal'
            confidence = 0.9
            explanation.append("Match fingerprint within normal baseline variation.")
        
        elif global_z < 2.5 and len(high_channels) <= 1:
            verdict = 'suspicious'
            confidence = 0.6
            explanation.append("Moderate deviation from baseline detected.")
            if high_channels:
                explanation.append(f"Elevated activity in {high_channels[0]} channel.")
        
        elif len(critical_channels) > 0 or global_z >= 3.5:
            verdict = 'manipulated_certain'
            confidence = 0.95
            explanation.append("Severe deviation from baseline across multiple channels.")
            if critical_channels:
                explanation.append(f"Critical anomalies: {', '.join(critical_channels)}.")
        
        else:
            verdict = 'manipulated_likely'
            confidence = 0.75
            explanation.append("Significant deviation from baseline detected.")
            if len(high_channels) > 1:
                explanation.append(f"Multiple suspect channels active: {', '.join(high_channels)}.")
        
        # Add channel-specific explanations
        if channel_scores['aim_assist'] >= 0.6:
            explanation.append("Aim-assist channel: visual smoothing exceeds baseline with low micro-adjust input.")
        
        if channel_scores['recoil_compensation'] >= 0.6:
            explanation.append("Recoil-comp channel: high-frequency visual energy suppressed during sustained fire.")
        
        if channel_scores['eomm_lag'] >= 0.6:
            explanation.append("EOMM channel: stutter/lag correlated with high-intensity engagement periods.")
        
        if channel_scores['network_priority_bias'] >= 0.6:
            explanation.append("Network bias channel: RTT/jitter spikes during high input activity.")
        
        return verdict, confidence, explanation
    
    def analyze(self, fingerprint: Dict) -> RecognitionReport:
        """
        Analyze a match fingerprint.
        
        Args:
            fingerprint: Match fingerprint dict with 'vector', 'meta', etc.
        
        Returns:
            RecognitionReport with full analysis
        """
        if self.baseline is None:
            raise ValueError("No baseline loaded. Call load_baseline() first.")
        
        vector = np.array(fingerprint['vector'])
        
        # Compute z-scores
        z_scores = self._compute_z_scores(vector)
        
        # Block z-scores
        visual_z = self._compute_block_z(z_scores, *self.VISUAL_BLOCK)
        gamepad_z = self._compute_block_z(z_scores, *self.GAMEPAD_BLOCK)
        network_z = self._compute_block_z(z_scores, *self.NETWORK_BLOCK)
        crossmodal_z = self._compute_block_z(z_scores, *self.CROSSMODAL_BLOCK)
        meta_z = self._compute_block_z(z_scores, *self.META_BLOCK)
        
        # Global anomaly score (weighted average of blocks)
        global_z = np.sqrt(
            0.25 * visual_z**2 +
            0.20 * gamepad_z**2 +
            0.20 * network_z**2 +
            0.25 * crossmodal_z**2 +
            0.10 * meta_z**2
        )
        
        # Channel scores
        aim_score, aim_dims = self._compute_channel_score(z_scores, self.AIM_ASSIST_DIMS)
        recoil_score, recoil_dims = self._compute_channel_score(z_scores, self.RECOIL_COMP_DIMS)
        eomm_score, eomm_dims = self._compute_channel_score(z_scores, self.EOMM_LAG_DIMS)
        network_score, network_dims = self._compute_channel_score(z_scores, self.NETWORK_BIAS_DIMS)
        
        channel_scores = {
            'aim_assist': aim_score,
            'recoil_compensation': recoil_score,
            'eomm_lag': eomm_score,
            'network_priority_bias': network_score
        }
        
        # Verdict
        verdict, confidence, explanation = self._compute_verdict(global_z, channel_scores)
        
        # Build report
        report = RecognitionReport(
            match_id=fingerprint.get('capture_timestamp', 'unknown'),
            profile=fingerprint.get('profile', 'unknown'),
            duration_seconds=fingerprint.get('duration', 0.0),
            frame_count=fingerprint.get('frame_count', 0),
            global_anomaly_score=global_z,
            visual_block_z=visual_z,
            gamepad_block_z=gamepad_z,
            network_block_z=network_z,
            crossmodal_block_z=crossmodal_z,
            meta_block_z=meta_z,
            aim_assist=ChannelScore(aim_score, self._score_to_level(aim_score), aim_dims),
            recoil_compensation=ChannelScore(recoil_score, self._score_to_level(recoil_score), recoil_dims),
            eomm_lag=ChannelScore(eomm_score, self._score_to_level(eomm_score), eomm_dims),
            network_priority_bias=ChannelScore(network_score, self._score_to_level(network_score), network_dims),
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            analysis_timestamp=datetime.now().isoformat(),
            baseline_count=self.baseline.count
        )
        
        return report
    
    def compare(self, fp_a: Dict, fp_b: Dict) -> Dict:
        """
        Compare two fingerprints directly.
        
        Args:
            fp_a: First fingerprint
            fp_b: Second fingerprint
        
        Returns:
            Dict with comparison metrics
        """
        vec_a = np.array(fp_a['vector'])
        vec_b = np.array(fp_b['vector'])
        
        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(vec_a - vec_b))
        
        # Cosine similarity
        cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        
        # Block-level distances
        visual_dist = float(np.linalg.norm(vec_a[0:128] - vec_b[0:128]))
        gamepad_dist = float(np.linalg.norm(vec_a[128:224] - vec_b[128:224]))
        network_dist = float(np.linalg.norm(vec_a[224:256] - vec_b[224:256]))
        crossmodal_dist = float(np.linalg.norm(vec_a[256:336] - vec_b[256:336]))
        
        return {
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim,
            'visual_block_distance': visual_dist,
            'gamepad_block_distance': gamepad_dist,
            'network_block_distance': network_dist,
            'crossmodal_block_distance': crossmodal_dist,
            'match_a_id': fp_a.get('capture_timestamp', 'unknown'),
            'match_b_id': fp_b.get('capture_timestamp', 'unknown')
        }


# Quick test
if __name__ == "__main__":
    print("RecognitionField - Test Harness")
    print("=" * 60)
    
    # Create synthetic baseline
    print("\n[Creating synthetic baseline]")
    baseline_vectors = []
    for i in range(10):
        vec = np.random.randn(365) * 0.1 + np.linspace(0, 1, 365)
        baseline_vectors.append(vec)
    
    mean = np.mean(baseline_vectors, axis=0)
    std = np.std(baseline_vectors, axis=0)
    std = np.maximum(std, 1e-6)
    
    baseline = BaselineIndex(
        layout_version='v1',
        count=10,
        mean=mean,
        std=std
    )
    
    # Create Recognition Field
    rf = RecognitionField(baseline)
    
    # Create synthetic match (normal)
    normal_match = {
        'vector': (np.random.randn(365) * 0.1 + mean).tolist(),
        'capture_timestamp': '2025-11-26T14:00:00',
        'profile': 'forensic',
        'duration': 600.0,
        'frame_count': 3600
    }
    
    # Analyze normal match
    print("\n[Analyzing normal match]")
    report = rf.analyze(normal_match)
    print(f"  Verdict: {report.verdict}")
    print(f"  Confidence: {report.confidence:.2f}")
    print(f"  Global anomaly: {report.global_anomaly_score:.2f}")
    print(f"  Aim assist: {report.aim_assist.level} ({report.aim_assist.score:.2f})")
    print(f"  Recoil comp: {report.recoil_compensation.level} ({report.recoil_compensation.score:.2f})")
    
    # Create synthetic match (suspicious)
    sus_match = {
        'vector': (np.random.randn(365) * 0.5 + mean + 3.0).tolist(),
        'capture_timestamp': '2025-11-26T14:30:00',
        'profile': 'forensic',
        'duration': 600.0,
        'frame_count': 3600
    }
    
    # Analyze suspicious match
    print("\n[Analyzing suspicious match]")
    report = rf.analyze(sus_match)
    print(f"  Verdict: {report.verdict}")
    print(f"  Confidence: {report.confidence:.2f}")
    print(f"  Global anomaly: {report.global_anomaly_score:.2f}")
    print(f"  Aim assist: {report.aim_assist.level} ({report.aim_assist.score:.2f})")
    print(f"  EOMM lag: {report.eomm_lag.level} ({report.eomm_lag.score:.2f})")
    print(f"\n  Explanation:")
    for line in report.explanation:
        print(f"    - {line}")
    
    print("\n" + "=" * 60)
    print("✓ Phase 7 RecognitionField ready")
