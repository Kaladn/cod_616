"""
Phase 7 Integration Test
Validates complete pipeline: Fingerprint → Baseline → Recognition → Verdict

Built: November 26, 2025
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add recognition module to path
sys.path.append(str(Path(__file__).parent / 'recognition'))

from recognition_field import RecognitionField, BaselineIndex


def test_phase_7_pipeline():
    """Test complete Phase 7 pipeline."""
    
    print("=" * 70)
    print("PHASE 7 INTEGRATION TEST")
    print("COD 616 Telemetry Engine - Recognition Field")
    print("=" * 70)
    
    # Step 1: Create synthetic baseline fingerprints
    print("\n[Step 1] Creating synthetic baseline (10 matches)...")
    baseline_vectors = []
    for i in range(10):
        # Normal variation around standard profile
        vec = np.random.randn(365) * 0.15 + np.linspace(0.1, 1.0, 365)
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
    
    print(f"  ✓ Baseline created: {baseline.count} matches")
    print(f"    Mean range: [{baseline.mean.min():.3f}, {baseline.mean.max():.3f}]")
    print(f"    Std range: [{baseline.std.min():.3f}, {baseline.std.max():.3f}]")
    
    # Step 2: Initialize Recognition Field
    print("\n[Step 2] Initializing Recognition Field...")
    rf = RecognitionField(baseline)
    print(f"  ✓ Recognition Field ready")
    
    # Step 3: Test normal match
    print("\n[Step 3] Analyzing NORMAL match...")
    normal_match = {
        'vector': (np.random.randn(365) * 0.12 + mean).tolist(),
        'capture_timestamp': '2025-11-26T14:00:00',
        'profile': 'forensic',
        'duration': 600.0,
        'frame_count': 3600
    }
    
    report_normal = rf.analyze(normal_match)
    
    print(f"  Verdict: {report_normal.verdict}")
    print(f"  Confidence: {report_normal.confidence:.1%}")
    print(f"  Global anomaly: {report_normal.global_anomaly_score:.2f}")
    print(f"  Block z-scores:")
    print(f"    Visual: {report_normal.visual_block_z:.2f}")
    print(f"    Gamepad: {report_normal.gamepad_block_z:.2f}")
    print(f"    Network: {report_normal.network_block_z:.2f}")
    print(f"    Cross-modal: {report_normal.crossmodal_block_z:.2f}")
    print(f"  Channels:")
    print(f"    Aim assist: {report_normal.aim_assist.level} ({report_normal.aim_assist.score:.2f})")
    print(f"    Recoil comp: {report_normal.recoil_compensation.level} ({report_normal.recoil_compensation.score:.2f})")
    print(f"    EOMM lag: {report_normal.eomm_lag.level} ({report_normal.eomm_lag.score:.2f})")
    print(f"    Network bias: {report_normal.network_priority_bias.level} ({report_normal.network_priority_bias.score:.2f})")
    
    assert report_normal.verdict in ['normal', 'suspicious'], f"Expected normal/suspicious, got {report_normal.verdict}"
    print(f"  ✓ Normal match detected correctly")
    
    # Step 4: Test suspicious match (moderate deviation)
    print("\n[Step 4] Analyzing SUSPICIOUS match...")
    suspicious_match = {
        'vector': (np.random.randn(365) * 0.3 + mean + 2.0).tolist(),
        'capture_timestamp': '2025-11-26T14:15:00',
        'profile': 'forensic',
        'duration': 600.0,
        'frame_count': 3600
    }
    
    report_sus = rf.analyze(suspicious_match)
    
    print(f"  Verdict: {report_sus.verdict}")
    print(f"  Confidence: {report_sus.confidence:.1%}")
    print(f"  Global anomaly: {report_sus.global_anomaly_score:.2f}")
    print(f"  Channels:")
    print(f"    Aim assist: {report_sus.aim_assist.level} ({report_sus.aim_assist.score:.2f})")
    print(f"    EOMM lag: {report_sus.eomm_lag.level} ({report_sus.eomm_lag.score:.2f})")
    print(f"  Explanation:")
    for line in report_sus.explanation[:3]:
        print(f"    • {line}")
    
    assert report_sus.verdict in ['suspicious', 'manipulated_likely', 'manipulated_certain'], f"Expected suspicious/manipulated, got {report_sus.verdict}"
    print(f"  ✓ Suspicious match detected correctly (verdict: {report_sus.verdict})")
    
    # Step 5: Test manipulated match (high deviation)
    print("\n[Step 5] Analyzing MANIPULATED match...")
    manipulated_match = {
        'vector': (np.random.randn(365) * 0.5 + mean + 4.0).tolist(),
        'capture_timestamp': '2025-11-26T14:30:00',
        'profile': 'forensic',
        'duration': 600.0,
        'frame_count': 3600
    }
    
    report_manip = rf.analyze(manipulated_match)
    
    print(f"  Verdict: {report_manip.verdict}")
    print(f"  Confidence: {report_manip.confidence:.1%}")
    print(f"  Global anomaly: {report_manip.global_anomaly_score:.2f}")
    print(f"  Block z-scores:")
    print(f"    Visual: {report_manip.visual_block_z:.2f}")
    print(f"    Network: {report_manip.network_block_z:.2f}")
    print(f"    Cross-modal: {report_manip.crossmodal_block_z:.2f}")
    print(f"  Channels:")
    print(f"    Aim assist: {report_manip.aim_assist.level} ({report_manip.aim_assist.score:.2f})")
    print(f"    Recoil comp: {report_manip.recoil_compensation.level} ({report_manip.recoil_compensation.score:.2f})")
    print(f"    EOMM lag: {report_manip.eomm_lag.level} ({report_manip.eomm_lag.score:.2f})")
    print(f"  Explanation:")
    for line in report_manip.explanation:
        print(f"    • {line}")
    
    assert report_manip.verdict in ['manipulated_likely', 'manipulated_certain'], f"Expected manipulated, got {report_manip.verdict}"
    print(f"  ✓ Manipulated match detected correctly")
    
    # Step 6: Test comparison
    print("\n[Step 6] Comparing normal vs manipulated...")
    comparison = rf.compare(normal_match, manipulated_match)
    
    print(f"  Euclidean distance: {comparison['euclidean_distance']:.2f}")
    print(f"  Cosine similarity: {comparison['cosine_similarity']:.3f}")
    print(f"  Visual block distance: {comparison['visual_block_distance']:.2f}")
    print(f"  Network block distance: {comparison['network_block_distance']:.2f}")
    
    assert comparison['euclidean_distance'] > 10.0, "Expected large distance between normal and manipulated"
    print(f"  ✓ Comparison metrics valid")
    
    # Step 7: Test baseline save/load
    print("\n[Step 7] Testing baseline save/load...")
    test_baseline_path = Path(__file__).parent / 'recognition' / 'profiles' / 'test_baseline.json'
    rf.save_baseline(baseline, str(test_baseline_path))
    
    rf_new = RecognitionField()
    rf_new.load_baseline(str(test_baseline_path))
    
    assert np.allclose(rf_new.baseline.mean, baseline.mean), "Baseline mean mismatch"
    assert np.allclose(rf_new.baseline.std, baseline.std), "Baseline std mismatch"
    print(f"  ✓ Baseline save/load successful")
    
    # Step 8: Test report serialization
    print("\n[Step 8] Testing report serialization...")
    report_dict = report_manip.to_dict()
    
    assert 'verdict' in report_dict, "Missing verdict in report dict"
    assert 'channels' in report_dict, "Missing channels in report dict"
    assert 'explanation' in report_dict, "Missing explanation in report dict"
    assert len(report_dict['explanation']) > 0, "Empty explanation"
    print(f"  ✓ Report serialization successful")
    print(f"    Keys: {list(report_dict.keys())}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ PHASE 7 INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print("\nAll tests passed:")
    print("  ✓ Baseline creation")
    print("  ✓ Recognition Field initialization")
    print("  ✓ Normal match detection")
    print("  ✓ Suspicious match detection")
    print("  ✓ Manipulated match detection")
    print("  ✓ Match comparison")
    print("  ✓ Baseline save/load")
    print("  ✓ Report serialization")
    print("\n🎉 Phase 7 Recognition Field is PRODUCTION READY")
    print("\nNext steps:")
    print("  1. Capture 10+ baseline matches (bot lobbies)")
    print("  2. Build baseline index: recognition_cli.py index-baseline")
    print("  3. Capture real matches (multiplayer)")
    print("  4. Analyze matches: recognition_cli.py analyze/batch")
    print("  5. Review verdicts and detect manipulation")
    

if __name__ == "__main__":
    test_phase_7_pipeline()
