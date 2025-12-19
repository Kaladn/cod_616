import numpy as np
from match_fingerprint_builder import MatchFingerprintBuilder


def make_fused_frame(t: float):
    return {
        'timestamp': t,
        'visual_resonance': {
            'vis_energy_total': float(np.abs(np.sin(t * 2.0))),
            'vis_energy_mean': 0.1 + float(np.random.rand() * 0.05),
            'vis_highfreq_energy': float(np.abs(np.sin(t * 4.0)))
        },
        'gamepad': {
            'button_press_count': 1.0 if int(t*10) % 10 < 2 else 0.0,
            'stick_magnitude': 0.5 + 0.1 * float(np.sin(t))
        },
        'network': {
            'rtt': 20.0 + float(np.random.rand() * 5.0),
            'packet_loss': 0.0,
            'jitter': 1.0 + float(np.random.rand())
        }
    }


def test_bounded_sampling_and_build():
    builder = MatchFingerprintBuilder(max_frames=100, sample_size=50)

    # Feed 200 frames (exceeds sample_size)
    for i in range(200):
        t = i / 20.0
        fused = make_fused_frame(t)
        builder.update(fused)

    assert builder.frame_count == 200

    # Ensure stored sample lengths are bounded by sample_size
    for name, stats in builder.visual_stats.items():
        assert len(stats['values']) <= 50

    # Correlation sample bounds
    for pair, stats in builder.correlation_stats.items():
        assert len(stats['x_vals']) <= 50
        assert len(stats['y_vals']) <= 50

    # Build fingerprint should succeed and have expected shape
    fp = builder.build()
    assert 'vector' in fp
    assert fp['vector'].shape == (525,)
    assert fp['frame_count'] == 200


def test_small_run_no_errors():
    builder = MatchFingerprintBuilder(sample_size=10)
    for i in range(30):
        t = i / 30.0
        builder.update(make_fused_frame(t))

    fp = builder.build()
    assert fp['vector'].shape == (525,)
    assert fp['frame_count'] == 30
