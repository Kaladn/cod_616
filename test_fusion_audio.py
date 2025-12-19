import numpy as np
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('fusion_616_engine', str(Path(__file__).resolve().parent / 'modules' / 'fusion_616_engine.py'))
fusion_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_mod)
Fusion616Engine = fusion_mod.Fusion616Engine


def make_audio_state():
    # Simple audio state with consistent keys sorted order
    keys = [
        'aud_energy_total', 'aud_energy_var', 'aud_spectral_centroid_mean', 'aud_spectral_centroid_std',
        'aud_low_mid_high_balance_low', 'aud_low_mid_high_balance_high', 'aud_transient_rate',
        'aud_transient_to_total_ratio', 'aud_gunshot_peak_index', 'aud_impact_burst_score',
        'aud_sustain_ratio', 'aud_harmonicity_mean', 'aud_harmonicity_std', 'aud_voice_activity_ratio',
        'aud_noise_dominance_ratio', 'aud_stereo_width', 'aud_left_right_bias', 'aud_occlusion_index',
        'aud_clarity_index', 'aud_ui_presence_score'
    ]
    return {k: float(i % 5) for i, k in enumerate(keys)}


def test_fuse_with_audio():
    engine = Fusion616Engine()

    screen = np.zeros(100)
    visual = {f'vis_feature_{i}': 0.1 * i for i in range(20)}
    audio = make_audio_state()
    gamepad = np.zeros(54)
    network = np.zeros(8)

    result = engine.fuse(screen_features=screen, visual_resonance=visual, audio_resonance=audio, gamepad_features=gamepad, network_features=network)

    assert 'fused_vector' in result
    assert result['fused_vector'].shape[0] == engine.total_dim
    assert result['resonance_vector'].shape[0] == 12
    assert 'manipulation_score' in result
    assert isinstance(result['manipulation_score'], float)

    # Ensure audio slice is present in fused_vector at expected position
    # Screen(100) + Visual(20) => audio starts at idx 120
    audio_slice = result['fused_vector'][100+20:100+20+20]
    assert np.any(audio_slice != 0)
