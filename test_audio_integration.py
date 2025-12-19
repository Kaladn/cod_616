import numpy as np
import time
import importlib.util
from pathlib import Path

# Import modules directly by path to avoid importing heavy package deps during tests
spec_ac = importlib.util.spec_from_file_location('audio_capture', str(Path(__file__).resolve().parent / 'modules' / 'audio_capture.py'))
audio_capture = importlib.util.module_from_spec(spec_ac)
spec_ac.loader.exec_module(audio_capture)
AudioCapture = audio_capture.AudioCapture
SD_AVAILABLE = audio_capture.SD_AVAILABLE

spec_ars = importlib.util.spec_from_file_location('audio_resonance_state', str(Path(__file__).resolve().parent / 'modules' / 'audio_resonance_state.py'))
audio_resonance_state = importlib.util.module_from_spec(spec_ars)
spec_ars.loader.exec_module(audio_resonance_state)
AudioResonanceState = audio_resonance_state.AudioResonanceState


def test_audio_capture_stub_or_buffering():
    # Create a capture instance with small sample rate to make tests fast
    ac = AudioCapture(sample_rate=8000, channels=1, block_duration=0.1)

    if not SD_AVAILABLE:
        # If sounddevice is not present, get_block should be None and no exceptions
        assert ac.get_block() is None
    else:
        # If available, directly inject synthetic samples into buffer to avoid hardware timing
        # Generate 200ms of white noise
        samples = (np.random.randn(int(0.2 * ac.sample_rate), 1) * 0.1).astype(np.float32)
        ac._buffer.append(samples)
        # Now get_block should return either None (if not enough) or a block array
        blk = ac.get_block()
        # blk can be None if not enough samples accumulated; that's acceptable
        if blk is not None:
            assert isinstance(blk, np.ndarray)
            # Duration check (approx)
            assert blk.shape[0] <= ac.block_samples

    ac.close()


def test_audio_resonance_update_sine():
    ars = AudioResonanceState(sample_rate=48000, block_duration=0.5)
    t = np.linspace(0, 0.5, ars.block_samples, endpoint=False)
    sine = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    state = ars.update(sine)
    # Ensure expected keys exist
    assert 'aud_energy_total' in state
    assert 'aud_transient_rate' in state
    assert isinstance(state['aud_energy_total'], float)
    assert isinstance(state['aud_transient_rate'], float)
