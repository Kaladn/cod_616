"""Audio capture helper with graceful fallback.

Provides a background input stream when sounddevice is available and
returns recent blocks of audio for processing by AudioResonanceState.

If sounddevice is not available or initialization fails, this module
runs in "stub" mode and `get_block()` returns `None`.
"""

from collections import deque
from typing import Optional
import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False


class AudioCapture:
    def __init__(self, sample_rate: int = 48000, channels: int = 1, block_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_duration = block_duration
        self.block_samples = int(self.sample_rate * self.block_duration)

        self.available = SD_AVAILABLE
        self._buffer = deque()
        self._stream = None

        if SD_AVAILABLE:
            try:
                self._stream = sd.InputStream(samplerate=self.sample_rate,
                                              channels=self.channels,
                                              callback=self._callback)
                self._stream.start()
                print(f"[AudioCapture] Input stream started ({self.sample_rate} Hz, {self.channels} ch)")
            except Exception as e:
                print(f"[AudioCapture] Failed to start input stream: {e}")
                self.available = False
        else:
            print("[AudioCapture] sounddevice not available; running in stub mode")

    def _callback(self, indata, frames, time, status):
        # Copy chunk into buffer (callback must be fast)
        try:
            self._buffer.append(indata.copy())
            # Keep buffer from growing indefinitely
            if len(self._buffer) > 100:
                self._buffer.popleft()
        except Exception:
            # Swallow exceptions in callback
            return

    def get_block(self) -> Optional[np.ndarray]:
        """Return the most recent contiguous block of audio with length `block_samples`.

        Returns:
            np.ndarray shape (samples,) for mono or (samples, channels) for stereo, or
            None if no block is available or capture is unavailable.
        """
        if not self.available:
            return None

        if not self._buffer:
            return None

        try:
            # Concatenate chunks
            data = np.concatenate(list(self._buffer), axis=0)
            if data.shape[0] < self.block_samples:
                # Not enough samples yet
                return None

            # Take the most recent block
            block = data[-self.block_samples:]

            # Trim buffer to keep recent trailing data (avoid memory growth)
            # Keep at most 1 block of trailing samples
            self._buffer.clear()
            self._buffer.append(block[-int(self.sample_rate * 0.1):])  # keep 100ms

            # If mono channels==1, return 1d array
            if block.ndim == 2 and block.shape[1] == 1:
                return block[:, 0]

            return block
        except Exception:
            return None

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
