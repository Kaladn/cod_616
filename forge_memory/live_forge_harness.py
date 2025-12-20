from dataclasses import dataclass
from typing import Any

class BinaryLog:
    def __init__(self):
        self._count = 0
        self._closed = False

    def get_record_count(self) -> int:
        return self._count

    def increment(self, n: int = 1):
        self._count += n

    def close(self):
        self._closed = True

class PulseWriter:
    def __init__(self):
        self.binary_log = BinaryLog()
        self._closed = False

    def submit_window(self, window: Any):
        # Simulate writing a record
        self.binary_log.increment(1)

    def flush(self, reason: str = None):
        return True

    def close(self):
        self._closed = True

@dataclass
class PulseConfig:
    max_records_per_pulse: int = 128
    max_bytes_per_pulse: int = 512 * 1024
    max_age_ms_per_pulse: int = 250


def build_truevision_forge_pipeline(data_dir: str, pulse_config: PulseConfig):
    # Return a PulseWriter stub
    return PulseWriter()
