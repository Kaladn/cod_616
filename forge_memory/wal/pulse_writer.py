from dataclasses import dataclass

@dataclass
class PulseConfig:
    max_records_per_pulse: int = 128
    max_bytes_per_pulse: int = 512 * 1024
    max_age_ms_per_pulse: int = 250
