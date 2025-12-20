"""Simple worker to stream ScreenVectorState objects from a TrueVision JSONL
telemetry file using the adapter. Useful for local playback and tests.
"""
from typing import Iterator
import json
from .truevision_adapter import telemetry_window_to_screen_vector_state
from .screen_vector_state import ScreenVectorState


def stream_states_from_jsonl(path: str) -> Iterator[ScreenVectorState]:
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            window = json.loads(line)
            yield telemetry_window_to_screen_vector_state(window)
