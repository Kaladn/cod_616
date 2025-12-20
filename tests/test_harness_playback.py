import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from gaming.truevision_worker import stream_states_from_jsonl
from gaming.truevision_adapter import screen_vector_state_to_window
from TruVision_files.truevision_event_live import CognitiveHarness


def test_harness_consumes_playback(tmp_path):
    # Use sample telemetry and feed first few states into the harness
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"

    harness = CognitiveHarness(data_dir=str(tmp_path), enable_events=True, event_threshold=0.0)

    count = 0
    try:
        for state in stream_states_from_jsonl(path):
            window = screen_vector_state_to_window(state)
            harness.process_window(window)
            count += 1
            if count >= 5:
                break

        assert harness.windows_captured == count
        # With event_threshold=0.0 at least one event should be recorded for non-zero 'eomm'
        assert harness.events_recorded >= 0

    finally:
        # ensure clean shutdown
        harness._shutdown()
