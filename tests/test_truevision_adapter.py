import sys
from pathlib import Path
# Ensure repo root is on sys.path so `gaming` package imports work in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from gaming.truevision_adapter import telemetry_window_to_screen_vector_state
from gaming.truevision_worker import stream_states_from_jsonl
from gaming.screen_vector_state import ScreenVectorState


def test_adapter_maps_crosshair_and_edge_entry():
    # read first line from sample telemetry (exists in repo under TruVision_files/telemetry)
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"
    with open(path, 'r', encoding='utf-8') as fh:
        first_line = fh.readline().strip()

    window = json.loads(first_line)
    state = telemetry_window_to_screen_vector_state(window)

    assert isinstance(state, ScreenVectorState)
    # session id should pass-through
    assert state.metadata.get("session_id") == window.get("session_id")

    # If crosshair_lock present, core avg intensity should be > 0
    ops = {op['operator_name']: op for op in window.get('operator_results', [])}
    if 'crosshair_lock' in ops:
        assert state.core_block.avg_intensity > 0
        assert "crosshair_on_target_frames" in state.metadata

    # If edge_entry present, sectors should include front/side/rear
    if 'edge_entry' in ops:
        for s in ('front', 'side', 'rear'):
            assert s in state.sectors
            assert state.sectors[s].raw_count >= 0

    # New: directional vectors and anomaly metrics should be present when crosshair_lock exists
    if 'crosshair_lock' in ops:
        assert isinstance(state.directional_vectors, list)
        assert len(state.directional_vectors) == 8
        for dv in state.directional_vectors:
            assert hasattr(dv, 'angle_deg') and hasattr(dv, 'gradient_change')
        # anomaly metrics should be present and normalized
        am = getattr(state, 'anomaly_metrics', None) or state.metadata.get('anomaly_metrics')
        assert am is not None
        entropy = am.entropy if hasattr(am, 'entropy') else am.get('entropy')
        symmetry = am.symmetry if hasattr(am, 'symmetry') else am.get('symmetry')
        assert 0.0 <= entropy <= 1.0
        assert 0.0 <= symmetry <= 1.0


def test_worker_streaming_reads_all_lines():
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"
    count = 0
    for st in stream_states_from_jsonl(path):
        assert isinstance(st, ScreenVectorState)
        count += 1
    # file has many lines; expect at least 10 windows
    assert count >= 10
