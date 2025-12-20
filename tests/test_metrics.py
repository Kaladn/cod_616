import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaming.session_playback import session_to_features
from gaming.metrics import sector_time_series, directional_variance_per_window, directional_stability_summary, sector_stability_summary


def test_sector_time_series_and_directional_variance():
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"
    features = session_to_features(path)

    series = sector_time_series(features)
    assert "front" in series and "side" in series and "rear" in series
    assert len(series["front"]) == len(features)

    vars = directional_variance_per_window(features)
    assert isinstance(vars, list)

    summary = directional_stability_summary(vars)
    assert "mean" in summary and "median" in summary

    ssummary = sector_stability_summary(series)
    assert all(k in ssummary for k in ("front", "side", "rear"))
