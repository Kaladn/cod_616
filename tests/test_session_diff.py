import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaming.session_playback import session_to_features, generate_baseline
from gaming.session_diff import diff_baselines, export_diff_json, export_diff_csv, diff_features


def test_baseline_diff_and_exports(tmp_path):
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"

    features = session_to_features(path)
    # split into two deterministic halves
    mid = len(features) // 2
    a = features[:mid]
    b = features[mid:mid + mid]

    baseline_a = generate_baseline(a)
    baseline_b = generate_baseline(b)

    d = diff_baselines(baseline_a, baseline_b)
    assert d["schema_version"] == "1.0"
    assert "core" in d

    # exports
    jsonp = tmp_path / "diff.json"
    csvp = tmp_path / "diff.csv"
    export_diff_json(d, str(jsonp))
    export_diff_csv(d, str(csvp))

    assert jsonp.exists()
    assert csvp.exists()


def test_feature_diff_basic():
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"
    features = session_to_features(path)
    a = features[:10]
    b = features[10:20]

    fd = diff_features(a, b, metric="core_avg_intensity")
    assert fd["schema_version"] == "1.0"
    assert fd["summary"]["count"] == min(len(a), len(b))
    assert "pairs" in fd
