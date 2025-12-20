import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaming.session_playback import session_to_features, generate_baseline, export_anomaly_curve
from pathlib import Path
import json


def test_session_to_features_and_baseline(tmp_path):
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"

    features = session_to_features(path)
    assert len(features) > 0

    baseline = generate_baseline(features)
    assert baseline["schema_version"] == "1.0"
    assert "core" in baseline

    # Export an anomaly curve
    csvp = tmp_path / "core.csv"
    jsnp = tmp_path / "core.json"
    export_anomaly_curve(features, "core_avg_intensity", str(csvp), str(jsnp))

    data = json.loads(jsnp.read_text())
    assert data["schema_version"] == "1.0"
    assert data["metric"] == "core_avg_intensity"
    assert isinstance(data["data"], list)
