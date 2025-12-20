import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaming.session_playback import session_to_features
from gaming.metrics import sector_time_series
from gaming.plotting import plot_metric_curve, plot_sector_timeseries
from examples.generate_example_artifacts import run_example


def test_plot_and_example_generation(tmp_path):
    path = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl"
    features = session_to_features(path)
    series = sector_time_series(features)

    out1 = tmp_path / "core.png"
    plot_metric_curve([f.core_avg_intensity for f in features], 'core_avg_intensity', str(out1))
    assert out1.exists() and out1.stat().st_size > 0

    out2 = tmp_path / "sectors.png"
    plot_sector_timeseries(series, str(out2))
    assert out2.exists() and out2.stat().st_size > 0

    # run example artifacts (writes to folder)
    outdir = tmp_path / "examples"
    run_example(jsonl_path=path, out_dir=str(outdir))
    assert (outdir / "baseline.json").exists()
    assert (outdir / "core.png").exists()
    assert (outdir / "sectors.csv").exists()
