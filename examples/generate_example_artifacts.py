"""Generate example artifacts (baseline, sector CSV, plots) using sample telemetry."""
from pathlib import Path
from gaming.session_playback import session_to_features, generate_baseline, write_baseline_json
from gaming.metrics import sector_time_series, export_series_csv, export_series_json
from gaming.plotting import plot_metric_curve, plot_sector_timeseries


def run_example(jsonl_path: str = "TruVision_files/telemetry/truevision_live_20251202_155530.jsonl", out_dir: str = "examples/out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = session_to_features(jsonl_path)
    baseline = generate_baseline(features)
    write_baseline_json(baseline, str(out / "baseline.json"))

    series = sector_time_series(features)
    export_series_csv(series, str(out / "sectors.csv"))
    export_series_json(series, str(out / "sectors.json"))

    # plot core and sectors
    plot_metric_curve([f.core_avg_intensity for f in features], 'core_avg_intensity', str(out / 'core.png'))
    plot_sector_timeseries(series, str(out / 'sectors.png'))

    print(f"Example artifacts written to {out}")


if __name__ == "__main__":
    run_example()
