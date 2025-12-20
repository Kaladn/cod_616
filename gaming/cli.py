"""Thin CLI for session playback and artifact export.
Thin (parsing + dispatch) only; heavy lifting in pure functions.
"""
import argparse
from pathlib import Path
from .session_playback import session_to_features, generate_baseline, write_baseline_json, export_anomaly_curve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="session_playback")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # run subcommand
    run = sub.add_parser("run", help="Run session playback and export artifacts")
    run.add_argument("--jsonl", required=True, help="Path to telemetry JSONL file")
    run.add_argument("--out-dir", required=True, help="Directory to write artifacts")
    run.add_argument("--metric", default="core_avg_intensity", choices=["core_avg_intensity","entropy","symmetry","directional_mean"], help="Metric to export as anomaly curve")

    # diff subcommand (baseline paths)
    diff = sub.add_parser("diff", help="Diff two baseline JSON files and export artifacts")
    diff.add_argument("--a", required=True, help="Baseline A JSON path")
    diff.add_argument("--b", required=True, help="Baseline B JSON path")
    diff.add_argument("--out-dir", required=True, help="Directory to write diff artifacts")

    args = parser.parse_args(argv)

    if args.cmd == "diff":
        from .session_diff import diff_baselines, export_diff_json, export_diff_csv
        import json

        with open(args.a, 'r', encoding='utf-8') as fh:
            a = json.load(fh)
        with open(args.b, 'r', encoding='utf-8') as fh:
            b = json.load(fh)

        d = diff_baselines(a, b)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        export_diff_json(d, str(Path(args.out_dir) / "diff.json"))
        export_diff_csv(d, str(Path(args.out_dir) / "diff.csv"))
        print(f"Diff artifacts written to {args.out_dir}")
        return

    # default to run
    if args.cmd == "run" or args.cmd is None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        features = session_to_features(args.jsonl)
        baseline = generate_baseline(features)
        write_baseline_json(baseline, str(out_dir / "baseline.json"))
        export_anomaly_curve(features, args.metric, str(out_dir / f"{args.metric}.csv"), str(out_dir / f"{args.metric}.json"))

        print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
