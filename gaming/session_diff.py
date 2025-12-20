from typing import Dict, Any, List
import json
import csv
from pathlib import Path

from .session_schema import PLAYBACK_SCHEMA_VERSION


def diff_baselines(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Pure deterministic baseline diff between two baseline dicts.
    Returns a dict with schema_version and a deterministic set of numeric deltas.
    """
    diff = {
        "schema_version": PLAYBACK_SCHEMA_VERSION,
        "baseline_a_schema": a.get("schema_version"),
        "baseline_b_schema": b.get("schema_version"),
        "windows_a": int(a.get("windows", 0)),
        "windows_b": int(b.get("windows", 0)),
        "core": {},
        "entropy_mean_diff": float(b.get("entropy_mean", 0.0)) - float(a.get("entropy_mean", 0.0)),
        "symmetry_mean_diff": float(b.get("symmetry_mean", 0.0)) - float(a.get("symmetry_mean", 0.0))
    }

    # core fields (mean, median, min, max)
    core_a = a.get("core", {})
    core_b = b.get("core", {})
    for key in ("mean", "median", "min", "max"):
        va = float(core_a.get(key, 0.0))
        vb = float(core_b.get(key, 0.0))
        diff["core"][key] = {"a": va, "b": vb, "delta": vb - va}

    return diff


def export_diff_json(d: Dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def export_diff_csv(d: Dict[str, Any], out_path: str) -> None:
    """Write a small deterministic CSV summarizing core diffs and summary fields."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Basic info
    rows.append({"metric": "windows_a", "a": d.get("windows_a"), "b": d.get("windows_b"), "delta": d.get("windows_b") - d.get("windows_a")})
    rows.append({"metric": "entropy_mean_diff", "a": None, "b": None, "delta": d.get("entropy_mean_diff")})
    rows.append({"metric": "symmetry_mean_diff", "a": None, "b": None, "delta": d.get("symmetry_mean_diff")})

    # Core fields
    core = d.get("core", {})
    for key in ("mean", "median", "min", "max"):
        entry = core.get(key, {})
        rows.append({"metric": f"core_{key}", "a": entry.get("a"), "b": entry.get("b"), "delta": entry.get("delta")})

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "a", "b", "delta"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def diff_features(features_a: List[Any], features_b: List[Any], metric: str = "core_avg_intensity") -> Dict[str, Any]:
    """Compute per-window diffs for a given metric. Align by index up to min length.
    Returns a dict containing per-window deltas and summary statistics (mean_delta, median_delta).
    """
    n = min(len(features_a), len(features_b))
    pairs = []
    deltas = []
    for i in range(n):
        va = float(getattr(features_a[i], metric))
        vb = float(getattr(features_b[i], metric))
        delta = vb - va
        pairs.append({"index": i, "a": va, "b": vb, "delta": delta})
        deltas.append(delta)

    # summary
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    s = sorted(deltas)
    median_delta = (s[len(s)//2] if len(s) % 2 == 1 else ((s[len(s)//2 - 1] + s[len(s)//2]) / 2.0)) if s else 0.0

    return {
        "schema_version": PLAYBACK_SCHEMA_VERSION,
        "metric": metric,
        "pairs": pairs,
        "summary": {"mean_delta": mean_delta, "median_delta": median_delta, "count": len(pairs)}
    }


def export_feature_diff_json(d: Dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def export_feature_diff_csv(d: Dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "a", "b", "delta"])
        writer.writeheader()
        for p in d.get("pairs", []):
            writer.writerow(p)
