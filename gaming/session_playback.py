from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import csv
import json
from pathlib import Path

from .truevision_worker import stream_states_from_jsonl
from .truevision_adapter import telemetry_window_to_screen_vector_state
from .session_schema import PLAYBACK_SCHEMA_VERSION


@dataclass
class WindowFeatures:
    index: int
    core_avg_intensity: float
    sectors: Dict[str, float]
    directional_mean: float
    entropy: float
    symmetry: float
    metadata: Dict[str, Any]


def session_to_features(jsonl_path: str) -> List[WindowFeatures]:
    """Pure function: reads JSONL and returns ordered list of WindowFeatures.
    Deterministic: iteration order preserved.
    """
    features: List[WindowFeatures] = []

    # Note: stream_states_from_jsonl already yields ScreenVectorState objects
    for i, state in enumerate(stream_states_from_jsonl(jsonl_path)):
        core = float(state.core_block.avg_intensity)
        # sectors: front, side, rear normalized [0..1]
        sectors = {}
        for name in ("front", "side", "rear"):
            sec = state.sectors.get(name) if name in state.sectors else None
            sectors[name] = float(sec.weighted_avg_intensity / 100.0) if sec is not None else 0.0

        # directional mean (mean gradient_change) if present
        dv = getattr(state, 'directional_vectors', None) or []
        dv_vals = [float(d.gradient_change) for d in dv]
        directional_mean = (sum(dv_vals) / len(dv_vals)) if dv_vals else 0.0

        # anomaly metrics
        am = getattr(state, 'anomaly_metrics', None)
        if am is not None:
            entropy = float(am.entropy)
            symmetry = float(am.symmetry)
        else:
            meta_am = state.metadata.get('anomaly_metrics', {}) if hasattr(state, 'metadata') else {}
            entropy = float(meta_am.get('entropy', 0.0))
            symmetry = float(meta_am.get('symmetry', 0.0))

        features.append(WindowFeatures(
            index=i,
            core_avg_intensity=core,
            sectors=sectors,
            directional_mean=directional_mean,
            entropy=entropy,
            symmetry=symmetry,
            metadata=state.metadata if hasattr(state, 'metadata') else {}
        ))

    return features


def generate_baseline(features: List[WindowFeatures]) -> Dict[str, Any]:
    """Deterministic baseline generator returning aggregates and schema version."""
    cores = [f.core_avg_intensity for f in features]
    entropies = [f.entropy for f in features]
    symmetries = [f.symmetry for f in features]

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    def median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0

    baseline = {
        "schema_version": PLAYBACK_SCHEMA_VERSION,
        "windows": len(features),
        "core": {
            "mean": mean(cores),
            "median": median(cores),
            "min": min(cores) if cores else 0.0,
            "max": max(cores) if cores else 0.0
        },
        "entropy_mean": mean(entropies),
        "symmetry_mean": mean(symmetries)
    }
    return baseline


def export_anomaly_curve(features: List[WindowFeatures], metric: str, csv_path: str, json_path: str) -> None:
    """Export per-window metric curve to CSV and JSON.
    metric must be one of: 'core_avg_intensity','entropy','symmetry','directional_mean'
    """
    assert metric in ("core_avg_intensity", "entropy", "symmetry", "directional_mean")

    rows = []
    for f in features:
        val = getattr(f, metric)
        rows.append({"index": f.index, "value": val})

    # Write CSV deterministically ordered by index
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "value"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSON with schema
    out = {
        "schema_version": PLAYBACK_SCHEMA_VERSION,
        "metric": metric,
        "data": rows
    }
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)


def write_baseline_json(baseline: Dict[str, Any], out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(baseline, fh, indent=2)
