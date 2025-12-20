from typing import List, Dict, Any
import statistics

from .session_playback import WindowFeatures


def sector_time_series(features: List[WindowFeatures]) -> Dict[str, List[float]]:
    """Returns deterministic sector time series as floats in order."""
    out = {"front": [], "side": [], "rear": []}
    for f in features:
        for k in out.keys():
            out[k].append(float(f.sectors.get(k, 0.0)))
    return out


def directional_variance_per_window(features: List[WindowFeatures]) -> List[float]:
    """Compute per-window variance of directional vectors (gradient_change).
    If no directional vectors are present for the window, returns 0.0.
    """
    vars = []
    for f in features:
        dv = getattr(f, 'metadata', {}).get('directional_vectors') or []
        # The stored WindowFeatures doesn't include raw directional vectors; this helper
        # expects features' metadata to optionally contain list of directional dicts
        vals = [float(d.get('gradient_change', 0.0)) for d in dv]
        if len(vals) <= 1:
            vars.append(0.0)
        else:
            vars.append(float(statistics.pvariance(vals)))  # population variance deterministic
    return vars


def directional_stability_summary(vars_list: List[float]) -> Dict[str, float]:
    """Compute deterministic stability summary for directional variance list."""
    if not vars_list:
        return {"mean": 0.0, "median": 0.0}
    mean = float(statistics.mean(vars_list))
    median = float(statistics.median(vars_list))
    return {"mean": mean, "median": median}


def sector_stability_summary(series: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in series.items():
        if not v:
            out[k] = {"mean": 0.0, "median": 0.0}
        else:
            out[k] = {"mean": float(statistics.mean(v)), "median": float(statistics.median(v))}
    return out


def export_series_csv(series: Dict[str, List[float]], path: str) -> None:
    import csv
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Write header: index,front,side,rear
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "front", "side", "rear"])
        n = max(len(v) for v in series.values()) if series else 0
        for i in range(n):
            row = [i]
            for k in ("front", "side", "rear"):
                row.append(series.get(k, [])[i] if i < len(series.get(k, [])) else 0.0)
            writer.writerow(row)


def export_series_json(series: Dict[str, List[float]], path: str) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({"schema_version": "1.0", "series": series}, fh, indent=2)
