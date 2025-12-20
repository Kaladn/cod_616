import json
from pathlib import Path


def validate_baseline_json(path: str) -> bool:
    data = json.loads(Path(path).read_text())
    # Required keys
    keys = {"schema_version", "windows", "core", "entropy_mean", "symmetry_mean"}
    if not keys.issubset(set(data.keys())):
        return False
    core_keys = {"mean", "median", "min", "max"}
    return core_keys.issubset(set(data.get("core", {}).keys()))


def validate_diff_json(path: str) -> bool:
    data = json.loads(Path(path).read_text())
    req = {"schema_version", "core", "entropy_mean_diff", "symmetry_mean_diff"}
    return req.issubset(set(data.keys()))


def validate_feature_diff_json(path: str) -> bool:
    data = json.loads(Path(path).read_text())
    return "schema_version" in data and "pairs" in data and "summary" in data
