"""Adapter to convert TrueVision telemetry windows (JSON) into
ScreenVectorState objects so operators can consume telemetry-only sessions.
"""
from typing import Dict
from .screen_vector_state import ScreenVectorState, CoreBlock, SectorDefinition, DirectionalVector, AnomalyMetrics


def telemetry_window_to_screen_vector_state(window: Dict) -> ScreenVectorState:
    """Convert a single telemetry `window` dict (one JSONL line) into
    a minimal `ScreenVectorState`.

    Mapping rules (conservative and explainable):
    - If `crosshair_lock` operator exists, use its `intersection_ratio`
      as a proxy for `core_block.avg_intensity` (scaled 0..100).
    - If `edge_entry` operator exists, use its `direction_counts` to
      populate three sectors: `front`, `side`, `rear` (normalized counts).
    - Preserve useful metadata (session, window start/end, operator flags)

    This adapter does not attempt to reconstruct raw grids — it produces
    a lightweight SVE-compatible state for operator consumption and testing.
    """
    state = ScreenVectorState()

    # Metadata
    state.metadata["session_id"] = window.get("session_id")
    state.metadata["window_start_epoch"] = window.get("window_start_epoch")
    state.metadata["window_end_epoch"] = window.get("window_end_epoch")
    state.metadata["window_duration_ms"] = window.get("window_duration_ms")

    # Map operator results
    op_results = {op["operator_name"]: op for op in window.get("operator_results", [])}

    # Crosshair proxy -> core avg intensity
    if "crosshair_lock" in op_results:
        cl = op_results["crosshair_lock"]
        metrics = cl.get("metrics", {})
        # intersection_ratio is in [0,1]; scale to 0..100
        intersection = float(metrics.get("intersection_ratio", 0.0))
        state.core_block.avg_intensity = intersection * 100.0
        # keep hit efficiency / frames in metadata for traceability
        state.metadata["crosshair_on_target_frames"] = metrics.get("on_target_frames")
        state.metadata["crosshair_hit_marker_frames"] = metrics.get("hit_marker_frames")
        state.metadata["crosshair_hit_efficiency"] = metrics.get("hit_efficiency")
        state.metadata["crosshair_flags"] = cl.get("flags", [])

    # Edge entry -> sectors
    if "edge_entry" in op_results:
        ee = op_results["edge_entry"]
        md = ee.get("metadata", {})
        dir_counts = md.get("direction_counts", {})
        total = sum(dir_counts.values()) or 1
        # Create sectors for 'front','side','rear' if present
        for name in ("front", "side", "rear"):
            cnt = int(dir_counts.get(name, 0))
            # normalized intensity proxy
            weighted = (cnt / total) * 100.0
            state.sectors[name] = SectorDefinition(name=name, weighted_avg_intensity=weighted, raw_count=cnt)
        state.metadata["edge_entry_flags"] = ee.get("flags", [])

    # Compute directional vectors using aim_resistance_score and sector weights
    import math
    aim_res = None
    if "crosshair_lock" in op_results:
        aim_res = float(op_results["crosshair_lock"].get("metrics", {}).get("aim_resistance_score", 0.0))

    # sector weights normalized to [0,1]
    front = state.sectors.get("front").weighted_avg_intensity / 100.0 if "front" in state.sectors else 0.0
    side = state.sectors.get("side").weighted_avg_intensity / 100.0 if "side" in state.sectors else 0.0
    rear = state.sectors.get("rear").weighted_avg_intensity / 100.0 if "rear" in state.sectors else 0.0

    # entropy over direction counts (front/side/rear), normalized to [0,1]
    counts = [state.sectors.get(k).raw_count if k in state.sectors else 0 for k in ("front", "side", "rear")]
    total = sum(counts) or 1
    ps = [c / total for c in counts]
    entropy = -sum(p * math.log2(p) for p in ps if p > 0) / math.log2(3) if total > 0 else 0.0
    symmetry = 1.0 - abs(front - rear)  # 1.0 when front and rear equal, 0 when maximally different

    # If aim_res provided, build directional vectors (8 rays)
    if aim_res is not None:
        angles = [i * 45 for i in range(8)]
        dirs = []
        for a in angles:
            if a % 180 == 0:  # 0 or 180
                sector_weight = front if a == 0 else rear
            elif a % 90 == 0:  # 90 or 270
                sector_weight = side
            else:
                # diagonals: average adjacent sectors
                if a == 45:
                    sector_weight = (front + side) / 2
                elif a == 135:
                    sector_weight = (side + rear) / 2
                elif a == 225:
                    sector_weight = (rear + side) / 2
                elif a == 315:
                    sector_weight = (side + front) / 2
                else:
                    sector_weight = (front + side + rear) / 3
            # deterministic gradient proxy: scale aim_res to 0..100 and factor by sector_weight
            gradient_change = aim_res * 100.0 * (0.5 + 0.5 * sector_weight)
            dirs.append(DirectionalVector(angle_deg=a, gradient_change=gradient_change))
        state.directional_vectors = dirs

    # Populate anomaly metrics dataclass
    try:
        state.anomaly_metrics = AnomalyMetrics(entropy=entropy, symmetry=symmetry, flags=[])
    except Exception:
        # If AnomalyMetrics class isn't available, fall back to metadata dict
        state.metadata["anomaly_metrics"] = {"entropy": entropy, "symmetry": symmetry, "flags": []}

    # Record operator summaries for traceability
    for name, op in op_results.items():
        state.metadata.setdefault("operators", {})[name] = {
            "confidence": op.get("confidence"),
            "flags": op.get("flags", []),
            "metrics": op.get("metrics", {})
        }

    return state


def screen_vector_state_to_window(state: dict) -> dict:
    """Convert a `ScreenVectorState` into a minimal telemetry window dict
    compatible with `CognitiveHarness.process_window()` for smoke testing.

    Rules:
    - `eomm_composite`: max operator confidence (0..1) if available, else 0.0
    - `operator_flags`: aggregate operator flags
    - `window_id` / `epoch` / `session_id`: preserved from state.metadata when present
    - `grid` shape: synthetic [32,32] to satisfy downstream fields
    """
    operators = state.metadata.get("operators", {}) if hasattr(state, 'metadata') else state.get('metadata', {}).get('operators', {})

    max_conf = 0.0
    flags = []
    for name, info in operators.items():
        try:
            c = float(info.get('confidence', 0.0))
        except Exception:
            c = 0.0
        max_conf = max(max_conf, c)
        flags.extend(info.get('flags', []))

    window = {
        "eomm_composite": max_conf,
        "eomm_composite_score": max_conf,
        "operator_flags": list(set(flags)),
        "window_id": state.metadata.get('window_id') if hasattr(state, 'metadata') else None,
        "epoch": state.metadata.get('window_start_epoch') if hasattr(state, 'metadata') else None,
        "session_id": state.metadata.get('session_id') if hasattr(state, 'metadata') else None,
        "grid": {"shape": [32, 32]},
    }

    return window
