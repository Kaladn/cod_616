# 🧬 TRUEVISION → FORGERECORD SCHEMA MAP

This is the **canonical, final-word specification** for how TrueVision windows are translated into ForgeRecord binary memory cells.

This document is the **DNA layer** — the biological blueprint that defines what goes inside each memory cell.

**Last Updated**: December 3, 2025  
**Schema Version**: 1  
**Status**: LOCKED

---

## 1. Canonical TrueVision Window Shape

Every window TrueVision emits into Forge must be a dict with this minimum structure:

```python
TrueVisionWindow = {
    "ts_start": float,          # epoch seconds
    "ts_end": float,            # epoch seconds
    "grid_shape": [int, int],   # [H, W] visual grid
    "grid_color_count": int,    # number of non-empty cells / intensity buckets

    "operator_scores": {        # per-operator numeric scores
        "<OP_NAME>": float,     # e.g. "INSTA_MELT": 0.87
        ...
    },

    "operator_flags": [         # names of operators that fired as flags
        "INSTA_MELT",
        "SPAWN_BIAS",
        ...
    ],

    "eomm_score": float,        # final composite score for this window

    "baseline_stats": {         # Welford stats / session baselines
        # e.g. "ttk_mean": ..., "ttk_var": ..., etc.
        ...
    },

    "telemetry": {              # aggregated per-window runtime metrics
        "reflex_latency_ms": float,
        "gpu_frametime_ms": float,
        "packet_loss_pct": float,
        "input_magnitude": float,
        "cpu_usage_pct": float,
        # plus any extra numeric/boolean telemetry fields
        ...
    },

    "session_context": {        # semantic tags for this run/window
        "session_id": str,
        "match_id": str,
        "source": str,          # "live", "vod", "benchmark"
        "map_name": str | None,
        "mode_name": str | None,
        "bot_difficulty": str | None,  # "casual", "hard", etc.
        "perspective": str | None,     # "pov", "killcam", "benchmark_bot"
        "run_id": str | None,
        # any additional high-level context fields
        ...
    }
}
```

**Rule**: If extra fields exist, they must be folded into `session_context` or `telemetry`, not invented at Forge layer.

---

## 2. ForgeRecord Fields — Exact Mapping

From the canonical `ForgeRecord`:

```python
ForgeRecord(
    pulse_id: int,
    worker_id: int,
    seq: int,
    timestamp: float,
    success: bool,
    task_id: str,
    engine_id: str,
    transform_id: str,
    failure_reason: str | None,
    grid_shape_in: (int, int),
    grid_shape_out: (int, int),
    color_count: int,
    train_pair_indices: list[int],
    error_metrics: dict,
    params: dict,
    context: dict,
)
```

### 2.1 Fixed Header Fields

**These are fully determined for TrueVision v2:**

| ForgeRecord field    | Source / Value                                                              |
| -------------------- | --------------------------------------------------------------------------- |
| `pulse_id`           | Assigned by PulseWriter (monotonic per pulse)                               |
| `worker_id`          | Ingestion worker ID (0 if single-process)                                   |
| `seq`                | Per-worker sequence (incrementing per record)                               |
| `timestamp`          | `window["ts_start"]`                                                        |
| `success`            | `window["eomm_score"] <= EOMM_THRESHOLD` (see below)                        |
| `task_id`            | `"truevision_window"`                                                       |
| `engine_id`          | `"truevision_v2"`                                                           |
| `transform_id`       | e.g. `"eomm_compositor_v2.0.0"` (constant per build, set in config)         |
| `failure_reason`     | `None` if `success` else `"eomm_above_threshold"` (or more specific string) |
| `grid_shape_in`      | `tuple(window["grid_shape"])`                                               |
| `grid_shape_out`     | same as `grid_shape_in` (TrueVision doesn't reshape yet)                    |
| `color_count`        | `window["grid_color_count"]`                                                |
| `train_pair_indices` | `[0]` for TrueVision (required non-empty, single-stream source)             |

**EOMM Threshold Constant:**

Define once in config (not inline):

```python
EOMM_THRESHOLD = 0.5  # or whatever baseline says is "too manipulated"
```

**Semantics:**

* `success = True` → window considered "within normal band"
* `success = False` → window considered "manipulated / anomalous" for indexing

---

### 2.2 `error_metrics` Blob (MessagePack)

This is the **numeric anomaly fingerprint** for the window.

**Schema:**

```python
error_metrics = {
    "eomm_score": float,              # the composite
    "operators": {
        "<OP_NAME>": float,           # per-operator score, e.g. "INSTA_MELT": 0.92
        ...
    },
    "flags": list[str],               # operator_flags echoed here for convenience
}
```

**Mapping:**

* `error_metrics["eomm_score"] = window["eomm_score"]`
* `error_metrics["operators"] = window["operator_scores"]` (shallow copy)
* `error_metrics["flags"] = window["operator_flags"]`

**Invariant:**

* All values in `error_metrics["operators"]` MUST be numeric and serializable.

---

### 2.3 `params` Blob (MessagePack)

This is **static-ish configuration and per-window run parameters**, not telemetry.

**Schema:**

```python
params = {
    "schema_version": 1,
    "window_duration_ms": (window["ts_end"] - window["ts_start"]) * 1000.0,
    "grid_shape": list[int],         # copy of window["grid_shape"]
    "grid_color_count": int,         # copy of window["grid_color_count"]
    "eomm_threshold": float,         # the threshold used to derive success
    "engine_version": str,           # e.g. "truevision_v2.0.0"
    "compositor_version": str,       # e.g. "eomm_v2.0.0"
    "operator_set": list[str],       # sorted list(window["operator_scores"].keys())
    # optional static run-level knobs:
    "render_scale": float | None,
    "capture_source": str | None,    # "screen", "video_file", "stream"
    "roi": dict | None,              # region of interest config, if any
}
```

**Mapping Rules:**

* Everything in `params` must be:
  * either constant for the run, or
  * derived from window timestamps / static configuration
* **No live telemetry** goes here (that's `context`).

---

### 2.4 `context` Blob (MessagePack)

This is the **dynamic context + telemetry + session metadata**.

**Schema:**

```python
context = {
    "session": {
        # everything from window["session_context"]
        # normalized to strings / simple scalars
        "session_id": str,
        "match_id": str,
        "source": str,           # "live", "vod", "benchmark"
        "map_name": str | None,
        "mode_name": str | None,
        "bot_difficulty": str | None,
        "perspective": str | None,
        "run_id": str | None,
        # plus any extra session tags
        ...
    },
    "telemetry": {
        # direct copy of window["telemetry"],
        # with any complex structures simplified to scalars
        "reflex_latency_ms": float,
        "gpu_frametime_ms": float,
        "packet_loss_pct": float,
        "input_magnitude": float,
        "cpu_usage_pct": float,
        ...
    },
    "flags": {
        "operator_flags": list[str],    # same as window["operator_flags"]
        "is_baseline_bot": bool,        # derived from session_context
        "is_human": bool,               # derived from session_context
        "is_benchmark": bool,           # derived from session_context
    }
}
```

**Derivation Rules for Flags:**

* `is_baseline_bot = (session_context["source"] in {"live", "vod"}) and session_context["bot_difficulty"] is not None`
* `is_human = (session_context["bot_difficulty"] is None)` (or explicit field if you add it)
* `is_benchmark = (session_context["source"] == "benchmark")`

**Key Principle:**

> **No interpretation inside TrueVision.
> Just classification rules baked into schema_map.**

---

## 3. Invariants and Rules

To keep the whole system sane, enforce these rules inside schema_map (and only there):

### 3.1 No Missing Basics

A window is **INVALID** for Forge if any of these are missing:

* `ts_start`, `ts_end`
* `grid_shape`, `grid_color_count`
* `operator_scores`
* `eomm_score`
* `telemetry`
* `session_context`

### 3.2 grid_shape Length MUST Be 2

Anything else is a bug.

### 3.3 TRAIN_PAIR_INDICES Must Be Non-Empty

For TrueVision, we always use `[0]`.

### 3.4 operator_scores Keys Must Be Strings

No numeric keys or nested dicts here.

### 3.5 All Dicts Must Be MessagePack-Safe

No unserializable types (no custom classes, sets, etc.).

### 3.6 schema_version Is Required in `params`

Start at `1`, bump when structure changes.
Query layer can branch safely on this.

### 3.7 task_id / engine_id / transform_id Must Be Stable Names

These string values drive `StringDictionary` and let you filter by engine version later.

---

## 4. Example Mapping (Baseline Bot Window)

**Given:**

```python
window = {
    "ts_start": 1733160000.0,
    "ts_end": 1733160000.4,
    "grid_shape": [1080, 1920],  # Native resolution (H, W)
    "grid_color_count": 12,
    "operator_scores": {
        "INSTA_MELT": 0.91,
        "SPAWN_BIAS": 0.43,
    },
    "operator_flags": ["INSTA_MELT"],
    "eomm_score": 0.62,
    "baseline_stats": {
        "ttk_mean": 248.0,
        "ttk_var": 12.3,
    },
    "telemetry": {
        "reflex_latency_ms": 24.5,
        "gpu_frametime_ms": 7.1,
        "packet_loss_pct": 0.0,
        "input_magnitude": 0.87,
        "cpu_usage_pct": 31.0,
    },
    "session_context": {
        "session_id": "S123",
        "match_id": "M456",
        "source": "live",
        "map_name": "Rust",
        "mode_name": "TDM",
        "bot_difficulty": "casual",
        "perspective": "pov",
        "run_id": "RUN_001",
    },
}
```

**With `EOMM_THRESHOLD = 0.5`, mapping yields:**

### Fixed Header Fields:

* `pulse_id`: (assigned by PulseWriter)
* `worker_id`: 0
* `seq`: (incrementing)
* `timestamp`: `1733160000.0`
* `success`: `False` (0.62 > 0.5)
* `task_id`: `"truevision_window"`
* `engine_id`: `"truevision_v2"`
* `transform_id`: `"eomm_compositor_v2.0.0"`
* `failure_reason`: `"eomm_above_threshold"`
* `grid_shape_in`: `(1080, 1920)` (native resolution)
* `grid_shape_out`: `(1080, 1920)` (native resolution)
* `color_count`: `12`
* `train_pair_indices`: `[0]`

### `error_metrics` Blob:

```python
{
    "eomm_score": 0.62,
    "operators": {
        "INSTA_MELT": 0.91,
        "SPAWN_BIAS": 0.43,
    },
    "flags": ["INSTA_MELT"],
}
```

### `params` Blob:

```python
{
    "schema_version": 1,
    "window_duration_ms": 400.0,
    "grid_shape": [1080, 1920],  # Native resolution
    "grid_color_count": 12,
    "eomm_threshold": 0.5,
    "engine_version": "truevision_v2.0.0",
    "compositor_version": "eomm_v2.0.0",
    "operator_set": ["INSTA_MELT", "SPAWN_BIAS"],
    "render_scale": None,
    "capture_source": "screen",
    "roi": None,
}
```

### `context` Blob:

```python
{
    "session": {
        "session_id": "S123",
        "match_id": "M456",
        "source": "live",
        "map_name": "Rust",
        "mode_name": "TDM",
        "bot_difficulty": "casual",
        "perspective": "pov",
        "run_id": "RUN_001",
    },
    "telemetry": {
        "reflex_latency_ms": 24.5,
        "gpu_frametime_ms": 7.1,
        "packet_loss_pct": 0.0,
        "input_magnitude": 0.87,
        "cpu_usage_pct": 31.0,
    },
    "flags": {
        "operator_flags": ["INSTA_MELT"],
        "is_baseline_bot": True,
        "is_human": False,
        "is_benchmark": False,
    },
}
```

**Result**: A **fully defined memory cell** for one TrueVision window.

---

## 5. Audit Checklist

When implementing `schema_map.py`, verify:

- [ ] Window validation: all required fields present
- [ ] `grid_shape` is exactly 2 elements
- [ ] `train_pair_indices = [0]` always for TrueVision
- [ ] `success` derived from `eomm_score <= EOMM_THRESHOLD`
- [ ] `failure_reason` set correctly based on `success`
- [ ] `error_metrics` contains only serializable numeric values
- [ ] `params` contains no live telemetry
- [ ] `context` contains all session + telemetry + flags
- [ ] `schema_version = 1` in `params`
- [ ] All string refs (`task_id`, `engine_id`, `transform_id`) are stable constants
- [ ] `timestamp` uses `ts_start` (not `ts_end`)
- [ ] `grid_shape_in == grid_shape_out` for TrueVision
- [ ] All blobs are MessagePack-safe (no sets, custom classes)
- [ ] Flags logic: `is_baseline_bot`, `is_human`, `is_benchmark` computed correctly

---

## 6. Next Steps

This schema_map specification is now **LOCKED**.

Next layer to spec: **PulseWriter**

* How windows are grouped into pulses
* How `pulse_id` increments
* When to flush
* How WAL + BinaryLog see those batches

---

**END OF SCHEMA MAP SPECIFICATION**
