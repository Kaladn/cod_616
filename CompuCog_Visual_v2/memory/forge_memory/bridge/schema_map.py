from __future__ import annotations

from typing import Dict, Any, Tuple, List
from forge_memory.core.record import ForgeRecord


class TrueVisionSchemaMap:
    """
    Canonical translator:
        TrueVisionWindow (dict) → ForgeRecord.from_dict(...) payload (dict)

    This MUST NOT invent fields. It MUST mirror TRUEVISION_SCHEMA_MAP.md exactly.
    """

    def __init__(
        self,
        *,
        eomm_threshold: float,
        engine_version: str,
        compositor_version: str,
    ) -> None:
        self.eomm_threshold = eomm_threshold
        self.engine_version = engine_version
        self.compositor_version = compositor_version

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------
    def window_to_record_dict(self, w: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a TrueVision window into a ForgeRecord.from_dict() dictionary.
        PulseWriter will later fill pulse_id, worker_id, seq.
        """
        self._validate_window(w)

        ts_start = float(w["ts_start"])
        ts_end = float(w["ts_end"])
        grid_shape = tuple(w["grid_shape"])     # (H, W)
        color_count = int(w["grid_color_count"])

        eomm_score = float(w["eomm_score"])
        operator_scores = dict(w["operator_scores"])
        operator_flags = list(w["operator_flags"])
        telemetry = dict(w["telemetry"])
        session_ctx = dict(w["session_context"])

        success, failure_reason = self._compute_success_and_failure(eomm_score)

        # ------------------------------------------------------------------
        # error_metrics blob
        # ------------------------------------------------------------------
        error_metrics = {
            "eomm_score": eomm_score,
            "operators": operator_scores,
            "flags": operator_flags,
        }

        # ------------------------------------------------------------------
        # params blob
        # ------------------------------------------------------------------
        params = {
            "schema_version": 1,
            "window_duration_ms": (ts_end - ts_start) * 1000.0,
            "grid_shape": list(grid_shape),
            "grid_color_count": color_count,
            "eomm_threshold": self.eomm_threshold,
            "engine_version": self.engine_version,
            "compositor_version": self.compositor_version,
            "operator_set": sorted(operator_scores.keys()),
            "render_scale": session_ctx.get("render_scale"),
            "capture_source": session_ctx.get("source"),
            "roi": session_ctx.get("roi"),
        }

        # ------------------------------------------------------------------
        # context blob
        # ------------------------------------------------------------------
        flags = self._derive_context_flags(session_ctx, operator_flags)

        context = {
            "session": session_ctx,
            "telemetry": telemetry,
            "flags": flags,
        }

        # ------------------------------------------------------------------
        # FINAL FORGERECORD DICT
        # ------------------------------------------------------------------
        return {
            # PulseWriter will fill pulse_id, worker_id, seq
            "timestamp": ts_start,
            "success": success,
            "task_id": "truevision_window",
            "engine_id": "truevision_v2",
            "transform_id": self.compositor_version,
            "failure_reason": failure_reason,

            "grid_shape_in": grid_shape,
            "grid_shape_out": grid_shape,
            "color_count": color_count,
            "train_pair_indices": [0],  # mandatory non-empty list for TrueVision
            "error_metrics": error_metrics,
            "params": params,
            "context": context,
        }

    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------

    def _compute_success_and_failure(self, eomm_score: float) -> Tuple[bool, str | None]:
        """Success if eomm_score <= threshold."""
        if eomm_score <= self.eomm_threshold:
            return True, None
        return False, "eomm_above_threshold"

    def _derive_context_flags(self, session_ctx: Dict[str, Any], operator_flags: List[str]) -> Dict[str, Any]:
        source = session_ctx.get("source")
        difficulty = session_ctx.get("bot_difficulty")

        is_benchmark = (source == "benchmark")
        is_baseline_bot = (difficulty is not None)
        is_human = not is_baseline_bot and not is_benchmark

        return {
            "operator_flags": operator_flags,
            "is_baseline_bot": bool(is_baseline_bot),
            "is_human": bool(is_human),
            "is_benchmark": bool(is_benchmark),
        }

    def _validate_window(self, w: Dict[str, Any]) -> None:
        """Strict validation matching TRUEVISION_SCHEMA_MAP invariants."""
        required = [
            "ts_start", "ts_end",
            "grid_shape", "grid_color_count",
            "operator_scores", "operator_flags",
            "eomm_score", "telemetry", "session_context",
        ]
        for key in required:
            if key not in w:
                raise ValueError(f"Missing required window field: {key}")

        grid_shape = w["grid_shape"]
        if (not isinstance(grid_shape, (list, tuple))) or len(grid_shape) != 2:
            raise ValueError(f"grid_shape must be [H, W], got: {grid_shape}")

        if not isinstance(w["operator_scores"], dict):
            raise ValueError("operator_scores must be a dict")

        if not isinstance(w["operator_flags"], list):
            raise ValueError("operator_flags must be a list")

        if not isinstance(w["telemetry"], dict):
            raise ValueError("telemetry must be a dict")

        if not isinstance(w["session_context"], dict):
            raise ValueError("session_context must be a dict")
