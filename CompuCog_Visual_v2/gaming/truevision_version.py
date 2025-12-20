"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog — Sovereign Cognitive Defense System                           ║
║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
║                                                                              ║
║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
║                                                                              ║
║     "We use unconventional digital wisdom —                                  ║
║        because conventional digital wisdom doesn't protect anyone."         ║
║                                                                              ║
║     This software is proprietary and confidential.                           ║
║     Unauthorized access, copying, modification, or distribution             ║
║     is strictly prohibited and may violate applicable laws.                  ║
║                                                                              ║
║     File automatically watermarked on: 2025-11-29 00:00:00                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

TRUEVISION_VERSION = "v1.0.0"

# Version history
VERSION_HISTORY = {
    "v1.0.0": {
        "date": "2025-11-29",
        "description": "Initial production release - Full EOMM manipulation detection pipeline",
        "components": [
            "Unified telemetry schema (OperatorResult + TelemetryWindow)",
            "4 manipulation detection operators (crosshair_lock, hit_registration, death_event, edge_entry)",
            "Session baseline tracker with Welford's algorithm",
            "EOMM composite scoring with weighted aggregation",
            "Config-driven threshold tuning (truevision_config.yaml)"
        ],
        "capabilities": [
            "Screen-only detection (no input logger dependency)",
            "Session-normalized baselines (no weapon database required)",
            "1-second telemetry windows (~30 frames at 30 FPS)",
            "Deterministic ARC-style palette detection",
            "JSONL forensic evidence export"
        ],
        "limitations": [
            "No input delta correlation (v2.0 feature)",
            "No HUD compass parsing (v2.0 feature)",
            "No per-weapon baselines (v2.0 feature)",
            "No automatic palette calibration (v2.0 feature)"
        ]
    }
}


def get_version() -> str:
    """Get current TrueVision version string"""
    return TRUEVISION_VERSION


def get_version_info() -> dict:
    """Get detailed version information"""
    return VERSION_HISTORY.get(TRUEVISION_VERSION, {})


if __name__ == "__main__":
    import json
    print(f"TrueVision {TRUEVISION_VERSION}")
    print(json.dumps(get_version_info(), indent=2))
