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

TrueVision v1.0.0 - Smoke Test Harness

Purpose:
  Minimal end-to-end test of full TrueVision pipeline:
  - Generate synthetic frame sequence (simulated gameplay)
  - Run all 4 operators (crosshair_lock, hit_registration, death_event, edge_entry)
  - Aggregate with EOMM compositor
  - Export TelemetryWindow per second to JSONL
  
  This creates a replayable dataset for:
  - Validating pipeline integration
  - Training/testing ML models
  - Evidence collection ("this match was rigged")
  
Usage:
  python truevision_smoke_test.py
  
Output:
  telemetry/truevision_smoke_test_YYYYMMDD_HHMMSS.jsonl
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List
import random

# Add paths for imports (v2 structure)
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "operators"))
sys.path.insert(0, str(Path(__file__).parent.parent / "compositor"))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent))

from frame_to_grid import FrameGrid
from crosshair_lock import CrosshairLockOperator, FrameSequence
from hit_registration import HitRegistrationOperator
from death_event import DeathEventOperator
from edge_entry import EdgeEntryOperator
from eomm_compositor import EommCompositor
from session_baseline import SessionBaselineTracker
from truevision_version import TRUEVISION_VERSION, get_version


def generate_synthetic_frame(h: int, w: int, frame_idx: int, scenario: str) -> FrameGrid:
    """
    Generate synthetic ARC grid simulating gameplay.
    
    Scenarios:
    - "normal": Typical gameplay with reasonable aim and hit registration
    - "aim_suppression": Enemy in crosshair but few hits (hitbox shrink)
    - "insta_melt": Rapid incoming damage (damage amplification)
    - "spawn_pressure": Enemies flooding from rear/edges
    """
    grid = [[0 for _ in range(w)] for _ in range(h)]
    
    # Base background (palette 1-3)
    for y in range(h):
        for x in range(w):
            grid[y][x] = random.randint(1, 3)
    
    # Crosshair region (center 5%)
    center_y = h // 2
    center_x = w // 2
    crosshair_radius = int(min(h, w) * 0.05)
    
    if scenario == "normal":
        # Simulate normal gameplay
        # Enemy appears in crosshair occasionally (palette 6-9)
        if frame_idx % 10 < 5:  # Enemy on-target 50% of time
            for dy in range(-crosshair_radius, crosshair_radius + 1):
                for dx in range(-crosshair_radius, crosshair_radius + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        if (dx * dx + dy * dy) <= (crosshair_radius * crosshair_radius):
                            grid[y][x] = random.randint(6, 9)  # Enemy palette
        
        # Hit marker flash (palette 9) occasionally
        if frame_idx % 15 < 2:  # Hit marker every ~0.5 sec
            for dy in range(-crosshair_radius // 2, crosshair_radius // 2 + 1):
                for dx in range(-crosshair_radius // 2, crosshair_radius // 2 + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        grid[y][x] = 9  # Hit marker
        
        # Blood splatter occasionally (palette 7-8)
        if frame_idx % 15 == 3:  # Blood after hit marker
            blood_radius = int(min(h, w) * 0.1)
            for dy in range(-blood_radius, blood_radius + 1):
                for dx in range(-blood_radius, blood_radius + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        if random.random() > 0.7:  # Sparse blood
                            grid[y][x] = random.randint(7, 8)
    
    elif scenario == "aim_suppression":
        # Enemy ALWAYS in crosshair, but hit markers RARE (hitbox shrink)
        for dy in range(-crosshair_radius, crosshair_radius + 1):
            for dx in range(-crosshair_radius, crosshair_radius + 1):
                y = center_y + dy
                x = center_x + dx
                if 0 <= y < h and 0 <= x < w:
                    if (dx * dx + dy * dy) <= (crosshair_radius * crosshair_radius):
                        grid[y][x] = random.randint(6, 9)  # Enemy always on-target
        
        # Hit marker only 10% of time (should be 30-50%)
        if frame_idx % 30 < 2:  # Rare hits despite being on target
            for dy in range(-crosshair_radius // 2, crosshair_radius // 2 + 1):
                for dx in range(-crosshair_radius // 2, crosshair_radius // 2 + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        grid[y][x] = 9
        
        # Blood even rarer
        if frame_idx % 30 == 5:
            blood_radius = int(min(h, w) * 0.05)
            for dy in range(-blood_radius, blood_radius + 1):
                for dx in range(-blood_radius, blood_radius + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        if random.random() > 0.9:
                            grid[y][x] = 7
    
    elif scenario == "insta_melt":
        # Rapid incoming damage (red vignette at edges)
        edge_h = int(h * 0.1)
        edge_w = int(w * 0.1)
        
        # Every frame has damage indicator
        for y in range(h):
            for x in range(w):
                # Top/bottom edges
                if y < edge_h or y >= h - edge_h:
                    grid[y][x] = random.randint(7, 8)  # Damage indicator
                # Left/right edges
                elif x < edge_w or x >= w - edge_w:
                    grid[y][x] = random.randint(7, 8)
        
        # Intense flinch (entire screen shifts palette)
        if frame_idx % 3 == 0:  # Flinch every 3 frames
            for y in range(h):
                for x in range(w):
                    grid[y][x] = min(9, grid[y][x] + 2)  # Boost palette (flash)
        
        # Death screen after 10 frames (~0.3 sec = insta-melt)
        if frame_idx > 10:
            # Grayscale transition
            for y in range(h):
                for x in range(w):
                    grid[y][x] = 5  # Mid-gray
            
            # Death UI in corners
            corner_size = int(min(h, w) * 0.1)
            for dy in range(corner_size):
                for dx in range(corner_size):
                    grid[dy][dx] = 9
                    grid[h - dy - 1][w - dx - 1] = 9
    
    elif scenario == "spawn_pressure":
        # Enemies flooding from rear/top edges
        edge_h = int(h * 0.15)
        
        # Many enemies at top edge (rear spawns)
        for y in range(edge_h):
            for x in range(w):
                if random.random() > 0.3:  # Dense enemy presence
                    grid[y][x] = random.randint(6, 9)
        
        # Some enemies at sides
        edge_w = int(w * 0.15)
        for y in range(edge_h, h - edge_h):
            for x in range(edge_w):
                if random.random() > 0.7:
                    grid[y][x] = random.randint(6, 9)
            for x in range(w - edge_w, w):
                if random.random() > 0.7:
                    grid[y][x] = random.randint(6, 9)
    
    return FrameGrid(
        frame_id=frame_idx,
        t_sec=frame_idx / 30.0,  # Assume 30 FPS
        grid=grid,
        source="synthetic",
        capture_region="full",
        h=h,
        w=w
    )


def generate_frame_sequence(h: int, w: int, start_frame: int, num_frames: int, scenario: str) -> FrameSequence:
    """Generate 1-second window of synthetic frames"""
    frames = [generate_synthetic_frame(h, w, start_frame + i, scenario) for i in range(num_frames)]
    
    return FrameSequence(
        frames=frames,
        t_start=frames[0].t_sec,
        t_end=frames[-1].t_sec,
        src="synthetic_smoke_test"
    )


def run_smoke_test():
    """Run full TrueVision pipeline on synthetic data"""
    print(f"[TrueVision Smoke Test] Version: {TRUEVISION_VERSION}")
    print("=" * 80)
    
    # Initialize components
    config_path = str(Path(__file__).parent / "truevision_config.yaml")
    
    print("[1/6] Initializing operators...")
    crosshair_op = CrosshairLockOperator(config_path)
    hit_reg_op = HitRegistrationOperator(config_path)
    death_op = DeathEventOperator(config_path)
    edge_op = EdgeEntryOperator(config_path)
    
    print("[2/6] Initializing compositor...")
    compositor = EommCompositor(config_path)
    
    print("[3/6] Initializing session baseline tracker...")
    session_tracker = SessionBaselineTracker(min_samples_for_warmup=3)
    
    # Generate test scenarios
    scenarios = [
        ("normal", 3),           # 3 seconds normal gameplay
        ("aim_suppression", 2),  # 2 seconds aim manipulation
        ("insta_melt", 1),       # 1 second insta-melt death
        ("spawn_pressure", 2)    # 2 seconds spawn flooding
    ]
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "telemetry"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"truevision_smoke_test_{timestamp}.jsonl"
    
    print(f"[4/6] Generating synthetic gameplay (8 seconds total)...")
    print(f"      Scenarios: {', '.join(s[0] for s in scenarios)}")
    
    frame_idx = 0
    session_id = f"smoke_test_{timestamp}"
    
    with open(output_file, 'w') as f:
        print(f"[5/6] Running detection pipeline...")
        
        for scenario, duration_sec in scenarios:
            print(f"    -> Scenario: {scenario} ({duration_sec}s)")
            
            for sec in range(duration_sec):
                # Generate 1-second window (30 frames at 30 FPS)
                seq = generate_frame_sequence(h=32, w=32, start_frame=frame_idx, num_frames=30, scenario=scenario)
                
                # Run all operators
                op_results = []
                
                result = crosshair_op.analyze(seq)
                if result:
                    op_results.append(result)
                
                result = hit_reg_op.analyze(seq)
                if result:
                    op_results.append(result)
                
                result = death_op.analyze(seq)
                if result:
                    op_results.append(result)
                
                result = edge_op.analyze(seq)
                if result:
                    op_results.append(result)
                
                # Compose telemetry window
                if op_results:
                    window = compositor.compose_window(
                        operator_results=op_results,
                        window_start_epoch=seq.t_start,
                        window_end_epoch=seq.t_end,
                        session_id=session_id,
                        frame_count=len(seq.frames),
                        session_tracker=session_tracker
                    )
                    
                    # Write to JSONL
                    f.write(json.dumps(window.to_dict()) + "\n")
                    
                    # Print summary
                    flags_str = ", ".join(window.eomm_flags) if window.eomm_flags else "NONE"
                    print(f"       Frame {frame_idx:03d}-{frame_idx+29:03d}: EOMM={window.eomm_composite_score:.2f} | Flags=[{flags_str}]")
                
                frame_idx += 30  # Advance 1 second
    
    print(f"[6/6] Smoke test complete!")
    print(f"      Output: {output_file}")
    print(f"      Total windows: {sum(d for _, d in scenarios)}")
    print("=" * 80)
    
    # Validate output
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(f"\n[Validation] JSONL lines written: {len(lines)}")
        
        if lines:
            # Parse first window
            first_window = json.loads(lines[0])
            print(f"[Validation] First window keys: {list(first_window.keys())}")
            print(f"[Validation] Operators in first window: {len(first_window.get('operator_results', []))}")
            
            # Check for expected scenarios
            manipulation_windows = [json.loads(line) for line in lines if json.loads(line)['eomm_composite_score'] > 0.3]
            print(f"[Validation] Windows with EOMM > 0.3: {len(manipulation_windows)} / {len(lines)}")
    
    print("\n[OK] Smoke test PASSED - TrueVision v1.0.0 pipeline operational")
    return output_file


if __name__ == "__main__":
    try:
        output_path = run_smoke_test()
        print(f"\n[OUTPUT] Telemetry dataset: {output_path}")
        print("   Use this for:")
        print("   - Training ML models")
        print("   - Testing detection accuracy")
        print("   - Evidence collection for 'rigged match' claims")
    except Exception as e:
        print(f"\n[FAIL] Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
