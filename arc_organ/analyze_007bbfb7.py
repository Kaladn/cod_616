#!/usr/bin/env python3
"""Analyze exact transformation for 007bbfb7."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

# Load task
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)

task = tasks["007bbfb7"]
ex = task.train[0]

print("Example 1:")
print("\nInput (3×3):")
print(ex.input_grid)

print("\nOutput (9×9):")
print(ex.output_grid)

print("\nSimple tiling (3×3):")
simple_tiled = np.tile(ex.input_grid, (3, 3))
print(simple_tiled)

print(f"\nSimple tiling matches output: {np.array_equal(simple_tiled, ex.output_grid)}")

# Check what's different
print("\nDifferences (output != tiled):")
diff = ex.output_grid != simple_tiled
print(f"Number of differing pixels: {np.sum(diff)}")
if np.sum(diff) > 0:
    coords = np.argwhere(diff)
    print("First few differences:")
    for i, (r, c) in enumerate(coords[:10]):
        print(f"  ({r},{c}): output={ex.output_grid[r,c]}, tiled={simple_tiled[r,c]}")
