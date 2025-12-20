#!/usr/bin/env python3
"""Quick analysis of the 4 failed tasks to identify patterns."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

tasks = load_arc_tasks(
    Path('arc-prize-2024/arc-agi_training_challenges.json'),
    Path('arc-prize-2024/arc-agi_training_solutions.json')
)

failed_tasks = ['00d62c1b', '017c7c7b', '025d127b', '045e512c']

print("="*70)
print("QUICK PATTERN ANALYSIS - 4 FAILED TASKS")
print("="*70)

for task_id in failed_tasks:
    task = tasks[task_id]
    print(f"\n{'='*70}")
    print(f"Task: {task_id}")
    print(f"{'='*70}")
    print(f"Training examples: {len(task.train)}")
    
    # Check size patterns
    for i, ex in enumerate(task.train[:2], 1):  # First 2 examples
        in_shape = ex.input_grid.shape
        out_shape = ex.output_grid.shape
        print(f"\nExample {i}:")
        print(f"  Input:  {in_shape[0]:2d}×{in_shape[1]:2d}")
        print(f"  Output: {out_shape[0]:2d}×{out_shape[1]:2d}")
        
        # Same size?
        if in_shape == out_shape:
            print(f"  → Same size (recolor/paint/flip likely)")
            
            # Check if simple color mapping
            unique_in = len(np.unique(ex.input_grid))
            unique_out = len(np.unique(ex.output_grid))
            print(f"  → Colors: {unique_in} input → {unique_out} output")
            
            # Check color mapping consistency
            color_changes = 0
            for val in np.unique(ex.input_grid):
                out_vals = ex.output_grid[ex.input_grid == val]
                if len(np.unique(out_vals)) > 1:
                    color_changes += 1
            
            if color_changes == 0:
                print(f"  → Simple color remapping (1-to-1)")
            else:
                print(f"  → Complex transformation (position-dependent)")
        
        else:
            print(f"  → Size change ({out_shape[0]/in_shape[0]:.1f}x h, {out_shape[1]/in_shape[1]:.1f}x w)")
            
            # Check if expansion/contraction
            if out_shape[0] > in_shape[0] or out_shape[1] > in_shape[1]:
                print(f"  → EXPANSION")
            else:
                print(f"  → CONTRACTION/CROP")
        
        # Show small sample
        print(f"\n  Input (top-left 3×3):")
        print("  " + str(ex.input_grid[:3, :3]).replace('\n', '\n  '))
        print(f"\n  Output (top-left 3×3):")
        print("  " + str(ex.output_grid[:3, :3]).replace('\n', '\n  '))

print("\n" + "="*70)
print("PATTERN SUMMARY")
print("="*70)
