#!/usr/bin/env python3
"""Analyze 017c7c7b transformation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

# Load task
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)

task = tasks["017c7c7b"]

print("Task 017c7c7b - Color Mapping Analysis\n")

for i, ex in enumerate(task.train):
    print(f"Example {i+1}:")
    print(f"  Input shape: {ex.input_grid.shape}")
    print(f"  Output shape: {ex.output_grid.shape}")
    print(f"  Same shape: {ex.input_grid.shape == ex.output_grid.shape}")
    
    if ex.output_grid is not None:
        # Extract color mapping
        mapping = {}
        for r in range(ex.input_grid.shape[0]):
            for c in range(ex.input_grid.shape[1]):
                in_color = int(ex.input_grid[r, c])
                out_color = int(ex.output_grid[r, c])
                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        print(f"  WARNING: Inconsistent mapping for color {in_color}")
                else:
                    mapping[in_color] = out_color
        
        print(f"  Color mapping: {mapping}")
        
        # Check if spatial structure preserved
        same_positions = np.sum(ex.input_grid == ex.output_grid)
        total = ex.input_grid.size
        print(f"  Spatial structure: {same_positions}/{total} pixels unchanged")
        
        print()

print("\nTest case:")
test_input = task.test[0].input_grid
test_output = task.test[0].output_grid
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape if test_output is not None else 'Unknown'}")

print("\nInput:")
print(test_input)

if test_output is not None:
    print("\nExpected output:")
    print(test_output)
    
    # Extract mapping
    mapping = {}
    for r in range(test_input.shape[0]):
        for c in range(test_input.shape[1]):
            in_color = int(test_input[r, c])
            out_color = int(test_output[r, c])
            if in_color not in mapping:
                mapping[in_color] = out_color
    
    print(f"\nTest mapping: {mapping}")
