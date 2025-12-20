#!/usr/bin/env python3
"""Deep dive on 025d127b - appears to be gravity/shift operation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

tasks = load_arc_tasks(
    Path('arc-prize-2024/arc-agi_training_challenges.json'),
    Path('arc-prize-2024/arc-agi_training_solutions.json')
)

task = tasks['025d127b']

print("="*70)
print("TASK 025d127b - Deep Analysis")
print("="*70)

for ex_num, ex in enumerate(task.train, 1):
    print(f"\n{'='*70}")
    print(f"Example {ex_num}")
    print(f"{'='*70}")
    
    print(f"\nInput ({ex.input_grid.shape[0]}×{ex.input_grid.shape[1]}):")
    print(ex.input_grid)
    
    print(f"\nOutput ({ex.output_grid.shape[0]}×{ex.output_grid.shape[1]}):")
    print(ex.output_grid)
    
    # Analyze column by column
    print("\nColumn-by-column analysis:")
    for col in range(ex.input_grid.shape[1]):
        in_col = ex.input_grid[:, col]
        out_col = ex.output_grid[:, col]
        
        in_nonzero = in_col[in_col != 0]
        out_nonzero = out_col[out_col != 0]
        
        if len(in_nonzero) > 0 or len(out_nonzero) > 0:
            print(f"\n  Col {col}:")
            print(f"    Input non-zeros: {list(in_nonzero)} at rows {list(np.where(in_col != 0)[0])}")
            print(f"    Output non-zeros: {list(out_nonzero)} at rows {list(np.where(out_col != 0)[0])}")
            
            # Check if values moved down (gravity)
            if len(in_nonzero) == len(out_nonzero):
                if np.array_equal(sorted(in_nonzero), sorted(out_nonzero)):
                    print(f"    → Same colors, different positions (MOVED)")
                    
                    # Check if moved to bottom
                    in_rows = np.where(in_col != 0)[0]
                    out_rows = np.where(out_col != 0)[0]
                    
                    if len(out_rows) > 0:
                        out_bottom = max(out_rows)
                        in_bottom = max(in_rows) if len(in_rows) > 0 else 0
                        
                        if out_bottom > in_bottom:
                            print(f"    → MOVED DOWN (gravity)")
                        elif out_bottom < in_bottom:
                            print(f"    → MOVED UP")
                        else:
                            print(f"    → SHIFTED horizontally?")

print("\n" + "="*70)
print("Test case:")
print("="*70)
test_input = task.test[0].input_grid
test_output = task.test[0].output_grid

print(f"\nTest Input ({test_input.shape[0]}×{test_input.shape[1]}):")
print(test_input)

if test_output is not None:
    print(f"\nTest Output ({test_output.shape[0]}×{test_output.shape[1]}):")
    print(test_output)
