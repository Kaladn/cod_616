#!/usr/bin/env python3
"""Understand the REAL pattern of 007bbfb7 by checking each 3×3 block."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

tasks = load_arc_tasks(
    Path('arc-prize-2024/arc-agi_training_challenges.json'),
    Path('arc-prize-2024/arc-agi_training_solutions.json')
)

task = tasks['007bbfb7']
ex = task.train[0]
tile = ex.input_grid
output = ex.output_grid

print('Input tile (3×3):')
print(tile)
print('\nOutput (9×9):')
print(output)
print('\n' + '='*60)
print('Analyzing each 3×3 block in output:')
print('='*60 + '\n')

for i in range(3):
    for j in range(3):
        block = output[i*3:(i+1)*3, j*3:(j+1)*3]
        print(f'Block (row={i}, col={j}):')
        print(block)
        
        # Check if it matches tile with any 2D shift
        found = False
        for shift_h in range(3):
            for shift_w in range(3):
                shifted = np.roll(np.roll(tile, shift_h, axis=0), shift_w, axis=1)
                if np.array_equal(block, shifted):
                    print(f'  ✓ Matches tile shifted by ({shift_h}, {shift_w})')
                    found = True
                    break
            if found:
                break
        
        if not found:
            if np.all(block == 0):
                print('  ⚠ ALL ZEROS (background)')
            else:
                print('  ✗ NO MATCH to any tile shift')
        print()
