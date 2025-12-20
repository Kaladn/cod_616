#!/usr/bin/env python3
"""Check if masking correlates with input tile's background pattern."""

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

print('Correlating input background (0s) with masked output blocks:\n')
print('='*70)

for ex_num, ex in enumerate(task.train, 1):
    tile = ex.input_grid
    output = ex.output_grid
    
    print(f'\nExample {ex_num}:')
    print(f'Input tile:')
    print(tile)
    
    # Find background positions in tile
    bg_positions = []
    for i in range(3):
        for j in range(3):
            if tile[i, j] == 0:
                bg_positions.append((i, j))
    
    print(f'\nBackground (0) positions in tile: {bg_positions}')
    
    # Find masked blocks in output
    masked_blocks = []
    for i in range(3):
        for j in range(3):
            block = output[i*3:(i+1)*3, j*3:(j+1)*3]
            if np.all(block == 0):
                masked_blocks.append((i, j))
    
    print(f'Masked blocks in output (3×3 grid): {masked_blocks}')
    
    # Check if they match
    if set(bg_positions) == set(masked_blocks):
        print('✓ MATCH! Masked blocks = background positions in tile')
    else:
        print('✗ No direct match')
        print(f'  Difference: bg={set(bg_positions)} vs masked={set(masked_blocks)}')

print('\n' + '='*70)
print('HYPOTHESIS: Output = tile in 3×3 grid, where blocks at (i,j)')
print('that correspond to background (0) positions in the tile are zeroed!')
