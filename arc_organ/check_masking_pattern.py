#!/usr/bin/env python3
"""Check if the masking pattern is consistent across all examples."""

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

print('Checking masked tiling pattern across all examples:\n')

all_masked_patterns = []

for ex_num, ex in enumerate(task.train, 1):
    tile = ex.input_grid
    output = ex.output_grid
    
    print(f'Example {ex_num}:')
    masked_blocks = []
    
    for i in range(3):
        for j in range(3):
            block = output[i*3:(i+1)*3, j*3:(j+1)*3]
            
            # Check if it's the unshifted tile
            if np.array_equal(block, tile):
                pass  # Normal tile
            elif np.all(block == 0):
                masked_blocks.append((i, j))
                print(f'  Block ({i},{j}): ALL ZEROS (masked)')
    
    all_masked_patterns.append(tuple(sorted(masked_blocks)))
    print()

print('='*60)
print('Summary:')
print('='*60)
for i, pattern in enumerate(all_masked_patterns, 1):
    print(f'Example {i}: {pattern}')

if len(set(all_masked_patterns)) == 1:
    print(f'\n✓ CONSISTENT PATTERN: {all_masked_patterns[0]}')
    print('\nThis is "masked tiling" - simple tile with specific blocks zeroed out!')
else:
    print('\n✗ Patterns differ across examples')
