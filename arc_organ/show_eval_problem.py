#!/usr/bin/env python3
"""Display an ARC evaluation problem."""

import json
import sys
from pathlib import Path

# Load evaluation challenges
eval_challenges = json.loads(Path('arc-prize-2024/arc-agi_evaluation_challenges.json').read_text())

# Get first task
first_task_id = list(eval_challenges.keys())[0]
task = eval_challenges[first_task_id]

print('='*70)
print(f'ARC EVALUATION TASK: {first_task_id}')
print('='*70)
print(f'\nTraining Examples: {len(task["train"])}')
print(f'Test Cases: {len(task["test"])}')

# Show training examples
for i, example in enumerate(task['train'], 1):
    inp = example['input']
    out = example['output']
    
    print(f'\n{"="*70}')
    print(f'TRAINING EXAMPLE {i}')
    print(f'{"="*70}')
    print(f'\nInput ({len(inp)}×{len(inp[0])}):')
    for row in inp:
        print('  ' + ' '.join(str(x) for x in row))
    
    print(f'\nOutput ({len(out)}×{len(out[0])}):')
    for row in out:
        print('  ' + ' '.join(str(x) for x in row))

# Show test input
print(f'\n{"="*70}')
print(f'TEST CASE (must predict output)')
print(f'{"="*70}')
test_inp = task['test'][0]['input']
print(f'\nInput ({len(test_inp)}×{len(test_inp[0])}):')
for row in test_inp:
    print('  ' + ' '.join(str(x) for x in row))

print(f'\n{"="*70}')
print('CHALLENGE: Learn the pattern from training examples,')
print('then predict the correct output for the test input.')
print('='*70)
