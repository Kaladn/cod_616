#!/usr/bin/env python3
"""Find tasks with same input/output dimensions - truly simple transformations."""

import json
from pathlib import Path

challenges = json.loads(Path('arc-prize-2024/arc-agi_training_challenges.json').read_text())
solutions = json.loads(Path('arc-prize-2024/arc-agi_training_solutions.json').read_text())

same_size_tasks = []
for task_id in challenges:
    task = challenges[task_id]
    solutions_task = solutions[task_id]
    
    all_same_size = True
    for i, example in enumerate(task['train']):
        inp = example['input']
        out = solutions_task[i] if i < len(solutions_task) else None
        if out is None:
            all_same_size = False
            break
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            all_same_size = False
            break
    
    if all_same_size:
        same_size_tasks.append(task_id)
        if len(same_size_tasks) <= 10:
            ex = task['train'][0]
            h, w = len(ex['input']), len(ex['input'][0])
            print(f'{task_id}: {h}×{w} (same size)')

print(f'\nTotal same-size tasks: {len(same_size_tasks)} / {len(challenges)}')
print(f'\nFirst 5 same-size task IDs:')
for tid in same_size_tasks[:5]:
    print(f'  {tid}')
