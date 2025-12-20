#!/usr/bin/env python3
"""Test current system on random 20 training tasks to estimate full coverage."""

import sys
import json
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cod_616.arc_organ.arc_task_runner import load_arc_tasks, ARCTask
from cod_616.arc_organ.arc_two_attempt_sampler import ARCTwoAttemptSampler

# Load all training tasks
tasks = load_arc_tasks(
    Path('arc-prize-2024/arc-agi_training_challenges.json'),
    Path('arc-prize-2024/arc-agi_training_solutions.json')
)

# Load solutions for evaluation
solutions = json.loads(Path('arc-prize-2024/arc-agi_training_solutions.json').read_text())

# Sample 20 random tasks
task_ids = list(tasks.keys())
random.seed(42)
sample_ids = random.sample(task_ids, min(20, len(task_ids)))

print("="*70)
print(f"TESTING ON {len(sample_ids)} RANDOM TRAINING TASKS")
print("="*70)

sampler = ARCTwoAttemptSampler()
correct = 0
total = 0

for task_id in sample_ids:
    task = tasks[task_id]
    expected_outputs = solutions[task_id]
    
    for test_idx, test_case in enumerate(task.test):
        test_input = test_case.input_grid
        expected = expected_outputs[test_idx]
        
        # Generate attempts
        attempt_1, attempt_2 = sampler.generate_attempts(task.train, test_input)
        
        # Check if either matches
        match_1 = (attempt_1.tolist() == expected)
        match_2 = (attempt_2.tolist() == expected)
        
        total += 1
        if match_1 or match_2:
            correct += 1
            print(f"✓ {task_id}")
        else:
            print(f"✗ {task_id}")

print(f"\n{'='*70}")
accuracy = (correct / total * 100) if total > 0 else 0
print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
print(f"{'='*70}")

# Extrapolate to full 400
estimated_full = correct * (400 / len(sample_ids))
print(f"\nEstimated correct on full 400 tasks: ~{estimated_full:.0f} ({estimated_full/400*100:.1f}%)")
